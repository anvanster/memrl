// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
//! LLM-based extraction using Anthropic API
//!
//! This module provides intent extraction and session analysis using Claude.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::episode::{Claim, ClaimCategory, TaskType};

/// Anthropic API client
pub struct AnthropicClient {
    api_key: String,
    client: reqwest::Client,
    model: String,
}

/// Extracted intent from a prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedIntent {
    /// Concise summary of the intent
    pub summary: String,
    /// Task type classification
    pub task_type: TaskType,
    /// Domain tags
    pub tags: Vec<String>,
    /// Key entities (files, functions, concepts)
    pub entities: Vec<String>,
    /// Estimated complexity (1-5)
    pub complexity: u8,
    /// Falsifiability + category, piggybacked in the same extract call so
    /// we don't pay a second round trip. `None` if the LLM declined to
    /// produce them (older deployments, parse failure on those fields).
    /// Added in v0.6.2.
    pub claim: Option<Claim>,
}

/// Message for Anthropic API
#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

/// Anthropic API request
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    system: Option<String>,
}

/// Anthropic API response
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY environment variable not set")?;

        Ok(Self {
            api_key,
            client: reqwest::Client::new(),
            model: "claude-3-haiku-20240307".to_string(), // Use Haiku for speed/cost
        })
    }

    /// Create client with a specific model
    pub fn with_model(model: &str) -> Result<Self> {
        let mut client = Self::new()?;
        client.model = model.to_string();
        Ok(client)
    }

    /// Low-level completion call: send a single user message under the
    /// given system prompt and return the assistant's text. Used by
    /// dream-cycle phases (triage / reflect / patterns) that handle their
    /// own JSON parsing. The model is whatever this client was constructed
    /// with — pair with `AnthropicClient::with_model("claude-sonnet-4-6")`
    /// for authorship phases, `with_model("claude-haiku-4-5-20251001")`
    /// for gating.
    pub async fn raw_completion(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<String> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            messages: vec![Message {
                role: "user".to_string(),
                content: user.to_string(),
            }],
            system: Some(system.to_string()),
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send raw completion request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error ({}): {}", status, text);
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .context("Failed to parse raw completion response")?;
        Ok(api_response
            .content
            .into_iter()
            .next()
            .map(|c| c.text)
            .unwrap_or_default())
    }

    /// Extract structured intent from a prompt
    pub async fn extract_intent(&self, prompt: &str) -> Result<ExtractedIntent> {
        let system = r#"You are an expert at analyzing coding task descriptions.
Extract structured information from the user's prompt.

Respond with a JSON object containing:
- summary: A concise 1-2 sentence summary of what the user wants to accomplish
- task_type: One of: bugfix, feature, refactor, test, docs, research, debug, setup, unknown
- tags: Array of relevant domain tags (e.g., "authentication", "database", "frontend", "api")
- entities: Key entities mentioned (files, functions, concepts, technologies)
- complexity: Estimated complexity from 1 (trivial) to 5 (very complex)
- falsifiability: 0.0 to 1.0 — can the central claim of this episode be checked
  against future reality or future code?
    1.0 = specific, testable assertion ("function X always returns Y when Z",
          "approach A is faster than B")
    0.7 = strong directional claim with concrete shape
    0.5 = soft pattern ("usually", "tends to")
    0.0 = pure logistics ("bumped version", "ran migration", "added comment")
- claim_category: one of
    api_contract | performance | structural | conventional |
    workaround | logistics | other
- validity_scope: one of these colon-encoded strings — pick the MOST SPECIFIC
  scope that the central insight is actually true within:
    "forever"                — mathematical or universal truth
                               (e.g. "BFS visits nodes in layers")
    "language:<lang>"        — true at the language level
                               (e.g. "language:rust" for borrow-checker facts)
    "crate:<name>@<version>" — true for a specific library version
                               (e.g. "crate:sqlx@0.8", "crate:tokio")
                               Omit @version for "any version".
    "domain:<tag>"           — true within a problem domain
                               (e.g. "domain:async-rust", "domain:auth-middleware")
    "workaround:<ref>"       — bug-specific, expires when the issue closes
                               (e.g. "workaround:rust-lang/cargo#12345")
    "project"                — bound to THIS codebase only (project conventions,
                               internal APIs, naming choices). Pick this when in
                               doubt — it's the safe default that won't bleed
                               into cross-project retrieval.

Respond ONLY with valid JSON, no other text."#;

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 500,
            messages: vec![Message {
                role: "user".to_string(),
                content: format!("Analyze this coding task:\n\n{}", prompt),
            }],
            system: Some(system.to_string()),
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error ({}): {}", status, text);
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic API response")?;

        let text = api_response
            .content
            .first()
            .map(|c| c.text.as_str())
            .unwrap_or("{}");

        // Parse the JSON response
        let parsed: serde_json::Value =
            serde_json::from_str(text).context("Failed to parse LLM response as JSON")?;

        Ok(ExtractedIntent {
            summary: parsed["summary"].as_str().unwrap_or("").to_string(),
            task_type: parse_task_type(parsed["task_type"].as_str().unwrap_or("unknown")),
            tags: parsed["tags"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            entities: parsed["entities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            complexity: parsed["complexity"].as_u64().unwrap_or(3) as u8,
            claim: parse_claim(&parsed),
        })
    }

    /// Analyze a session transcript and extract key information
    pub async fn analyze_session(&self, transcript: &str) -> Result<SessionAnalysis> {
        let system = r#"You are an expert at analyzing coding session transcripts.
Extract structured information about what happened during the session.

Respond with a JSON object containing:
- summary: A concise summary of what was accomplished
- task_type: One of: bugfix, feature, refactor, test, docs, research, debug, setup, unknown
- outcome: One of: success, partial, failure
- tags: Array of relevant domain tags
- files_modified: Array of files that were modified (based on context)
- errors_resolved: Array of objects with "error" and "resolution" fields for any errors that were fixed
- key_learnings: Array of important insights or patterns from the session
- falsifiability: 0.0 to 1.0 — can the *central insight* of this session be
  checked against future reality or future code?
    1.0 = specific, testable assertion ("function X always returns Y when Z")
    0.7 = strong directional claim with concrete shape
    0.5 = soft pattern ("usually", "tends to")
    0.0 = pure logistics ("bumped version", "ran migration", "added comment")
- claim_category: one of
    api_contract | performance | structural | conventional |
    workaround | logistics | other
- validity_scope: one of these colon-encoded strings — pick the MOST SPECIFIC
  scope that the central insight is actually true within:
    "forever"                — mathematical or universal truth
    "language:<lang>"        — true at the language level (e.g. "language:rust")
    "crate:<name>@<version>" — true for a specific library version
                               (omit @version for "any version")
    "domain:<tag>"           — true within a problem domain
                               (e.g. "domain:async-rust", "domain:auth-middleware")
    "workaround:<ref>"       — bug-specific, expires when the referenced issue closes
    "project"                — bound to THIS codebase only. Pick this when in
                               doubt — it's the safe default that won't bleed
                               into cross-project retrieval.

Respond ONLY with valid JSON, no other text."#;

        // Truncate transcript if too long
        let truncated = if transcript.len() > 10000 {
            format!(
                "{}...\n\n[TRUNCATED - showing first and last portions]\n\n...{}",
                &transcript[..5000],
                &transcript[transcript.len() - 4000..]
            )
        } else {
            transcript.to_string()
        };

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 1000,
            messages: vec![Message {
                role: "user".to_string(),
                content: format!("Analyze this coding session transcript:\n\n{}", truncated),
            }],
            system: Some(system.to_string()),
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error ({}): {}", status, text);
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic API response")?;

        let text = api_response
            .content
            .first()
            .map(|c| c.text.as_str())
            .unwrap_or("{}");

        let parsed: serde_json::Value =
            serde_json::from_str(text).context("Failed to parse LLM response as JSON")?;

        Ok(SessionAnalysis {
            summary: parsed["summary"].as_str().unwrap_or("").to_string(),
            task_type: parse_task_type(parsed["task_type"].as_str().unwrap_or("unknown")),
            outcome: parse_outcome(parsed["outcome"].as_str().unwrap_or("partial")),
            tags: parsed["tags"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            files_modified: parsed["files_modified"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            errors_resolved: parsed["errors_resolved"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            Some(ErrorResolution {
                                error: v["error"].as_str()?.to_string(),
                                resolution: v["resolution"].as_str().map(String::from),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default(),
            key_learnings: parsed["key_learnings"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            claim: parse_claim(&parsed),
        })
    }
}

/// Session analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAnalysis {
    pub summary: String,
    pub task_type: TaskType,
    pub outcome: crate::episode::OutcomeStatus,
    pub tags: Vec<String>,
    pub files_modified: Vec<String>,
    pub errors_resolved: Vec<ErrorResolution>,
    pub key_learnings: Vec<String>,
    /// Falsifiability + category for the session's central claim. Added in v0.6.2.
    /// `None` when the LLM omits the fields.
    #[serde(default)]
    pub claim: Option<Claim>,
}

/// Error resolution pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResolution {
    pub error: String,
    pub resolution: Option<String>,
}

/// Parse task type from string
fn parse_task_type(s: &str) -> TaskType {
    match s.to_lowercase().as_str() {
        "bugfix" | "bug" | "fix" => TaskType::Bugfix,
        "feature" | "feat" => TaskType::Feature,
        "refactor" | "refactoring" => TaskType::Refactor,
        "test" | "testing" => TaskType::Test,
        "docs" | "documentation" => TaskType::Docs,
        "research" => TaskType::Research,
        "debug" | "debugging" => TaskType::Debug,
        "setup" | "config" | "configuration" => TaskType::Setup,
        _ => TaskType::Unknown,
    }
}

/// Parse outcome status from string
fn parse_outcome(s: &str) -> crate::episode::OutcomeStatus {
    match s.to_lowercase().as_str() {
        "success" | "complete" | "done" => crate::episode::OutcomeStatus::Success,
        "partial" | "incomplete" => crate::episode::OutcomeStatus::Partial,
        "failure" | "failed" => crate::episode::OutcomeStatus::Failure,
        _ => crate::episode::OutcomeStatus::Partial,
    }
}

fn parse_claim_category(s: &str) -> ClaimCategory {
    match s.to_lowercase().replace('-', "_").as_str() {
        "api_contract" => ClaimCategory::ApiContract,
        "performance" | "perf" => ClaimCategory::Performance,
        "structural" => ClaimCategory::Structural,
        "conventional" | "convention" => ClaimCategory::Conventional,
        "workaround" => ClaimCategory::Workaround,
        "logistics" => ClaimCategory::Logistics,
        _ => ClaimCategory::Other,
    }
}

/// Parse the LLM-supplied `validity_scope` string into a `ValidityScope`
/// (v0.10.3). The colon-encoded format mirrors what the prompt asks for:
///
///   "forever"                → ValidityScope::Forever
///   "language:rust"          → Language { name: "rust" }
///   "crate:sqlx@0.8"         → Crate { name: "sqlx", version: "0.8" }
///   "crate:tokio"            → Crate { name: "tokio", version: "" }
///   "domain:async-rust"      → Domain { tag: "async-rust" }
///   "workaround:foo#123"     → Workaround { ref_: "foo#123", expires: None }
///   "project"                → Project { name: "" }     (caller fills name)
///
/// Returns `None` on unrecognised input — the rest of the Claim still
/// loads. The `Project { name: "" }` placeholder is a contract with the
/// caller: capture.rs patches the empty name with the episode's actual
/// project so llm.rs stays project-agnostic.
pub(crate) fn parse_validity_scope(s: &str) -> Option<crate::episode::ValidityScope> {
    use crate::episode::ValidityScope;
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if s.eq_ignore_ascii_case("forever") {
        return Some(ValidityScope::Forever);
    }
    if s.eq_ignore_ascii_case("project") {
        return Some(ValidityScope::Project {
            name: String::new(),
        });
    }
    let (kind, value) = s.split_once(':')?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    match kind.trim().to_lowercase().as_str() {
        "language" => Some(ValidityScope::Language {
            name: value.to_string(),
        }),
        "crate" => {
            // Optional @version after the name.
            let (name, version) = match value.split_once('@') {
                Some((n, v)) => (n.trim().to_string(), v.trim().to_string()),
                None => (value.to_string(), String::new()),
            };
            if name.is_empty() {
                return None;
            }
            Some(ValidityScope::Crate { name, version })
        }
        "domain" => Some(ValidityScope::Domain {
            tag: value.to_string(),
        }),
        "workaround" => Some(ValidityScope::Workaround {
            ref_: value.to_string(),
            expires: None,
        }),
        "project" => Some(ValidityScope::Project {
            name: value.to_string(),
        }),
        _ => None,
    }
}

/// Extract `claim` from the JSON response. Returns `None` when the LLM
/// didn't produce either field (older deployments, partial outputs) — the
/// rest of `ExtractedIntent` still loads.
///
/// As of v0.10.3 the LLM is also asked to suggest a `validity_scope`.
/// When the model emits a recognisable colon-string, it gets parsed
/// here; if the value is a `Project { name: "" }` placeholder the
/// caller (capture.rs) is responsible for filling in the project
/// name before the Claim is persisted.
fn parse_claim(parsed: &serde_json::Value) -> Option<Claim> {
    let falsifiability = parsed.get("falsifiability").and_then(|v| v.as_f64())?;
    let category = parsed
        .get("claim_category")
        .and_then(|v| v.as_str())
        .map(parse_claim_category)
        .unwrap_or(ClaimCategory::Other);
    let validity_scope = parsed
        .get("validity_scope")
        .and_then(|v| v.as_str())
        .and_then(parse_validity_scope);
    Some(Claim {
        falsifiability: (falsifiability as f32).clamp(0.0, 1.0),
        category,
        validity_scope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_task_type() {
        assert_eq!(parse_task_type("bugfix"), TaskType::Bugfix);
        assert_eq!(parse_task_type("Feature"), TaskType::Feature);
        assert_eq!(parse_task_type("unknown"), TaskType::Unknown);
    }

    #[test]
    fn test_parse_outcome() {
        use crate::episode::OutcomeStatus;
        assert_eq!(parse_outcome("success"), OutcomeStatus::Success);
        assert_eq!(parse_outcome("partial"), OutcomeStatus::Partial);
        assert_eq!(parse_outcome("failure"), OutcomeStatus::Failure);
    }

    #[test]
    fn parse_validity_scope_forever() {
        use crate::episode::ValidityScope;
        assert!(matches!(
            parse_validity_scope("forever"),
            Some(ValidityScope::Forever)
        ));
        assert!(matches!(
            parse_validity_scope("FOREVER"),
            Some(ValidityScope::Forever)
        ));
    }

    #[test]
    fn parse_validity_scope_language() {
        use crate::episode::ValidityScope;
        let s = parse_validity_scope("language:rust").unwrap();
        match s {
            ValidityScope::Language { name } => assert_eq!(name, "rust"),
            _ => panic!("expected Language"),
        }
    }

    #[test]
    fn parse_validity_scope_crate_with_version() {
        use crate::episode::ValidityScope;
        let s = parse_validity_scope("crate:sqlx@0.8").unwrap();
        match s {
            ValidityScope::Crate { name, version } => {
                assert_eq!(name, "sqlx");
                assert_eq!(version, "0.8");
            }
            _ => panic!("expected Crate"),
        }
    }

    #[test]
    fn parse_validity_scope_crate_without_version() {
        use crate::episode::ValidityScope;
        let s = parse_validity_scope("crate:tokio").unwrap();
        match s {
            ValidityScope::Crate { name, version } => {
                assert_eq!(name, "tokio");
                assert_eq!(version, "");
            }
            _ => panic!("expected Crate"),
        }
    }

    #[test]
    fn parse_validity_scope_domain() {
        use crate::episode::ValidityScope;
        let s = parse_validity_scope("domain:async-rust").unwrap();
        match s {
            ValidityScope::Domain { tag } => assert_eq!(tag, "async-rust"),
            _ => panic!("expected Domain"),
        }
    }

    #[test]
    fn parse_validity_scope_workaround() {
        use crate::episode::ValidityScope;
        let s = parse_validity_scope("workaround:rust-lang/cargo#12345").unwrap();
        match s {
            ValidityScope::Workaround { ref_, expires } => {
                assert_eq!(ref_, "rust-lang/cargo#12345");
                assert!(expires.is_none());
            }
            _ => panic!("expected Workaround"),
        }
    }

    #[test]
    fn parse_validity_scope_project_uses_placeholder() {
        use crate::episode::ValidityScope;
        let s = parse_validity_scope("project").unwrap();
        match s {
            ValidityScope::Project { name } => assert_eq!(name, ""),
            _ => panic!("expected Project"),
        }
    }

    #[test]
    fn parse_validity_scope_malformed_returns_none() {
        assert!(parse_validity_scope("").is_none());
        assert!(parse_validity_scope("bogus").is_none());
        assert!(parse_validity_scope("language:").is_none());
        assert!(parse_validity_scope("crate:").is_none());
        assert!(parse_validity_scope("crate:@1.0").is_none());
    }

    #[test]
    fn parse_claim_picks_up_validity_scope() {
        use crate::episode::ValidityScope;
        let json: serde_json::Value = serde_json::from_str(
            r#"{ "falsifiability": 0.8, "claim_category": "api_contract",
                 "validity_scope": "language:rust" }"#,
        )
        .unwrap();
        let claim = parse_claim(&json).unwrap();
        match claim.validity_scope {
            Some(ValidityScope::Language { name }) => assert_eq!(name, "rust"),
            other => panic!("expected Language, got {:?}", other),
        }
    }

    #[test]
    fn parse_claim_without_validity_scope_field() {
        // Legacy LLM output (no validity_scope field) — claim still loads,
        // scope stays None.
        let json: serde_json::Value =
            serde_json::from_str(r#"{ "falsifiability": 0.8, "claim_category": "api_contract" }"#)
                .unwrap();
        let claim = parse_claim(&json).unwrap();
        assert!(claim.validity_scope.is_none());
    }
}
