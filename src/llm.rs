#![allow(dead_code)]
//! LLM-based extraction using Anthropic API
//!
//! This module provides intent extraction and session analysis using Claude.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::episode::TaskType;

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
}
