// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Current on-disk schema version for `Episode`. Bump when adding fields that
/// cannot be filled by serde defaults — then add a `migrate_v{n}_to_v{n+1}`
/// arm in `Episode::migrate`. See the v0.5.4 roadmap entry.
pub const CURRENT_EPISODE_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    // Old episodes on disk pre-date the schema_version field. They are
    // implicitly v1 — the shape that has been stable since v0.4.0.
    1
}

/// A coding episode - a single session of work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Schema version. Added in v0.4.5. Missing on older files → defaults to 1.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub id: String,
    pub timestamp_start: DateTime<Utc>,
    pub timestamp_end: DateTime<Utc>,
    pub project: String,
    pub intent: Intent,
    pub context: Context,
    pub outcome: Outcome,
    pub utility: Utility,
    #[serde(default)]
    pub retrieval_history: Vec<RetrievalRecord>,
    /// Groups episodes in the same logical session (multi-step task)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Explicit links to related episodes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_episodes: Vec<RelatedEpisode>,
}

impl Episode {
    /// Walk this episode forward through the migration chain to
    /// `CURRENT_EPISODE_VERSION`. Idempotent — already-current episodes
    /// pass through unchanged. Each `v_n → v_{n+1}` step is its own match arm.
    ///
    /// Episodes from a *future* version (downgrade scenario) bail; the safer
    /// failure mode is "refuse to load" rather than "drop fields silently".
    pub fn migrate(self) -> Result<Self> {
        // No real upgrade paths exist yet — every episode on disk is already
        // v1. When CURRENT_EPISODE_VERSION bumps to 2, restore the
        // `while self.schema_version < CURRENT_EPISODE_VERSION { ... }` loop
        // with a `1 => migrate_v1_to_v2(self)?` arm. Until then there's
        // nothing to walk forward.
        if self.schema_version < CURRENT_EPISODE_VERSION {
            bail!(
                "no migration from episode schema v{} to v{}",
                self.schema_version,
                CURRENT_EPISODE_VERSION
            );
        }
        if self.schema_version > CURRENT_EPISODE_VERSION {
            bail!(
                "episode schema_version {} is newer than this binary supports (v{}); refusing to load",
                self.schema_version,
                CURRENT_EPISODE_VERSION
            );
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// The raw first prompt from the user
    pub raw_prompt: String,
    /// LLM-extracted intent summary
    pub extracted_intent: String,
    /// Task type classification
    pub task_type: TaskType,
    /// Domain tags
    pub domain: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Bugfix,
    Feature,
    Refactor,
    Test,
    Docs,
    Research,
    Debug,
    Setup,
    Unknown,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Bugfix => write!(f, "bugfix"),
            TaskType::Feature => write!(f, "feature"),
            TaskType::Refactor => write!(f, "refactor"),
            TaskType::Test => write!(f, "test"),
            TaskType::Docs => write!(f, "docs"),
            TaskType::Research => write!(f, "research"),
            TaskType::Debug => write!(f, "debug"),
            TaskType::Setup => write!(f, "setup"),
            TaskType::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub files_read: Vec<String>,
    pub files_modified: Vec<String>,
    pub tools_invoked: Vec<String>,
    pub errors_encountered: Vec<ErrorRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub error_type: String,
    pub message: String,
    pub resolved: bool,
    pub resolution: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub status: OutcomeStatus,
    pub tests_before: Option<TestResults>,
    pub tests_after: Option<TestResults>,
    pub commit_sha: Option<String>,
    pub pr_number: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutcomeStatus {
    Success,
    Partial,
    Failure,
}

impl std::fmt::Display for OutcomeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutcomeStatus::Success => write!(f, "✅ success"),
            OutcomeStatus::Partial => write!(f, "⚠️ partial"),
            OutcomeStatus::Failure => write!(f, "❌ failure"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub passed: u32,
    pub failed: u32,
    pub skipped: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Utility {
    /// Learned utility score (0.0 - 1.0)
    pub score: Option<f32>,
    /// Number of times this episode was retrieved
    pub retrieval_count: u32,
    /// Number of times marked as helpful
    pub helpful_count: u32,
}

impl Utility {
    /// Calculate utility score using Wilson score interval (lower bound)
    /// This handles uncertainty for low-sample episodes
    pub fn calculate_score(&self) -> f32 {
        let n = self.retrieval_count as f64;
        if n == 0.0 {
            return 0.5; // Default for unretreived episodes
        }

        let p = self.helpful_count as f64 / n;
        let z = 1.96; // 95% confidence

        // Wilson score lower bound
        let score = (p + z * z / (2.0 * n) - z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt())
            / (1.0 + z * z / n);

        score as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRecord {
    pub timestamp: DateTime<Utc>,
    pub project: String,
    pub task_description: String,
    pub was_helpful: Option<bool>,
}

/// A link to a related episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedEpisode {
    pub id: String,
    pub relationship: EpisodeRelation,
}

/// The type of relationship between two episodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeRelation {
    /// Same task, next step
    Continuation,
    /// This episode was needed before the other
    Prerequisite,
    /// Different approach to the same problem
    Alternative,
    /// Loosely related
    Related,
}

impl Episode {
    pub fn new(project: String, raw_prompt: String) -> Self {
        Self {
            schema_version: CURRENT_EPISODE_VERSION,
            id: uuid::Uuid::new_v4().to_string(),
            timestamp_start: Utc::now(),
            timestamp_end: Utc::now(),
            project,
            intent: Intent {
                raw_prompt,
                extracted_intent: String::new(),
                task_type: TaskType::Unknown,
                domain: vec![],
            },
            context: Context {
                files_read: vec![],
                files_modified: vec![],
                tools_invoked: vec![],
                errors_encountered: vec![],
            },
            outcome: Outcome {
                status: OutcomeStatus::Partial,
                tests_before: None,
                tests_after: None,
                commit_sha: None,
                pr_number: None,
            },
            utility: Utility::default(),
            retrieval_history: vec![],
            session_id: None,
            related_episodes: vec![],
        }
    }

    /// Convert to markdown format for human-readable storage
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!(
            "# Episode: {}\n\n",
            if self.intent.extracted_intent.is_empty() {
                &self.intent.raw_prompt
            } else {
                &self.intent.extracted_intent
            }
        ));

        md.push_str(&format!("**ID**: {}\n", &self.id[..8]));
        md.push_str(&format!(
            "**Date**: {}\n",
            self.timestamp_start.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        md.push_str(&format!("**Project**: {}\n", self.project));
        md.push_str(&format!("**Outcome**: {}\n", self.outcome.status));
        if let Some(sid) = &self.session_id {
            md.push_str(&format!("**Session**: {}\n", &sid[..8.min(sid.len())]));
        }
        md.push('\n');

        md.push_str("## Intent\n\n");
        md.push_str(&format!("{}\n\n", self.intent.raw_prompt));

        md.push_str("## Context\n\n");
        md.push_str("### Files Read\n");
        if self.context.files_read.is_empty() {
            md.push_str("- None\n");
        } else {
            for f in &self.context.files_read {
                md.push_str(&format!("- {}\n", f));
            }
        }
        md.push_str("\n");

        md.push_str("### Files Modified\n");
        if self.context.files_modified.is_empty() {
            md.push_str("- None\n");
        } else {
            for f in &self.context.files_modified {
                md.push_str(&format!("- {}\n", f));
            }
        }
        md.push_str("\n");

        md.push_str("### Commands/Tools Used\n");
        if self.context.tools_invoked.is_empty() {
            md.push_str("- None\n");
        } else {
            for t in &self.context.tools_invoked {
                md.push_str(&format!("- {}\n", t));
            }
        }
        md.push_str("\n");

        if !self.context.errors_encountered.is_empty() {
            md.push_str("## Errors → Resolutions\n\n");
            md.push_str("| Error | Resolution |\n");
            md.push_str("|-------|------------|\n");
            for e in &self.context.errors_encountered {
                let resolution = e.resolution.as_deref().unwrap_or("unresolved");
                md.push_str(&format!("| {} | {} |\n", e.message, resolution));
            }
            md.push_str("\n");
        }

        md.push_str("## Tags\n\n");
        md.push_str(&format!("{}\n\n", self.intent.domain.join(", ")));

        if !self.related_episodes.is_empty() {
            md.push_str("## Related Episodes\n\n");
            for rel in &self.related_episodes {
                let rel_type = match rel.relationship {
                    EpisodeRelation::Continuation => "continuation",
                    EpisodeRelation::Prerequisite => "prerequisite",
                    EpisodeRelation::Alternative => "alternative",
                    EpisodeRelation::Related => "related",
                };
                md.push_str(&format!(
                    "- {} ({})\n",
                    &rel.id[..8.min(rel.id.len())],
                    rel_type
                ));
            }
            md.push('\n');
        }

        if !self.retrieval_history.is_empty() {
            md.push_str("## Retrieval History\n\n");
            md.push_str("| Date | Project | Task | Helpful |\n");
            md.push_str("|------|---------|------|--------|\n");
            for r in &self.retrieval_history {
                let helpful = match r.was_helpful {
                    Some(true) => "✅",
                    Some(false) => "❌",
                    None => "?",
                };
                md.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    r.timestamp.format("%Y-%m-%d"),
                    r.project,
                    r.task_description,
                    helpful
                ));
            }
        }

        md
    }

    /// Parse from markdown (basic implementation)
    pub fn from_markdown(content: &str, _file_path: &std::path::Path) -> anyhow::Result<Self> {
        // Basic parsing - extract key fields from markdown
        // This is a simplified implementation

        let mut episode = Episode::new(
            extract_field(content, "**Project**:").unwrap_or_default(),
            extract_section(content, "## Intent").unwrap_or_default(),
        );

        if let Some(id) = extract_field(content, "**ID**:") {
            episode.id = id;
        }

        if let Some(outcome) = extract_field(content, "**Outcome**:") {
            episode.outcome.status = match outcome.to_lowercase().as_str() {
                s if s.contains("success") => OutcomeStatus::Success,
                s if s.contains("partial") => OutcomeStatus::Partial,
                s if s.contains("failure") => OutcomeStatus::Failure,
                _ => OutcomeStatus::Partial,
            };
        }

        if let Some(tags) = extract_section(content, "## Tags") {
            episode.intent.domain = tags
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        Ok(episode)
    }
}

fn extract_field(content: &str, field: &str) -> Option<String> {
    for line in content.lines() {
        if line.starts_with(field) {
            return Some(line.trim_start_matches(field).trim().to_string());
        }
    }
    None
}

fn extract_section(content: &str, header: &str) -> Option<String> {
    let mut in_section = false;
    let mut section_content = String::new();

    for line in content.lines() {
        if line.starts_with(header) {
            in_section = true;
            continue;
        }
        if in_section {
            if line.starts_with("## ") {
                break;
            }
            section_content.push_str(line);
            section_content.push('\n');
        }
    }

    if section_content.is_empty() {
        None
    } else {
        Some(section_content.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utility_score_calculation() {
        // No retrievals = 0.5
        let utility = Utility::default();
        assert!((utility.calculate_score() - 0.5).abs() < 0.01);

        // 10 retrievals, 10 helpful = high score
        let utility = Utility {
            score: None,
            retrieval_count: 10,
            helpful_count: 10,
        };
        assert!(utility.calculate_score() > 0.7);

        // 10 retrievals, 0 helpful = low score
        let utility = Utility {
            score: None,
            retrieval_count: 10,
            helpful_count: 0,
        };
        assert!(utility.calculate_score() < 0.3);
    }

    #[test]
    fn test_backward_compat_deserialization() {
        // Old JSON without session_id and related_episodes should deserialize fine
        let json = r#"{
            "id": "test-1234-5678-abcd",
            "timestamp_start": "2026-03-01T10:00:00Z",
            "timestamp_end": "2026-03-01T11:00:00Z",
            "project": "test-proj",
            "intent": {
                "raw_prompt": "fix the bug",
                "extracted_intent": "fix auth bug",
                "task_type": "bugfix",
                "domain": ["rust", "auth"]
            },
            "context": {
                "files_read": [],
                "files_modified": ["src/auth.rs"],
                "tools_invoked": [],
                "errors_encountered": []
            },
            "outcome": {
                "status": "success",
                "tests_before": null,
                "tests_after": null,
                "commit_sha": null,
                "pr_number": null
            },
            "utility": {
                "score": null,
                "retrieval_count": 3,
                "helpful_count": 2
            }
        }"#;
        let ep: Episode = serde_json::from_str(json).unwrap();
        assert_eq!(ep.id, "test-1234-5678-abcd");
        assert!(ep.session_id.is_none());
        assert!(ep.related_episodes.is_empty());
        assert!(ep.retrieval_history.is_empty());
        // Older files lack `schema_version`; serde default puts them at v1.
        assert_eq!(ep.schema_version, 1);
    }

    #[test]
    fn migrate_passthroughs_current_version() {
        let ep = Episode::new("p".to_string(), "test".to_string());
        assert_eq!(ep.schema_version, CURRENT_EPISODE_VERSION);
        let migrated = ep.clone().migrate().unwrap();
        assert_eq!(migrated.schema_version, CURRENT_EPISODE_VERSION);
    }

    #[test]
    fn migrate_bails_on_future_version() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.schema_version = CURRENT_EPISODE_VERSION + 1;
        let err = ep.migrate().unwrap_err();
        assert!(
            err.to_string().contains("newer than this binary supports"),
            "got: {err}"
        );
    }

    #[test]
    fn new_episode_has_current_schema_version() {
        let ep = Episode::new("p".to_string(), "test".to_string());
        assert_eq!(ep.schema_version, CURRENT_EPISODE_VERSION);
    }

    #[test]
    fn schema_version_roundtrips_through_serde() {
        let mut ep = Episode::new("p".to_string(), "test".to_string());
        ep.schema_version = 1;
        let json = serde_json::to_string(&ep).unwrap();
        assert!(
            json.contains("\"schema_version\":1"),
            "schema_version not in output: {json}"
        );
        let parsed: Episode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema_version, 1);
    }

    #[test]
    fn test_session_fields_serialization() {
        let mut ep = Episode::new("test".to_string(), "prompt".to_string());
        ep.session_id = Some("session-abc".to_string());
        ep.related_episodes.push(RelatedEpisode {
            id: "related-123".to_string(),
            relationship: EpisodeRelation::Continuation,
        });

        let json = serde_json::to_string(&ep).unwrap();
        let parsed: Episode = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.session_id, Some("session-abc".to_string()));
        assert_eq!(parsed.related_episodes.len(), 1);
        assert_eq!(
            parsed.related_episodes[0].relationship,
            EpisodeRelation::Continuation
        );
    }

    #[test]
    fn test_session_in_markdown() {
        let mut ep = Episode::new("test".to_string(), "prompt".to_string());
        ep.session_id = Some("abcdef12-3456-7890".to_string());
        ep.related_episodes.push(RelatedEpisode {
            id: "related-123456789".to_string(),
            relationship: EpisodeRelation::Prerequisite,
        });

        let md = ep.to_markdown();
        assert!(md.contains("**Session**: abcdef12"));
        assert!(md.contains("related- (prerequisite)"));
    }
}
