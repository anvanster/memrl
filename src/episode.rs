#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A coding episode - a single session of work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
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

impl Episode {
    pub fn new(project: String, raw_prompt: String) -> Self {
        Self {
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
        md.push_str(&format!("**Outcome**: {}\n\n", self.outcome.status));

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
}
