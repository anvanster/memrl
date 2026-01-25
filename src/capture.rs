#![allow(dead_code)]
use anyhow::{Context, Result};
use chrono::Utc;
use git2::Repository;
use regex::Regex;
use std::path::PathBuf;

use crate::config::Config;
use crate::episode::{Context as EpisodeContext, Episode, ErrorRecord, OutcomeStatus, TaskType};
use crate::llm::{AnthropicClient, SessionAnalysis};
use crate::store::EpisodeStore;

/// Run the capture command
pub async fn run(
    session: Option<PathBuf>,
    project: Option<PathBuf>,
    extract_intent: bool,
    capture_diff: bool,
    _config: &Config,
) -> Result<()> {
    let project_dir = project.unwrap_or_else(|| std::env::current_dir().unwrap());
    let project_name = extract_project_name(&project_dir);

    println!("📝 Capturing episode for project: {}", project_name);

    // Parse session transcript if provided
    let (raw_prompt, context, outcome_status) = if let Some(session_path) = &session {
        let transcript = std::fs::read_to_string(session_path)
            .with_context(|| format!("Failed to read session file: {}", session_path.display()))?;
        let prompt = extract_first_prompt(&transcript);
        let ctx = extract_context_from_transcript(&transcript, &project_dir);
        let status = determine_outcome(&transcript);
        (prompt, ctx, status)
    } else {
        // Interactive mode: ask user for information
        println!("No session transcript provided. Creating episode interactively.");
        let prompt = prompt_user("Enter the main task/intent for this session:")?;
        let context = EpisodeContext {
            files_read: vec![],
            files_modified: get_modified_files(&project_dir)?,
            tools_invoked: vec![],
            errors_encountered: vec![],
        };
        (prompt, context, OutcomeStatus::Partial)
    };

    // Create the episode
    let mut episode = Episode::new(project_name.clone(), raw_prompt.clone());
    episode.timestamp_end = Utc::now();
    episode.context = context;
    episode.outcome.status = outcome_status;

    // Extract intent with classification
    if extract_intent {
        // Try LLM-based extraction first
        match extract_intent_with_llm(&raw_prompt, session.as_ref()).await {
            Ok(analysis) => {
                println!("🤖 Using LLM-based intent extraction...");
                episode.intent.extracted_intent = analysis.summary;
                episode.intent.task_type = analysis.task_type;
                episode.intent.domain = analysis.tags;
                if !analysis.files_modified.is_empty() {
                    episode
                        .context
                        .files_modified
                        .extend(analysis.files_modified);
                    episode.context.files_modified.sort();
                    episode.context.files_modified.dedup();
                }
                for err in analysis.errors_resolved {
                    episode.context.errors_encountered.push(ErrorRecord {
                        error_type: "runtime".to_string(),
                        message: err.error,
                        resolved: err.resolution.is_some(),
                        resolution: err.resolution,
                    });
                }
                // Use LLM-determined outcome if available
                episode.outcome.status = analysis.outcome;
            }
            Err(e) => {
                println!("⚠️  LLM extraction failed ({}), using simple extraction", e);
                episode.intent.extracted_intent = extract_intent_simple(&raw_prompt);
                episode.intent.task_type = classify_task_type(&raw_prompt);
                episode.intent.domain = extract_domain_tags(&raw_prompt, &episode.context);
            }
        }
    } else {
        // Always do basic extraction
        episode.intent.extracted_intent = extract_intent_simple(&raw_prompt);
        episode.intent.task_type = classify_task_type(&raw_prompt);
        episode.intent.domain = extract_domain_tags(&raw_prompt, &episode.context);
    }

    // Try to get git info
    if let Ok(repo) = Repository::open(&project_dir) {
        if let Ok(head) = repo.head() {
            if let Some(oid) = head.target() {
                episode.outcome.commit_sha = Some(oid.to_string()[..8].to_string());
            }
        }
    }

    // Save the episode
    let store = EpisodeStore::new()?;
    let episode_path = store.save(&episode)?;
    println!("✅ Episode saved: {}", episode_path.display());

    // Capture git diff if requested
    if capture_diff {
        if let Ok(diff) = capture_git_diff(&project_dir) {
            if !diff.is_empty() {
                let diff_path = store.save_diff(&episode, &diff)?;
                println!("📄 Diff saved: {}", diff_path.display());
            }
        }
    }

    // Display summary
    println!("\n📋 Episode Summary:");
    println!("   ID: {}", &episode.id[..8]);
    println!("   Intent: {}", episode.intent.extracted_intent);
    println!("   Type: {}", episode.intent.task_type);
    println!("   Tags: {}", episode.intent.domain.join(", "));
    println!("   Outcome: {}", episode.outcome.status);
    if !episode.context.files_modified.is_empty() {
        println!(
            "   Files modified: {}",
            episode.context.files_modified.len()
        );
    }

    Ok(())
}

/// Extract project name from path
fn extract_project_name(path: &PathBuf) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Extract the first user prompt from a transcript
fn extract_first_prompt(transcript: &str) -> String {
    // Look for patterns like "Human:", "User:", or first paragraph
    let patterns = [
        r"(?i)^Human:\s*(.+?)(?:\n\n|\nAssistant:|\z)",
        r"(?i)^User:\s*(.+?)(?:\n\n|\nAssistant:|\z)",
        r"(?i)^>\s*(.+?)(?:\n\n|\z)",
    ];

    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(caps) = re.captures(transcript) {
                if let Some(m) = caps.get(1) {
                    return m.as_str().trim().to_string();
                }
            }
        }
    }

    // Fall back to first non-empty line
    transcript
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .to_string()
}

/// Extract context from transcript
fn extract_context_from_transcript(transcript: &str, project_dir: &PathBuf) -> EpisodeContext {
    EpisodeContext {
        files_read: extract_file_reads(transcript),
        files_modified: get_modified_files(project_dir).unwrap_or_default(),
        tools_invoked: extract_tool_calls(transcript),
        errors_encountered: extract_errors(transcript),
    }
}

/// Extract files that were read (mentioned in transcript)
fn extract_file_reads(transcript: &str) -> Vec<String> {
    let mut files = Vec::new();

    // Look for file path patterns
    let patterns = [
        r#"(?:Read|Reading|read|reading)\s+[`'"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`'"]?"#,
        r#"(?:file|File)\s+[`'"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`'"]?"#,
        r#"(?:cat|less|head|tail)\s+([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)"#,
    ];

    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            for caps in re.captures_iter(transcript) {
                if let Some(m) = caps.get(1) {
                    files.push(m.as_str().to_string());
                }
            }
        }
    }

    files.sort();
    files.dedup();
    files
}

/// Get modified files from git
fn get_modified_files(project_dir: &PathBuf) -> Result<Vec<String>> {
    let mut files = Vec::new();

    if let Ok(repo) = Repository::open(project_dir) {
        // Get status of working directory
        let statuses = repo.statuses(None)?;

        for entry in statuses.iter() {
            if let Some(path) = entry.path() {
                let status = entry.status();
                if status.is_wt_modified()
                    || status.is_wt_new()
                    || status.is_index_modified()
                    || status.is_index_new()
                {
                    files.push(path.to_string());
                }
            }
        }
    }

    files.sort();
    files.dedup();
    Ok(files)
}

/// Extract tool calls from transcript
fn extract_tool_calls(transcript: &str) -> Vec<String> {
    let mut tools = Vec::new();

    // Common patterns for tool invocations
    let patterns = [
        r#"(?:Running|Executing|run|exec):\s*[`'"]?([a-zA-Z0-9_\-]+)"#,
        r#"(?:cargo|npm|yarn|pip|python|node|git)\s+([a-zA-Z0-9_\-]+)"#,
        r#"(?:\$|>)\s*([a-zA-Z0-9_\-]+)\s"#,
    ];

    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            for caps in re.captures_iter(transcript) {
                if let Some(m) = caps.get(0) {
                    tools.push(m.as_str().trim().to_string());
                }
            }
        }
    }

    tools.sort();
    tools.dedup();
    tools
}

/// Extract errors from transcript
fn extract_errors(transcript: &str) -> Vec<ErrorRecord> {
    let mut errors = Vec::new();

    // Look for error patterns
    let error_patterns = [
        r"(?i)error(?:\[E\d+\])?:\s*(.+?)(?:\n|$)",
        r"(?i)failed:\s*(.+?)(?:\n|$)",
        r"(?i)exception:\s*(.+?)(?:\n|$)",
        r"(?i)panic:\s*(.+?)(?:\n|$)",
    ];

    for pattern in error_patterns {
        if let Ok(re) = Regex::new(pattern) {
            for caps in re.captures_iter(transcript) {
                if let Some(m) = caps.get(1) {
                    let message = m.as_str().trim().to_string();
                    // Check if this error was resolved (simple heuristic)
                    let resolved = transcript.contains("fixed")
                        || transcript.contains("resolved")
                        || transcript.contains("success");

                    errors.push(ErrorRecord {
                        error_type: "unknown".to_string(),
                        message,
                        resolved,
                        resolution: None,
                    });
                }
            }
        }
    }

    errors
}

/// Determine outcome from transcript
fn determine_outcome(transcript: &str) -> OutcomeStatus {
    let lower = transcript.to_lowercase();

    // Check for success indicators
    let success_indicators = [
        "tests pass",
        "all tests pass",
        "successfully",
        "completed successfully",
        "build successful",
        "✅",
    ];
    let failure_indicators = [
        "failed",
        "error",
        "tests fail",
        "build failed",
        "❌",
        "panic",
    ];

    let success_count = success_indicators
        .iter()
        .filter(|&s| lower.contains(s))
        .count();
    let failure_count = failure_indicators
        .iter()
        .filter(|&s| lower.contains(s))
        .count();

    if success_count > failure_count {
        OutcomeStatus::Success
    } else if failure_count > success_count {
        OutcomeStatus::Failure
    } else {
        OutcomeStatus::Partial
    }
}

/// Simple intent extraction without LLM
fn extract_intent_simple(prompt: &str) -> String {
    // Take first sentence or first 100 chars
    let first_sentence = prompt.split('.').next().unwrap_or(prompt);
    if first_sentence.len() <= 100 {
        first_sentence.trim().to_string()
    } else {
        format!("{}...", &first_sentence[..97].trim())
    }
}

/// Classify task type from prompt
fn classify_task_type(prompt: &str) -> TaskType {
    let lower = prompt.to_lowercase();

    if lower.contains("fix") || lower.contains("bug") || lower.contains("broken") {
        TaskType::Bugfix
    } else if lower.contains("test") || lower.contains("spec") {
        TaskType::Test
    } else if lower.contains("refactor") || lower.contains("clean") || lower.contains("improve") {
        TaskType::Refactor
    } else if lower.contains("doc") || lower.contains("readme") || lower.contains("comment") {
        TaskType::Docs
    } else if lower.contains("debug") || lower.contains("investigate") || lower.contains("why") {
        TaskType::Debug
    } else if lower.contains("setup") || lower.contains("install") || lower.contains("config") {
        TaskType::Setup
    } else if lower.contains("research") || lower.contains("explore") || lower.contains("learn") {
        TaskType::Research
    } else if lower.contains("add")
        || lower.contains("implement")
        || lower.contains("create")
        || lower.contains("build")
        || lower.contains("feature")
    {
        TaskType::Feature
    } else {
        TaskType::Unknown
    }
}

/// Extract domain tags from prompt and context
fn extract_domain_tags(prompt: &str, context: &EpisodeContext) -> Vec<String> {
    let mut tags = Vec::new();
    let combined = format!(
        "{} {}",
        prompt.to_lowercase(),
        context.files_modified.join(" ").to_lowercase()
    );

    // Language/framework tags
    let domain_keywords = [
        ("rust", "rust"),
        ("python", "python"),
        ("javascript", "javascript"),
        ("typescript", "typescript"),
        (".rs", "rust"),
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        ("react", "react"),
        ("vue", "vue"),
        ("angular", "angular"),
        ("async", "async"),
        ("api", "api"),
        ("database", "database"),
        ("sql", "sql"),
        ("test", "testing"),
        ("cli", "cli"),
        ("web", "web"),
        ("ui", "ui"),
        ("auth", "auth"),
        ("security", "security"),
        ("performance", "performance"),
        ("docker", "docker"),
        ("ci", "ci-cd"),
        ("deploy", "deployment"),
    ];

    for (keyword, tag) in domain_keywords {
        if combined.contains(keyword) {
            tags.push(tag.to_string());
        }
    }

    tags.sort();
    tags.dedup();
    tags
}

/// Capture git diff
fn capture_git_diff(project_dir: &PathBuf) -> Result<String> {
    let repo = Repository::open(project_dir)?;
    let mut diff_output = String::new();

    // Get diff of working directory
    let diff = repo.diff_index_to_workdir(None, None)?;

    diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
        let prefix = match line.origin() {
            '+' => "+",
            '-' => "-",
            ' ' => " ",
            _ => "",
        };
        if let Ok(content) = std::str::from_utf8(line.content()) {
            diff_output.push_str(prefix);
            diff_output.push_str(content);
        }
        true
    })?;

    // Also get staged changes
    let head = repo.head()?.peel_to_tree()?;
    let staged_diff = repo.diff_tree_to_index(Some(&head), None, None)?;

    staged_diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
        let prefix = match line.origin() {
            '+' => "+",
            '-' => "-",
            ' ' => " ",
            _ => "",
        };
        if let Ok(content) = std::str::from_utf8(line.content()) {
            diff_output.push_str(prefix);
            diff_output.push_str(content);
        }
        true
    })?;

    Ok(diff_output)
}

/// Prompt user for input
fn prompt_user(message: &str) -> Result<String> {
    println!("{}", message);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

/// Extract intent using LLM
async fn extract_intent_with_llm(
    prompt: &str,
    session_path: Option<&PathBuf>,
) -> Result<SessionAnalysis> {
    let client = AnthropicClient::new()?;

    // If we have a session file, analyze the whole session
    if let Some(path) = session_path {
        let transcript = std::fs::read_to_string(path)?;
        client.analyze_session(&transcript).await
    } else {
        // Just analyze the prompt
        let intent = client.extract_intent(prompt).await?;

        // Convert ExtractedIntent to SessionAnalysis
        Ok(SessionAnalysis {
            summary: intent.summary,
            task_type: intent.task_type,
            outcome: OutcomeStatus::Partial, // Unknown without full session
            tags: intent.tags,
            files_modified: vec![],
            errors_resolved: vec![],
            key_learnings: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_task_type() {
        assert_eq!(classify_task_type("fix the bug"), TaskType::Bugfix);
        assert_eq!(classify_task_type("add new feature"), TaskType::Feature);
        assert_eq!(classify_task_type("write tests"), TaskType::Test);
        assert_eq!(classify_task_type("refactor code"), TaskType::Refactor);
    }

    #[test]
    fn test_extract_first_prompt() {
        let transcript = "Human: Fix the login bug\n\nAssistant: I'll help you fix that.";
        assert_eq!(extract_first_prompt(transcript), "Fix the login bug");
    }

    #[test]
    fn test_determine_outcome() {
        assert_eq!(determine_outcome("All tests pass!"), OutcomeStatus::Success);
        assert_eq!(
            determine_outcome("Build failed with errors"),
            OutcomeStatus::Failure
        );
    }
}
