use serde_json::Value;

use crate::mcp::helpers::{extract_project, extract_string_array};
use crate::{episode, indexer, store, utility};

/// Similarity threshold for BKM consolidation.
/// Above this, a new capture updates the existing episode instead of creating a new one.
/// Must be high (0.85+) because project-scoped searches inflate similarity.
const CONSOLIDATION_THRESHOLD: f32 = 0.85;

/// Capture a new episode, consolidating with existing BKMs when similar
pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let summary = args
        .get("summary")
        .and_then(|v| v.as_str())
        .ok_or("Missing summary parameter")?;

    let task_type_str = args
        .get("task_type")
        .and_then(|v| v.as_str())
        .ok_or("Missing task_type parameter")?;

    let outcome_str = args
        .get("outcome")
        .and_then(|v| v.as_str())
        .ok_or("Missing outcome parameter")?;

    let files_modified = extract_string_array(args, "files_modified");
    let tags = extract_string_array(args, "tags");

    let project = extract_project(args);

    let task_type = match task_type_str {
        "bugfix" => episode::TaskType::Bugfix,
        "feature" => episode::TaskType::Feature,
        "refactor" => episode::TaskType::Refactor,
        "test" => episode::TaskType::Test,
        "docs" => episode::TaskType::Docs,
        "research" => episode::TaskType::Research,
        "debug" => episode::TaskType::Debug,
        "setup" => episode::TaskType::Setup,
        _ => episode::TaskType::Unknown,
    };

    let outcome = match outcome_str {
        "success" => episode::OutcomeStatus::Success,
        "partial" => episode::OutcomeStatus::Partial,
        "failure" => episode::OutcomeStatus::Failure,
        _ => episode::OutcomeStatus::Partial,
    };

    // Parse errors if provided
    let mut errors = Vec::new();
    if let Some(error_arr) = args.get("errors_resolved").and_then(|v| v.as_array()) {
        for err in error_arr {
            if let (Some(error_msg), resolution) = (
                err.get("error").and_then(|v| v.as_str()),
                err.get("resolution").and_then(|v| v.as_str()),
            ) {
                errors.push(episode::ErrorRecord {
                    error_type: "runtime".to_string(),
                    message: error_msg.to_string(),
                    resolved: resolution.is_some(),
                    resolution: resolution.map(String::from),
                });
            }
        }
    }

    let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;

    // Try to find a similar existing episode to consolidate with
    if let Some(result) = try_consolidate(
        &store,
        summary,
        &project,
        &task_type,
        &outcome,
        &tags,
        &files_modified,
        &errors,
    )
    .await
    {
        return Ok(result);
    }

    // No consolidation match — create new episode
    let mut ep = episode::Episode::new(project.clone(), summary.to_string());
    ep.intent.task_type = task_type;
    ep.outcome.status = outcome;
    ep.context.files_modified = files_modified;
    ep.intent.domain = tags;
    ep.intent.extracted_intent = summary.to_string();
    ep.context.errors_encountered = errors;
    ep.timestamp_end = chrono::Utc::now();

    store.save(&ep).map_err(|e| e.to_string())?;

    // Index the new episode
    if let Ok(mut indexer) = indexer::EpisodeIndexer::new().await {
        let _ = indexer.index_episode(&ep).await;
    }

    let mut output = format!(
        "Episode captured successfully!\n\
         - ID: {}\n\
         - Project: {}\n\
         - Type: {}\n\
         - Outcome: {}\n",
        &ep.id[..8],
        ep.project,
        ep.intent.task_type,
        ep.outcome.status
    );

    // Auto-propagate utility to spread value
    output.push_str("\n📈 Running auto-propagation...\n");
    let params = utility::UtilityParams::default();
    match utility::run_bellman_propagation(&store, &params, Some(project.as_str())).await {
        Ok((count, _)) => output.push_str(&format!("  Propagated value to {} episode(s)\n", count)),
        Err(e) => output.push_str(&format!("  (propagation skipped: {})\n", e)),
    }

    output.push_str("\nThis experience is now stored for future reference.");
    Ok(output)
}

/// Try to find and consolidate with a similar existing episode.
/// Returns Some(output) if consolidation happened, None if no match found.
#[allow(clippy::too_many_arguments)]
async fn try_consolidate(
    store: &store::EpisodeStore,
    summary: &str,
    project: &str,
    task_type: &episode::TaskType,
    outcome: &episode::OutcomeStatus,
    tags: &[String],
    files_modified: &[String],
    errors: &[episode::ErrorRecord],
) -> Option<String> {
    // Try vector search first
    let mut indexer = indexer::EpisodeIndexer::new().await.ok()?;

    if !indexer.is_indexed().await {
        // Fall back to tag-based matching
        return try_tag_consolidate(
            store,
            summary,
            project,
            task_type,
            outcome,
            tags,
            files_modified,
            errors,
        );
    }

    let results = indexer.search(summary, 3, Some(project)).await.ok()?;

    // Find the best match above threshold
    let best = results
        .into_iter()
        .find(|r| r.similarity_score >= CONSOLIDATION_THRESHOLD)?;

    // Load the existing episode
    let mut existing = store.load(&best.id).ok()?;

    let similarity_pct = (best.similarity_score * 100.0) as u32;
    let short_id = &existing.id[..8];

    // Merge: newer summary wins (latest knowledge = best known method)
    existing.intent.extracted_intent = summary.to_string();
    existing.intent.raw_prompt = summary.to_string();

    // Update task type and outcome from latest capture
    existing.intent.task_type = task_type.clone();
    existing.outcome.status = outcome.clone();

    // Union-merge tags
    for tag in tags {
        if !existing.intent.domain.contains(tag) {
            existing.intent.domain.push(tag.clone());
        }
    }

    // Union-merge files_modified
    for f in files_modified {
        if !existing.context.files_modified.contains(f) {
            existing.context.files_modified.push(f.clone());
        }
    }

    // Append new errors (preserves full error history)
    for err in errors {
        existing.context.errors_encountered.push(err.clone());
    }

    // Update timestamp to mark when BKM was last refined
    existing.timestamp_end = chrono::Utc::now();

    // Save updated episode (utility counts preserved from existing)
    store.update(&existing).ok()?;

    // Re-index with new content
    let _ = indexer.index_episode(&existing).await;

    let mut output = format!(
        "🔄 Updated existing BKM ({}% similarity)\n\
         - ID: {}\n\
         - Project: {}\n\
         - Type: {}\n\
         - Outcome: {}\n\
         - Tags: {}\n",
        similarity_pct,
        short_id,
        existing.project,
        existing.intent.task_type,
        existing.outcome.status,
        existing.intent.domain.join(", ")
    );

    output
        .push_str("\nExisting episode refined with new insights instead of creating a duplicate.");
    Some(output)
}

/// Fallback: match by tags when vector index is unavailable
#[allow(clippy::too_many_arguments)]
fn try_tag_consolidate(
    store: &store::EpisodeStore,
    summary: &str,
    project: &str,
    task_type: &episode::TaskType,
    outcome: &episode::OutcomeStatus,
    tags: &[String],
    files_modified: &[String],
    errors: &[episode::ErrorRecord],
) -> Option<String> {
    if tags.len() < 2 {
        return None; // Not enough tags to match on
    }

    let episodes = store.list_all().ok()?;
    let project_lower = project.to_lowercase();

    // Find an episode in the same project with ≥3 matching tags and same task type
    let best = episodes.into_iter().find(|ep| {
        if !ep.project.to_lowercase().contains(&project_lower) {
            return false;
        }
        if ep.intent.task_type != *task_type {
            return false;
        }
        let matching_tags = tags
            .iter()
            .filter(|t| ep.intent.domain.iter().any(|d| d.eq_ignore_ascii_case(t)))
            .count();
        matching_tags >= 3
    })?;

    let mut existing = best;
    let short_id = existing.id[..8].to_string();

    // Same merge strategy
    existing.intent.extracted_intent = summary.to_string();
    existing.intent.raw_prompt = summary.to_string();
    existing.intent.task_type = task_type.clone();
    existing.outcome.status = outcome.clone();

    for tag in tags {
        if !existing.intent.domain.contains(tag) {
            existing.intent.domain.push(tag.clone());
        }
    }
    for f in files_modified {
        if !existing.context.files_modified.contains(f) {
            existing.context.files_modified.push(f.clone());
        }
    }
    for err in errors {
        existing.context.errors_encountered.push(err.clone());
    }
    existing.timestamp_end = chrono::Utc::now();

    store.update(&existing).ok()?;

    let mut output = format!(
        "🔄 Updated existing BKM (tag match)\n\
         - ID: {}\n\
         - Project: {}\n\
         - Type: {}\n\
         - Outcome: {}\n\
         - Tags: {}\n",
        short_id,
        existing.project,
        existing.intent.task_type,
        existing.outcome.status,
        existing.intent.domain.join(", ")
    );

    output
        .push_str("\nExisting episode refined with new insights instead of creating a duplicate.");
    Some(output)
}
