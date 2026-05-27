// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

use crate::mcp::helpers::record_mcp_retrieval;
use crate::{config, retrieve, store};

/// Retrieve relevant episodes
pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let query = args.get("query").and_then(|v| v.as_str());
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let project = args.get("project").and_then(|v| v.as_str());
    let list_all = args.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
    let scope = match args.get("scope").and_then(|v| v.as_str()) {
        Some(s)
            if s.eq_ignore_ascii_case("cross-project")
                || s.eq_ignore_ascii_case("cross_project") =>
        {
            retrieve::RetrievalScope::CrossProject
        }
        _ => retrieve::RetrievalScope::Project,
    };

    let config = config::Config::load().map_err(|e| e.to_string())?;
    let store = store::EpisodeStore::new().map_err(|e| e.to_string())?;

    // Case 1: List all episodes
    if list_all {
        return list_all_episodes(&store, limit, project);
    }

    // Need query for other cases
    let query = query.ok_or("Missing query parameter (or use all: true to list episodes)")?;

    // Case 2: Query looks like an episode ID - show full details
    if looks_like_episode_id(query) {
        if let Some(output) = show_episode_by_id(&store, query)? {
            return Ok(output);
        }
        // If not found by ID, fall through to search
    }

    // Case 3: Semantic search (scope-aware — v0.10).
    let episodes = match retrieve::try_search_scoped(query, limit, project, scope, &config).await {
        Ok(eps) if !eps.is_empty() => eps,
        _ => {
            // Fallback to text search. CrossProject scope widens the
            // pull and filters by transferable-claim status; Project
            // scope preserves the prior behaviour.
            let raw = retrieve::retrieve_episodes_text(
                query,
                if scope == retrieve::RetrievalScope::CrossProject {
                    limit.saturating_mul(3).max(limit)
                } else {
                    limit
                },
                if scope == retrieve::RetrievalScope::CrossProject {
                    None
                } else {
                    project
                },
                &config,
                &store,
            )
            .map_err(|e| e.to_string())?;
            if scope == retrieve::RetrievalScope::CrossProject {
                let current = project.map(String::from);
                let mut filtered: Vec<retrieve::ScoredEpisode> = raw
                    .into_iter()
                    .filter(|s| {
                        if let Some(p) = &current
                            && s.episode.project == *p
                        {
                            return true;
                        }
                        crate::episode::is_episode_transferable(&s.episode)
                    })
                    .collect();
                filtered.truncate(limit);
                filtered
            } else {
                raw
            }
        }
    };

    if episodes.is_empty() {
        return Ok("No relevant episodes found in memory.".to_string());
    }

    // Format results
    let mut output = format!("Found {} relevant past experiences:\n\n", episodes.len());

    for (i, scored) in episodes.iter().enumerate() {
        let ep = &scored.episode;
        output.push_str(&format!(
            "{}. **{}**\n",
            i + 1,
            if ep.intent.extracted_intent.is_empty() {
                &ep.intent.raw_prompt
            } else {
                &ep.intent.extracted_intent
            }
        ));
        output.push_str(&format!("   - ID: {}\n", &ep.id[..8]));
        output.push_str(&format!("   - Project: {}\n", ep.project));
        output.push_str(&format!("   - Type: {}\n", ep.intent.task_type));
        output.push_str(&format!("   - Outcome: {}\n", ep.outcome.status));
        // Show utility with confidence level based on retrieval count
        let confidence = match ep.utility.retrieval_count {
            0 => "untested",
            1..=2 => "low confidence",
            3..=5 => "moderate confidence",
            _ => "high confidence",
        };
        output.push_str(&format!(
            "   - Relevance: {:.0}% similarity, {:.0}% utility ({}, {} retrievals)\n",
            scored.similarity_score * 100.0,
            scored.utility_score * 100.0,
            confidence,
            ep.utility.retrieval_count
        ));

        if !ep.context.files_modified.is_empty() {
            output.push_str(&format!(
                "   - Files: {}\n",
                ep.context.files_modified.join(", ")
            ));
        }

        if !ep.intent.domain.is_empty() {
            output.push_str(&format!("   - Tags: {}\n", ep.intent.domain.join(", ")));
        }

        // Show resolved errors if any
        let resolved: Vec<_> = ep
            .context
            .errors_encountered
            .iter()
            .filter(|e| e.resolved)
            .collect();
        if !resolved.is_empty() {
            output.push_str("   - Errors resolved:\n");
            for err in resolved.iter().take(2) {
                output.push_str(&format!("     - {}\n", err.message));
                if let Some(res) = &err.resolution {
                    output.push_str(&format!("       Resolution: {}\n", res));
                }
            }
        }

        output.push('\n');
    }

    // Surface session context: if any retrieved episode has a session, mention related episodes
    let mut session_context_shown = std::collections::HashSet::new();
    for scored in &episodes {
        if let Some(sid) = &scored.episode.session_id {
            if session_context_shown.insert(sid.clone()) {
                if let Ok(session_eps) = store.list_by_session(sid) {
                    let others: Vec<_> = session_eps
                        .iter()
                        .filter(|e| e.id != scored.episode.id)
                        .collect();
                    if !others.is_empty() {
                        output.push_str(&format!(
                            "📎 Session {} has {} related episode(s):\n",
                            &sid[..8],
                            others.len()
                        ));
                        for other in others.iter().take(3) {
                            let summary: String =
                                other.intent.extracted_intent.chars().take(50).collect();
                            output.push_str(&format!("   - {} ({}...)\n", &other.id[..8], summary));
                        }
                        output.push('\n');
                    }
                }
            }
        }
    }

    output.push_str("Use tempera_feedback to indicate if these were helpful.");

    // Record retrieval for tracking
    let _ = record_mcp_retrieval(&episodes, query, &store);

    // v0.11: append structured recall attribution so downstream tools
    // (codegraph, etc.) can programmatically detect whether memory
    // contributed and thread it into their own responses.
    let attr = retrieve::compute_recall_attribution(&episodes);
    output.push('\n');
    output.push_str(&retrieve::format_recall_attribution(&attr));

    Ok(output)
}

/// Check if a string looks like an episode ID
fn looks_like_episode_id(s: &str) -> bool {
    let s = s.trim();
    if s.len() == 8 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    if s.len() == 36 && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        return true;
    }
    false
}

/// List all episodes
fn list_all_episodes(
    store: &store::EpisodeStore,
    limit: usize,
    project: Option<&str>,
) -> Result<String, String> {
    let mut episodes = store.list_all().map_err(|e| e.to_string())?;

    if let Some(proj) = project {
        episodes.retain(|e| e.project.to_lowercase().contains(&proj.to_lowercase()));
    }

    episodes.sort_by(|a, b| b.timestamp_start.cmp(&a.timestamp_start));
    episodes.truncate(limit);

    if episodes.is_empty() {
        return Ok("No episodes found in memory.".to_string());
    }

    let mut output = format!("Listing {} episode(s):\n\n", episodes.len());

    for (i, ep) in episodes.iter().enumerate() {
        let summary = if ep.intent.extracted_intent.is_empty() {
            &ep.intent.raw_prompt
        } else {
            &ep.intent.extracted_intent
        };
        let summary_short: String = summary.chars().take(60).collect();
        let ellipsis = if summary.len() > 60 { "..." } else { "" };

        output.push_str(&format!("{}. **{}{}**\n", i + 1, summary_short, ellipsis));
        output.push_str(&format!("   - ID: {}\n", &ep.id[..8]));
        output.push_str(&format!("   - Project: {}\n", ep.project));
        output.push_str(&format!(
            "   - Type: {} | Outcome: {}\n",
            ep.intent.task_type, ep.outcome.status
        ));
        output.push_str(&format!(
            "   - Date: {}\n",
            ep.timestamp_start.format("%Y-%m-%d %H:%M")
        ));
        if !ep.intent.domain.is_empty() {
            output.push_str(&format!("   - Tags: {}\n", ep.intent.domain.join(", ")));
        }
        output.push('\n');
    }

    Ok(output)
}

/// Show full episode details by ID
fn show_episode_by_id(store: &store::EpisodeStore, id: &str) -> Result<Option<String>, String> {
    let episodes = store.list_all().map_err(|e| e.to_string())?;

    let episode = episodes
        .iter()
        .find(|e| e.id.starts_with(id) || e.id[..8] == *id);

    let ep = match episode {
        Some(e) => e,
        None => return Ok(None),
    };

    let mut output = String::from("Episode Details\n");
    output.push_str("===============\n\n");

    output.push_str(&format!("**ID**: {}\n", ep.id));
    output.push_str(&format!("**Project**: {}\n", ep.project));
    output.push_str(&format!("**Type**: {}\n", ep.intent.task_type));
    output.push_str(&format!("**Outcome**: {}\n", ep.outcome.status));
    output.push_str(&format!(
        "**Date**: {} - {}\n",
        ep.timestamp_start.format("%Y-%m-%d %H:%M"),
        ep.timestamp_end.format("%H:%M")
    ));
    output.push_str(&format!(
        "**Utility**: {:.0}%\n",
        ep.utility.calculate_score() * 100.0
    ));
    if let Some(sid) = &ep.session_id {
        output.push_str(&format!("**Session**: {}\n", &sid[..8]));
    }
    output.push('\n');

    output.push_str("## Intent\n");
    if !ep.intent.extracted_intent.is_empty() {
        output.push_str(&format!("{}\n\n", ep.intent.extracted_intent));
    }
    output.push_str(&format!("**Raw prompt**: {}\n\n", ep.intent.raw_prompt));

    if !ep.intent.domain.is_empty() {
        output.push_str(&format!("**Tags**: {}\n\n", ep.intent.domain.join(", ")));
    }

    if !ep.context.files_modified.is_empty() {
        output.push_str("## Files Modified\n");
        for f in &ep.context.files_modified {
            output.push_str(&format!("- {}\n", f));
        }
        output.push('\n');
    }

    if !ep.context.errors_encountered.is_empty() {
        output.push_str("## Errors Encountered\n");
        for err in &ep.context.errors_encountered {
            output.push_str(&format!("- **{}**: {}\n", err.error_type, err.message));
            if let Some(res) = &err.resolution {
                output.push_str(&format!("  - Resolution: {}\n", res));
            }
        }
        output.push('\n');
    }

    output.push_str("## Retrieval Stats\n");
    output.push_str(&format!(
        "- Retrieved: {} times\n",
        ep.utility.retrieval_count
    ));
    output.push_str(&format!(
        "- Marked helpful: {} times\n",
        ep.utility.helpful_count
    ));

    Ok(Some(output))
}
