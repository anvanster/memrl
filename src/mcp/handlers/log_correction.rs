// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `tempera_log_correction` MCP handler — agent-facing entry point for
//! the anchored-mistakes index (v0.8.2).

use serde_json::Value;

use crate::mcp::helpers::{extract_project, extract_string_array};
use crate::{mistakes, store as episode_store};

pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let category_raw = args
        .get("category")
        .and_then(|v| v.as_str())
        .ok_or("Missing category parameter")?;
    let description = args
        .get("description")
        .and_then(|v| v.as_str())
        .ok_or("Missing description parameter")?;
    let correction = args
        .get("correction")
        .and_then(|v| v.as_str())
        .ok_or("Missing correction parameter")?;

    let project = extract_project(args);
    let files = extract_string_array(args, "files");
    let episode_id = args
        .get("episode_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Resolve a short episode-id prefix to the full id if the agent
    // passed an 8-char form. Falls back to whatever it gave us if the
    // lookup fails — better to log with a short id than to lose the
    // signal entirely.
    let episode_id = if let Some(id) = episode_id {
        episode_store::EpisodeStore::new()
            .ok()
            .and_then(|s| s.load(&id).ok())
            .map(|ep| ep.id)
            .or(Some(id))
    } else {
        None
    };

    let category = mistakes::normalize_category(category_raw);
    if category.is_empty() {
        return Err("category cannot be empty after normalization".to_string());
    }

    let m = mistakes::Mistake {
        id: None,
        project: project.clone(),
        category: category.clone(),
        episode_id,
        files,
        description: description.to_string(),
        correction: correction.to_string(),
        created_at: chrono::Utc::now(),
    };

    let store = mistakes::MistakeStore::open_default()
        .await
        .map_err(|e| e.to_string())?;
    let id = store.record(&m).await.map_err(|e| e.to_string())?;

    let top = store
        .top_categories(Some(&project), 5)
        .await
        .unwrap_or_default();

    let mut out = format!(
        "Logged correction #{id} in [{project}] / category {category}.\n\
         - Description: {desc_preview}\n",
        desc_preview = truncate(description, 120),
    );
    if !top.is_empty() {
        out.push_str(&format!(
            "\nTop categories in {project} so far:\n",
            project = project
        ));
        for c in &top {
            out.push_str(&format!("  - {} ({} occurrence(s))\n", c.category, c.count));
        }
    }
    Ok(out)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}…")
    }
}
