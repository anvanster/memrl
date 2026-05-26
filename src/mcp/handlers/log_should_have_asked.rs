// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `tempera_log_should_have_asked` MCP handler — agent-facing entry
//! point for the should-have-asked log (v0.8.4).

use serde_json::Value;

use crate::asks;
use crate::mcp::helpers::{extract_project, extract_string_array};
use crate::store as episode_store;

pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let trigger_raw = args
        .get("trigger")
        .and_then(|v| v.as_str())
        .ok_or("Missing trigger parameter")?;
    let question = args
        .get("question")
        .and_then(|v| v.as_str())
        .ok_or("Missing question parameter")?;
    let answer = args
        .get("answer")
        .and_then(|v| v.as_str())
        .ok_or("Missing answer parameter")?;

    let project = extract_project(args);
    let files = extract_string_array(args, "files");
    let episode_id = args
        .get("episode_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Resolve a short episode-id prefix to the full id. Falls back to
    // whatever was passed if lookup fails — better to log with a short
    // id than to lose the signal entirely.
    let episode_id = if let Some(id) = episode_id {
        episode_store::EpisodeStore::new()
            .ok()
            .and_then(|s| s.load(&id).ok())
            .map(|ep| ep.id)
            .or(Some(id))
    } else {
        None
    };

    let trigger = asks::normalize_trigger(trigger_raw);
    if trigger.is_empty() {
        return Err("trigger cannot be empty after normalization".to_string());
    }

    let sha = asks::ShouldHaveAsked {
        id: None,
        project: project.clone(),
        trigger: trigger.clone(),
        question: question.to_string(),
        answer: answer.to_string(),
        episode_id,
        files,
        created_at: chrono::Utc::now(),
    };

    let store = asks::AsksStore::open_default()
        .await
        .map_err(|e| e.to_string())?;
    let id = store.record(&sha).await.map_err(|e| e.to_string())?;

    let top = store
        .top_triggers(Some(&project), 5)
        .await
        .unwrap_or_default();

    let mut out = format!(
        "Logged should-have-asked #{id} in [{project}] / trigger {trigger}.\n\
         - Question: {q_preview}\n\
         - Answer:   {a_preview}\n",
        q_preview = truncate(question, 120),
        a_preview = truncate(answer, 120),
    );
    if !top.is_empty() {
        out.push_str(&format!("\nTop triggers in {project} so far:\n"));
        for c in &top {
            out.push_str(&format!("  - {} ({} occurrence(s))\n", c.trigger, c.count));
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
