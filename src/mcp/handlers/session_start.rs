// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `tempera_session_start` MCP handler — surfaces any pending ask-back
//! drafted by capture for this project at the start of a new session
//! (v0.8.5). Marking the row served happens here so the slot frees up
//! for the *next* failure to queue a new question.

use serde_json::Value;

use crate::ask_backs::AskBackStore;
use crate::mcp::helpers::extract_project;

pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let project = extract_project(args);

    let store = AskBackStore::open_default()
        .await
        .map_err(|e| e.to_string())?;

    let Some(ab) = store
        .get_pending_for_project(&project)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Ok(format!(
            "No pending ask-back for {project}. Proceed with the task.\n\
             (When tempera captures a failure/partial episode with vague intent, \
             it queues one clarifying question for the next session in that project.)"
        ));
    };

    // Mark served so the slot frees up for future failures. Done before
    // returning so a crashed agent still consumes the question rather
    // than receiving it forever.
    if let Some(id) = ab.id {
        let _ = store.mark_served(id).await;
    }

    Ok(format!(
        "Tempera has one question for you to ask the user before starting work in {project}:\n\n\
         📌 {question}\n\n\
         (Drafted after episode {ep8} ended in failure or partial success with vague intent. \
         Asking the user this up front — even briefly — beats guessing again.)",
        question = ab.question,
        ep8 = &ab.episode_id[..8.min(ab.episode_id.len())],
    ))
}
