// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `tempera_brief` MCP handler — joins the agent's working set
//! against every v0.8 surface in one call. The payoff for the
//! mistakes / asks / templates / ask_backs / calibration plumbing.
//!
//! Call this AT TASK START once the agent knows which files it's
//! about to touch (and ideally also the task_type + domain so the
//! reasoning-template section fires). Best paired with
//! `tempera_session_start` (called even earlier — before files are
//! known) as the standard session-warmup pair.

use serde_json::Value;

use crate::brief;
use crate::mcp::helpers::{extract_project, extract_string_array};

pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let project = extract_project(args);
    let files = extract_string_array(args, "files");
    let task_type = args
        .get("task_type")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let domain = args
        .get("domain")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let b = brief::build_brief(&project, &files, task_type.as_deref(), domain.as_deref())
        .await
        .map_err(|e| e.to_string())?;

    Ok(brief::render_text(&b))
}
