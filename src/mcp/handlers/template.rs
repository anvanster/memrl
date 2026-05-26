// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `tempera_template` MCP handler — pulls the reasoning template
//! extracted for a given `(task_type, domain)` pair (v0.8.3).
//!
//! The agent calls this at task start once it knows what kind of task
//! it's about to do and which domain it touches. If a template exists
//! the response is the imperative step sequence plus pointers back to
//! the supporting episodes — future Claude can drill in via
//! `tempera_retrieve` for the full episodes.

use serde_json::Value;

use crate::templates::TemplateStore;

pub(crate) async fn handle(args: &Value) -> Result<String, String> {
    let task_type = args
        .get("task_type")
        .and_then(|v| v.as_str())
        .ok_or("Missing task_type parameter")?;
    let domain = args
        .get("domain")
        .and_then(|v| v.as_str())
        .ok_or("Missing domain parameter")?;

    // Normalize casing — task_type is stored lowercase (matches the
    // TaskType::Display impl).
    let task_type_norm = task_type.to_lowercase();

    let store = TemplateStore::open_default()
        .await
        .map_err(|e| e.to_string())?;

    let Some(template) = store
        .get_by_pair(&task_type_norm, domain)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Ok(format!(
            "No reasoning template stored yet for ({task_type_norm}, {domain}). \
             Templates accrue during the dream cycle once at least 3 successful \
             episodes share this (task_type, domain) bucket. \
             For now, fall back to tempera_retrieve."
        ));
    };

    // Best-effort: bump times_used + last_used. If it fails we still
    // return the template — the counter is for ranking, not correctness.
    let _ = store.touch_used(&task_type_norm, domain).await;

    let mut out = format!(
        "Reasoning template for ({tt}, {dom}): {name}\n\
         (success_rate {sr:.2}, evidence: {ev_n}, used {used}×)\n\n\
         Steps:\n",
        tt = template.task_type,
        dom = template.domain,
        name = template.name,
        sr = template.success_rate,
        ev_n = template.evidence_episodes.len(),
        used = template.times_used + 1,
    );
    for (i, step) in template.steps.iter().enumerate() {
        out.push_str(&format!("  {}. {}\n", i + 1, step));
    }
    if !template.evidence_episodes.is_empty() {
        out.push_str("\nEvidence episodes (tempera_retrieve for full content):\n");
        for id in template.evidence_episodes.iter().take(8) {
            out.push_str(&format!("  - {id}\n"));
        }
    }
    Ok(out)
}
