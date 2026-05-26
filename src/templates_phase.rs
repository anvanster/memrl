// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reasoning templates — authoring pipeline (v0.8.3).
//!
//! This module hosts the Sonnet-backed pipeline that runs during the
//! dream cycle. The store and grouping primitives live in
//! `templates.rs` (re-exported and shared by both binaries); the
//! authorship code is gated to the `tempera` binary because it pulls
//! in `crate::dream::CostBudget` and `crate::llm::AnthropicClient`.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::config::Config;
use crate::dream::CostBudget;
use crate::episode::{Episode, TaskType};
use crate::llm::AnthropicClient;
use crate::store::EpisodeStore;
use crate::templates::{
    self, TaskDomainKey, Template, TemplateStore, cluster_success_rate, group_by_task_domain,
    is_eligible,
};

const TEMPLATE_SYSTEM: &str = r#"You are looking at several successful episodes — past task completions that share a (task_type, domain) bucket. Your job: extract the REUSABLE REASONING TEMPLATE — the step sequence the agent followed when it succeeded — so future Claude can pull it before starting a similar task.

Rules:
- 3-7 ordered steps. Each step is ONE concrete action — not a vague intention.
  Bad:  "Understand the problem"
  Good: "Read the failing test and identify which assertion fails first"
- Each step should describe WHAT to do, not WHY. The steps are imperatives.
- Drop any step that only appeared in one or two episodes — generalize across the cluster.
- Name the template as a short phrase the agent will recognize (≤6 words).
- Output JSON ONLY:
  {
    "name": "short phrase",
    "steps": ["step 1", "step 2", "step 3"],
    "evidence": ["<episode-id>", ...]
  }
- If the episodes don't share a clear repeatable pattern (only superficially similar), output:
  {"name": "none", "steps": [], "evidence": []}
"#;

#[derive(Debug, Deserialize)]
struct AuthoredTemplate {
    name: String,
    #[serde(default)]
    steps: Vec<String>,
    #[serde(default)]
    evidence: Vec<String>,
}

fn build_user_message(cluster: &[&Episode], task_type: &str, domain: &str) -> String {
    let mut s = format!(
        "Cluster: task_type={}, domain={}, {} successful episodes.\n\n",
        task_type,
        domain,
        cluster.len()
    );
    for ep in cluster {
        let intent = if ep.intent.extracted_intent.is_empty() {
            ep.intent.raw_prompt.as_str()
        } else {
            ep.intent.extracted_intent.as_str()
        };
        let intent_preview: String = intent.chars().take(280).collect();

        let tools: String = ep.context.tools_invoked.join(", ");
        let tools_line = if tools.is_empty() {
            String::new()
        } else {
            format!("Tools: {tools}\n")
        };

        let files: Vec<String> = ep.context.files_modified.iter().take(8).cloned().collect();
        let files_line = if files.is_empty() {
            String::new()
        } else {
            format!("Files modified: {}\n", files.join(", "))
        };

        let alts: String = ep
            .alternatives_considered
            .iter()
            .take(3)
            .map(|a| {
                format!(
                    "  - tried {}: {} (didn't pick because {})",
                    a.approach.chars().take(80).collect::<String>(),
                    a.how_close.label(),
                    a.why_not.chars().take(120).collect::<String>(),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let alts_line = if alts.is_empty() {
            String::new()
        } else {
            format!("Alternatives:\n{alts}\n")
        };

        s.push_str(&format!(
            "--- {} ---\nIntent: {}\n{}{}{}\n",
            ep.id, intent_preview, tools_line, files_line, alts_line
        ));
    }
    s
}

async fn author_template(
    cluster: &[&Episode],
    task_type: &str,
    domain: &str,
    config: &Config,
    budget: Option<&CostBudget>,
) -> Result<Option<AuthoredTemplate>> {
    if let Some(b) = budget {
        b.try_spend(templates::TEMPLATE_ESTIMATED_COST_USD)?;
    }
    let user = build_user_message(cluster, task_type, domain);
    let client = AnthropicClient::with_model(&config.dream.reflect_model)?;
    let raw = client.raw_completion(TEMPLATE_SYSTEM, &user, 500).await?;
    let trimmed = strip_json_fences(&raw);
    let authored: AuthoredTemplate = serde_json::from_str(trimmed)
        .with_context(|| format!("failed to parse template JSON: {raw}"))?;
    if authored.name.eq_ignore_ascii_case("none") || authored.steps.is_empty() {
        return Ok(None);
    }
    Ok(Some(authored))
}

fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    s.trim_end_matches("```").trim()
}

// ===== Sidecar =====

fn templates_dir() -> Result<PathBuf> {
    Ok(Config::data_dir()?.join("templates"))
}

fn write_sidecar(t: &Template) -> Result<()> {
    let dir = templates_dir()?;
    std::fs::create_dir_all(&dir)?;
    let filename = format!("{}__{}.md", t.task_type, t.domain.replace('/', "-"));
    let path = dir.join(filename);
    let steps_lines: String = t
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n");
    let evidence_lines: String = t
        .evidence_episodes
        .iter()
        .map(|id| format!("- {id}"))
        .collect::<Vec<_>>()
        .join("\n");
    let content = format!(
        "+++\nid = \"{id}\"\ntask_type = \"{tt}\"\ndomain = \"{dom}\"\nname = \"{name}\"\nsuccess_rate = {sr:.3}\ntimes_used = {tu}\nmodel = \"{model}\"\ncreated_at = \"{ts}\"\n+++\n\n# Template: {name} ({tt} / {dom})\n\n## Steps\n\n{steps}\n\n## Evidence episodes\n\n{ev}\n",
        id = t.id,
        tt = t.task_type,
        dom = t.domain,
        name = t.name,
        sr = t.success_rate,
        tu = t.times_used,
        model = t.model,
        ts = t.created_at.to_rfc3339(),
        steps = steps_lines,
        ev = evidence_lines,
    );
    std::fs::write(&path, content)
        .with_context(|| format!("Failed to write template sidecar at {}", path.display()))?;
    Ok(())
}

// ===== Pipeline =====

#[derive(Debug, Clone, Serialize)]
pub struct TemplatesReport {
    pub episodes_examined: usize,
    pub episodes_eligible: usize,
    pub buckets_found: usize,
    pub buckets_above_min: usize,
    pub templates_written: usize,
    pub buckets_no_template: usize,
}

/// Run the templates phase. Returns a report even when no templates
/// were written.
pub async fn run_templates(
    config: &Config,
    budget: Option<&CostBudget>,
) -> Result<TemplatesReport> {
    let episode_store = EpisodeStore::new()?;
    let template_store = TemplateStore::open_default().await?;

    let all = episode_store.list_all()?;
    let n_total = all.len();

    let min_w = config.dream.templates_min_verification_weight;
    let eligible: Vec<&Episode> = all.iter().filter(|e| is_eligible(e, min_w)).collect();
    let n_eligible = eligible.len();

    let min_evidence = config.dream.templates_min_evidence;

    if n_eligible < min_evidence {
        return Ok(TemplatesReport {
            episodes_examined: n_total,
            episodes_eligible: n_eligible,
            buckets_found: 0,
            buckets_above_min: 0,
            templates_written: 0,
            buckets_no_template: 0,
        });
    }

    let buckets = group_by_task_domain(eligible.iter().copied());
    let buckets_found = buckets.len();
    let buckets_above_min = buckets.values().filter(|v| v.len() >= min_evidence).count();

    let mut report = TemplatesReport {
        episodes_examined: n_total,
        episodes_eligible: n_eligible,
        buckets_found,
        buckets_above_min,
        templates_written: 0,
        buckets_no_template: 0,
    };

    // Sort buckets by size desc so we author the densest first — if the
    // budget runs out, we've covered the highest-evidence pairs.
    let mut buckets_vec: Vec<(TaskDomainKey, Vec<&Episode>)> = buckets.into_iter().collect();
    buckets_vec.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (key, cluster) in buckets_vec {
        if cluster.len() < min_evidence {
            continue;
        }
        // Skip Unknown task type — too generic for a useful template.
        if key.task_type == TaskType::Unknown.to_string() {
            continue;
        }

        let authored =
            match author_template(&cluster, &key.task_type, &key.domain, config, budget).await {
                Ok(Some(a)) => a,
                Ok(None) => {
                    report.buckets_no_template += 1;
                    continue;
                }
                Err(e) if e.to_string().contains("cost cap reached") => break,
                Err(e) => return Err(e),
            };

        if authored.steps.is_empty() {
            report.buckets_no_template += 1;
            continue;
        }

        let evidence_ids: Vec<String> = if authored.evidence.is_empty() {
            cluster.iter().map(|e| e.id.clone()).collect()
        } else {
            let known: std::collections::HashSet<&str> =
                cluster.iter().map(|e| e.id.as_str()).collect();
            authored
                .evidence
                .into_iter()
                .filter(|e| known.contains(e.as_str()))
                .collect()
        };

        let success_rate = cluster_success_rate(&cluster);
        let now = Utc::now();
        let template = Template {
            id: format!("{}-{}-{}", key.task_type, key.domain, now.timestamp()),
            task_type: key.task_type.clone(),
            domain: key.domain.clone(),
            name: authored.name,
            steps: authored.steps,
            evidence_episodes: evidence_ids,
            success_rate,
            times_used: 0,
            model: config.dream.reflect_model.clone(),
            created_at: now,
            last_used: None,
        };
        template_store.put(&template).await?;
        write_sidecar(&template)?;
        report.templates_written += 1;
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::{
        Context, Intent, Outcome, OutcomeStatus, TaskType, Utility, VerificationState,
    };

    fn mk_episode(id: &str) -> Episode {
        Episode {
            schema_version: 5,
            id: id.to_string(),
            timestamp_start: Utc::now(),
            timestamp_end: Utc::now(),
            project: "p".into(),
            intent: Intent {
                raw_prompt: "raw".into(),
                extracted_intent: "extracted".into(),
                task_type: TaskType::Bugfix,
                domain: vec!["rust".into()],
                claim: None,
            },
            context: Context {
                files_read: vec![],
                files_modified: vec!["src/foo.rs".into()],
                tools_invoked: vec!["Read".into(), "Edit".into()],
                errors_encountered: vec![],
            },
            outcome: Outcome {
                status: OutcomeStatus::Success,
                tests_before: None,
                tests_after: None,
                commit_sha: None,
                pr_number: None,
                verification: VerificationState::Untested,
            },
            utility: Utility::default(),
            retrieval_history: vec![],
            session_id: None,
            related_episodes: vec![],
            alternatives_considered: vec![],
        }
    }

    #[test]
    fn strip_json_fences_handles_markdown() {
        assert_eq!(
            strip_json_fences("```json\n{\"name\":\"x\"}\n```"),
            "{\"name\":\"x\"}"
        );
        assert_eq!(strip_json_fences("  {\"x\":1}  "), "{\"x\":1}");
    }

    #[test]
    fn build_user_message_includes_intent_and_tools() {
        let ep = mk_episode("ep-1");
        let msg = build_user_message(&[&ep], "bugfix", "rust");
        assert!(msg.contains("ep-1"));
        assert!(msg.contains("extracted"));
        assert!(msg.contains("Read, Edit"));
        assert!(msg.contains("src/foo.rs"));
    }
}
