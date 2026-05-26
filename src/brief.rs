// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Brief surface (v0.9) — single MCP call that joins the agent's
//! working set (files about to be edited + optional task_type/domain)
//! against every signal v0.8 accrues:
//!
//! - **Pending ask-back** for this project (v0.8.5) — most actionable.
//! - **Reasoning template** for `(task_type, domain)` (v0.8.3).
//! - **Top correction categories** for these files (v0.8.2).
//! - **Should-have-asked triggers** matching these files (v0.8.4).
//! - **Calibration warning** if the bucket's verified ratio is low
//!   relative to declared-success volume (v0.8.1).
//!
//! Every section is independently optional: store-open failures are
//! silently absorbed (`Ok(None)` / empty vec) so a partial brief is
//! better than no brief. The agent calls this ONCE at task start once
//! the file set is known and gets a single block of "what tempera has
//! learned about this exact patch of code." Pairs naturally with
//! `tempera_session_start` (called even earlier — before the file set
//! is known).

#![allow(dead_code)]

use anyhow::Result;
use serde::Serialize;

use crate::ask_backs::{AskBack, AskBackStore};
use crate::asks::{AsksStore, ShouldHaveAsked};
use crate::calibration::{CalibrationBucket, CalibrationStore};
use crate::mistakes::{CategoryCount, MistakeStore};
use crate::templates::{Template, TemplateStore};

/// Warn at retrieval surfaces when the verified ratio is below this
/// threshold AND the bucket has enough samples to trust the metric.
/// 0.50 matches the salience tilt the design doc proposed.
const VERIFIED_RATIO_WARN_THRESHOLD: f32 = 0.50;

/// Don't warn on calibration when the declared-success count is below
/// this. Low samples inflate overconfidence rate; ranking shouldn't
/// react. Matches the per-bucket gate hinted at in calibration.rs.
const CALIBRATION_MIN_SAMPLE_FOR_WARN: i64 = 5;

/// Maximum mistakes top-N to surface in a brief.
const BRIEF_TOP_CATEGORIES: i64 = 5;

/// Maximum should-have-asked rows to surface in a brief.
const BRIEF_TOP_ASKS: i64 = 5;

#[derive(Debug, Clone, Serialize)]
pub struct Brief {
    pub project: String,
    pub files: Vec<String>,
    pub task_type: Option<String>,
    pub domain: Option<String>,
    pub pending_ask_back: Option<AskBack>,
    pub template: Option<Template>,
    pub top_correction_categories: Vec<CategoryCount>,
    pub should_have_asked_triggers: Vec<ShouldHaveAsked>,
    pub calibration_warning: Option<CalibrationWarning>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationWarning {
    pub task_type: String,
    pub project: String,
    pub declared_success: i64,
    pub verified_ratio: f32,
}

impl Brief {
    pub fn is_empty(&self) -> bool {
        self.pending_ask_back.is_none()
            && self.template.is_none()
            && self.top_correction_categories.is_empty()
            && self.should_have_asked_triggers.is_empty()
            && self.calibration_warning.is_none()
    }
}

/// Assemble the brief. Each section opens its own store independently;
/// any store-open or query failure produces a `None` / empty for that
/// section without failing the whole brief. The agent gets partial
/// information rather than an error response.
///
/// Pre-v0.10.2 signature retained — defaults to project-scoped brief.
pub async fn build_brief(
    project: &str,
    files: &[String],
    task_type: Option<&str>,
    domain: Option<&str>,
) -> Result<Brief> {
    build_brief_scoped(project, files, task_type, domain, false).await
}

/// Scoped variant (v0.10.2): when `cross_project = true`, the
/// mistakes and should-have-asked sections gain rows from projects
/// OTHER than `project` whose files overlap with `files`. Each
/// foreign row carries its source project for the renderer to tag.
/// The pending ask-back, calibration warning, and template sections
/// always stay project-scoped — those signals don't generalise.
pub async fn build_brief_scoped(
    project: &str,
    files: &[String],
    task_type: Option<&str>,
    domain: Option<&str>,
    cross_project: bool,
) -> Result<Brief> {
    let pending_ask_back = match AskBackStore::open_default().await {
        Ok(s) => s.get_pending_for_project(project).await.ok().flatten(),
        Err(_) => None,
    };

    let template = match (task_type, domain) {
        (Some(tt), Some(d)) => match TemplateStore::open_default().await {
            Ok(s) => s.get_by_pair(&tt.to_lowercase(), d).await.ok().flatten(),
            Err(_) => None,
        },
        _ => None,
    };

    let top_correction_categories = if files.is_empty() {
        Vec::new()
    } else {
        match MistakeStore::open_default().await {
            Ok(s) => {
                let mut current = s
                    .top_categories_for_files(project, files, BRIEF_TOP_CATEGORIES)
                    .await
                    .unwrap_or_default();
                if cross_project {
                    let foreign = s
                        .top_categories_for_files_cross_project(
                            project,
                            files,
                            BRIEF_TOP_CATEGORIES,
                        )
                        .await
                        .unwrap_or_default();
                    current.extend(foreign);
                }
                current
            }
            Err(_) => Vec::new(),
        }
    };

    let should_have_asked_triggers = if files.is_empty() {
        Vec::new()
    } else {
        match AsksStore::open_default().await {
            Ok(s) => {
                let mut current = s
                    .triggers_for_files(project, files, BRIEF_TOP_ASKS)
                    .await
                    .unwrap_or_default();
                if cross_project {
                    let foreign = s
                        .triggers_for_files_cross_project(project, files, BRIEF_TOP_ASKS)
                        .await
                        .unwrap_or_default();
                    current.extend(foreign);
                }
                current
            }
            Err(_) => Vec::new(),
        }
    };

    let calibration_warning = match task_type {
        Some(tt) => match CalibrationStore::open_default().await {
            Ok(s) => calibration_warning_for(&s, tt, project).await,
            Err(_) => None,
        },
        None => None,
    };

    Ok(Brief {
        project: project.to_string(),
        files: files.to_vec(),
        task_type: task_type.map(str::to_string),
        domain: domain.map(str::to_string),
        pending_ask_back,
        template,
        top_correction_categories,
        should_have_asked_triggers,
        calibration_warning,
    })
}

async fn calibration_warning_for(
    store: &CalibrationStore,
    task_type: &str,
    project: &str,
) -> Option<CalibrationWarning> {
    let buckets = store.list_by_project(project).await.ok()?;
    let bucket: &CalibrationBucket = buckets
        .iter()
        .find(|b| b.task_type.eq_ignore_ascii_case(task_type))?;
    if bucket.declared_success < CALIBRATION_MIN_SAMPLE_FOR_WARN {
        return None;
    }
    let ratio = bucket.verified_ratio();
    if ratio >= VERIFIED_RATIO_WARN_THRESHOLD {
        return None;
    }
    Some(CalibrationWarning {
        task_type: bucket.task_type.clone(),
        project: bucket.project.clone(),
        declared_success: bucket.declared_success,
        verified_ratio: ratio,
    })
}

/// Render the brief as a plain-text block suitable for an MCP response.
/// Sections are omitted entirely when empty — the agent's context
/// window shouldn't fill with "no mistakes for these files" noise.
pub fn render_text(brief: &Brief) -> String {
    let mut out = String::new();
    let header_files = if brief.files.is_empty() {
        "no files specified".to_string()
    } else if brief.files.len() == 1 {
        brief.files[0].clone()
    } else {
        format!("{} files", brief.files.len())
    };
    out.push_str(&format!(
        "Tempera brief for {project} ({header_files})",
        project = brief.project,
    ));
    if let (Some(tt), Some(dom)) = (&brief.task_type, &brief.domain) {
        out.push_str(&format!(" — task: {tt} / domain: {dom}"));
    } else if let Some(tt) = &brief.task_type {
        out.push_str(&format!(" — task: {tt}"));
    }
    out.push_str(":\n\n");

    if brief.is_empty() {
        out.push_str(
            "Nothing tempera has surfaced yet for this context. \
             Proceed normally — and consider calling tempera_retrieve \
             with a task-specific query for episode-level recall.\n",
        );
        return out;
    }

    if let Some(ab) = &brief.pending_ask_back {
        let ep8 = &ab.episode_id[..8.min(ab.episode_id.len())];
        out.push_str(&format!(
            "🔴 PENDING ASK-BACK\n  Ask the user first: \"{q}\"\n  (drafted after episode {ep8})\n\n",
            q = ab.question,
        ));
    }

    if let Some(t) = &brief.template {
        out.push_str(&format!(
            "📋 REASONING TEMPLATE ({tt} / {dom})\n  {name}  (success_rate {sr:.2}, evidence {ev}, used {used}×)\n",
            tt = t.task_type,
            dom = t.domain,
            name = t.name,
            sr = t.success_rate,
            ev = t.evidence_episodes.len(),
            used = t.times_used,
        ));
        for (i, step) in t.steps.iter().enumerate() {
            out.push_str(&format!("    {}. {}\n", i + 1, step));
        }
        out.push('\n');
    }

    if !brief.top_correction_categories.is_empty() {
        out.push_str("⚠️  TOP CORRECTION CATEGORIES for these files\n");
        for c in &brief.top_correction_categories {
            let project_tag = match &c.source_project {
                Some(p) if p != &brief.project => format!("[from {p}] "),
                _ => String::new(),
            };
            out.push_str(&format!(
                "  - {project_tag}{category} ({count}×, last {last})\n",
                category = c.category,
                count = c.count,
                last = c.last_seen.format("%Y-%m-%d"),
            ));
        }
        out.push('\n');
    }

    if !brief.should_have_asked_triggers.is_empty() {
        out.push_str("📌 SHOULD-HAVE-ASKED triggers for these files\n");
        for s in &brief.should_have_asked_triggers {
            let project_tag = if s.project != brief.project {
                format!("[from {}] ", s.project)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "  - {project_tag}[{trig}] {q}\n    → {a}\n",
                trig = s.trigger,
                q = s.question,
                a = truncate(&s.answer, 120),
            ));
        }
        out.push('\n');
    }

    if let Some(cw) = &brief.calibration_warning {
        out.push_str(&format!(
            "📊 CALIBRATION ({tt} / {project})\n  \
             {pct:.0}% verified rate over {n} declared successes — treat your success \
             claims here with skepticism. Verify before declaring done.\n",
            tt = cw.task_type,
            project = cw.project,
            pct = cw.verified_ratio * 100.0,
            n = cw.declared_success,
        ));
    }

    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ask_backs::AskBackStatus;
    use chrono::Utc;

    fn empty_brief() -> Brief {
        Brief {
            project: "p".to_string(),
            files: vec!["src/foo.rs".to_string()],
            task_type: None,
            domain: None,
            pending_ask_back: None,
            template: None,
            top_correction_categories: Vec::new(),
            should_have_asked_triggers: Vec::new(),
            calibration_warning: None,
        }
    }

    #[test]
    fn is_empty_true_when_no_signals() {
        assert!(empty_brief().is_empty());
    }

    #[test]
    fn is_empty_false_when_any_signal_present() {
        let mut b = empty_brief();
        b.calibration_warning = Some(CalibrationWarning {
            task_type: "bugfix".into(),
            project: "p".into(),
            declared_success: 10,
            verified_ratio: 0.3,
        });
        assert!(!b.is_empty());
    }

    #[test]
    fn render_text_empty_brief_says_so() {
        let out = render_text(&empty_brief());
        assert!(out.contains("Tempera brief for p"));
        assert!(out.contains("src/foo.rs"));
        assert!(out.contains("Nothing tempera has surfaced"));
    }

    #[test]
    fn render_text_includes_pending_ask_back() {
        let mut b = empty_brief();
        b.pending_ask_back = Some(AskBack {
            id: Some(1),
            project: "p".into(),
            episode_id: "abcd1234-rest-of-uuid".into(),
            question: "Which auth provider is wired up?".into(),
            status: AskBackStatus::Pending,
            model: "haiku".into(),
            created_at: Utc::now(),
            served_at: None,
        });
        let out = render_text(&b);
        assert!(out.contains("PENDING ASK-BACK"));
        assert!(out.contains("Which auth provider is wired up?"));
        assert!(out.contains("abcd1234"));
    }

    #[test]
    fn render_text_includes_template_steps() {
        let mut b = empty_brief();
        b.task_type = Some("bugfix".into());
        b.domain = Some("rust".into());
        b.template = Some(Template {
            id: "bugfix-rust-1".into(),
            task_type: "bugfix".into(),
            domain: "rust".into(),
            name: "spawn_blocking deadlock dance".into(),
            steps: vec![
                "Find every tokio::spawn_blocking site".into(),
                "Check Drop ordering for guards held across awaits".into(),
            ],
            evidence_episodes: vec!["a".into(), "b".into()],
            success_rate: 0.42,
            times_used: 3,
            model: "sonnet".into(),
            created_at: Utc::now(),
            last_used: None,
        });
        let out = render_text(&b);
        assert!(out.contains("REASONING TEMPLATE"));
        assert!(out.contains("spawn_blocking deadlock dance"));
        assert!(out.contains("1. Find every tokio::spawn_blocking site"));
        assert!(out.contains("2. Check Drop ordering"));
        assert!(out.contains("0.42"));
    }

    #[test]
    fn render_text_includes_correction_categories() {
        let mut b = empty_brief();
        b.top_correction_categories = vec![
            CategoryCount {
                category: "lifetime_annotations".into(),
                count: 3,
                last_seen: Utc::now(),
                source_project: None,
            },
            CategoryCount {
                category: "auth_order".into(),
                count: 2,
                last_seen: Utc::now(),
                source_project: None,
            },
        ];
        let out = render_text(&b);
        assert!(out.contains("TOP CORRECTION CATEGORIES"));
        assert!(out.contains("lifetime_annotations (3×"));
        assert!(out.contains("auth_order (2×"));
    }

    #[test]
    fn render_text_includes_should_have_asked() {
        let mut b = empty_brief();
        b.should_have_asked_triggers = vec![ShouldHaveAsked {
            id: Some(1),
            project: "p".into(),
            trigger: "edit_auth_middleware".into(),
            question: "Which auth provider is wired up?".into(),
            answer: "tempera uses no auth.".into(),
            episode_id: None,
            files: vec!["src/auth.rs".into()],
            created_at: Utc::now(),
        }];
        let out = render_text(&b);
        assert!(out.contains("SHOULD-HAVE-ASKED"));
        assert!(out.contains("edit_auth_middleware"));
        assert!(out.contains("Which auth provider is wired up?"));
    }

    #[test]
    fn render_text_includes_calibration_warning() {
        let mut b = empty_brief();
        b.task_type = Some("bugfix".into());
        b.calibration_warning = Some(CalibrationWarning {
            task_type: "bugfix".into(),
            project: "p".into(),
            declared_success: 12,
            verified_ratio: 0.33,
        });
        let out = render_text(&b);
        assert!(out.contains("CALIBRATION"));
        assert!(out.contains("33% verified rate"));
        assert!(out.contains("12 declared successes"));
    }

    #[test]
    fn render_text_tags_foreign_project_categories() {
        let mut b = empty_brief();
        b.top_correction_categories = vec![
            CategoryCount {
                category: "lifetime_annotations".into(),
                count: 2,
                last_seen: Utc::now(),
                source_project: None, // current project
            },
            CategoryCount {
                category: "auth_order".into(),
                count: 3,
                last_seen: Utc::now(),
                source_project: Some("smelt".into()),
            },
        ];
        let out = render_text(&b);
        // Current-project row has no tag.
        assert!(out.contains("- lifetime_annotations (2×"));
        // Foreign-project row is prefixed.
        assert!(out.contains("- [from smelt] auth_order (3×"));
    }

    #[test]
    fn render_text_tags_foreign_project_asks() {
        let mut b = empty_brief();
        b.should_have_asked_triggers = vec![
            ShouldHaveAsked {
                id: Some(1),
                project: "p".into(), // matches brief.project
                trigger: "local_trig".into(),
                question: "local Q?".into(),
                answer: "local A".into(),
                episode_id: None,
                files: vec![],
                created_at: Utc::now(),
            },
            ShouldHaveAsked {
                id: Some(2),
                project: "smelt".into(),
                trigger: "foreign_trig".into(),
                question: "foreign Q?".into(),
                answer: "foreign A".into(),
                episode_id: None,
                files: vec![],
                created_at: Utc::now(),
            },
        ];
        let out = render_text(&b);
        assert!(out.contains("- [local_trig] local Q?"));
        assert!(out.contains("- [from smelt] [foreign_trig] foreign Q?"));
    }

    #[test]
    fn render_text_header_handles_no_files() {
        let mut b = empty_brief();
        b.files = vec![];
        let out = render_text(&b);
        assert!(out.contains("no files specified"));
    }

    #[test]
    fn render_text_header_multiple_files() {
        let mut b = empty_brief();
        b.files = vec!["a".into(), "b".into(), "c".into()];
        let out = render_text(&b);
        assert!(out.contains("3 files"));
    }
}
