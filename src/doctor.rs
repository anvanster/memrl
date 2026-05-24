// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Read-only health diagnostic for a tempera install.
//!
//! Runs a fixed set of dimensions, each producing a 0.0–1.0 score and a
//! short human summary. Weights sum to 100 so the aggregate is the
//! "Health" number a user reads first. No remediation in v1 — that ships
//! with v0.7's dream cycle. The goal here is to point a finger at what's
//! wrong; the user (or a future `--remediate` plan walker) does the fixing.

#![allow(dead_code)]

use anyhow::Result;
use chrono::{DateTime, Utc};
use colored::Colorize;
use serde::Serialize;
use std::collections::HashSet;
use std::path::Path;

use crate::episode::Episode;
use crate::indexer;
use crate::jobs::{JobQueue, JobStatus};
use crate::store::EpisodeStore;

/// Weight contributions per dimension. Sum = 100.
const W_INDEX_FRESHNESS: u32 = 25;
const W_EMBEDDING_COVERAGE: u32 = 25;
const W_KEYWORD_COVERAGE: u32 = 15;
const W_SESSION_LINKS: u32 = 15;
const W_EVAL_PASS_RATE: u32 = 10;
const W_JOB_QUEUE: u32 = 10;

/// Index is "fresh" within 24h, decays linearly to 0 over 7 days.
const INDEX_FRESH_WINDOW_DAYS: f32 = 7.0;

#[derive(Debug, Clone, Serialize)]
pub struct HealthReport {
    pub score: u32,
    pub dimensions: Vec<DimensionReport>,
    /// Tempera version that produced this report.
    pub version: String,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DimensionReport {
    pub name: &'static str,
    pub weight: u32,
    /// Raw 0.0–1.0 score for this dimension.
    pub score: f32,
    /// Points contributed to the aggregate (round(weight × score)).
    pub points: u32,
    pub status: DimensionStatus,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DimensionStatus {
    Pass,
    Warn,
    Fail,
}

impl DimensionStatus {
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 {
            Self::Pass
        } else if score >= 0.5 {
            Self::Warn
        } else {
            Self::Fail
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Self::Pass => "✓",
            Self::Warn => "~",
            Self::Fail => "✗",
        }
    }
}

/// Compute the full health report.
pub async fn check() -> Result<HealthReport> {
    let store = EpisodeStore::new()?;
    let episodes = store.list_all().unwrap_or_default();
    let episode_count = episodes.len();

    let mut dims = Vec::new();

    dims.push(make_dim(
        "index freshness",
        W_INDEX_FRESHNESS,
        check_index_freshness().await,
    ));
    dims.push(make_dim(
        "embedding coverage",
        W_EMBEDDING_COVERAGE,
        check_embedding_coverage(episode_count).await,
    ));
    dims.push(make_dim(
        "keyword coverage",
        W_KEYWORD_COVERAGE,
        check_keyword_coverage(episode_count),
    ));
    dims.push(make_dim(
        "session links",
        W_SESSION_LINKS,
        check_session_links(&episodes),
    ));
    dims.push(make_dim(
        "eval pass rate",
        W_EVAL_PASS_RATE,
        check_eval_pass_rate(),
    ));
    dims.push(make_dim("job queue", W_JOB_QUEUE, check_job_queue().await));

    let total: u32 = dims.iter().map(|d| d.points).sum();

    Ok(HealthReport {
        score: total.min(100),
        dimensions: dims,
        version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at: Utc::now(),
    })
}

fn make_dim(name: &'static str, weight: u32, result: (f32, String)) -> DimensionReport {
    let (score, summary) = result;
    let clamped = score.clamp(0.0, 1.0);
    let points = (clamped * weight as f32).round() as u32;
    DimensionReport {
        name,
        weight,
        score: clamped,
        points,
        status: DimensionStatus::from_score(clamped),
        summary,
    }
}

// ===== Dimensions =====

async fn check_index_freshness() -> (f32, String) {
    let path = match indexer::vector_index_path() {
        Ok(p) => p,
        Err(e) => return (0.0, format!("path error: {e}")),
    };
    if !path.exists() {
        return (0.0, "no vector index directory".to_string());
    }
    match latest_mtime(&path) {
        Ok(Some(mtime)) => {
            let age_days = (Utc::now() - mtime).num_seconds() as f32 / 86400.0;
            let score =
                ((INDEX_FRESH_WINDOW_DAYS - age_days) / INDEX_FRESH_WINDOW_DAYS).clamp(0.0, 1.0);
            let summary = if age_days < 1.0 {
                format!("updated {:.1}h ago", age_days * 24.0)
            } else {
                format!("updated {:.1}d ago", age_days)
            };
            (score, summary)
        }
        Ok(None) => (0.0, "no files in index".to_string()),
        Err(e) => (0.0, format!("mtime error: {e}")),
    }
}

async fn check_embedding_coverage(episode_count: usize) -> (f32, String) {
    if episode_count == 0 {
        return (1.0, "no episodes to index".to_string());
    }
    let (indexed, has_index) = match indexer::inspect_vector_index().await {
        Ok(v) => v,
        Err(e) => return (0.0, format!("inspect error: {e}")),
    };
    if !has_index {
        return (0.0, format!("0/{episode_count} episodes indexed"));
    }
    use std::cmp::Ordering;
    match indexed.cmp(&episode_count) {
        Ordering::Less => {
            let ratio = indexed as f32 / episode_count as f32;
            (ratio, format!("{indexed}/{episode_count} episodes indexed"))
        }
        Ordering::Equal => (1.0, format!("{indexed}/{episode_count} episodes indexed")),
        Ordering::Greater => {
            // Vector index has more rows than there are episode files. Usually
            // means the JSON for some episodes was deleted by hand but the
            // index wasn't rebuilt. Searches will return ghost IDs that
            // `store.load` can't resolve.
            let stale = indexed - episode_count;
            let ratio = 1.0 - (stale as f32 / indexed as f32);
            (
                ratio.clamp(0.0, 1.0),
                format!(
                    "{indexed} indexed but only {episode_count} on disk — {stale} stale; consider `tempera index --reindex`"
                ),
            )
        }
    }
}

fn check_keyword_coverage(episode_count: usize) -> (f32, String) {
    if episode_count == 0 {
        return (1.0, "no episodes".to_string());
    }
    let indexed = match indexer::inspect_keyword_index() {
        Ok(n) => n,
        Err(e) => return (0.0, format!("inspect error: {e}")),
    };
    if indexed == 0 {
        return (
            0.0,
            format!("0/{episode_count} in keyword index — run `tempera index`"),
        );
    }
    let ratio = (indexed as f32 / episode_count as f32).min(1.0);
    let summary = format!("{indexed}/{episode_count} docs in BM25 index");
    (ratio, summary)
}

fn check_session_links(episodes: &[Episode]) -> (f32, String) {
    let known: HashSet<&str> = episodes.iter().map(|e| e.id.as_str()).collect();
    let mut total_refs: u32 = 0;
    let mut broken_refs: u32 = 0;
    for ep in episodes {
        for related in &ep.related_episodes {
            total_refs += 1;
            if !known.contains(related.id.as_str()) {
                broken_refs += 1;
            }
        }
    }
    if total_refs == 0 {
        return (1.0, "no related-episode links yet".to_string());
    }
    let resolved = total_refs - broken_refs;
    let score = resolved as f32 / total_refs as f32;
    let summary = if broken_refs == 0 {
        format!("{total_refs}/{total_refs} refs resolve")
    } else {
        format!("{resolved}/{total_refs} refs resolve ({broken_refs} broken)")
    };
    (score, summary)
}

fn check_eval_pass_rate() -> (f32, String) {
    let baseline_dir = std::path::Path::new("evals/baselines");
    if !baseline_dir.exists() {
        return (
            0.0,
            "no baselines — run `tempera eval baseline --fixture <path>`".to_string(),
        );
    }
    let mut files: Vec<_> = match std::fs::read_dir(baseline_dir) {
        Ok(r) => r
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect(),
        Err(e) => return (0.0, format!("eval dir read: {e}")),
    };
    if files.is_empty() {
        return (0.0, "no baselines yet".to_string());
    }
    files.sort();
    let latest = files.last().expect("non-empty after check");
    let Ok(content) = std::fs::read_to_string(latest) else {
        return (0.0, "couldn't read latest baseline".to_string());
    };
    let Ok(json): std::result::Result<serde_json::Value, _> = serde_json::from_str(&content) else {
        return (0.0, "couldn't parse latest baseline".to_string());
    };
    // MRR is the right metric here: naturally 0-1, interpretable as
    // "average rank^-1 of first relevant hit". Insensitive to fixture's
    // relevant-doc density (unlike P@K, which is capped by it).
    let mrr = json
        .pointer("/aggregate/mrr")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let p_at_k = json
        .pointer("/aggregate/mean_p_at_k")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let mode = json
        .pointer("/mode")
        .and_then(|v| v.as_str())
        .unwrap_or("hybrid");
    let k = json.pointer("/k").and_then(|v| v.as_u64()).unwrap_or(5);
    let summary = format!("MRR={:.3}, P@{k}={:.3} ({mode})", mrr, p_at_k);
    (mrr.clamp(0.0, 1.0), summary)
}

async fn check_job_queue() -> (f32, String) {
    let queue = match JobQueue::open_default().await {
        Ok(q) => q,
        Err(_) => return (1.0, "queue not yet initialized".to_string()),
    };
    let all = match queue.list(None, 10_000).await {
        Ok(v) => v,
        Err(e) => return (0.0, format!("list error: {e}")),
    };
    if all.is_empty() {
        return (1.0, "no jobs in queue".to_string());
    }
    let dead = all.iter().filter(|j| j.status == JobStatus::Dead).count();
    let total = all.len();
    let score = 1.0 - (dead as f32 / total as f32);
    let summary = if dead == 0 {
        format!("{total} jobs, 0 dead")
    } else {
        format!("{dead} dead / {total} total")
    };
    (score, summary)
}

// ===== Helpers =====

/// Find the most recent mtime of any file under `dir` (recursive).
fn latest_mtime(dir: &Path) -> Result<Option<DateTime<Utc>>> {
    let mut best: Option<std::time::SystemTime> = None;
    walk(dir, &mut |p| {
        if let Ok(meta) = p.metadata()
            && let Ok(t) = meta.modified()
        {
            best = match best {
                None => Some(t),
                Some(prev) if t > prev => Some(t),
                Some(prev) => Some(prev),
            };
        }
    })?;
    Ok(best.map(|t| t.into()))
}

fn walk(dir: &Path, f: &mut dyn FnMut(&Path)) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk(&p, f)?;
        } else if ft.is_file() {
            f(&p);
        }
    }
    Ok(())
}

// ===== Output =====

pub fn print_human(report: &HealthReport) {
    let score = report.score;
    let header_color = if score >= 90 {
        format!("{score}").green()
    } else if score >= 60 {
        format!("{score}").yellow()
    } else {
        format!("{score}").red()
    };
    println!();
    println!(
        "{} {} / 100 (tempera v{})",
        "Health:".bold(),
        header_color,
        report.version
    );
    println!();
    let name_w = report
        .dimensions
        .iter()
        .map(|d| d.name.len())
        .max()
        .unwrap_or(20);
    for d in &report.dimensions {
        let glyph = match d.status {
            DimensionStatus::Pass => d.status.glyph().green(),
            DimensionStatus::Warn => d.status.glyph().yellow(),
            DimensionStatus::Fail => d.status.glyph().red(),
        };
        println!(
            "  {glyph}  {name:<name_w$}  {pts:>3}/{w:<3}  {summary}",
            name = d.name,
            pts = d.points,
            w = d.weight,
            summary = d.summary,
            name_w = name_w,
        );
    }
    println!();
    if score < 90 {
        println!("{}", "Hints to lift the score:".dimmed());
        for d in report
            .dimensions
            .iter()
            .filter(|d| d.status != DimensionStatus::Pass)
        {
            println!("  · {}: {}", d.name, hint_for(d.name).dimmed());
        }
        println!();
    }
    println!(
        "{}",
        "Run `tempera doctor --json` for machine-readable output.".dimmed()
    );
}

fn hint_for(name: &str) -> &'static str {
    match name {
        "index freshness" => {
            "run `tempera index` (or submit an index job via `tempera job submit index`)"
        }
        "embedding coverage" => {
            "run `tempera index` — episodes captured since last index aren't searchable yet"
        }
        "keyword coverage" => "run `tempera index` — the BM25 side rebuilds from scratch each time",
        "session links" => {
            "broken refs usually mean an episode was deleted by hand; safe to ignore unless reproducible"
        }
        "eval pass rate" => {
            "label more queries in your fixture, or investigate per-query regressions"
        }
        "job queue" => "inspect dead jobs with `tempera job list --status dead`",
        _ => "no hint available",
    }
}

// ===== Remediation (v0.7.6) =====

/// A single remediation action that can lift the health score.
///
/// v0.7.6 ships exactly one variant — `Reindex` — because one
/// `tempera index --reindex` covers the three index-related
/// dimensions (freshness, embedding coverage, keyword coverage) in
/// one shot. Future variants land here as new dimensions and fix
/// paths are added; the planner just maps low dimensions → steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RemediationStep {
    /// Full rebuild of the vector + keyword indexes. Cheap (free, ~30s)
    /// and idempotent.
    Reindex,
}

impl RemediationStep {
    pub fn kind(self) -> &'static str {
        match self {
            Self::Reindex => "reindex",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Reindex => "rebuild vector + keyword indexes from scratch",
        }
    }

    pub fn estimated_usd(self) -> f32 {
        match self {
            Self::Reindex => 0.0,
        }
    }

    pub fn estimated_seconds(self) -> u32 {
        match self {
            Self::Reindex => 30,
        }
    }

    /// Run the step. Always synchronous from the caller's perspective —
    /// no daemon dependency. Returns a one-line summary.
    pub async fn execute(self) -> Result<String> {
        match self {
            Self::Reindex => {
                let mut idx = crate::indexer::EpisodeIndexer::new().await?;
                let n = idx.index_all(true).await?;
                Ok(format!("reindexed {n} episodes"))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RemediationPlan {
    pub steps: Vec<RemediationStep>,
    pub estimated_usd: f32,
    pub estimated_seconds: u32,
}

/// Map low dimensions to the steps that would lift them. Deduplicates:
/// when multiple dimensions share a fix (e.g. `Reindex` covers index
/// freshness AND embedding coverage AND keyword coverage), the step
/// only appears once.
///
/// Steps are ordered so prerequisites land first — currently trivial
/// (single step), but the structure is here for v0.7.6+ additions.
pub fn plan_remediation(report: &HealthReport, max_usd: f32) -> RemediationPlan {
    use std::collections::BTreeSet;
    let mut chosen: BTreeSet<RemediationStep> = BTreeSet::new();
    let mut spent: f32 = 0.0;

    let by_name: std::collections::HashMap<&str, &DimensionReport> =
        report.dimensions.iter().map(|d| (d.name, d)).collect();

    let dim_below = |name: &str, threshold: f32| -> bool {
        by_name.get(name).is_some_and(|d| d.score < threshold)
    };

    // Reindex covers index freshness, embedding coverage, and keyword
    // coverage. If any of the three is below 0.9 (anything but Pass),
    // run it.
    if dim_below("index freshness", 0.9)
        || dim_below("embedding coverage", 0.9)
        || dim_below("keyword coverage", 0.9)
    {
        let step = RemediationStep::Reindex;
        if spent + step.estimated_usd() <= max_usd {
            chosen.insert(step);
            spent += step.estimated_usd();
        }
    }

    let steps: Vec<RemediationStep> = chosen.into_iter().collect();
    let estimated_seconds = steps.iter().map(|s| s.estimated_seconds()).sum();
    RemediationPlan {
        steps,
        estimated_usd: spent,
        estimated_seconds,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AppliedStep {
    pub step: RemediationStep,
    pub summary: String,
    pub elapsed_seconds: f64,
    pub cost_usd: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RemediationOutcome {
    pub initial_score: u32,
    pub final_score: u32,
    pub target_score: u32,
    pub target_reached: bool,
    pub spent_usd: f32,
    pub applied: Vec<AppliedStep>,
    pub skipped: Vec<RemediationSkip>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RemediationSkip {
    pub step: RemediationStep,
    pub reason: String,
}

/// Walk the plan: for each step, check budget + target, run, re-check
/// health, decide whether to continue. Stops on:
///   - score >= target_score
///   - next step would exceed max_usd
///   - a step errors (logged, recorded, plan continues to next step)
pub async fn execute_remediation(
    plan: RemediationPlan,
    target_score: u32,
    max_usd: f32,
) -> Result<RemediationOutcome> {
    let initial = check().await?;
    let initial_score = initial.score;
    let mut current_score = initial_score;
    let mut spent = 0.0;
    let mut applied = Vec::new();
    let mut skipped = Vec::new();

    for step in plan.steps {
        if current_score >= target_score {
            skipped.push(RemediationSkip {
                step,
                reason: format!("target {target_score} already reached"),
            });
            continue;
        }
        if spent + step.estimated_usd() > max_usd {
            skipped.push(RemediationSkip {
                step,
                reason: format!(
                    "would spend ${:.4} (cap ${:.4})",
                    spent + step.estimated_usd(),
                    max_usd
                ),
            });
            continue;
        }
        let started = std::time::Instant::now();
        match step.execute().await {
            Ok(summary) => {
                let elapsed = started.elapsed().as_secs_f64();
                spent += step.estimated_usd();
                applied.push(AppliedStep {
                    step,
                    summary,
                    elapsed_seconds: elapsed,
                    cost_usd: step.estimated_usd(),
                });
                // Re-check score after each step.
                if let Ok(rep) = check().await {
                    current_score = rep.score;
                }
            }
            Err(e) => {
                skipped.push(RemediationSkip {
                    step,
                    reason: format!("error: {e:#}"),
                });
            }
        }
    }

    Ok(RemediationOutcome {
        initial_score,
        final_score: current_score,
        target_score,
        target_reached: current_score >= target_score,
        spent_usd: spent,
        applied,
        skipped,
    })
}

pub fn print_remediation_plan(plan: &RemediationPlan) {
    println!();
    println!("{}", "Remediation plan".bold());
    if plan.steps.is_empty() {
        println!("  (no steps needed — every dimension at/above its threshold)");
    } else {
        for (i, step) in plan.steps.iter().enumerate() {
            println!(
                "  {}. {:<10}  ~{}s, ~${:.4}    {}",
                i + 1,
                step.kind(),
                step.estimated_seconds(),
                step.estimated_usd(),
                step.description()
            );
        }
        println!(
            "  total:        ~{}s, ~${:.4}",
            plan.estimated_seconds, plan.estimated_usd
        );
    }
    println!();
}

pub fn print_remediation_outcome(outcome: &RemediationOutcome) {
    println!();
    println!("{}", "Remediation outcome".bold());
    let arrow = if outcome.final_score > outcome.initial_score {
        format!("{} → {}", outcome.initial_score, outcome.final_score).green()
    } else if outcome.final_score < outcome.initial_score {
        format!("{} → {}", outcome.initial_score, outcome.final_score).red()
    } else {
        format!("{} → {}", outcome.initial_score, outcome.final_score).normal()
    };
    let target = if outcome.target_reached {
        format!("target {} ✓", outcome.target_score).green()
    } else {
        format!("target {}", outcome.target_score).yellow()
    };
    println!("  score:   {arrow}  ({target})");
    println!("  spent:   ${:.4}", outcome.spent_usd);
    if !outcome.applied.is_empty() {
        println!("  applied:");
        for a in &outcome.applied {
            println!(
                "    ✓ {:<10}  {}  ({:.2}s)",
                a.step.kind(),
                a.summary,
                a.elapsed_seconds
            );
        }
    }
    if !outcome.skipped.is_empty() {
        println!("  skipped:");
        for s in &outcome.skipped {
            println!("    ~ {:<10}  {}", s.step.kind(), s.reason);
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_status_thresholds() {
        assert_eq!(DimensionStatus::from_score(0.95), DimensionStatus::Pass);
        assert_eq!(DimensionStatus::from_score(0.90), DimensionStatus::Pass);
        assert_eq!(DimensionStatus::from_score(0.89), DimensionStatus::Warn);
        assert_eq!(DimensionStatus::from_score(0.50), DimensionStatus::Warn);
        assert_eq!(DimensionStatus::from_score(0.49), DimensionStatus::Fail);
        assert_eq!(DimensionStatus::from_score(0.0), DimensionStatus::Fail);
    }

    #[test]
    fn make_dim_clamps_and_rounds() {
        let d = make_dim("test", 10, (0.5, "x".into()));
        assert_eq!(d.points, 5);
        let d2 = make_dim("test", 10, (1.5, "x".into())); // out of range
        assert_eq!(d2.score, 1.0);
        assert_eq!(d2.points, 10);
        let d3 = make_dim("test", 10, (-0.3, "x".into()));
        assert_eq!(d3.score, 0.0);
        assert_eq!(d3.points, 0);
    }

    #[test]
    fn make_dim_status_from_score() {
        let pass = make_dim("a", 10, (0.95, "".into()));
        assert_eq!(pass.status, DimensionStatus::Pass);
        let warn = make_dim("a", 10, (0.7, "".into()));
        assert_eq!(warn.status, DimensionStatus::Warn);
        let fail = make_dim("a", 10, (0.1, "".into()));
        assert_eq!(fail.status, DimensionStatus::Fail);
    }

    #[test]
    fn weights_sum_to_100() {
        let total = W_INDEX_FRESHNESS
            + W_EMBEDDING_COVERAGE
            + W_KEYWORD_COVERAGE
            + W_SESSION_LINKS
            + W_EVAL_PASS_RATE
            + W_JOB_QUEUE;
        assert_eq!(total, 100, "doctor weights must sum to 100, got {total}");
    }

    #[test]
    fn check_session_links_no_links() {
        let ep = Episode::new("p".into(), "test".into());
        let (score, summary) = check_session_links(&[ep]);
        assert_eq!(score, 1.0);
        assert!(summary.contains("no related"));
    }

    #[test]
    fn check_session_links_all_resolve() {
        use crate::episode::{EpisodeRelation, RelatedEpisode};
        let ep1 = Episode::new("p".into(), "a".into());
        let mut ep2 = Episode::new("p".into(), "b".into());
        ep2.related_episodes.push(RelatedEpisode {
            id: ep1.id.clone(),
            relationship: EpisodeRelation::Continuation,
        });
        let (score, summary) = check_session_links(&[ep1, ep2]);
        assert_eq!(score, 1.0);
        assert!(summary.contains("1/1"));
    }

    #[test]
    fn check_session_links_broken_ref() {
        use crate::episode::{EpisodeRelation, RelatedEpisode};
        let mut ep = Episode::new("p".into(), "a".into());
        ep.related_episodes.push(RelatedEpisode {
            id: "does-not-exist".into(),
            relationship: EpisodeRelation::Continuation,
        });
        let (score, summary) = check_session_links(&[ep]);
        assert_eq!(score, 0.0);
        assert!(summary.contains("1 broken"));
    }

    // ===== Remediation plan tests =====

    fn make_dim_synthetic(name: &'static str, score: f32) -> DimensionReport {
        DimensionReport {
            name,
            weight: 25,
            score,
            points: (score * 25.0).round() as u32,
            status: DimensionStatus::from_score(score),
            summary: String::new(),
        }
    }

    fn make_report_synthetic(dims: Vec<DimensionReport>) -> HealthReport {
        let score: u32 = dims.iter().map(|d| d.points).sum::<u32>().min(100);
        HealthReport {
            score,
            dimensions: dims,
            version: "0.0.0-test".to_string(),
            generated_at: Utc::now(),
        }
    }

    #[test]
    fn plan_includes_reindex_when_index_freshness_low() {
        let r = make_report_synthetic(vec![
            make_dim_synthetic("index freshness", 0.2),
            make_dim_synthetic("embedding coverage", 1.0),
            make_dim_synthetic("keyword coverage", 1.0),
        ]);
        let plan = plan_remediation(&r, 1.0);
        assert_eq!(plan.steps, vec![RemediationStep::Reindex]);
    }

    #[test]
    fn plan_dedups_reindex_across_three_dimensions() {
        // Multiple dimensions failing → still only one Reindex.
        let r = make_report_synthetic(vec![
            make_dim_synthetic("index freshness", 0.2),
            make_dim_synthetic("embedding coverage", 0.3),
            make_dim_synthetic("keyword coverage", 0.4),
        ]);
        let plan = plan_remediation(&r, 1.0);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0], RemediationStep::Reindex);
    }

    #[test]
    fn plan_empty_when_everything_passes() {
        let r = make_report_synthetic(vec![
            make_dim_synthetic("index freshness", 0.99),
            make_dim_synthetic("embedding coverage", 1.0),
            make_dim_synthetic("keyword coverage", 1.0),
        ]);
        let plan = plan_remediation(&r, 1.0);
        assert!(plan.steps.is_empty());
        assert_eq!(plan.estimated_usd, 0.0);
    }

    #[test]
    fn plan_respects_max_usd_budget() {
        let r = make_report_synthetic(vec![make_dim_synthetic("index freshness", 0.2)]);
        // Reindex is free, so a $0 cap still includes it; only zero-cost
        // steps fit. (Future paid steps would be dropped.)
        let plan = plan_remediation(&r, 0.0);
        assert_eq!(plan.steps, vec![RemediationStep::Reindex]);
    }

    #[test]
    fn remediation_step_metadata_present() {
        let s = RemediationStep::Reindex;
        assert_eq!(s.kind(), "reindex");
        assert!(!s.description().is_empty());
        assert!(s.estimated_seconds() > 0);
        assert_eq!(s.estimated_usd(), 0.0);
    }
}
