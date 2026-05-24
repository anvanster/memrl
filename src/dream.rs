// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dream cycle orchestrator.
//!
//! A "dream" is a chain of phases that runs while no user-facing work is
//! happening: time-advance verifications, decay stale utility, reflect on
//! captures (v0.7.2+), find patterns (v0.7.3+), surface contradictions
//! (v0.7.4+), re-embed new artifacts.
//!
//! v0.7.1 ships the framework + two cheap phases:
//!   - `verify_advance`: promote Merged episodes past the stability window
//!     into `StableNoRevert`.
//!   - `decay`: scope-aware utility decay (the same math `tempera propagate`
//!     uses, callable as a phase so dream cycles can apply it without
//!     also re-running Bellman propagation).
//!
//! No LLM-bearing phases yet; the `CostBudget` plumbing is in place so the
//! authorship phases that ship in v0.7.2 just plug in.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::config::Config;
use crate::episode::VerificationState;
use crate::patterns;
use crate::reflect;
use crate::store::EpisodeStore;
use crate::triage;

/// All dream phases that exist today. Add a variant + a match arm in
/// `dispatch` + a row in `dependency_chain` to introduce a new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseName {
    /// Time-based promotion of `Merged` → `StableNoRevert`.
    VerifyAdvance,
    /// Scope-aware utility decay (per `episode::decay_rate_per_day`).
    Decay,
    /// Author a daily reflection page when triage approves (v0.7.3+).
    /// Runs Haiku triage internally; skips below-threshold days.
    Reflect,
    /// Cluster recent reflections and author pattern pages for clusters
    /// with `>= patterns_min_evidence` members (v0.7.4+).
    Patterns,
}

impl PhaseName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VerifyAdvance => "verify_advance",
            Self::Decay => "decay",
            Self::Reflect => "reflect",
            Self::Patterns => "patterns",
        }
    }
}

impl FromStr for PhaseName {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().replace('-', "_").as_str() {
            "verify_advance" => Self::VerifyAdvance,
            "decay" => Self::Decay,
            "reflect" => Self::Reflect,
            "patterns" => Self::Patterns,
            other => anyhow::bail!(
                "unknown phase '{other}' (expected: verify_advance | decay | reflect | patterns)"
            ),
        })
    }
}

impl std::fmt::Display for PhaseName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Phase execution order. Full-cycle runs walk this list top-to-bottom;
/// adding a phase = appending here. Dependencies are implicit in the
/// list order — if A reads what B wrote, A goes after B.
const FULL_CYCLE_ORDER: &[PhaseName] = &[
    PhaseName::VerifyAdvance,
    PhaseName::Decay,
    PhaseName::Reflect,
    PhaseName::Patterns,
];

pub fn all_phases() -> &'static [PhaseName] {
    FULL_CYCLE_ORDER
}

/// Estimated LLM/network cost a phase will incur. v0.7.1 phases are all
/// free; the structure is wire-ready for v0.7.2 authorship.
#[derive(Debug, Clone, Serialize)]
pub struct CostEstimate {
    pub usd: f32,
    pub estimated_seconds: u32,
}

impl CostEstimate {
    pub fn free(estimated_seconds: u32) -> Self {
        Self {
            usd: 0.0,
            estimated_seconds,
        }
    }
}

/// Shared budget enforced across an entire dream cycle. Phases check
/// `try_spend` before kicking off any paid work; the dream loop also
/// checks it before invoking a phase.
pub struct CostBudget {
    max_usd: f32,
    spent_cents: AtomicU64, // store as cents × 100 to keep atomic
}

impl CostBudget {
    pub fn new(max_usd: f32) -> Self {
        Self {
            max_usd,
            spent_cents: AtomicU64::new(0),
        }
    }

    pub fn max_usd(&self) -> f32 {
        self.max_usd
    }

    pub fn spent_usd(&self) -> f32 {
        self.spent_cents.load(Ordering::Relaxed) as f32 / 10_000.0
    }

    pub fn remaining_usd(&self) -> f32 {
        (self.max_usd - self.spent_usd()).max(0.0)
    }

    /// Attempt to charge `amount` USD. Returns Err when it would exceed
    /// the cap — callers should bail rather than partially run.
    pub fn try_spend(&self, amount_usd: f32) -> Result<()> {
        // Convert to fixed-point so the atomic addition is safe.
        let amount_cents = (amount_usd * 10_000.0).round() as u64;
        let prior = self.spent_cents.fetch_add(amount_cents, Ordering::Relaxed);
        let post_usd = (prior + amount_cents) as f32 / 10_000.0;
        if post_usd > self.max_usd + 1e-6 {
            // Roll back: best-effort undo. Concurrent spenders may still
            // exceed the cap by a small margin, which is acceptable.
            self.spent_cents.fetch_sub(amount_cents, Ordering::Relaxed);
            anyhow::bail!(
                "cost cap reached: would spend ${:.4} (cap ${:.4})",
                amount_usd,
                self.max_usd
            );
        }
        Ok(())
    }
}

/// Context handed to each phase.
pub struct PhaseContext<'a> {
    pub config: &'a Config,
    pub budget: &'a CostBudget,
    pub store: &'a EpisodeStore,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseReport {
    pub name: PhaseName,
    pub status: PhaseStatus,
    /// One-line human summary of what the phase did.
    pub summary: String,
    /// Dollars actually spent. Always 0.0 for v0.7.1 phases.
    pub cost_usd: f32,
    /// Wallclock seconds.
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    Ok,
    Skipped,
    Failed,
}

/// Outcome of running an entire cycle (one or more phases).
#[derive(Debug, Clone, Serialize)]
pub struct CycleReport {
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub budget_max_usd: f32,
    pub budget_spent_usd: f32,
    pub phases: Vec<PhaseReport>,
}

// ===== Dispatch =====

fn estimate(phase: PhaseName) -> CostEstimate {
    match phase {
        PhaseName::VerifyAdvance => CostEstimate::free(2),
        PhaseName::Decay => CostEstimate::free(5),
        // Reflect estimate covers one Sonnet authorship + one Haiku triage.
        // Days that triage skips cost only the triage call (~$0.002).
        PhaseName::Reflect => CostEstimate {
            usd: reflect::REFLECT_ESTIMATED_COST_USD + triage::TRIAGE_ESTIMATED_COST_USD,
            estimated_seconds: 20,
        },
        // Patterns: assume up to 3 clusters to author per run. Many days
        // will produce 0–1; this is a worst-case budget guard.
        PhaseName::Patterns => CostEstimate {
            usd: patterns::PATTERN_ESTIMATED_COST_USD * 3.0,
            estimated_seconds: 30,
        },
    }
}

async fn dispatch(phase: PhaseName, ctx: &PhaseContext<'_>) -> Result<PhaseReport> {
    let started = Instant::now();
    let (status, summary, cost) = match phase {
        PhaseName::VerifyAdvance => {
            let (s, m) = run_verify_advance(ctx)?;
            (s, m, 0.0)
        }
        PhaseName::Decay => {
            let (s, m) = run_decay(ctx).await?;
            (s, m, 0.0)
        }
        PhaseName::Reflect => run_reflect(ctx).await?,
        PhaseName::Patterns => run_patterns(ctx).await?,
    };
    Ok(PhaseReport {
        name: phase,
        status,
        summary,
        cost_usd: cost,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

// ===== Runner =====

/// Run a single phase by name. Used by `tempera dream --phase <name>`.
pub async fn run_one(phase: PhaseName, config: &Config, max_usd: f32) -> Result<CycleReport> {
    run_phases(&[phase], config, max_usd, false).await
}

/// Run the full default cycle.
pub async fn run_cycle(config: &Config, max_usd: f32) -> Result<CycleReport> {
    run_phases(all_phases(), config, max_usd, false).await
}

/// Plan-only: report what would run and what it would cost, without
/// actually running. Useful for `--dry-run`.
pub fn plan(phases: &[PhaseName], max_usd: f32) -> CycleReport {
    let started = Utc::now();
    let mut reports = Vec::new();
    let mut total_seconds = 0_u32;
    for p in phases {
        let est = estimate(*p);
        total_seconds = total_seconds.saturating_add(est.estimated_seconds);
        reports.push(PhaseReport {
            name: *p,
            status: PhaseStatus::Skipped,
            summary: format!("[dry-run] ~{}s, ~${:.4}", est.estimated_seconds, est.usd),
            cost_usd: est.usd,
            elapsed_seconds: 0.0,
        });
    }
    let finished = Utc::now();
    CycleReport {
        started_at: started,
        finished_at: finished,
        budget_max_usd: max_usd,
        budget_spent_usd: 0.0,
        phases: reports,
    }
}

async fn run_phases(
    phases: &[PhaseName],
    config: &Config,
    max_usd: f32,
    is_dry: bool,
) -> Result<CycleReport> {
    if is_dry {
        return Ok(plan(phases, max_usd));
    }
    let started = Utc::now();
    let budget = CostBudget::new(max_usd);
    let store = EpisodeStore::new()?;
    let ctx = PhaseContext {
        config,
        budget: &budget,
        store: &store,
    };

    let mut reports = Vec::new();
    for &phase in phases {
        let est = estimate(phase);
        if est.usd > 0.0 && budget.remaining_usd() < est.usd {
            reports.push(PhaseReport {
                name: phase,
                status: PhaseStatus::Skipped,
                summary: format!(
                    "skipped: estimated ${:.4} exceeds remaining ${:.4}",
                    est.usd,
                    budget.remaining_usd()
                ),
                cost_usd: 0.0,
                elapsed_seconds: 0.0,
            });
            continue;
        }
        match dispatch(phase, &ctx).await {
            Ok(r) => reports.push(r),
            Err(e) => reports.push(PhaseReport {
                name: phase,
                status: PhaseStatus::Failed,
                summary: format!("error: {e:#}"),
                cost_usd: 0.0,
                elapsed_seconds: 0.0,
            }),
        }
    }

    Ok(CycleReport {
        started_at: started,
        finished_at: Utc::now(),
        budget_max_usd: max_usd,
        budget_spent_usd: budget.spent_usd(),
        phases: reports,
    })
}

// ===== Phase implementations =====

/// v0.7.1: promote `Merged` episodes past the stability window to
/// `StableNoRevert`. Idempotent. Reads `config.dream.stable_threshold_days`.
fn run_verify_advance(ctx: &PhaseContext<'_>) -> Result<(PhaseStatus, String)> {
    let threshold_days = ctx.config.dream.stable_threshold_days as i64;
    let now = Utc::now();
    let episodes = ctx.store.list_all()?;
    let mut promoted = 0_usize;
    let mut already_stable = 0_usize;
    let mut not_yet = 0_usize;
    for ep in episodes {
        if let VerificationState::Merged { commit: _, at } = &ep.outcome.verification {
            let days = (now - *at).num_days();
            if days >= threshold_days {
                let mut updated = ep.clone();
                updated.set_verification(VerificationState::StableNoRevert {
                    days: days as u32,
                    since: *at,
                });
                ctx.store
                    .update(&updated)
                    .context("Failed to update verification state")?;
                promoted += 1;
            } else {
                not_yet += 1;
            }
        } else if matches!(
            ep.outcome.verification,
            VerificationState::StableNoRevert { .. }
                | VerificationState::ValidatedCrossProject { .. }
        ) {
            already_stable += 1;
        }
    }
    let summary = format!(
        "{promoted} promoted to stable, {not_yet} pending (<{threshold_days}d), {already_stable} already stable"
    );
    Ok((PhaseStatus::Ok, summary))
}

/// v0.7.1: scope-aware utility decay. Wraps the same math used by
/// `tempera propagate` but doesn't run the Bellman propagation step.
async fn run_decay(ctx: &PhaseContext<'_>) -> Result<(PhaseStatus, String)> {
    let report = crate::utility::run_decay_only(ctx.config)
        .await
        .context("decay phase failed")?;
    let summary = format!(
        "{} episodes decayed (Δ {:+.3} total utility)",
        report.decayed_episodes, report.total_utility_change
    );
    Ok((PhaseStatus::Ok, summary))
}

/// v0.7.3: reflect on yesterday's captures.
///
/// Triages first (Haiku) — if score < MIN_SYNTHESIZE_SCORE, skip the
/// expensive Sonnet call. If a reflection already exists for the day,
/// skip (idempotent). Otherwise author + persist.
///
/// "Yesterday" is the natural target for a nightly cron: today's
/// captures aren't done yet. CLI lets users override the date.
async fn run_reflect(ctx: &PhaseContext<'_>) -> Result<(PhaseStatus, String, f32)> {
    let target_date = (Utc::now() - chrono::Duration::days(1)).date_naive();
    let all = ctx.store.list_all()?;
    let day_eps: Vec<_> = all
        .into_iter()
        .filter(|e| e.timestamp_start.date_naive() == target_date)
        .collect();

    // Skip if no captures.
    if day_eps.is_empty() {
        return Ok((
            PhaseStatus::Skipped,
            format!("no captures on {target_date}"),
            0.0,
        ));
    }

    // Skip if we've already authored a reflection for this day.
    let reflect_store = reflect::ReflectionStore::open_default().await?;
    let existing_id = reflect::Reflection::id_for(&target_date, None);
    if reflect_store.get(&existing_id).await?.is_some() {
        return Ok((
            PhaseStatus::Skipped,
            format!("reflection already exists for {target_date}"),
            0.0,
        ));
    }

    // Triage.
    let triage_store = triage::TriageStore::open_default().await?;
    let (verdict, from_cache) = triage::triage_day_with_model(
        &target_date,
        &day_eps,
        &triage_store,
        Some(ctx.budget),
        false,
        &ctx.config.dream.triage_model,
    )
    .await
    .context("reflect: triage failed")?;
    let triage_cost = if from_cache {
        0.0
    } else {
        triage::TRIAGE_ESTIMATED_COST_USD
    };

    if !verdict.worth_synthesizing() {
        return Ok((
            PhaseStatus::Skipped,
            format!(
                "triage said skip ({} ep, score {:.2}, {} signals)",
                day_eps.len(),
                verdict.score,
                verdict.signals.len()
            ),
            triage_cost,
        ));
    }

    // Author.
    let reflection = reflect::author_reflection(
        &target_date,
        &day_eps,
        &verdict.signals,
        verdict.score,
        ctx.config,
        Some(ctx.budget),
        &reflect_store,
    )
    .await
    .context("reflect: authorship failed")?;
    let total_cost = triage_cost + reflect::REFLECT_ESTIMATED_COST_USD;
    Ok((
        PhaseStatus::Ok,
        format!(
            "authored {} ({} citations, triage {:.2})",
            reflection.id,
            reflection.citations.len(),
            verdict.score
        ),
        total_cost,
    ))
}

/// v0.7.4: cluster reflections from the last N days and author pattern
/// pages for clusters with shared themes.
async fn run_patterns(ctx: &PhaseContext<'_>) -> Result<(PhaseStatus, String, f32)> {
    let report = patterns::run_patterns(ctx.config, Some(ctx.budget))
        .await
        .context("patterns phase failed")?;
    let cost_usd = report.patterns_written as f32 * patterns::PATTERN_ESTIMATED_COST_USD;
    let summary = if report.reflections_examined < ctx.config.dream.patterns_min_evidence {
        format!(
            "{} reflection(s) — below min_evidence {} (no clustering)",
            report.reflections_examined, ctx.config.dream.patterns_min_evidence
        )
    } else {
        format!(
            "examined {} reflections, {} clusters ({} above min), {} pattern(s) written, {} existing, {} themeless",
            report.reflections_examined,
            report.clusters_found,
            report.clusters_above_min,
            report.patterns_written,
            report.patterns_skipped_existing,
            report.clusters_with_no_theme,
        )
    };
    Ok((PhaseStatus::Ok, summary, cost_usd))
}

// ===== Output =====

pub fn print_cycle(report: &CycleReport, dry: bool) {
    use colored::Colorize;
    let elapsed = (report.finished_at - report.started_at).num_milliseconds() as f64 / 1000.0;
    println!();
    let header = if dry {
        "Dream cycle plan (dry-run)".bold()
    } else {
        "Dream cycle".bold()
    };
    println!("{}", header);
    println!(
        "  started:  {}",
        report.started_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    if !dry {
        println!("  elapsed:  {:.2}s", elapsed);
    }
    println!(
        "  budget:   ${:.4} cap, ${:.4} spent",
        report.budget_max_usd, report.budget_spent_usd
    );
    println!();
    for p in &report.phases {
        let (glyph, label) = match p.status {
            PhaseStatus::Ok => ("✓".green(), "ok".green()),
            PhaseStatus::Skipped => ("~".yellow(), "skipped".yellow()),
            PhaseStatus::Failed => ("✗".red(), "failed".red()),
        };
        println!(
            "  {glyph} {name:<18} [{label}] {summary} ({elapsed:.2}s)",
            name = p.name.as_str(),
            summary = p.summary,
            elapsed = p.elapsed_seconds,
        );
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_name_roundtrip() {
        for p in all_phases() {
            let parsed: PhaseName = p.as_str().parse().unwrap();
            assert_eq!(parsed, *p);
        }
    }

    #[test]
    fn phase_name_parses_dashed_input() {
        let p: PhaseName = "verify-advance".parse().unwrap();
        assert_eq!(p, PhaseName::VerifyAdvance);
    }

    #[test]
    fn phase_name_rejects_unknown() {
        assert!("synthesize".parse::<PhaseName>().is_err());
    }

    #[test]
    fn budget_spend_then_remaining() {
        let b = CostBudget::new(1.00);
        assert_eq!(b.spent_usd(), 0.0);
        b.try_spend(0.30).unwrap();
        assert!((b.spent_usd() - 0.30).abs() < 1e-3);
        assert!((b.remaining_usd() - 0.70).abs() < 1e-3);
    }

    #[test]
    fn budget_rejects_when_over_cap() {
        let b = CostBudget::new(0.10);
        b.try_spend(0.08).unwrap();
        let err = b.try_spend(0.05).unwrap_err();
        assert!(err.to_string().contains("cost cap reached"));
        // Failed spend rolled back — second small spend still works.
        b.try_spend(0.01).unwrap();
        assert!(b.spent_usd() < 0.10);
    }

    #[test]
    fn budget_accepts_exact_cap() {
        let b = CostBudget::new(0.10);
        b.try_spend(0.10).unwrap();
        assert!((b.spent_usd() - 0.10).abs() < 1e-3);
        // One more cent should fail.
        assert!(b.try_spend(0.01).is_err());
    }

    #[test]
    fn plan_produces_skipped_status_per_phase() {
        let r = plan(all_phases(), 1.00);
        assert_eq!(r.phases.len(), all_phases().len());
        for p in &r.phases {
            assert_eq!(p.status, PhaseStatus::Skipped);
        }
    }

    #[test]
    fn estimate_free_phases_remain_free() {
        for p in &[PhaseName::VerifyAdvance, PhaseName::Decay] {
            let e = estimate(*p);
            assert_eq!(e.usd, 0.0, "free phase {p:?} regressed to paid");
        }
    }

    #[test]
    fn estimate_paid_phases_have_positive_cost() {
        // Catches regressions if someone accidentally zeroes a paid
        // phase's estimate.
        for p in &[PhaseName::Reflect, PhaseName::Patterns] {
            let e = estimate(*p);
            assert!(e.usd > 0.0, "paid phase {p:?} has zero estimate");
        }
    }
}
