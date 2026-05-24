// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Contradiction probe.
//!
//! Most of what's wrong with a memory isn't missing data — it's stale
//! data quietly disagreeing with newer data. The probe samples
//! frequently-retrieved BKM pairs whose embeddings are *related but not
//! duplicate* (cosine in `[contradict_min_similarity,
//! contradict_max_similarity]`), asks Haiku whether they contradict on
//! a factual claim, and stores the verdicts with a severity grading.
//!
//! Output is **data, not action**: a list of findings + a calibrated
//! Wilson 95% confidence interval on the probe-wide contradiction rate.
//! The operator (or future doctor remediation) decides what to do.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;

use crate::config::Config;
use crate::dream::CostBudget;
use crate::episode::Episode;
use crate::indexer::EpisodeIndexer;
use crate::llm::AnthropicClient;
use crate::patterns::cosine_similarity;
use crate::store::EpisodeStore;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// USD per judge call. Haiku 4.5 with ~600 input + ~100 output tokens
/// → ~$0.001. Doubled here as a budget guard.
pub const JUDGE_ESTIMATED_COST_USD: f32 = 0.002;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Same claim phrased differently, or a factual nit ("Alice Smith" vs "A. Smith").
    Low,
    /// Values may be stale (revenue / headcount / version-pinned counts).
    Medium,
    /// Identity- or structural-claim conflict (founder, CEO, "module X owns Y").
    High,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "high" => Self::High,
            "medium" | "mid" => Self::Medium,
            _ => Self::Low,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub id: Option<i64>,
    pub episode_a: String,
    pub episode_b: String,
    pub severity: Severity,
    pub confidence: f32,
    pub explanation: String,
    pub resolution_hint: Option<String>,
    pub similarity: f32,
    pub found_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolved_action: Option<String>,
}

// ===== Store =====

#[derive(Clone)]
pub struct ContradictionStore {
    pool: SqlitePool,
}

impl ContradictionStore {
    pub async fn open_default() -> Result<Self> {
        let path = Config::data_dir()?.join("jobs.sqlite");
        Self::open(&path).await
    }

    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts)
            .await
            .with_context(|| format!("Failed to open contradictions DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run contradiction migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Upsert by (episode_a, episode_b) — re-runs of the probe refresh
    /// existing rows rather than duplicating.
    pub async fn put(&self, c: &Contradiction) -> Result<()> {
        sqlx::query(
            "INSERT INTO contradictions
                (episode_a, episode_b, severity, confidence, explanation,
                 resolution_hint, similarity, found_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(episode_a, episode_b) DO UPDATE SET
                severity = excluded.severity,
                confidence = excluded.confidence,
                explanation = excluded.explanation,
                resolution_hint = excluded.resolution_hint,
                similarity = excluded.similarity,
                found_at = excluded.found_at",
        )
        .bind(&c.episode_a)
        .bind(&c.episode_b)
        .bind(c.severity.as_str())
        .bind(c.confidence)
        .bind(&c.explanation)
        .bind(c.resolution_hint.as_deref())
        .bind(c.similarity)
        .bind(c.found_at.timestamp())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_active(&self, limit: i64) -> Result<Vec<Contradiction>> {
        let rows = sqlx::query(
            "SELECT id, episode_a, episode_b, severity, confidence, explanation,
                    resolution_hint, similarity, found_at, resolved_at, resolved_action
             FROM contradictions
             WHERE resolved_at IS NULL
             ORDER BY
                 CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                 found_at DESC
             LIMIT ?1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_contradiction).collect()
    }

    pub async fn mark_resolved(&self, id: i64, action: &str) -> Result<()> {
        sqlx::query(
            "UPDATE contradictions
             SET resolved_at = ?1, resolved_action = ?2
             WHERE id = ?3",
        )
        .bind(Utc::now().timestamp())
        .bind(action)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    fn row_to_contradiction(row: &sqlx::sqlite::SqliteRow) -> Result<Contradiction> {
        let severity_str: String = row.get("severity");
        let found_ts: i64 = row.get("found_at");
        let resolved_ts: Option<i64> = row.get("resolved_at");
        Ok(Contradiction {
            id: Some(row.get("id")),
            episode_a: row.get("episode_a"),
            episode_b: row.get("episode_b"),
            severity: Severity::from_str_lossy(&severity_str),
            confidence: row.get("confidence"),
            explanation: row.get("explanation"),
            resolution_hint: row.get("resolution_hint"),
            similarity: row.get("similarity"),
            found_at: DateTime::<Utc>::from_timestamp(found_ts, 0).unwrap_or_default(),
            resolved_at: resolved_ts.and_then(|t| DateTime::<Utc>::from_timestamp(t, 0)),
            resolved_action: row.get("resolved_action"),
        })
    }
}

// ===== Wilson CI =====

/// Wilson score 95% confidence interval for a binomial proportion.
/// Better than the naive Normal-approximation for small n / extreme p.
pub fn wilson_ci_95(positives: u32, total: u32) -> (f32, f32) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let n = total as f32;
    let p = positives as f32 / n;
    let z: f32 = 1.96;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half_width = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
    let lower = (center - half_width).clamp(0.0, 1.0);
    let upper = (center + half_width).clamp(0.0, 1.0);
    (lower, upper)
}

// ===== Pair selection =====

/// A candidate pair returned by `select_pairs`.
struct PairCandidate {
    a: Episode,
    b: Episode,
    similarity: f32,
}

/// Take top N episodes by retrieval count, then build all pairs with
/// embedding cosine in `[min_sim, max_sim]`, sorted by similarity DESC
/// (most-likely-related-but-different first). Truncated to `max_pairs`.
async fn select_pairs(
    top_n: usize,
    min_sim: f32,
    max_sim: f32,
    max_pairs: usize,
    store: &EpisodeStore,
    indexer: &EpisodeIndexer,
) -> Result<Vec<PairCandidate>> {
    let mut all = store.list_all()?;
    all.sort_by(|a, b| b.utility.retrieval_count.cmp(&a.utility.retrieval_count));
    all.truncate(top_n);

    // Embed each one. Reuse the same embedding text that the indexer
    // uses for the vector index so similarity here matches what
    // retrieval sees.
    let embedded: Vec<(Episode, Vec<f32>)> = all
        .into_iter()
        .filter_map(|ep| {
            let text = episode_to_intent_text(&ep);
            indexer.embed(&text).ok().map(|v| (ep, v))
        })
        .collect();

    let mut pairs: Vec<PairCandidate> = Vec::new();
    for i in 0..embedded.len() {
        for j in (i + 1)..embedded.len() {
            let sim = cosine_similarity(&embedded[i].1, &embedded[j].1);
            if sim >= min_sim && sim <= max_sim {
                pairs.push(PairCandidate {
                    a: embedded[i].0.clone(),
                    b: embedded[j].0.clone(),
                    similarity: sim,
                });
            }
        }
    }
    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(max_pairs);
    Ok(pairs)
}

fn episode_to_intent_text(ep: &Episode) -> String {
    if !ep.intent.extracted_intent.is_empty() {
        ep.intent.extracted_intent.clone()
    } else {
        ep.intent.raw_prompt.clone()
    }
}

// ===== Judge =====

const JUDGE_SYSTEM: &str = r#"Two episodes appear to answer questions about the same topic. Do they CONTRADICT on a factual claim about code or behavior?

Score each pair:
- contradicts:      true if they disagree on a verifiable fact (function returns X vs Y; module M is in dir A vs B; approach P is faster vs slower; convention is X vs Y). Disagreement on values that change over time (version pinned, but versions changed) still counts.
- confidence:       0.0-1.0. Be conservative — false positives are worse than missed signals.
- severity:
    "high":   structural / identity claims (who owns this, where this lives, what this is). Wrong answers mislead future debugging.
    "medium": values that may be stale (counts, versions, performance numbers).
    "low":    phrasing / naming / format nits.
- explanation:      one sentence. What's the specific factual disagreement?
- resolution_hint:  "supersede" | "keep_both" | "needs_review"
    supersede:    the newer one wins, the older should retire.
    keep_both:    intentional disagreement (e.g. two valid approaches for different contexts).
    needs_review: you can't tell from the captures alone.

Output JSON only:
{
  "contradicts": bool,
  "confidence": 0.0-1.0,
  "severity": "low" | "medium" | "high",
  "explanation": "one sentence",
  "resolution_hint": "supersede" | "keep_both" | "needs_review"
}
"#;

fn intent_of(ep: &Episode) -> &str {
    if !ep.intent.extracted_intent.is_empty() {
        ep.intent.extracted_intent.as_str()
    } else {
        ep.intent.raw_prompt.as_str()
    }
}

fn build_judge_user_message(a: &Episode, b: &Episode) -> String {
    format!(
        "Episode A (id: {a_id}, captured {a_date}):\n  Claim: {a_claim}\n  Outcome: {a_outcome}\n\nEpisode B (id: {b_id}, captured {b_date}):\n  Claim: {b_claim}\n  Outcome: {b_outcome}\n",
        a_id = &a.id[..8.min(a.id.len())],
        a_date = a.timestamp_start.format("%Y-%m-%d"),
        a_claim = intent_of(a),
        a_outcome = a.outcome.status,
        b_id = &b.id[..8.min(b.id.len())],
        b_date = b.timestamp_start.format("%Y-%m-%d"),
        b_claim = intent_of(b),
        b_outcome = b.outcome.status,
    )
}

#[derive(Debug, Deserialize)]
struct JudgeVerdict {
    contradicts: bool,
    confidence: f32,
    #[serde(default)]
    severity: Option<String>,
    explanation: String,
    #[serde(default)]
    resolution_hint: Option<String>,
}

async fn judge_pair(
    a: &Episode,
    b: &Episode,
    model: &str,
    budget: Option<&CostBudget>,
) -> Result<JudgeVerdict> {
    if let Some(bud) = budget {
        bud.try_spend(JUDGE_ESTIMATED_COST_USD)?;
    }
    let user = build_judge_user_message(a, b);
    let client = AnthropicClient::with_model(model)?;
    let raw = client.raw_completion(JUDGE_SYSTEM, &user, 250).await?;
    let trimmed = strip_json_fences(&raw);
    let verdict: JudgeVerdict = serde_json::from_str(trimmed)
        .with_context(|| format!("failed to parse judge JSON: {raw}"))?;
    Ok(verdict)
}

fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    s.trim_end_matches("```").trim()
}

// ===== Pipeline =====

#[derive(Debug, Clone, Serialize)]
pub struct ContradictionReport {
    pub pairs_evaluated: u32,
    pub contradictions_found: u32,
    /// Wilson-95 CI on contradictions_found / pairs_evaluated.
    pub rate_ci_lower: f32,
    pub rate_ci_upper: f32,
    pub by_severity: SeverityCounts,
    /// `n < 30` flag — CI is too wide to act on individual numbers.
    pub small_sample: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SeverityCounts {
    pub low: u32,
    pub medium: u32,
    pub high: u32,
}

/// Run a contradiction probe. Returns a report even when no pairs were
/// found (e.g. brain is too small or has no related-but-not-duplicate
/// episode pairs).
pub async fn run_probe(
    config: &Config,
    budget: Option<&CostBudget>,
) -> Result<ContradictionReport> {
    let store = EpisodeStore::new()?;
    let indexer = EpisodeIndexer::new()
        .await
        .context("contradict: failed to open embedder")?;

    let pairs = select_pairs(
        config.dream.contradict_top_n,
        config.dream.contradict_min_similarity,
        config.dream.contradict_max_similarity,
        config.dream.contradict_max_pairs,
        &store,
        &indexer,
    )
    .await?;

    if pairs.is_empty() {
        return Ok(ContradictionReport {
            pairs_evaluated: 0,
            contradictions_found: 0,
            rate_ci_lower: 0.0,
            rate_ci_upper: 0.0,
            by_severity: SeverityCounts::default(),
            small_sample: true,
        });
    }

    let contradiction_store = ContradictionStore::open_default().await?;
    let mut positives: u32 = 0;
    let mut by_severity = SeverityCounts::default();

    for pair in &pairs {
        let a = &pair.a;
        let b = &pair.b;
        let sim = pair.similarity;
        let verdict = match judge_pair(a, b, &config.dream.triage_model, budget).await {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "contradict: judge failed on pair ({}, {}): {e}",
                    &a.id[..8],
                    &b.id[..8]
                );
                continue;
            }
        };
        if verdict.confidence < config.dream.contradict_min_confidence {
            continue;
        }
        if !verdict.contradicts {
            continue;
        }
        positives += 1;
        let severity = verdict
            .severity
            .as_deref()
            .map(Severity::from_str_lossy)
            .unwrap_or(Severity::Low);
        match severity {
            Severity::Low => by_severity.low += 1,
            Severity::Medium => by_severity.medium += 1,
            Severity::High => by_severity.high += 1,
        }
        let row = Contradiction {
            id: None,
            episode_a: a.id.clone(),
            episode_b: b.id.clone(),
            severity,
            confidence: verdict.confidence,
            explanation: verdict.explanation,
            resolution_hint: verdict.resolution_hint,
            similarity: sim,
            found_at: Utc::now(),
            resolved_at: None,
            resolved_action: None,
        };
        if let Err(e) = contradiction_store.put(&row).await {
            eprintln!("contradict: failed to store finding: {e}");
        }
    }

    let total = pairs.len() as u32;
    let (lower, upper) = wilson_ci_95(positives, total);
    Ok(ContradictionReport {
        pairs_evaluated: total,
        contradictions_found: positives,
        rate_ci_lower: lower,
        rate_ci_upper: upper,
        by_severity,
        small_sample: total < 30,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_zero_total_returns_zero() {
        let (l, u) = wilson_ci_95(0, 0);
        assert_eq!(l, 0.0);
        assert_eq!(u, 0.0);
    }

    #[test]
    fn wilson_all_positives() {
        // 10/10 → centre near 1.0, upper clamped at 1.0, lower well above 0.5.
        let (l, u) = wilson_ci_95(10, 10);
        assert!(l > 0.6, "lower should be > 0.6, got {l}");
        assert!(u <= 1.0 && u > 0.9, "upper should be near 1, got {u}");
    }

    #[test]
    fn wilson_zero_positives() {
        // 0/10 → centre near 0, lower clamped at 0.0.
        let (l, u) = wilson_ci_95(0, 10);
        assert!(l <= 0.05, "lower should be near 0, got {l}");
        assert!(u > 0.2 && u < 0.5, "upper should be ~0.3, got {u}");
    }

    #[test]
    fn wilson_half_positives_n_30() {
        // 15/30 → CI roughly 0.33-0.67.
        let (l, u) = wilson_ci_95(15, 30);
        assert!(l > 0.3 && l < 0.4, "lower ~0.33, got {l}");
        assert!(u > 0.6 && u < 0.7, "upper ~0.67, got {u}");
    }

    #[test]
    fn wilson_small_sample_has_wide_ci() {
        // 1/3 → CI roughly 0.06-0.79.
        let (l, u) = wilson_ci_95(1, 3);
        assert!(l < 0.15);
        assert!(u > 0.7);
    }

    #[test]
    fn severity_str_roundtrip() {
        for s in &[Severity::Low, Severity::Medium, Severity::High] {
            assert_eq!(Severity::from_str_lossy(s.as_str()), *s);
        }
        // Bogus input falls back to Low (conservative).
        assert_eq!(Severity::from_str_lossy("nonsense"), Severity::Low);
    }

    #[test]
    fn severity_lossy_handles_aliases() {
        assert_eq!(Severity::from_str_lossy("HIGH"), Severity::High);
        assert_eq!(Severity::from_str_lossy("mid"), Severity::Medium);
    }

    #[test]
    fn strip_json_fences_works() {
        assert_eq!(
            strip_json_fences("```json\n{\"contradicts\":true}\n```"),
            "{\"contradicts\":true}"
        );
        assert_eq!(strip_json_fences("  {\"a\":1}  "), "{\"a\":1}");
    }

    #[tokio::test]
    async fn store_roundtrip() {
        let s = ContradictionStore::open_in_memory().await.unwrap();
        let c = Contradiction {
            id: None,
            episode_a: "ep_a_uuid".to_string(),
            episode_b: "ep_b_uuid".to_string(),
            severity: Severity::Medium,
            confidence: 0.85,
            explanation: "A says X is 5, B says X is 7.".to_string(),
            resolution_hint: Some("supersede".to_string()),
            similarity: 0.78,
            found_at: Utc::now(),
            resolved_at: None,
            resolved_action: None,
        };
        s.put(&c).await.unwrap();
        let active = s.list_active(10).await.unwrap();
        assert_eq!(active.len(), 1);
        let got = &active[0];
        assert_eq!(got.episode_a, "ep_a_uuid");
        assert_eq!(got.severity, Severity::Medium);
        assert!((got.confidence - 0.85).abs() < 1e-3);
    }

    #[tokio::test]
    async fn store_upsert_by_pair() {
        let s = ContradictionStore::open_in_memory().await.unwrap();
        let mk = |conf: f32| Contradiction {
            id: None,
            episode_a: "a".to_string(),
            episode_b: "b".to_string(),
            severity: Severity::Low,
            confidence: conf,
            explanation: "x".to_string(),
            resolution_hint: None,
            similarity: 0.7,
            found_at: Utc::now(),
            resolved_at: None,
            resolved_action: None,
        };
        s.put(&mk(0.5)).await.unwrap();
        s.put(&mk(0.9)).await.unwrap();
        let active = s.list_active(10).await.unwrap();
        assert_eq!(active.len(), 1);
        assert!((active[0].confidence - 0.9).abs() < 1e-3);
    }

    #[tokio::test]
    async fn store_list_active_sorts_high_severity_first() {
        let s = ContradictionStore::open_in_memory().await.unwrap();
        let mk = |a: &str, b: &str, sev: Severity| Contradiction {
            id: None,
            episode_a: a.to_string(),
            episode_b: b.to_string(),
            severity: sev,
            confidence: 0.9,
            explanation: "x".to_string(),
            resolution_hint: None,
            similarity: 0.7,
            found_at: Utc::now(),
            resolved_at: None,
            resolved_action: None,
        };
        s.put(&mk("a1", "b1", Severity::Low)).await.unwrap();
        s.put(&mk("a2", "b2", Severity::High)).await.unwrap();
        s.put(&mk("a3", "b3", Severity::Medium)).await.unwrap();
        let active = s.list_active(10).await.unwrap();
        assert_eq!(active.len(), 3);
        assert_eq!(active[0].severity, Severity::High);
        assert_eq!(active[1].severity, Severity::Medium);
        assert_eq!(active[2].severity, Severity::Low);
    }

    #[tokio::test]
    async fn store_mark_resolved_excludes_from_active() {
        let s = ContradictionStore::open_in_memory().await.unwrap();
        let c = Contradiction {
            id: None,
            episode_a: "a".to_string(),
            episode_b: "b".to_string(),
            severity: Severity::High,
            confidence: 0.9,
            explanation: "x".to_string(),
            resolution_hint: None,
            similarity: 0.7,
            found_at: Utc::now(),
            resolved_at: None,
            resolved_action: None,
        };
        s.put(&c).await.unwrap();
        let active = s.list_active(10).await.unwrap();
        let id = active[0].id.unwrap();
        s.mark_resolved(id, "supersede a").await.unwrap();
        let still_active = s.list_active(10).await.unwrap();
        assert!(still_active.is_empty());
    }
}
