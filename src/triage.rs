// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Two-tier LLM gating for dream-cycle authorship phases.
//!
//! Reflection / pattern / contradiction phases (v0.7.3+) are expensive —
//! a Sonnet call per worth-processing day. Routing every day's captures
//! straight to Sonnet wastes money on routine logistics. Triage runs a
//! cheap Haiku judge first: given a day's episodes, score 0-1 for "worth
//! synthesizing." Sonnet only fires when triage agrees.
//!
//! v0.7.2 ships:
//!   - the verdict cache (SQLite-backed, keyed by date + content hash)
//!   - the Haiku call + JSON-verdict parser
//!   - `tempera triage --date <YYYY-MM-DD>` for manual queries
//!
//! v0.7.3+ wire `triage_day()` into the reflect phase as the gate.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;

use crate::config::Config;
use crate::dream::CostBudget;
use crate::episode::Episode;
use crate::llm::AnthropicClient;

/// Embedded SQLite migrator (same DB file as jobs + fingerprints).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Output of one triage call. Scores below `MIN_SYNTHESIZE_SCORE` (default
/// 0.5) tell the reflect phase to skip the day entirely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageVerdict {
    /// 0.0 = pure logistics, no synthesis warranted.
    /// 1.0 = high-signal day; reflect / patterns should run.
    pub score: f32,
    /// Free-form signals the judge weighed (e.g. "multiple-projects",
    /// "novel-pattern", "high-falsifiability").
    pub signals: Vec<String>,
    /// One- or two-sentence rationale.
    pub reasoning: String,
}

/// Threshold below which dream-cycle authorship phases skip the day.
pub const MIN_SYNTHESIZE_SCORE: f32 = 0.5;

impl TriageVerdict {
    pub fn worth_synthesizing(&self) -> bool {
        self.score >= MIN_SYNTHESIZE_SCORE
    }
}

/// Estimated USD cost of one triage call. Haiku 4.5 is ~$1/M input,
/// ~$5/M output. Our triage prompt is ~500 input tokens and produces
/// ~100 output tokens → ~$0.001 per call. Generous estimate for budget
/// pre-check; actual cost recorded after the call.
pub const TRIAGE_ESTIMATED_COST_USD: f32 = 0.002;

// ===== Cache store =====

#[derive(Clone)]
pub struct TriageStore {
    pool: SqlitePool,
}

impl TriageStore {
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
            .with_context(|| format!("Failed to open triage DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run triage migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Lookup a cached verdict. Returns `None` when this exact `(date,
    /// captures_hash)` pair hasn't been triaged yet.
    pub async fn get(
        &self,
        date: &NaiveDate,
        captures_hash: &str,
    ) -> Result<Option<TriageVerdict>> {
        let row = sqlx::query(
            "SELECT verdict FROM dream_verdicts WHERE date = ?1 AND captures_hash = ?2",
        )
        .bind(date.to_string())
        .bind(captures_hash)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else { return Ok(None) };
        let json: String = row.get(0);
        let verdict: TriageVerdict = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse cached verdict for {date}"))?;
        Ok(Some(verdict))
    }

    pub async fn put(
        &self,
        date: &NaiveDate,
        captures_hash: &str,
        verdict: &TriageVerdict,
    ) -> Result<()> {
        let json = serde_json::to_string(verdict)?;
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO dream_verdicts (date, captures_hash, verdict, created_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(date, captures_hash) DO UPDATE SET
                 verdict = excluded.verdict,
                 created_at = excluded.created_at",
        )
        .bind(date.to_string())
        .bind(captures_hash)
        .bind(json)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

// ===== Captures hash =====

/// Stable content hash for the day's episodes. Sorting + the
/// `timestamp_end` field together mean any mutation (capture, feedback,
/// verification transition) invalidates the cache automatically.
pub fn captures_hash(episodes: &[Episode]) -> String {
    let mut pairs: Vec<(String, i64)> = episodes
        .iter()
        .map(|ep| (ep.id.clone(), ep.timestamp_end.timestamp()))
        .collect();
    pairs.sort();
    let canonical = pairs
        .iter()
        .map(|(id, ts)| format!("{id}|{ts}"))
        .collect::<Vec<_>>()
        .join("\n");
    blake3::hash(canonical.as_bytes()).to_hex().to_string()
}

// ===== Triage call =====

const TRIAGE_SYSTEM: &str = r#"You triage daily captures for a coding-session memory system.

Given a list of episodes captured on a single day, assess whether the
day's content is worth synthesizing into a reflection page (which costs
~$0.05 in Sonnet authorship to produce).

Score 0.0 to 1.0:
  1.0 = high-signal day. Multiple genuine BKMs (falsifiable claims),
        novel patterns, multi-project work, debugging insights with
        resolutions.
  0.7 = solid day, at least one clear BKM worth a reflection.
  0.5 = mixed: some logistics, some signal. Borderline; reflect optional.
  0.3 = mostly logistics with one or two takes.
  0.0 = pure logistics (commits bumped, configs touched, no insight).

Output JSON ONLY with keys:
  score:    f32 0.0-1.0
  signals:  array of short strings (e.g. "multiple-projects",
            "high-falsifiability", "novel-pattern", "debugging-insight",
            "mostly-logistics")
  reasoning: 1-2 sentences explaining the score."#;

/// Build the user message for a triage prompt — a compact summary of
/// each episode (intent, task type, outcome, claim falsifiability if
/// present, error count).
pub fn build_triage_user_message(date: &NaiveDate, episodes: &[Episode]) -> String {
    let mut s = format!("Date: {date}\nEpisodes: {} captured\n\n", episodes.len());
    for (i, ep) in episodes.iter().enumerate() {
        s.push_str(&format!("[{}] {}\n", i + 1, ep.project));
        let intent = if ep.intent.extracted_intent.is_empty() {
            &ep.intent.raw_prompt
        } else {
            &ep.intent.extracted_intent
        };
        let truncated: String = intent.chars().take(300).collect();
        s.push_str(&format!("  intent: {truncated}\n"));
        s.push_str(&format!(
            "  task: {} | outcome: {}\n",
            ep.intent.task_type, ep.outcome.status
        ));
        if let Some(claim) = &ep.intent.claim {
            s.push_str(&format!(
                "  claim: falsifiability={:.2} category={}\n",
                claim.falsifiability, claim.category
            ));
        }
        if !ep.context.errors_encountered.is_empty() {
            let resolved = ep
                .context
                .errors_encountered
                .iter()
                .filter(|e| e.resolved)
                .count();
            s.push_str(&format!(
                "  errors: {} ({} resolved)\n",
                ep.context.errors_encountered.len(),
                resolved
            ));
        }
        s.push('\n');
    }
    s
}

/// Triage a day's episodes. Returns the cached verdict if one exists,
/// otherwise calls the LLM, stores, and returns the result.
///
/// `force=true` skips the cache (re-runs the Haiku call).
pub async fn triage_day(
    date: &NaiveDate,
    episodes: &[Episode],
    store: &TriageStore,
    budget: Option<&CostBudget>,
    force: bool,
) -> Result<(TriageVerdict, bool /* from_cache */)> {
    triage_day_with_model(
        date,
        episodes,
        store,
        budget,
        force,
        "claude-haiku-4-5-20251001",
    )
    .await
}

/// Same as `triage_day` but with an explicit model name. The dream phase
/// and CLI both pass `config.dream.triage_model` here so model selection
/// stays in one place.
pub async fn triage_day_with_model(
    date: &NaiveDate,
    episodes: &[Episode],
    store: &TriageStore,
    budget: Option<&CostBudget>,
    force: bool,
    model: &str,
) -> Result<(TriageVerdict, bool /* from_cache */)> {
    let hash = captures_hash(episodes);

    if !force && let Some(cached) = store.get(date, &hash).await? {
        return Ok((cached, true));
    }

    // Empty days are trivially "not worth synthesizing" — short-circuit
    // before paying for an LLM call.
    if episodes.is_empty() {
        let verdict = TriageVerdict {
            score: 0.0,
            signals: vec!["empty-day".to_string()],
            reasoning: "No episodes captured.".to_string(),
        };
        store.put(date, &hash, &verdict).await?;
        return Ok((verdict, false));
    }

    if let Some(b) = budget {
        b.try_spend(TRIAGE_ESTIMATED_COST_USD)?;
    }

    let user_msg = build_triage_user_message(date, episodes);
    let client = AnthropicClient::with_model(model)?;
    let verdict = call_haiku_for_verdict(&client, &user_msg)
        .await
        .context("Haiku triage call failed")?;
    store.put(date, &hash, &verdict).await?;
    Ok((verdict, false))
}

/// Wraps the `reqwest`-based Anthropic client with the triage prompt.
async fn call_haiku_for_verdict(client: &AnthropicClient, user: &str) -> Result<TriageVerdict> {
    let raw = client.raw_completion(TRIAGE_SYSTEM, user, 200).await?;
    // The model usually emits clean JSON, but be tolerant of leading/
    // trailing whitespace and stray markdown fences.
    let trimmed = strip_json_fences(&raw);
    let verdict: TriageVerdict = serde_json::from_str(trimmed)
        .with_context(|| format!("Failed to parse triage JSON: {raw}"))?;
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

/// Group all episodes in the store by capture date (UTC). Returns
/// `(date, episodes)` pairs sorted newest first.
pub fn episodes_by_date(episodes: &[Episode]) -> Vec<(NaiveDate, Vec<Episode>)> {
    use std::collections::BTreeMap;
    let mut by_date: BTreeMap<NaiveDate, Vec<Episode>> = BTreeMap::new();
    for ep in episodes {
        let date = ep.timestamp_start.date_naive();
        by_date.entry(date).or_default().push(ep.clone());
    }
    // BTreeMap is ascending; reverse for newest-first.
    by_date.into_iter().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::Episode;
    use chrono::TimeZone;

    fn ep_at(id: &str, project: &str, ts: chrono::DateTime<Utc>) -> Episode {
        let mut e = Episode::new(project.to_string(), "x".to_string());
        e.id = id.to_string();
        e.timestamp_start = ts;
        e.timestamp_end = ts;
        e
    }

    #[test]
    fn captures_hash_stable_for_same_input() {
        let ts = Utc.with_ymd_and_hms(2026, 5, 24, 10, 0, 0).unwrap();
        let eps = vec![
            ep_at("ep1", "proj", ts),
            ep_at("ep2", "proj", ts + chrono::Duration::minutes(5)),
        ];
        assert_eq!(captures_hash(&eps), captures_hash(&eps));
    }

    #[test]
    fn captures_hash_order_independent() {
        let ts = Utc.with_ymd_and_hms(2026, 5, 24, 10, 0, 0).unwrap();
        let a = vec![ep_at("ep1", "p", ts), ep_at("ep2", "p", ts)];
        let b = vec![ep_at("ep2", "p", ts), ep_at("ep1", "p", ts)];
        assert_eq!(captures_hash(&a), captures_hash(&b));
    }

    #[test]
    fn captures_hash_changes_on_timestamp_change() {
        let ts = Utc.with_ymd_and_hms(2026, 5, 24, 10, 0, 0).unwrap();
        let a = vec![ep_at("ep1", "p", ts)];
        let b = vec![ep_at("ep1", "p", ts + chrono::Duration::seconds(1))];
        assert_ne!(captures_hash(&a), captures_hash(&b));
    }

    #[test]
    fn worth_synthesizing_threshold() {
        let v = |s: f32| TriageVerdict {
            score: s,
            signals: vec![],
            reasoning: "".into(),
        };
        assert!(!v(0.0).worth_synthesizing());
        assert!(!v(0.49).worth_synthesizing());
        assert!(v(0.5).worth_synthesizing());
        assert!(v(1.0).worth_synthesizing());
    }

    #[test]
    fn strip_json_fences_removes_markdown() {
        assert_eq!(
            strip_json_fences("  \n```json\n{\"a\":1}\n```\n"),
            "{\"a\":1}"
        );
        assert_eq!(strip_json_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_json_fences("  {\"a\":1}  "), "{\"a\":1}");
    }

    #[test]
    fn build_triage_user_message_compact() {
        let ts = Utc.with_ymd_and_hms(2026, 5, 24, 10, 0, 0).unwrap();
        let eps = vec![ep_at("ep1", "tempera", ts)];
        let msg = build_triage_user_message(&ts.date_naive(), &eps);
        assert!(msg.contains("2026-05-24"));
        assert!(msg.contains("tempera"));
        assert!(msg.contains("[1]"));
    }

    #[test]
    fn episodes_by_date_groups_and_sorts() {
        let d1 = Utc.with_ymd_and_hms(2026, 5, 23, 10, 0, 0).unwrap();
        let d2 = Utc.with_ymd_and_hms(2026, 5, 24, 10, 0, 0).unwrap();
        let eps = vec![
            ep_at("a", "p", d1),
            ep_at("b", "p", d2),
            ep_at("c", "p", d1),
        ];
        let grouped = episodes_by_date(&eps);
        assert_eq!(grouped.len(), 2);
        // Newest first.
        assert_eq!(grouped[0].0, d2.date_naive());
        assert_eq!(grouped[1].0, d1.date_naive());
        assert_eq!(grouped[0].1.len(), 1);
        assert_eq!(grouped[1].1.len(), 2);
    }

    #[tokio::test]
    async fn store_roundtrip() {
        let s = TriageStore::open_in_memory().await.unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        let v = TriageVerdict {
            score: 0.85,
            signals: vec!["high-falsifiability".into()],
            reasoning: "Two BKM-class captures.".into(),
        };
        s.put(&date, "hash1", &v).await.unwrap();
        let got = s.get(&date, "hash1").await.unwrap().unwrap();
        assert_eq!(got.score, 0.85);
        assert_eq!(got.signals, vec!["high-falsifiability".to_string()]);
    }

    #[tokio::test]
    async fn store_miss_returns_none() {
        let s = TriageStore::open_in_memory().await.unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        assert!(s.get(&date, "missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn store_upsert_replaces() {
        let s = TriageStore::open_in_memory().await.unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        let v1 = TriageVerdict {
            score: 0.3,
            signals: vec![],
            reasoning: "first".into(),
        };
        let v2 = TriageVerdict {
            score: 0.7,
            signals: vec![],
            reasoning: "second".into(),
        };
        s.put(&date, "hash", &v1).await.unwrap();
        s.put(&date, "hash", &v2).await.unwrap();
        let got = s.get(&date, "hash").await.unwrap().unwrap();
        assert_eq!(got.reasoning, "second");
    }

    #[tokio::test]
    async fn empty_day_short_circuits_to_zero_score() {
        let s = TriageStore::open_in_memory().await.unwrap();
        let date = NaiveDate::from_ymd_opt(2026, 5, 24).unwrap();
        let (v, from_cache) = triage_day(&date, &[], &s, None, false).await.unwrap();
        assert_eq!(v.score, 0.0);
        assert!(!from_cache);
        // Second call hits cache.
        let (_, from_cache2) = triage_day(&date, &[], &s, None, false).await.unwrap();
        assert!(from_cache2);
    }
}
