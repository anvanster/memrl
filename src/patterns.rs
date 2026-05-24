// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cross-day pattern detection.
//!
//! The reflect phase (v0.7.3) writes one page per high-signal day. Each
//! page captures what was salient that day. Over weeks of reflections,
//! the *same* themes keep surfacing — "you keep losing time to feature-
//! flag plumbing on Tuesdays," "the auth flow rewards keeping the
//! middleware ordered." A reader spotting that pattern by hand is what
//! v0.7.4 automates.
//!
//! Pipeline:
//!   1. Load reflections from the last `patterns_lookback_days`.
//!   2. Embed each (reuses BGE-Small via `EpisodeIndexer`).
//!   3. Agglomerative single-linkage cluster at
//!      `patterns_cluster_threshold` (cosine).
//!   4. For each cluster with `>= patterns_min_evidence` members that
//!      isn't already covered by an existing pattern (theme_slug
//!      collision), ask Sonnet to name the theme.
//!   5. Persist pattern + .md sidecar.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::dream::CostBudget;
use crate::indexer::EpisodeIndexer;
use crate::llm::AnthropicClient;
use crate::reflect::{Reflection, ReflectionStore};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// USD per Sonnet pattern call. Smaller than reflect because the
/// content surface is shorter — 3-5 reflection bodies in, one paragraph
/// out.
pub const PATTERN_ESTIMATED_COST_USD: f32 = 0.04;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub theme_slug: String,
    pub statement: String,
    pub evidence_reflection_ids: Vec<String>,
    pub first_seen: NaiveDate,
    pub last_reinforced: NaiveDate,
    pub occurrence_count: i64,
    pub model: String,
    pub created_at: DateTime<Utc>,
}

// ===== Store =====

#[derive(Clone)]
pub struct PatternStore {
    pool: SqlitePool,
}

impl PatternStore {
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
            .with_context(|| format!("Failed to open patterns DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run pattern migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn get_by_slug(&self, slug: &str) -> Result<Option<Pattern>> {
        let row = sqlx::query(
            "SELECT id, theme_slug, statement, evidence_reflection_ids,
                    first_seen, last_reinforced, occurrence_count, model, created_at
             FROM patterns WHERE theme_slug = ?1",
        )
        .bind(slug)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::row_to_pattern).transpose()
    }

    pub async fn list_all(&self) -> Result<Vec<Pattern>> {
        let rows = sqlx::query(
            "SELECT id, theme_slug, statement, evidence_reflection_ids,
                    first_seen, last_reinforced, occurrence_count, model, created_at
             FROM patterns ORDER BY last_reinforced DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_pattern).collect()
    }

    pub async fn put(&self, p: &Pattern) -> Result<()> {
        sqlx::query(
            "INSERT INTO patterns
                (id, theme_slug, statement, evidence_reflection_ids,
                 first_seen, last_reinforced, occurrence_count, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(theme_slug) DO UPDATE SET
                statement = excluded.statement,
                evidence_reflection_ids = excluded.evidence_reflection_ids,
                last_reinforced = excluded.last_reinforced,
                occurrence_count = excluded.occurrence_count,
                model = excluded.model",
        )
        .bind(&p.id)
        .bind(&p.theme_slug)
        .bind(&p.statement)
        .bind(serde_json::to_string(&p.evidence_reflection_ids)?)
        .bind(p.first_seen.to_string())
        .bind(p.last_reinforced.to_string())
        .bind(p.occurrence_count)
        .bind(&p.model)
        .bind(p.created_at.timestamp())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    fn row_to_pattern(row: &sqlx::sqlite::SqliteRow) -> Result<Pattern> {
        let ev_json: String = row.get("evidence_reflection_ids");
        let first: String = row.get("first_seen");
        let last: String = row.get("last_reinforced");
        let created_ts: i64 = row.get("created_at");
        Ok(Pattern {
            id: row.get("id"),
            theme_slug: row.get("theme_slug"),
            statement: row.get("statement"),
            evidence_reflection_ids: serde_json::from_str(&ev_json)?,
            first_seen: NaiveDate::parse_from_str(&first, "%Y-%m-%d")?,
            last_reinforced: NaiveDate::parse_from_str(&last, "%Y-%m-%d")?,
            occurrence_count: row.get("occurrence_count"),
            model: row.get("model"),
            created_at: DateTime::<Utc>::from_timestamp(created_ts, 0).unwrap_or_default(),
        })
    }
}

// ===== Clustering =====

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Agglomerative single-linkage clustering. Repeatedly merges the pair
/// of clusters whose closest members have cosine >= `threshold` until
/// no qualifying pair remains. O(N⁴) worst case — fine for N ≤ ~200
/// reflections; revisit if `patterns_lookback_days` grows huge.
pub fn cluster_agglomerative(embeddings: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = (0..embeddings.len()).map(|i| vec![i]).collect();
    loop {
        let mut best_sim = threshold;
        let mut best_merge: Option<(usize, usize)> = None;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let sim = single_linkage_sim(&clusters[i], &clusters[j], embeddings);
                if sim >= best_sim {
                    best_sim = sim;
                    best_merge = Some((i, j));
                }
            }
        }
        match best_merge {
            Some((i, j)) => {
                // Merge j into i; swap_remove j to keep linear cost.
                let merged = clusters.swap_remove(j);
                // After swap_remove, index i still points at the right
                // cluster only if i < j (true since we required i < j above).
                clusters[i].extend(merged);
            }
            None => break,
        }
    }
    clusters
}

fn single_linkage_sim(a: &[usize], b: &[usize], embeddings: &[Vec<f32>]) -> f32 {
    let mut best = -1.0_f32;
    for &i in a {
        for &j in b {
            let s = cosine_similarity(&embeddings[i], &embeddings[j]);
            if s > best {
                best = s;
            }
        }
    }
    best
}

// ===== Slug helpers =====

pub fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in s.chars().take(80) {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !out.is_empty() && !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

// ===== Pattern authorship =====

const PATTERNS_SYSTEM: &str = r#"You are reading several daily reflections that an embedding cluster found similar — they likely share a theme. Your job: name the theme as one concrete sentence, and list which reflections support it.

Rules:
- One sentence, present tense, concrete. Bad: "There seems to be a pattern around testing." Good: "Race conditions in tempera show up when capture and indexing share the keyword-index file."
- The theme should describe something REPEATABLE about how the codebase / user / workflow behaves, not a one-off event.
- Cite at least two reflections by their ID. List ALL the reflection IDs that genuinely support the theme.
- Output JSON ONLY:
  {
    "theme_slug": "kebab-case-slug",   // 2-6 words, descriptive
    "statement": "one concrete sentence",
    "evidence": ["<reflection-id>", ...]
  }
- If the cluster has no real shared theme (just topically adjacent), output:
  {"theme_slug": "none", "statement": "no shared theme", "evidence": []}
"#;

#[derive(Debug, Deserialize)]
struct AuthoredPattern {
    theme_slug: String,
    statement: String,
    #[serde(default)]
    evidence: Vec<String>,
}

fn build_user_message(cluster: &[&Reflection]) -> String {
    let mut s = format!(
        "Cluster of {} reflections likely sharing a theme.\n\n",
        cluster.len()
    );
    for r in cluster {
        let preview: String = r.body.lines().take(8).collect::<Vec<_>>().join("\n");
        s.push_str(&format!("--- {} ({}) ---\n{}\n\n", r.id, r.date, preview));
    }
    s
}

async fn author_pattern(
    cluster: &[&Reflection],
    config: &Config,
    budget: Option<&CostBudget>,
) -> Result<Option<AuthoredPattern>> {
    if let Some(b) = budget {
        b.try_spend(PATTERN_ESTIMATED_COST_USD)?;
    }
    let user = build_user_message(cluster);
    let client = AnthropicClient::with_model(&config.dream.reflect_model)?;
    let raw = client.raw_completion(PATTERNS_SYSTEM, &user, 400).await?;
    let trimmed = strip_json_fences(&raw);
    let authored: AuthoredPattern = serde_json::from_str(trimmed)
        .with_context(|| format!("failed to parse pattern JSON: {raw}"))?;
    if authored.theme_slug == "none" || authored.statement.eq_ignore_ascii_case("no shared theme") {
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

fn patterns_dir() -> Result<PathBuf> {
    Ok(Config::data_dir()?.join("patterns"))
}

fn write_sidecar(p: &Pattern) -> Result<()> {
    let dir = patterns_dir()?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.md", p.theme_slug));
    let evidence_lines: String = p
        .evidence_reflection_ids
        .iter()
        .map(|id| format!("- {id}"))
        .collect::<Vec<_>>()
        .join("\n");
    let content = format!(
        "+++\nid = \"{id}\"\ntheme_slug = \"{slug}\"\nfirst_seen = \"{first}\"\nlast_reinforced = \"{last}\"\noccurrence_count = {count}\nmodel = \"{model}\"\ncreated_at = \"{ts}\"\n+++\n\n# Pattern: {slug}\n\n**{stmt}**\n\n## Evidence\n\n{ev}\n",
        id = p.id,
        slug = p.theme_slug,
        first = p.first_seen,
        last = p.last_reinforced,
        count = p.occurrence_count,
        model = p.model,
        ts = p.created_at.to_rfc3339(),
        stmt = p.statement,
        ev = evidence_lines,
    );
    std::fs::write(&path, content)
        .with_context(|| format!("Failed to write pattern sidecar at {}", path.display()))?;
    Ok(())
}

// ===== Pipeline =====

#[derive(Debug, Clone, Serialize)]
pub struct PatternsReport {
    pub reflections_examined: usize,
    pub clusters_found: usize,
    pub clusters_above_min: usize,
    pub patterns_written: usize,
    pub patterns_skipped_existing: usize,
    pub clusters_with_no_theme: usize,
}

/// Cutoff date computed from `lookback_days` (inclusive bound).
fn cutoff_date(lookback_days: u32) -> NaiveDate {
    (Utc::now() - chrono::Duration::days(lookback_days as i64)).date_naive()
}

/// Run the patterns phase. Returns a report even when no patterns
/// were written (e.g. not enough reflections yet).
pub async fn run_patterns(config: &Config, budget: Option<&CostBudget>) -> Result<PatternsReport> {
    let reflect_store = ReflectionStore::open_default().await?;
    let pattern_store = PatternStore::open_default().await?;

    let cutoff = cutoff_date(config.dream.patterns_lookback_days);
    let reflections = reflect_store.list_since(&cutoff).await?;
    let n = reflections.len();

    let min_evidence = config.dream.patterns_min_evidence;
    if n < min_evidence {
        return Ok(PatternsReport {
            reflections_examined: n,
            clusters_found: 0,
            clusters_above_min: 0,
            patterns_written: 0,
            patterns_skipped_existing: 0,
            clusters_with_no_theme: 0,
        });
    }

    // Embed each reflection body. Reusing EpisodeIndexer for its
    // already-warm fastembed model.
    let indexer = EpisodeIndexer::new()
        .await
        .context("patterns: failed to open embedder")?;
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
    for r in &reflections {
        let v = indexer
            .embed(&r.body)
            .context("patterns: failed to embed reflection")?;
        embeddings.push(v);
    }

    let clusters = cluster_agglomerative(&embeddings, config.dream.patterns_cluster_threshold);
    let clusters_above_min = clusters.iter().filter(|c| c.len() >= min_evidence).count();

    let mut report = PatternsReport {
        reflections_examined: n,
        clusters_found: clusters.len(),
        clusters_above_min,
        patterns_written: 0,
        patterns_skipped_existing: 0,
        clusters_with_no_theme: 0,
    };

    for cluster in clusters {
        if cluster.len() < min_evidence {
            continue;
        }
        let refs: Vec<&Reflection> = cluster.iter().map(|&i| &reflections[i]).collect();

        let authored = match author_pattern(&refs, config, budget).await? {
            Some(a) => a,
            None => {
                report.clusters_with_no_theme += 1;
                continue;
            }
        };

        let slug = if authored.theme_slug.is_empty() {
            slugify(&authored.statement)
        } else {
            slugify(&authored.theme_slug)
        };
        if slug.is_empty() {
            report.clusters_with_no_theme += 1;
            continue;
        }

        if pattern_store.get_by_slug(&slug).await?.is_some() {
            report.patterns_skipped_existing += 1;
            continue;
        }

        let evidence_ids: Vec<String> = if authored.evidence.is_empty() {
            refs.iter().map(|r| r.id.clone()).collect()
        } else {
            // Filter to evidence IDs that match reflections we actually
            // know about; tolerate hallucinations from the model.
            let known: std::collections::HashSet<&str> =
                refs.iter().map(|r| r.id.as_str()).collect();
            authored
                .evidence
                .into_iter()
                .filter(|e| known.contains(e.as_str()))
                .collect()
        };

        let first_seen = refs.iter().map(|r| r.date).min().unwrap_or(cutoff);
        let last_reinforced = refs.iter().map(|r| r.date).max().unwrap_or(cutoff);
        let pattern = Pattern {
            id: format!("{}-{}", last_reinforced, slug),
            theme_slug: slug,
            statement: authored.statement,
            evidence_reflection_ids: evidence_ids,
            first_seen,
            last_reinforced,
            occurrence_count: refs.len() as i64,
            model: config.dream.reflect_model.clone(),
            created_at: Utc::now(),
        };
        pattern_store.put(&pattern).await?;
        write_sidecar(&pattern)?;
        report.patterns_written += 1;
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_mismatched_length_is_zero() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cluster_single_item_returns_one_cluster() {
        let v = vec![vec![1.0_f32, 0.0]];
        let c = cluster_agglomerative(&v, 0.5);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0], vec![0]);
    }

    #[test]
    fn cluster_groups_similar_items() {
        // Two pairs of near-duplicates + one outlier
        let v = vec![
            vec![1.0_f32, 0.0],
            vec![0.99_f32, 0.01],
            vec![0.0_f32, 1.0],
            vec![0.01_f32, 0.99],
            vec![-1.0_f32, 0.0],
        ];
        let c = cluster_agglomerative(&v, 0.95);
        // Expect 3 clusters: {0,1}, {2,3}, {4}
        assert_eq!(c.len(), 3);
        let sizes: Vec<usize> = c.iter().map(|c| c.len()).collect();
        let mut sizes_sorted = sizes;
        sizes_sorted.sort();
        assert_eq!(sizes_sorted, vec![1, 2, 2]);
    }

    #[test]
    fn cluster_threshold_strict_keeps_items_apart() {
        // [1,0] vs [0,1] are orthogonal (cosine = 0); they should never
        // merge for any positive threshold.
        let v = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let c = cluster_agglomerative(&v, 0.1);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn cluster_threshold_loose_merges_everything() {
        let v = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![-1.0_f32, 0.0]];
        let c = cluster_agglomerative(&v, -1.0);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 3);
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(
            slugify("auth middleware ordering"),
            "auth-middleware-ordering"
        );
    }

    #[test]
    fn slugify_strips_punctuation() {
        assert_eq!(
            slugify("tokio::spawn_blocking deadlock!"),
            "tokio-spawn-blocking-deadlock"
        );
    }

    #[test]
    fn slugify_handles_empty_and_punctuation_only() {
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("!!!"), "");
    }

    #[test]
    fn strip_json_fences_handles_markdown() {
        assert_eq!(strip_json_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_json_fences("  {\"a\":1}  "), "{\"a\":1}");
    }

    #[tokio::test]
    async fn store_roundtrip() {
        let s = PatternStore::open_in_memory().await.unwrap();
        let p = Pattern {
            id: "2026-05-24-auth-flow".to_string(),
            theme_slug: "auth-flow".to_string(),
            statement: "Auth middleware ordering matters.".to_string(),
            evidence_reflection_ids: vec!["2026-05-22-all".into(), "2026-05-23-all".into()],
            first_seen: NaiveDate::from_ymd_opt(2026, 5, 22).unwrap(),
            last_reinforced: NaiveDate::from_ymd_opt(2026, 5, 24).unwrap(),
            occurrence_count: 3,
            model: "claude-sonnet-4-6".to_string(),
            created_at: Utc::now(),
        };
        s.put(&p).await.unwrap();
        let got = s.get_by_slug("auth-flow").await.unwrap().unwrap();
        assert_eq!(got.theme_slug, "auth-flow");
        assert_eq!(got.evidence_reflection_ids.len(), 2);
        assert_eq!(got.occurrence_count, 3);
    }

    #[tokio::test]
    async fn store_upsert_by_slug() {
        let s = PatternStore::open_in_memory().await.unwrap();
        let mk = |stmt: &str, count: i64| Pattern {
            id: format!("2026-05-24-{}", "auth-flow"),
            theme_slug: "auth-flow".to_string(),
            statement: stmt.to_string(),
            evidence_reflection_ids: vec![],
            first_seen: NaiveDate::from_ymd_opt(2026, 5, 22).unwrap(),
            last_reinforced: NaiveDate::from_ymd_opt(2026, 5, 24).unwrap(),
            occurrence_count: count,
            model: "m".to_string(),
            created_at: Utc::now(),
        };
        s.put(&mk("First version", 2)).await.unwrap();
        s.put(&mk("Refined version", 5)).await.unwrap();
        let got = s.get_by_slug("auth-flow").await.unwrap().unwrap();
        assert_eq!(got.statement, "Refined version");
        assert_eq!(got.occurrence_count, 5);
        let all = s.list_all().await.unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn build_user_message_includes_ids_and_previews() {
        let r1 = Reflection {
            id: "2026-05-22-all".to_string(),
            date: NaiveDate::from_ymd_opt(2026, 5, 22).unwrap(),
            project: None,
            body: "Line one\nLine two\nLine three".to_string(),
            citations: vec![],
            signals: vec![],
            triage_score: 0.8,
            model: "m".to_string(),
            created_at: Utc::now(),
        };
        let msg = build_user_message(&[&r1]);
        assert!(msg.contains("2026-05-22-all"));
        assert!(msg.contains("Line one"));
    }
}
