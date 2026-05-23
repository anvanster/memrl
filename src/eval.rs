// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Retrieval evaluation harness.
//!
//! Loads a labeled fixture of `(query, relevant_episode_ids)` pairs, runs each
//! query through the production retrieval pipeline, and reports P@K, R@K, MRR,
//! and nDCG@K. Baselines are persisted to disk so subsequent runs can compute
//! deltas — this is the measurement surface that gates retrieval changes.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::retrieve::try_vector_search;

/// Default top-K cutoff for metrics if the caller doesn't specify.
pub const DEFAULT_K: usize = 5;

/// A single labeled query in an eval fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureEntry {
    /// Stable identifier for this query (e.g., "q001"). Used in diffs.
    pub id: String,
    /// The query string sent to the retriever.
    pub query: String,
    /// Episodes the labeler considers relevant, with optional grades for nDCG.
    #[serde(default)]
    pub relevant: Vec<RelevantEpisode>,
    /// Optional project filter, mirroring `tempera retrieve --project`.
    #[serde(default)]
    pub project: Option<String>,
    /// Free-form tags for slicing aggregate metrics later.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Human notes about labeling rationale.
    #[serde(default)]
    pub notes: Option<String>,
}

/// A relevant episode reference with optional graded relevance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevantEpisode {
    pub id: String,
    /// 0 = irrelevant, 1 = partial, 2 = relevant, 3 = highly relevant.
    /// Defaults to 3 (binary-relevance compatible).
    #[serde(default = "default_grade")]
    pub grade: u8,
}

fn default_grade() -> u8 {
    3
}

/// Per-query result from a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub fixture_id: String,
    pub query: String,
    pub retrieved: Vec<String>,
    pub p_at_k: f64,
    pub r_at_k: f64,
    pub rr: f64,
    pub ndcg_at_k: f64,
}

/// Aggregate metrics across a fixture run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub mean_p_at_k: f64,
    pub mean_r_at_k: f64,
    pub mrr: f64,
    pub mean_ndcg_at_k: f64,
    pub k: usize,
    pub queries_evaluated: usize,
}

/// Full record of a single eval invocation. Persisted as JSON to baselines/.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    pub timestamp: DateTime<Utc>,
    pub git_commit: Option<String>,
    pub fixture_path: String,
    pub k: usize,
    pub aggregate: AggregateMetrics,
    pub per_query: Vec<QueryResult>,
}

// ===== Fixture loading =====

/// Load a JSONL fixture file. Lines beginning with `#` or empty lines are skipped.
pub fn load_fixture<P: AsRef<Path>>(path: P) -> Result<Vec<FixtureEntry>> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read fixture from {}", path.display()))?;
    let mut entries = Vec::new();
    for (i, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let entry: FixtureEntry = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse fixture line {}: {}", i + 1, raw))?;
        entries.push(entry);
    }
    Ok(entries)
}

// ===== Metrics =====

/// `P@K = |relevant ∩ retrieved[..K]| / K`.
///
/// Note: divides by K, not `min(K, |retrieved|)` — this matches standard IR
/// convention and penalizes under-retrieval.
pub fn precision_at_k(retrieved: &[String], relevant_ids: &[String], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let top_k = top_k_slice(retrieved, k);
    let hits = top_k.iter().filter(|id| relevant_ids.contains(id)).count();
    hits as f64 / k as f64
}

/// `R@K = |relevant ∩ retrieved[..K]| / |relevant|`. Zero when relevant is empty.
pub fn recall_at_k(retrieved: &[String], relevant_ids: &[String], k: usize) -> f64 {
    if relevant_ids.is_empty() {
        return 0.0;
    }
    let top_k = top_k_slice(retrieved, k);
    let hits = top_k.iter().filter(|id| relevant_ids.contains(id)).count();
    hits as f64 / relevant_ids.len() as f64
}

/// Reciprocal rank: `1 / rank` of the first relevant result (1-indexed), or 0.
pub fn reciprocal_rank(retrieved: &[String], relevant_ids: &[String]) -> f64 {
    for (i, id) in retrieved.iter().enumerate() {
        if relevant_ids.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// `nDCG@K = DCG@K / IDCG@K` with gain `2^grade - 1` and discount `log2(i + 2)`.
/// Returns 0 when no graded items are present.
pub fn ndcg_at_k(retrieved: &[String], graded: &[(String, u8)], k: usize) -> f64 {
    if k == 0 || graded.is_empty() {
        return 0.0;
    }
    let grade_of = |id: &str| -> u8 {
        graded
            .iter()
            .find(|(g_id, _)| g_id == id)
            .map(|(_, g)| *g)
            .unwrap_or(0)
    };

    let top_k = top_k_slice(retrieved, k);
    let dcg: f64 = top_k
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let g = grade_of(id) as f64;
            (2_f64.powf(g) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    let mut ideal_grades: Vec<u8> = graded.iter().map(|(_, g)| *g).collect();
    ideal_grades.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal_grades
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &g)| {
            let g = g as f64;
            (2_f64.powf(g) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

fn top_k_slice(retrieved: &[String], k: usize) -> &[String] {
    if retrieved.len() < k {
        retrieved
    } else {
        &retrieved[..k]
    }
}

// ===== Runner =====

/// Run every fixture entry through the current retrieval pipeline and produce
/// an `EvalRun`. Failures on individual queries are recorded as zero-score
/// rather than aborting the whole run.
pub async fn run_eval(fixture_path: &Path, k: usize, config: &Config) -> Result<EvalRun> {
    let entries = load_fixture(fixture_path)?;
    if entries.is_empty() {
        anyhow::bail!(
            "fixture {} is empty — nothing to evaluate",
            fixture_path.display()
        );
    }
    println!(
        "Running {} queries from {} (K={})",
        entries.len(),
        fixture_path.display(),
        k
    );

    let mut per_query = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        eprint!("\r  [{}/{}] {}", i + 1, entries.len(), entry.id);
        use std::io::Write;
        std::io::stderr().flush().ok();

        let retrieved = retrieve_ids(entry, k, config).await;
        let relevant_ids: Vec<String> = entry.relevant.iter().map(|r| r.id.clone()).collect();
        let graded: Vec<(String, u8)> = entry
            .relevant
            .iter()
            .map(|r| (r.id.clone(), r.grade))
            .collect();

        per_query.push(QueryResult {
            fixture_id: entry.id.clone(),
            query: entry.query.clone(),
            p_at_k: precision_at_k(&retrieved, &relevant_ids, k),
            r_at_k: recall_at_k(&retrieved, &relevant_ids, k),
            rr: reciprocal_rank(&retrieved, &relevant_ids),
            ndcg_at_k: ndcg_at_k(&retrieved, &graded, k),
            retrieved,
        });
    }
    eprintln!();

    let agg = aggregate(&per_query, k);

    Ok(EvalRun {
        timestamp: Utc::now(),
        git_commit: try_git_commit(),
        fixture_path: fixture_path.display().to_string(),
        k,
        aggregate: agg,
        per_query,
    })
}

async fn retrieve_ids(entry: &FixtureEntry, k: usize, config: &Config) -> Vec<String> {
    match try_vector_search(&entry.query, k, entry.project.as_deref(), config).await {
        Ok(results) => results.into_iter().map(|s| s.episode.id).collect(),
        Err(_) => Vec::new(),
    }
}

fn aggregate(per_query: &[QueryResult], k: usize) -> AggregateMetrics {
    if per_query.is_empty() {
        return AggregateMetrics {
            mean_p_at_k: 0.0,
            mean_r_at_k: 0.0,
            mrr: 0.0,
            mean_ndcg_at_k: 0.0,
            k,
            queries_evaluated: 0,
        };
    }
    let n = per_query.len() as f64;
    AggregateMetrics {
        mean_p_at_k: per_query.iter().map(|q| q.p_at_k).sum::<f64>() / n,
        mean_r_at_k: per_query.iter().map(|q| q.r_at_k).sum::<f64>() / n,
        mrr: per_query.iter().map(|q| q.rr).sum::<f64>() / n,
        mean_ndcg_at_k: per_query.iter().map(|q| q.ndcg_at_k).sum::<f64>() / n,
        k,
        queries_evaluated: per_query.len(),
    }
}

fn try_git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
}

// ===== Baseline persistence =====

/// Directory where baselines for a given fixture are stored. By convention,
/// `evals/fixtures/foo.jsonl` → `evals/baselines/`.
pub fn baselines_dir(fixture_path: &Path) -> PathBuf {
    fixture_path
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| Path::new("."))
        .join("baselines")
}

/// Persist a run as `<timestamp>-<commit>.json` in `dir`. Creates `dir` if missing.
pub fn save_baseline(run: &EvalRun, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let ts = run.timestamp.format("%Y%m%dT%H%M%SZ");
    let commit_part = run.git_commit.as_deref().unwrap_or("nogit");
    let filename = format!("{ts}-{commit_part}.json");
    let path = dir.join(filename);
    let json = serde_json::to_string_pretty(run)?;
    std::fs::write(&path, json)
        .with_context(|| format!("Failed to write baseline to {}", path.display()))?;
    Ok(path)
}

/// Load the most recent baseline from `dir`, or `None` if the directory is
/// missing or empty. Filenames sort lexicographically by ISO timestamp.
pub fn load_latest_baseline(dir: &Path) -> Result<Option<EvalRun>> {
    if !dir.exists() {
        return Ok(None);
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    if entries.is_empty() {
        return Ok(None);
    }
    entries.sort();
    let last = entries.last().expect("non-empty after check");
    let content = std::fs::read_to_string(last)
        .with_context(|| format!("Failed to read baseline {}", last.display()))?;
    let run: EvalRun = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse baseline {}", last.display()))?;
    Ok(Some(run))
}

// ===== Output =====

/// Render a run to stdout, with green/red deltas against `baseline` if provided.
pub fn print_run(run: &EvalRun, baseline: Option<&EvalRun>) {
    let a = &run.aggregate;
    println!();
    println!("{}", "Eval run".bold());
    println!("  fixture: {}", run.fixture_path);
    println!(
        "  commit:  {}",
        run.git_commit.as_deref().unwrap_or("(none)")
    );
    println!("  queries: {}", a.queries_evaluated);
    if let Some(b) = baseline {
        let bts = b.timestamp.format("%Y-%m-%d %H:%M UTC");
        let bcommit = b.git_commit.as_deref().unwrap_or("(none)");
        println!(
            "  vs baseline: {} (commit {})",
            bts.to_string().dimmed(),
            bcommit.dimmed()
        );
    }
    println!();

    let prev = baseline.map(|b| &b.aggregate);
    let label_width = 8;
    let row = |label: String, cur: f64, prev_val: Option<f64>| {
        println!(
            "  {label:<label_width$} {}",
            format_delta(cur, prev_val),
            label = label,
            label_width = label_width,
        );
    };
    row(
        format!("P@{}:", a.k),
        a.mean_p_at_k,
        prev.map(|p| p.mean_p_at_k),
    );
    row(
        format!("R@{}:", a.k),
        a.mean_r_at_k,
        prev.map(|p| p.mean_r_at_k),
    );
    row("MRR:".to_string(), a.mrr, prev.map(|p| p.mrr));
    row(
        format!("nDCG@{}:", a.k),
        a.mean_ndcg_at_k,
        prev.map(|p| p.mean_ndcg_at_k),
    );
    println!();
}

fn format_delta(cur: f64, prev: Option<f64>) -> String {
    match prev {
        None => format!("{cur:.3}"),
        Some(p) => {
            let delta = cur - p;
            let body = format!("{cur:.3} ({:+.3})", delta);
            if delta > 0.001 {
                body.green().to_string()
            } else if delta < -0.001 {
                body.red().to_string()
            } else {
                body
            }
        }
    }
}

// ===== High-level CLI entrypoints =====

/// `tempera eval baseline --fixture <path>` — run and persist as a new baseline.
pub async fn run_baseline(fixture: &Path, k: usize, config: &Config) -> Result<()> {
    let run = run_eval(fixture, k, config).await?;
    let dir = baselines_dir(fixture);
    let path = save_baseline(&run, &dir)?;
    print_run(&run, None);
    println!("baseline saved: {}", path.display());
    Ok(())
}

/// `tempera eval run --fixture <path>` — run and diff against latest baseline.
pub async fn run_against_baseline(
    fixture: &Path,
    k: usize,
    save: bool,
    config: &Config,
) -> Result<()> {
    let run = run_eval(fixture, k, config).await?;
    let dir = baselines_dir(fixture);
    let baseline = load_latest_baseline(&dir)?;
    print_run(&run, baseline.as_ref());
    if save {
        let path = save_baseline(&run, &dir)?;
        println!("run saved: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn p_at_k_basic() {
        let retrieved = ids(&["a", "b", "c", "d", "e"]);
        let relevant = ids(&["a", "c", "f"]);
        assert!((precision_at_k(&retrieved, &relevant, 5) - 0.4).abs() < 1e-9);
        assert!((precision_at_k(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn p_at_k_no_hits() {
        let retrieved = ids(&["a", "b"]);
        let relevant = ids(&["x", "y"]);
        assert_eq!(precision_at_k(&retrieved, &relevant, 5), 0.0);
    }

    #[test]
    fn p_at_k_under_retrieved() {
        let retrieved = ids(&["a"]);
        let relevant = ids(&["a"]);
        // Standard IR convention: divide by K, penalizing under-retrieval
        assert!((precision_at_k(&retrieved, &relevant, 5) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn r_at_k_basic() {
        let retrieved = ids(&["a", "b", "c"]);
        let relevant = ids(&["a", "c", "d"]);
        assert!((recall_at_k(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn r_at_k_empty_relevant() {
        let retrieved = ids(&["a", "b"]);
        let relevant: Vec<String> = vec![];
        assert_eq!(recall_at_k(&retrieved, &relevant, 5), 0.0);
    }

    #[test]
    fn rr_basic() {
        let retrieved = ids(&["a", "b", "c"]);
        let relevant = ids(&["c"]);
        assert!((reciprocal_rank(&retrieved, &relevant) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn rr_first_hit() {
        let retrieved = ids(&["a", "b"]);
        let relevant = ids(&["a"]);
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 1.0);
    }

    #[test]
    fn rr_no_hit() {
        let retrieved = ids(&["a", "b"]);
        let relevant = ids(&["x"]);
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 0.0);
    }

    #[test]
    fn ndcg_perfect_order() {
        let retrieved = ids(&["a", "b", "c"]);
        let graded = vec![
            ("a".to_string(), 3),
            ("b".to_string(), 2),
            ("c".to_string(), 1),
        ];
        let ndcg = ndcg_at_k(&retrieved, &graded, 3);
        assert!((ndcg - 1.0).abs() < 1e-9, "expected 1.0, got {ndcg}");
    }

    #[test]
    fn ndcg_worst_order_below_one() {
        let retrieved = ids(&["c", "b", "a"]);
        let graded = vec![
            ("a".to_string(), 3),
            ("b".to_string(), 2),
            ("c".to_string(), 1),
        ];
        let ndcg = ndcg_at_k(&retrieved, &graded, 3);
        assert!(ndcg < 1.0 && ndcg > 0.0);
    }

    #[test]
    fn ndcg_no_relevant_retrieved() {
        let retrieved = ids(&["x", "y"]);
        let graded = vec![("a".to_string(), 3)];
        assert_eq!(ndcg_at_k(&retrieved, &graded, 5), 0.0);
    }

    #[test]
    fn load_fixture_skips_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.jsonl");
        let content = r#"# header comment
# another comment

{"id": "q1", "query": "first", "relevant": [{"id": "a", "grade": 3}]}

{"id": "q2", "query": "second", "relevant": [{"id": "b"}], "tags": ["t"]}
"#;
        std::fs::write(&path, content).unwrap();
        let entries = load_fixture(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "q1");
        assert_eq!(entries[0].relevant[0].grade, 3);
        // default grade applies when omitted
        assert_eq!(entries[1].relevant[0].grade, 3);
        assert_eq!(entries[1].tags, vec!["t".to_string()]);
    }

    #[test]
    fn save_and_load_baseline_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let run = EvalRun {
            timestamp: Utc::now(),
            git_commit: Some("abc1234".to_string()),
            fixture_path: "evals/fixtures/test.jsonl".to_string(),
            k: 5,
            aggregate: AggregateMetrics {
                mean_p_at_k: 0.42,
                mean_r_at_k: 0.61,
                mrr: 0.55,
                mean_ndcg_at_k: 0.7,
                k: 5,
                queries_evaluated: 10,
            },
            per_query: vec![],
        };
        let path = save_baseline(&run, dir.path()).unwrap();
        assert!(path.exists());
        let loaded = load_latest_baseline(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.git_commit, run.git_commit);
        assert!((loaded.aggregate.mean_p_at_k - 0.42).abs() < 1e-9);
    }

    #[test]
    fn load_latest_baseline_missing_dir_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert!(load_latest_baseline(&missing).unwrap().is_none());
    }

    #[test]
    fn aggregate_empty_returns_zeros() {
        let agg = aggregate(&[], 5);
        assert_eq!(agg.queries_evaluated, 0);
        assert_eq!(agg.mean_p_at_k, 0.0);
        assert_eq!(agg.mrr, 0.0);
    }

    #[test]
    fn aggregate_means_correctly() {
        let qs = vec![
            QueryResult {
                fixture_id: "1".into(),
                query: "".into(),
                retrieved: vec![],
                p_at_k: 0.4,
                r_at_k: 0.5,
                rr: 1.0,
                ndcg_at_k: 0.8,
            },
            QueryResult {
                fixture_id: "2".into(),
                query: "".into(),
                retrieved: vec![],
                p_at_k: 0.6,
                r_at_k: 0.7,
                rr: 0.5,
                ndcg_at_k: 0.6,
            },
        ];
        let agg = aggregate(&qs, 5);
        assert_eq!(agg.queries_evaluated, 2);
        assert!((agg.mean_p_at_k - 0.5).abs() < 1e-9);
        assert!((agg.mean_r_at_k - 0.6).abs() < 1e-9);
        assert!((agg.mrr - 0.75).abs() < 1e-9);
        assert!((agg.mean_ndcg_at_k - 0.7).abs() < 1e-9);
    }

    #[test]
    fn baselines_dir_sibling_of_fixtures() {
        let fixture = Path::new("evals/fixtures/general.jsonl");
        let dir = baselines_dir(fixture);
        assert_eq!(dir, PathBuf::from("evals/baselines"));
    }
}
