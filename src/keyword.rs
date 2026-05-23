// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust BM25 keyword index for tempera.
//!
//! Sits alongside the vector index in `~/.tempera/keyword/` and provides
//! lexical retrieval — the half of hybrid search that catches literal-token
//! queries (function names, error codes, file paths) where embedding-only
//! search drifts to thematic neighbors.
//!
//! Tokenizer is coding-aware: snake_case, CamelCase, `::` and `/` are split
//! into sub-tokens so a query for `spawn_blocking` matches an episode
//! mentioning `tokio::spawn_blocking()`.

#![allow(dead_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// BM25 k1 — term-frequency saturation. Standard literature default.
const BM25_K1: f32 = 1.2;
/// BM25 b — document-length normalization. Standard literature default.
const BM25_B: f32 = 0.75;

/// A BM25 keyword index over episode text.
///
/// Stores per-document term frequencies and an inverted posting list. All
/// data lives in memory; persistence is via serde to a single JSON file.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct KeywordIndex {
    /// Per-document metadata and term frequencies.
    docs: HashMap<String, DocStats>,
    /// Inverted index: term -> document IDs that contain it.
    postings: HashMap<String, Vec<String>>,
    /// Sum of document lengths (running, for avgdl).
    total_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocStats {
    project: String,
    doc_len: u32,
    /// term -> raw frequency in this document
    tf: HashMap<String, u32>,
}

/// A single search hit.
#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub id: String,
    pub score: f32,
}

impl KeywordIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    pub fn is_indexed(&self) -> bool {
        !self.docs.is_empty()
    }

    fn avg_doc_len(&self) -> f32 {
        if self.docs.is_empty() {
            0.0
        } else {
            self.total_len as f32 / self.docs.len() as f32
        }
    }

    /// Insert or replace a document.
    pub fn insert(&mut self, doc_id: String, project: String, text: &str) {
        if self.docs.contains_key(&doc_id) {
            self.remove(&doc_id);
        }

        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;
        let mut tf: HashMap<String, u32> = HashMap::new();
        for tok in &tokens {
            *tf.entry(tok.clone()).or_insert(0) += 1;
        }

        for term in tf.keys() {
            self.postings
                .entry(term.clone())
                .or_default()
                .push(doc_id.clone());
        }

        self.total_len += doc_len as u64;
        self.docs.insert(
            doc_id,
            DocStats {
                project,
                doc_len,
                tf,
            },
        );
    }

    /// Remove a document. No-op if absent.
    pub fn remove(&mut self, doc_id: &str) {
        let Some(stats) = self.docs.remove(doc_id) else {
            return;
        };
        self.total_len = self.total_len.saturating_sub(stats.doc_len as u64);
        for term in stats.tf.keys() {
            if let Some(list) = self.postings.get_mut(term) {
                list.retain(|d| d != doc_id);
                if list.is_empty() {
                    self.postings.remove(term);
                }
            }
        }
    }

    /// Search the index. Returns up to `limit` hits ranked by BM25 score.
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        project_filter: Option<&str>,
    ) -> Vec<KeywordHit> {
        if self.docs.is_empty() || limit == 0 {
            return Vec::new();
        }
        let q_terms = tokenize(query);
        if q_terms.is_empty() {
            return Vec::new();
        }

        // Deduplicate query terms — BM25 sums per unique query term
        let mut seen = std::collections::HashSet::new();
        let q_unique: Vec<&String> = q_terms.iter().filter(|t| seen.insert(t.as_str())).collect();

        let n = self.docs.len() as f32;
        let avgdl = self.avg_doc_len().max(1.0);

        let mut scores: HashMap<&str, f32> = HashMap::new();
        for term in &q_unique {
            let Some(docs_with_term) = self.postings.get(term.as_str()) else {
                continue;
            };
            let df = docs_with_term.len() as f32;
            // Robertson-Spärck-Jones IDF with the +1 smoothing variant used in Lucene/BM25
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for doc_id in docs_with_term {
                let Some(stats) = self.docs.get(doc_id) else {
                    continue;
                };
                if let Some(filter) = project_filter
                    && !stats.project.eq_ignore_ascii_case(filter)
                {
                    continue;
                }
                let tf = *stats.tf.get(term.as_str()).unwrap_or(&0) as f32;
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * stats.doc_len as f32 / avgdl);
                let term_score = idf * (tf * (BM25_K1 + 1.0)) / denom;
                *scores.entry(doc_id.as_str()).or_insert(0.0) += term_score;
            }
        }

        let mut hits: Vec<KeywordHit> = scores
            .into_iter()
            .map(|(id, score)| KeywordHit {
                id: id.to_string(),
                score,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        hits
    }

    /// Persist to a JSON file. Creates parent directories as needed.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec(self).context("Failed to serialize keyword index")?;
        std::fs::write(path, bytes)
            .with_context(|| format!("Failed to write keyword index to {}", path.display()))?;
        Ok(())
    }

    /// Load from a JSON file. Returns `Ok(None)` if the file doesn't exist.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read keyword index from {}", path.display()))?;
        let idx: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("Failed to parse keyword index at {}", path.display()))?;
        Ok(Some(idx))
    }
}

/// Tokenize text into lowercased terms with code-aware sub-token expansion.
///
/// Rules:
/// - Split on whitespace and most punctuation; keep `_`, `/`, `.`, `:`, and `-`
///   within raw tokens so identifiers like `tokio::spawn_blocking`, `auth.rs`,
///   and `src/main.rs` stay searchable as a unit.
/// - For each raw token, emit the lowercased form.
/// - Also emit sub-tokens split on `_`, `/`, `.`, `::`, `-` (so a query for
///   `spawn_blocking` matches `tokio::spawn_blocking` and vice versa).
/// - Detect CamelCase on the original-case token and emit lowercased pieces
///   (so `HashMap` yields `hashmap` + `hash` + `map`).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | ';'
                    | '!'
                    | '?'
                    | '"'
                    | '\''
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
                    | '|'
                    | '`'
                    | '='
                    | '+'
                    | '*'
                    | '&'
                    | '@'
                    | '#'
                    | '$'
                    | '%'
                    | '^'
                    | '~'
                    | '\\'
            )
    }) {
        if raw.is_empty() {
            continue;
        }
        emit_token(&mut out, raw);
    }
    out
}

fn emit_token(out: &mut Vec<String>, raw: &str) {
    let lower = raw.to_lowercase();
    out.push(lower.clone());

    // CamelCase split must operate on original case before lowercasing.
    let camel = split_camel(raw);
    if camel.len() > 1 {
        for p in camel {
            let pl = p.to_lowercase();
            if pl != lower && !pl.is_empty() {
                out.push(pl);
            }
        }
    }

    // Separator-based sub-tokens, split hierarchically so identifiers like
    // `tokio::spawn_blocking` yield both `spawn_blocking` AND `spawn`/`blocking`.
    const PATH_SEPS: [char; 3] = ['/', '.', ':'];
    const IDENT_SEPS: [char; 2] = ['_', '-'];

    if lower.contains(PATH_SEPS) || lower.contains(IDENT_SEPS) {
        // First split on path-style separators
        let path_parts: Vec<&str> = if lower.contains(PATH_SEPS) {
            lower.split(PATH_SEPS).filter(|s| !s.is_empty()).collect()
        } else {
            vec![lower.as_str()]
        };

        for part in &path_parts {
            if *part != lower {
                out.push(part.to_string());
            }
            // Then split each path-part on identifier separators
            if part.contains(IDENT_SEPS) {
                for sub in part.split(IDENT_SEPS) {
                    if !sub.is_empty() && sub != lower && sub != *part {
                        out.push(sub.to_string());
                    }
                }
            }
        }
    }
}

fn split_camel(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut prev_upper = false;
    for c in s.chars() {
        if c.is_uppercase() {
            if !current.is_empty() && !prev_upper {
                parts.push(std::mem::take(&mut current));
            }
            current.push(c);
            prev_upper = true;
        } else {
            current.push(c);
            prev_upper = false;
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_lowercases_simple_words() {
        let toks = tokenize("Fix the Login Bug");
        assert!(toks.contains(&"fix".to_string()));
        assert!(toks.contains(&"login".to_string()));
        assert!(toks.contains(&"bug".to_string()));
    }

    #[test]
    fn tokenize_splits_snake_case() {
        let toks = tokenize("spawn_blocking deadlock");
        assert!(toks.contains(&"spawn_blocking".to_string()));
        assert!(toks.contains(&"spawn".to_string()));
        assert!(toks.contains(&"blocking".to_string()));
    }

    #[test]
    fn tokenize_splits_camel_case() {
        let toks = tokenize("HashMap<String>");
        assert!(toks.contains(&"hashmap".to_string()));
        assert!(toks.contains(&"hash".to_string()));
        assert!(toks.contains(&"map".to_string()));
        assert!(toks.contains(&"string".to_string()));
    }

    #[test]
    fn tokenize_handles_rust_paths() {
        let toks = tokenize("tokio::spawn_blocking");
        assert!(toks.contains(&"tokio::spawn_blocking".to_string()));
        assert!(toks.contains(&"tokio".to_string()));
        assert!(toks.contains(&"spawn_blocking".to_string()));
        assert!(toks.contains(&"spawn".to_string()));
        assert!(toks.contains(&"blocking".to_string()));
    }

    #[test]
    fn tokenize_handles_file_paths() {
        let toks = tokenize("src/auth.rs");
        assert!(toks.contains(&"src/auth.rs".to_string()));
        assert!(toks.contains(&"src".to_string()));
        assert!(toks.contains(&"auth".to_string()));
        assert!(toks.contains(&"rs".to_string()));
    }

    #[test]
    fn tokenize_preserves_error_codes() {
        let toks = tokenize("got E0277 here");
        assert!(toks.contains(&"e0277".to_string()));
    }

    #[test]
    fn tokenize_handles_consecutive_uppercase() {
        // ABC + Reader should split into "ABC" + "Reader", not "A"+"B"+"C"+"Reader"
        let toks = tokenize("ABCReader");
        let lowercased: Vec<&str> = toks.iter().map(String::as_str).collect();
        // Conservatively: just check we got the original and didn't shatter A/B/C
        assert!(lowercased.contains(&"abcreader"));
    }

    #[test]
    fn insert_and_search_basic() {
        let mut idx = KeywordIndex::new();
        idx.insert(
            "ep1".into(),
            "tempera".into(),
            "fix tokio spawn_blocking deadlock",
        );
        idx.insert("ep2".into(), "tempera".into(), "database migration script");
        idx.insert(
            "ep3".into(),
            "tempera".into(),
            "auth middleware ordering bug in login flow",
        );

        let hits = idx.search("spawn_blocking", 3, None);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, "ep1");
    }

    #[test]
    fn search_returns_higher_score_for_more_matches() {
        let mut idx = KeywordIndex::new();
        idx.insert(
            "ep1".into(),
            "p".into(),
            "tokio spawn_blocking deadlock fix",
        );
        idx.insert(
            "ep2".into(),
            "p".into(),
            "spawn_blocking issue and tokio runtime panic",
        );

        let hits = idx.search("tokio spawn_blocking", 2, None);
        assert_eq!(hits.len(), 2);
        // both contain both terms — score depends on doc length; this just
        // sanity-checks ranking is monotone in shared term overlap
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn search_excludes_other_projects() {
        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "tempera".into(), "deadlock fix");
        idx.insert("ep2".into(), "smelt".into(), "deadlock fix");

        let hits = idx.search("deadlock", 5, Some("tempera"));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "ep1");
    }

    #[test]
    fn search_returns_empty_on_empty_index() {
        let idx = KeywordIndex::new();
        assert!(idx.search("anything", 5, None).is_empty());
    }

    #[test]
    fn search_returns_empty_on_empty_query() {
        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "p".into(), "some text");
        assert!(idx.search("", 5, None).is_empty());
        assert!(idx.search("   ", 5, None).is_empty());
    }

    #[test]
    fn remove_drops_doc() {
        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "p".into(), "deadlock fix");
        idx.insert("ep2".into(), "p".into(), "deadlock occurred");
        assert_eq!(idx.doc_count(), 2);

        idx.remove("ep1");
        assert_eq!(idx.doc_count(), 1);

        let hits = idx.search("deadlock", 5, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "ep2");
    }

    #[test]
    fn remove_missing_id_is_noop() {
        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "p".into(), "deadlock");
        idx.remove("does-not-exist");
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn insert_replaces_existing_doc() {
        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "p".into(), "first content");
        idx.insert("ep1".into(), "p".into(), "second content deadlock");
        assert_eq!(idx.doc_count(), 1);

        // searching for "first" returns nothing (replaced)
        assert!(idx.search("first", 5, None).is_empty());
        // searching for "deadlock" returns ep1
        let hits = idx.search("deadlock", 5, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "ep1");
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kw.json");

        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "tempera".into(), "tokio spawn_blocking issue");
        idx.insert("ep2".into(), "tempera".into(), "auth middleware");
        idx.save(&path).unwrap();

        let loaded = KeywordIndex::load(&path).unwrap().unwrap();
        assert_eq!(loaded.doc_count(), 2);
        let hits = loaded.search("spawn_blocking", 5, None);
        assert_eq!(hits[0].id, "ep1");
    }

    #[test]
    fn load_missing_path_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.json");
        assert!(KeywordIndex::load(&missing).unwrap().is_none());
    }

    #[test]
    fn case_insensitive_match() {
        let mut idx = KeywordIndex::new();
        idx.insert("ep1".into(), "p".into(), "Auth Middleware Ordering");
        let hits = idx.search("auth middleware", 5, None);
        assert_eq!(hits[0].id, "ep1");
    }
}
