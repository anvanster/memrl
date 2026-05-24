// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Stack-trace fingerprint index.
//!
//! Hash a normalized version of each error's text (stripped of hex
//! addresses and user-home paths) plus its error type, store the result
//! keyed to the episode it appeared in. When a new capture has an error
//! whose fingerprint matches one from a prior episode, surface that —
//! "you've seen this exact thing N times before, here are the resolutions
//! that worked" — like git blame but for bugs.
//!
//! Deliberately simple: blake3 over a tidy canonical string. No
//! language-specific frame parsing — normalizing the whole message gets
//! 80% of the value at 20% of the complexity, and works across Rust
//! panics, Python tracebacks, build-log errors, etc.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::Utc;
use regex::Regex;
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;
use std::sync::OnceLock;

use crate::config::Config;
use crate::episode::ErrorRecord;

/// Stable migrator (same DB file as the job queue).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Compute the fingerprint for an error record. Hex addresses and the
/// user's home-directory prefix are normalized so the same panic on two
/// machines produces the same hash. Stable across Rust versions (blake3).
pub fn fingerprint_error(err: &ErrorRecord) -> String {
    let normalized = normalize_message(&err.message);
    let canonical = format!("{}|{}", err.error_type, normalized);
    let hash = blake3::hash(canonical.as_bytes());
    hash.to_hex().to_string()
}

/// Apply the canonicalizing transforms used by `fingerprint_error`.
/// Exposed for tests and for any future code that needs to compare
/// normalized error text without going through the hash.
pub fn normalize_message(msg: &str) -> String {
    let stripped_hex = strip_hex_addresses(msg);
    strip_user_paths(&stripped_hex)
}

fn hex_addr_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"0x[0-9a-fA-F]+").expect("compile hex regex"))
}

fn strip_hex_addresses(s: &str) -> String {
    hex_addr_re().replace_all(s, "0xADDR").into_owned()
}

fn strip_user_paths(s: &str) -> String {
    let Some(home) = dirs::home_dir().and_then(|p| p.to_str().map(String::from)) else {
        return s.to_string();
    };
    s.replace(&home, "/USER")
}

// ===== SQLite-backed storage =====

/// A prior occurrence of a fingerprint in a *different* episode.
#[derive(Debug, Clone)]
pub struct FingerprintMatch {
    pub episode_id: String,
    pub occurrence_count: i64,
    pub last_seen: i64,
}

#[derive(Clone)]
pub struct FingerprintStore {
    pool: SqlitePool,
}

impl FingerprintStore {
    /// Open or create the fingerprint store at `~/.tempera/jobs.sqlite`
    /// (same file as the job queue — single SQLite DB for all tempera
    /// metadata). Migrations are idempotent via `_sqlx_migrations`.
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
            .with_context(|| format!("Failed to open fingerprint DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run fingerprint migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Upsert a (hash, episode_id) row. If the row exists, bump
    /// `occurrence_count` and refresh `last_seen`; otherwise insert with
    /// count=1. Returns the post-update occurrence count.
    pub async fn record(&self, hash: &str, episode_id: &str) -> Result<i64> {
        let now = Utc::now().timestamp();
        // Upsert via ON CONFLICT — SQLite supports this since 3.24.
        let row = sqlx::query(
            "INSERT INTO error_fingerprints (hash, episode_id, first_seen, last_seen, occurrence_count)
             VALUES (?1, ?2, ?3, ?3, 1)
             ON CONFLICT(hash, episode_id) DO UPDATE SET
                 occurrence_count = occurrence_count + 1,
                 last_seen = excluded.last_seen
             RETURNING occurrence_count",
        )
        .bind(hash)
        .bind(episode_id)
        .bind(now)
        .fetch_one(&self.pool)
        .await
        .context("Failed to record fingerprint")?;
        Ok(row.get(0))
    }

    /// Find episodes (other than `exclude_episode_id`) where this
    /// fingerprint appeared. Ordered by most-recently-seen first.
    pub async fn matches(
        &self,
        hash: &str,
        exclude_episode_id: &str,
    ) -> Result<Vec<FingerprintMatch>> {
        let rows = sqlx::query(
            "SELECT episode_id, occurrence_count, last_seen
             FROM error_fingerprints
             WHERE hash = ?1 AND episode_id != ?2
             ORDER BY last_seen DESC
             LIMIT 10",
        )
        .bind(hash)
        .bind(exclude_episode_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| FingerprintMatch {
                episode_id: r.get("episode_id"),
                occurrence_count: r.get("occurrence_count"),
                last_seen: r.get("last_seen"),
            })
            .collect())
    }

    /// Total occurrences of this fingerprint across all episodes.
    /// Useful for "seen N times" totals.
    pub async fn total_occurrences(&self, hash: &str) -> Result<i64> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(occurrence_count), 0) FROM error_fingerprints WHERE hash = ?1",
        )
        .bind(hash)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(error_type: &str, message: &str) -> ErrorRecord {
        ErrorRecord {
            error_type: error_type.to_string(),
            message: message.to_string(),
            resolved: false,
            resolution: None,
        }
    }

    #[test]
    fn fingerprint_is_stable_across_calls() {
        let e = err("panic", "thread 'main' panicked at src/foo.rs:42");
        assert_eq!(fingerprint_error(&e), fingerprint_error(&e));
    }

    #[test]
    fn fingerprint_changes_when_error_type_differs() {
        let a = err("panic", "thread 'main' panicked at src/foo.rs:42");
        let b = err("error", "thread 'main' panicked at src/foo.rs:42");
        assert_ne!(fingerprint_error(&a), fingerprint_error(&b));
    }

    #[test]
    fn fingerprint_ignores_hex_addresses() {
        let a = err("panic", "called at 0xdeadbeef on src/foo.rs");
        let b = err("panic", "called at 0xcafebabe on src/foo.rs");
        assert_eq!(
            fingerprint_error(&a),
            fingerprint_error(&b),
            "different hex addresses should not move the fingerprint"
        );
    }

    #[test]
    fn fingerprint_ignores_home_directory() {
        // Simulate two machines with different home dirs producing the
        // same panic. We can't change HOME inside the test, but we can
        // verify the strip applies whatever HOME currently is.
        let Some(home) = dirs::home_dir().and_then(|p| p.to_str().map(String::from)) else {
            return; // Test only meaningful when HOME is detectable
        };
        let with_home = err("io", &format!("file not found: {home}/projects/foo.rs"));
        let with_user = err("io", "file not found: /USER/projects/foo.rs");
        assert_eq!(fingerprint_error(&with_home), fingerprint_error(&with_user));
    }

    #[test]
    fn fingerprint_hex_output_length() {
        // blake3 outputs 32 bytes → 64 hex chars
        let f = fingerprint_error(&err("x", "y"));
        assert_eq!(f.len(), 64);
        assert!(f.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn normalize_strips_hex_and_paths() {
        let Some(home) = dirs::home_dir().and_then(|p| p.to_str().map(String::from)) else {
            return;
        };
        let raw = format!("0xdeadbeef at {home}/code/main.rs");
        let n = normalize_message(&raw);
        assert!(n.contains("0xADDR"));
        assert!(n.contains("/USER"));
        assert!(!n.contains("0xdeadbeef"));
        assert!(!n.contains(&home));
    }

    // ===== Store tests (against an in-memory SQLite) =====

    #[tokio::test]
    async fn record_inserts_then_increments() {
        let s = FingerprintStore::open_in_memory().await.unwrap();
        let c1 = s.record("hash1", "ep1").await.unwrap();
        let c2 = s.record("hash1", "ep1").await.unwrap();
        let c3 = s.record("hash1", "ep1").await.unwrap();
        assert_eq!(c1, 1);
        assert_eq!(c2, 2);
        assert_eq!(c3, 3);
    }

    #[tokio::test]
    async fn record_different_episodes_separate_rows() {
        let s = FingerprintStore::open_in_memory().await.unwrap();
        s.record("hash1", "ep1").await.unwrap();
        s.record("hash1", "ep2").await.unwrap();
        let total = s.total_occurrences("hash1").await.unwrap();
        assert_eq!(total, 2);
    }

    #[tokio::test]
    async fn matches_excludes_current_episode() {
        let s = FingerprintStore::open_in_memory().await.unwrap();
        s.record("hash1", "ep1").await.unwrap();
        s.record("hash1", "ep2").await.unwrap();
        s.record("hash1", "ep3").await.unwrap();

        let m = s.matches("hash1", "ep2").await.unwrap();
        assert_eq!(m.len(), 2);
        let ids: Vec<&str> = m.iter().map(|x| x.episode_id.as_str()).collect();
        assert!(ids.contains(&"ep1"));
        assert!(ids.contains(&"ep3"));
        assert!(!ids.contains(&"ep2"));
    }

    #[tokio::test]
    async fn matches_empty_when_no_others() {
        let s = FingerprintStore::open_in_memory().await.unwrap();
        s.record("hash1", "ep1").await.unwrap();
        let m = s.matches("hash1", "ep1").await.unwrap();
        assert!(m.is_empty());
    }

    #[tokio::test]
    async fn matches_orders_by_last_seen_desc() {
        let s = FingerprintStore::open_in_memory().await.unwrap();
        s.record("hash1", "old").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        s.record("hash1", "new").await.unwrap();

        let m = s.matches("hash1", "lookup").await.unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].episode_id, "new", "newest should come first");
        assert_eq!(m[1].episode_id, "old");
    }
}
