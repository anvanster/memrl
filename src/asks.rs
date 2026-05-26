// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Should-have-asked log (v0.8.4).
//!
//! Mirrors the v0.8.2 anchored-mistakes index, but for a different
//! failure mode. Where mistakes captures "I asserted X, you corrected
//! me to Y," should-have-asked captures "I went off and built/picked
//! something — when I finally asked, you told me Y, but the
//! observable context Z should have made me ask up front."
//!
//! Each row carries three pieces:
//! - `trigger`: a normalized kebab/snake_case label describing the
//!   observable context that should fire the prompt next time
//!   (e.g. "edit-auth-middleware", "new-rust-crate"). Same
//!   normalization rules as mistakes so similar triggers cluster.
//! - `question`: what the agent should ask first.
//! - `answer`: what the user said. Doubles as fallback knowledge: if
//!   asking would be disruptive, the prior answer is still a better
//!   default than guessing blindly.
//!
//! Storage layout matches [[mistakes]] — one row per realization,
//! indexed by `(project, trigger, created_at DESC)`. The future
//! `tempera_brief` surface (v0.9) joins file-paths-about-to-be-edited
//! against this table to remind the agent which questions to ask up
//! front in recurring contexts.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;

use crate::config::Config;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShouldHaveAsked {
    pub id: Option<i64>,
    pub project: String,
    pub trigger: String,
    pub question: String,
    pub answer: String,
    pub episode_id: Option<String>,
    pub files: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerCount {
    pub trigger: String,
    pub count: i64,
    pub last_seen: DateTime<Utc>,
}

/// Normalize a free-form context-trigger label so similar triggers
/// cluster. Same rules as [[mistakes]] `normalize_category`: lowercase,
/// internal whitespace and punctuation collapse to single `_`,
/// leading/trailing underscores stripped.
pub fn normalize_trigger(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_sep = true;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    out.trim_matches('_').to_string()
}

#[derive(Clone)]
pub struct AsksStore {
    pool: SqlitePool,
}

impl AsksStore {
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
            .with_context(|| format!("Failed to open asks DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run asks migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn record(&self, sha: &ShouldHaveAsked) -> Result<i64> {
        let files_json = serde_json::to_string(&sha.files)?;
        let result = sqlx::query(
            "INSERT INTO should_have_asked
                (project, trigger, question, answer, episode_id, files, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .bind(&sha.project)
        .bind(&sha.trigger)
        .bind(&sha.question)
        .bind(&sha.answer)
        .bind(&sha.episode_id)
        .bind(files_json)
        .bind(sha.created_at.timestamp())
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    /// List rows with optional project + trigger filters.
    pub async fn list(
        &self,
        project: Option<&str>,
        trigger: Option<&str>,
        limit: i64,
    ) -> Result<Vec<ShouldHaveAsked>> {
        let rows =
            match (project, trigger) {
                (Some(p), Some(t)) => sqlx::query(
                    "SELECT id, project, trigger, question, answer, episode_id, files, created_at
                 FROM should_have_asked WHERE project = ?1 AND trigger = ?2
                 ORDER BY created_at DESC LIMIT ?3",
                )
                .bind(p)
                .bind(t)
                .bind(limit)
                .fetch_all(&self.pool)
                .await?,
                (Some(p), None) => sqlx::query(
                    "SELECT id, project, trigger, question, answer, episode_id, files, created_at
                 FROM should_have_asked WHERE project = ?1
                 ORDER BY created_at DESC LIMIT ?2",
                )
                .bind(p)
                .bind(limit)
                .fetch_all(&self.pool)
                .await?,
                (None, Some(t)) => sqlx::query(
                    "SELECT id, project, trigger, question, answer, episode_id, files, created_at
                 FROM should_have_asked WHERE trigger = ?1
                 ORDER BY created_at DESC LIMIT ?2",
                )
                .bind(t)
                .bind(limit)
                .fetch_all(&self.pool)
                .await?,
                (None, None) => sqlx::query(
                    "SELECT id, project, trigger, question, answer, episode_id, files, created_at
                 FROM should_have_asked
                 ORDER BY created_at DESC LIMIT ?1",
                )
                .bind(limit)
                .fetch_all(&self.pool)
                .await?,
            };
        rows.iter().map(Self::row_to_sha).collect()
    }

    /// Top N triggers within a project (or globally if `project = None`),
    /// ordered by occurrence count desc.
    pub async fn top_triggers(&self, project: Option<&str>, n: i64) -> Result<Vec<TriggerCount>> {
        let rows = match project {
            Some(p) => {
                sqlx::query(
                    "SELECT trigger, COUNT(*) AS c, MAX(created_at) AS last
                 FROM should_have_asked WHERE project = ?1
                 GROUP BY trigger ORDER BY c DESC LIMIT ?2",
                )
                .bind(p)
                .bind(n)
                .fetch_all(&self.pool)
                .await?
            }
            None => {
                sqlx::query(
                    "SELECT trigger, COUNT(*) AS c, MAX(created_at) AS last
                 FROM should_have_asked
                 GROUP BY trigger ORDER BY c DESC LIMIT ?1",
                )
                .bind(n)
                .fetch_all(&self.pool)
                .await?
            }
        };
        rows.iter()
            .map(|r| {
                let ts: i64 = r.get("last");
                Ok(TriggerCount {
                    trigger: r.get("trigger"),
                    count: r.get("c"),
                    last_seen: DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_default(),
                })
            })
            .collect()
    }

    /// File-aware lookup for the v0.9 brief. Loads recent rows for the
    /// project and keeps only those whose files overlap with
    /// `target_files` (same basename rules as `mistakes::files_overlap`).
    /// Returns the rows directly — each trigger may have a different
    /// question, so aggregating by trigger here would hide signal.
    pub async fn triggers_for_files(
        &self,
        project: &str,
        target_files: &[String],
        limit: i64,
    ) -> Result<Vec<ShouldHaveAsked>> {
        let recent = self.list(Some(project), None, 500).await?;
        let mut out: Vec<ShouldHaveAsked> = recent
            .into_iter()
            .filter(|r| crate::mistakes::files_overlap(target_files, &r.files))
            .collect();
        out.truncate(limit as usize);
        Ok(out)
    }

    fn row_to_sha(row: &sqlx::sqlite::SqliteRow) -> Result<ShouldHaveAsked> {
        let files_json: String = row.get("files");
        let ts: i64 = row.get("created_at");
        Ok(ShouldHaveAsked {
            id: Some(row.get::<i64, _>("id")),
            project: row.get("project"),
            trigger: row.get("trigger"),
            question: row.get("question"),
            answer: row.get("answer"),
            episode_id: row.get("episode_id"),
            files: serde_json::from_str(&files_json)?,
            created_at: DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(project: &str, trigger: &str, question: &str, answer: &str) -> ShouldHaveAsked {
        ShouldHaveAsked {
            id: None,
            project: project.to_string(),
            trigger: trigger.to_string(),
            question: question.to_string(),
            answer: answer.to_string(),
            episode_id: None,
            files: vec![],
            created_at: Utc::now(),
        }
    }

    #[test]
    fn normalize_trigger_basic() {
        assert_eq!(
            normalize_trigger("Edit Auth Middleware"),
            "edit_auth_middleware"
        );
        assert_eq!(normalize_trigger("new-rust-crate"), "new_rust_crate");
    }

    #[test]
    fn normalize_trigger_collapses_separators() {
        assert_eq!(normalize_trigger("a   b---c"), "a_b_c");
        assert_eq!(normalize_trigger("__leading__trail__"), "leading_trail");
    }

    #[test]
    fn normalize_trigger_strips_punctuation() {
        assert_eq!(
            normalize_trigger("auth.middleware!ordering?"),
            "auth_middleware_ordering"
        );
    }

    #[test]
    fn normalize_trigger_empty_input() {
        assert_eq!(normalize_trigger(""), "");
        assert_eq!(normalize_trigger("---"), "");
    }

    #[tokio::test]
    async fn record_assigns_id_and_lists() {
        let s = AsksStore::open_in_memory().await.unwrap();
        let id = s
            .record(&mk("p", "t", "Why X?", "Because Y"))
            .await
            .unwrap();
        assert!(id > 0);
        let rows = s.list(None, None, 10).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].question, "Why X?");
    }

    #[tokio::test]
    async fn list_filters_by_project_and_trigger() {
        let s = AsksStore::open_in_memory().await.unwrap();
        s.record(&mk("a", "x", "q1", "a1")).await.unwrap();
        s.record(&mk("a", "y", "q2", "a2")).await.unwrap();
        s.record(&mk("b", "x", "q3", "a3")).await.unwrap();

        let by_proj = s.list(Some("a"), None, 10).await.unwrap();
        assert_eq!(by_proj.len(), 2);

        let by_trig = s.list(None, Some("x"), 10).await.unwrap();
        assert_eq!(by_trig.len(), 2);

        let by_both = s.list(Some("a"), Some("x"), 10).await.unwrap();
        assert_eq!(by_both.len(), 1);
        assert_eq!(by_both[0].question, "q1");
    }

    #[tokio::test]
    async fn top_triggers_orders_by_count_desc() {
        let s = AsksStore::open_in_memory().await.unwrap();
        s.record(&mk("p", "x", "q", "a")).await.unwrap();
        s.record(&mk("p", "x", "q", "a")).await.unwrap();
        s.record(&mk("p", "x", "q", "a")).await.unwrap();
        s.record(&mk("p", "y", "q", "a")).await.unwrap();
        s.record(&mk("p", "z", "q", "a")).await.unwrap();
        s.record(&mk("p", "z", "q", "a")).await.unwrap();

        let top = s.top_triggers(Some("p"), 10).await.unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].trigger, "x");
        assert_eq!(top[0].count, 3);
        assert_eq!(top[1].trigger, "z");
        assert_eq!(top[2].trigger, "y");
    }

    #[tokio::test]
    async fn top_triggers_scoped_to_project() {
        let s = AsksStore::open_in_memory().await.unwrap();
        s.record(&mk("p1", "x", "q", "a")).await.unwrap();
        s.record(&mk("p1", "x", "q", "a")).await.unwrap();
        s.record(&mk("p2", "x", "q", "a")).await.unwrap();

        let p1 = s.top_triggers(Some("p1"), 10).await.unwrap();
        assert_eq!(p1[0].count, 2);

        let global = s.top_triggers(None, 10).await.unwrap();
        assert_eq!(global[0].count, 3);
    }

    #[tokio::test]
    async fn triggers_for_files_filters_by_overlap() {
        let s = AsksStore::open_in_memory().await.unwrap();
        let mk_with_files = |trig: &str, q: &str, files: Vec<&str>| ShouldHaveAsked {
            id: None,
            project: "p".into(),
            trigger: trig.into(),
            question: q.into(),
            answer: "a".into(),
            episode_id: None,
            files: files.into_iter().map(String::from).collect(),
            created_at: Utc::now(),
        };
        s.record(&mk_with_files("edit_auth", "Q1", vec!["src/auth.rs"]))
            .await
            .unwrap();
        s.record(&mk_with_files("new_crate", "Q2", vec!["Cargo.toml"]))
            .await
            .unwrap();
        s.record(&mk_with_files(
            "sqlx_mig",
            "Q3",
            vec!["migrations/0001.sql"],
        ))
        .await
        .unwrap();

        let targets = vec!["src/auth.rs".into()];
        let rows = s.triggers_for_files("p", &targets, 10).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].trigger, "edit_auth");
        assert_eq!(rows[0].question, "Q1");

        let targets = vec!["Cargo.toml".into(), "migrations/0001.sql".into()];
        let rows = s.triggers_for_files("p", &targets, 10).await.unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[tokio::test]
    async fn files_roundtrip_as_json() {
        let s = AsksStore::open_in_memory().await.unwrap();
        let mut row = mk("p", "t", "q", "a");
        row.files = vec!["src/a.rs".into(), "src/b.rs".into()];
        s.record(&row).await.unwrap();
        let got = s.list(None, None, 10).await.unwrap();
        assert_eq!(got[0].files, vec!["src/a.rs", "src/b.rs"]);
    }
}
