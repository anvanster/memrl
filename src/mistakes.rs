// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Anchored mistakes (v0.8.2).
//!
//! Categorized log of corrections the user made to the agent's
//! assumptions, decisions, or code. The agent calls the MCP tool
//! `tempera_log_correction` when it recognizes a user turn as a
//! correction. Future v0.9 `tempera_brief` surfaces the top-N
//! categories for the files being touched, so the agent knows where
//! it tends to be wrong before it acts.

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
pub struct Mistake {
    pub id: Option<i64>,
    pub project: String,
    pub category: String,
    pub episode_id: Option<String>,
    pub files: Vec<String>,
    pub description: String,
    pub correction: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CategoryCount {
    pub category: String,
    pub count: i64,
    pub last_seen: DateTime<Utc>,
}

#[derive(Clone)]
pub struct MistakeStore {
    pool: SqlitePool,
}

impl MistakeStore {
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
            .with_context(|| format!("Failed to open mistakes DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run mistake migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Insert a new mistake. Returns the assigned id.
    pub async fn record(&self, m: &Mistake) -> Result<i64> {
        let now = m.created_at.timestamp();
        let files_json = serde_json::to_string(&m.files)?;
        let row = sqlx::query(
            "INSERT INTO mistakes (project, category, episode_id, files,
                                   description, correction, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             RETURNING id",
        )
        .bind(&m.project)
        .bind(&m.category)
        .bind(m.episode_id.as_deref())
        .bind(files_json)
        .bind(&m.description)
        .bind(&m.correction)
        .bind(now)
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert mistake")?;
        Ok(row.get(0))
    }

    pub async fn list(
        &self,
        project: Option<&str>,
        category: Option<&str>,
        limit: i64,
    ) -> Result<Vec<Mistake>> {
        let rows = match (project, category) {
            (Some(p), Some(c)) => {
                sqlx::query(
                    "SELECT id, project, category, episode_id, files, description,
                        correction, created_at
                 FROM mistakes
                 WHERE project = ?1 AND category = ?2
                 ORDER BY created_at DESC
                 LIMIT ?3",
                )
                .bind(p)
                .bind(c)
                .bind(limit)
                .fetch_all(&self.pool)
                .await?
            }
            (Some(p), None) => {
                sqlx::query(
                    "SELECT id, project, category, episode_id, files, description,
                        correction, created_at
                 FROM mistakes
                 WHERE project = ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
                )
                .bind(p)
                .bind(limit)
                .fetch_all(&self.pool)
                .await?
            }
            (None, Some(c)) => {
                sqlx::query(
                    "SELECT id, project, category, episode_id, files, description,
                        correction, created_at
                 FROM mistakes
                 WHERE category = ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
                )
                .bind(c)
                .bind(limit)
                .fetch_all(&self.pool)
                .await?
            }
            (None, None) => {
                sqlx::query(
                    "SELECT id, project, category, episode_id, files, description,
                        correction, created_at
                 FROM mistakes
                 ORDER BY created_at DESC
                 LIMIT ?1",
                )
                .bind(limit)
                .fetch_all(&self.pool)
                .await?
            }
        };
        rows.iter().map(Self::row_to_mistake).collect()
    }

    /// Top categories by count, optionally scoped to a project. Used by
    /// the future `tempera_brief` (v0.9) to warn about categories the
    /// agent tends to get wrong in this codebase.
    pub async fn top_categories(
        &self,
        project: Option<&str>,
        limit: i64,
    ) -> Result<Vec<CategoryCount>> {
        let rows = if let Some(p) = project {
            sqlx::query(
                "SELECT category, COUNT(*) as n, MAX(created_at) as last_ts
                 FROM mistakes
                 WHERE project = ?1
                 GROUP BY category
                 ORDER BY n DESC, last_ts DESC
                 LIMIT ?2",
            )
            .bind(p)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT category, COUNT(*) as n, MAX(created_at) as last_ts
                 FROM mistakes
                 GROUP BY category
                 ORDER BY n DESC, last_ts DESC
                 LIMIT ?1",
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };
        rows.iter()
            .map(|r| {
                let ts: i64 = r.get("last_ts");
                Ok(CategoryCount {
                    category: r.get("category"),
                    count: r.get("n"),
                    last_seen: DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_default(),
                })
            })
            .collect()
    }

    fn row_to_mistake(row: &sqlx::sqlite::SqliteRow) -> Result<Mistake> {
        let ts: i64 = row.get("created_at");
        let files_json: String = row.get("files");
        Ok(Mistake {
            id: Some(row.get("id")),
            project: row.get("project"),
            category: row.get("category"),
            episode_id: row.get("episode_id"),
            files: serde_json::from_str(&files_json).unwrap_or_default(),
            description: row.get("description"),
            correction: row.get("correction"),
            created_at: DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_default(),
        })
    }
}

/// Normalize a free-form category string. Lowercase, replace spaces +
/// hyphens with underscores, strip anything outside `[a-z0-9_]`. Keeps
/// related corrections from fragmenting into "Lifetime Annotations",
/// "lifetime-annotations", "Lifetime_Annotations" rows.
pub fn normalize_category(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_under = false;
    for c in s.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_under = false;
        } else if !out.is_empty() && !prev_under {
            out.push('_');
            prev_under = true;
        }
    }
    out.trim_end_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_mistake(project: &str, category: &str, desc: &str) -> Mistake {
        Mistake {
            id: None,
            project: project.into(),
            category: category.into(),
            episode_id: None,
            files: vec![],
            description: desc.into(),
            correction: "do it differently".into(),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn normalize_category_basic() {
        assert_eq!(
            normalize_category("Lifetime Annotations"),
            "lifetime_annotations"
        );
        assert_eq!(
            normalize_category("lifetime-annotations"),
            "lifetime_annotations"
        );
        assert_eq!(
            normalize_category("Lifetime_Annotations"),
            "lifetime_annotations"
        );
    }

    #[test]
    fn normalize_category_collapses_separators() {
        assert_eq!(normalize_category("foo  bar"), "foo_bar");
        assert_eq!(normalize_category("foo - bar"), "foo_bar");
        assert_eq!(normalize_category("__foo__"), "foo");
    }

    #[test]
    fn normalize_category_strips_punctuation() {
        assert_eq!(normalize_category("can't-find-it!"), "can_t_find_it");
    }

    #[test]
    fn normalize_category_handles_empty() {
        assert_eq!(normalize_category(""), "");
        assert_eq!(normalize_category("!!!"), "");
    }

    #[tokio::test]
    async fn record_assigns_id() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        let id = s
            .record(&mk_mistake("tempera", "test_setup", "missed a fixture"))
            .await
            .unwrap();
        assert!(id > 0);
    }

    #[tokio::test]
    async fn list_orders_newest_first() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        s.record(&mk_mistake("tempera", "a", "first"))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        s.record(&mk_mistake("tempera", "a", "second"))
            .await
            .unwrap();
        let rows = s.list(None, None, 10).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].description, "second");
    }

    #[tokio::test]
    async fn list_filters_by_project() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        s.record(&mk_mistake("tempera", "a", "x")).await.unwrap();
        s.record(&mk_mistake("smelt", "a", "y")).await.unwrap();
        let tempera_only = s.list(Some("tempera"), None, 10).await.unwrap();
        assert_eq!(tempera_only.len(), 1);
        assert_eq!(tempera_only[0].project, "tempera");
    }

    #[tokio::test]
    async fn list_filters_by_category() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        s.record(&mk_mistake("p", "lifetime_annotations", "1"))
            .await
            .unwrap();
        s.record(&mk_mistake("p", "test_setup", "2")).await.unwrap();
        let cat = s.list(None, Some("test_setup"), 10).await.unwrap();
        assert_eq!(cat.len(), 1);
        assert_eq!(cat[0].category, "test_setup");
    }

    #[tokio::test]
    async fn top_categories_returns_counts_desc() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        for _ in 0..3 {
            s.record(&mk_mistake("p", "lifetime_annotations", "x"))
                .await
                .unwrap();
        }
        s.record(&mk_mistake("p", "test_setup", "y")).await.unwrap();
        let top = s.top_categories(Some("p"), 10).await.unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].category, "lifetime_annotations");
        assert_eq!(top[0].count, 3);
        assert_eq!(top[1].category, "test_setup");
        assert_eq!(top[1].count, 1);
    }

    #[tokio::test]
    async fn files_field_roundtrip_as_json() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        let mut m = mk_mistake("tempera", "a", "x");
        m.files = vec!["src/foo.rs".into(), "src/bar.rs".into()];
        s.record(&m).await.unwrap();
        let rows = s.list(None, None, 10).await.unwrap();
        assert_eq!(rows[0].files, vec!["src/foo.rs", "src/bar.rs"]);
    }

    #[tokio::test]
    async fn top_categories_scoped_to_project() {
        let s = MistakeStore::open_in_memory().await.unwrap();
        s.record(&mk_mistake("tempera", "auth", "1")).await.unwrap();
        s.record(&mk_mistake("smelt", "db", "2")).await.unwrap();
        let tempera_top = s.top_categories(Some("tempera"), 10).await.unwrap();
        assert_eq!(tempera_top.len(), 1);
        assert_eq!(tempera_top[0].category, "auth");
    }
}
