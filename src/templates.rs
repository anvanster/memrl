// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reasoning templates — store + types (v0.8.3).
//!
//! Where reflection asks "what was salient yesterday?" and patterns asks
//! "what theme keeps coming back across days?", templates asks one more
//! thing: "*how* does the agent succeed at this kind of task in this
//! domain?" A template is the imperative step sequence the agent
//! followed when it worked, keyed on `(task_type, domain)` and pulled
//! at task start via the `tempera_template` MCP tool.
//!
//! This module holds the **storage + grouping primitives** so both
//! binaries (`tempera` and `tempera-mcp`) can read templates. The
//! authoring pipeline that runs during the dream cycle and calls
//! Sonnet lives in `templates_phase.rs` — only `tempera` registers it.
//!
//! Notes:
//! - `success_rate` is the **average verification weight** across the
//!   cluster, not the fraction of successful episodes — the phase
//!   pre-filters to Success status. Treats well-verified clusters as
//!   higher-confidence templates.
//! - Re-runs of the phase refresh templates in place keyed on
//!   `(task_type, domain)`. The sidecar gets overwritten too.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::collections::HashMap;
use std::path::Path;

use crate::config::Config;
use crate::episode::{Episode, OutcomeStatus};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// USD per Sonnet template authorship call. Lives here so `dream.rs`
/// can budget for the phase without depending on the (heavier)
/// `templates_phase` module. Slightly above patterns since each
/// cluster's input is denser.
pub const TEMPLATE_ESTIMATED_COST_USD: f32 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub id: String,
    pub task_type: String,
    pub domain: String,
    pub name: String,
    pub steps: Vec<String>,
    pub evidence_episodes: Vec<String>,
    /// Average verification weight across evidence episodes. Higher =
    /// more trustworthy template. Range [0.0, 1.0].
    pub success_rate: f32,
    pub times_used: i64,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub last_used: Option<DateTime<Utc>>,
}

// ===== Store =====

#[derive(Clone)]
pub struct TemplateStore {
    pool: SqlitePool,
}

impl TemplateStore {
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
            .with_context(|| format!("Failed to open templates DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run template migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn get_by_pair(&self, task_type: &str, domain: &str) -> Result<Option<Template>> {
        let row = sqlx::query(
            "SELECT id, task_type, domain, name, steps, evidence_episodes,
                    success_rate, times_used, model, created_at, last_used
             FROM reasoning_templates WHERE task_type = ?1 AND domain = ?2",
        )
        .bind(task_type)
        .bind(domain)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::row_to_template).transpose()
    }

    pub async fn list_all(&self) -> Result<Vec<Template>> {
        let rows = sqlx::query(
            "SELECT id, task_type, domain, name, steps, evidence_episodes,
                    success_rate, times_used, model, created_at, last_used
             FROM reasoning_templates ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_template).collect()
    }

    pub async fn put(&self, t: &Template) -> Result<()> {
        sqlx::query(
            "INSERT INTO reasoning_templates
                (id, task_type, domain, name, steps, evidence_episodes,
                 success_rate, times_used, model, created_at, last_used)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(task_type, domain) DO UPDATE SET
                name = excluded.name,
                steps = excluded.steps,
                evidence_episodes = excluded.evidence_episodes,
                success_rate = excluded.success_rate,
                model = excluded.model,
                created_at = excluded.created_at",
        )
        .bind(&t.id)
        .bind(&t.task_type)
        .bind(&t.domain)
        .bind(&t.name)
        .bind(serde_json::to_string(&t.steps)?)
        .bind(serde_json::to_string(&t.evidence_episodes)?)
        .bind(t.success_rate)
        .bind(t.times_used)
        .bind(&t.model)
        .bind(t.created_at.timestamp())
        .bind(t.last_used.map(|d| d.timestamp()))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Bump `times_used` and refresh `last_used` for a template. Best-
    /// effort: any error is propagated, callers may choose to ignore.
    pub async fn touch_used(&self, task_type: &str, domain: &str) -> Result<()> {
        sqlx::query(
            "UPDATE reasoning_templates
                SET times_used = times_used + 1,
                    last_used = ?3
              WHERE task_type = ?1 AND domain = ?2",
        )
        .bind(task_type)
        .bind(domain)
        .bind(Utc::now().timestamp())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    fn row_to_template(row: &sqlx::sqlite::SqliteRow) -> Result<Template> {
        let steps_json: String = row.get("steps");
        let ev_json: String = row.get("evidence_episodes");
        let created_ts: i64 = row.get("created_at");
        let last_ts: Option<i64> = row.get("last_used");
        Ok(Template {
            id: row.get("id"),
            task_type: row.get("task_type"),
            domain: row.get("domain"),
            name: row.get("name"),
            steps: serde_json::from_str(&steps_json)?,
            evidence_episodes: serde_json::from_str(&ev_json)?,
            success_rate: row.get("success_rate"),
            times_used: row.get("times_used"),
            model: row.get("model"),
            created_at: DateTime::<Utc>::from_timestamp(created_ts, 0).unwrap_or_default(),
            last_used: last_ts.and_then(|t| DateTime::<Utc>::from_timestamp(t, 0)),
        })
    }
}

// ===== Clustering primitives =====

/// `(task_type, domain)` key produced by `group_by_task_domain`. Owned
/// strings so we can use it across episode borrows.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskDomainKey {
    pub task_type: String,
    pub domain: String,
}

/// Group eligible episodes by `(task_type, domain_tag)`. An episode
/// tagged with multiple domains contributes to *each* bucket
/// independently. Episodes with no domain tag are skipped — we can't
/// generalize "the bugfix template for ???".
pub fn group_by_task_domain<'a>(
    episodes: impl IntoIterator<Item = &'a Episode>,
) -> HashMap<TaskDomainKey, Vec<&'a Episode>> {
    let mut buckets: HashMap<TaskDomainKey, Vec<&'a Episode>> = HashMap::new();
    for ep in episodes {
        for d in &ep.intent.domain {
            let key = TaskDomainKey {
                task_type: ep.intent.task_type.to_string(),
                domain: d.clone(),
            };
            buckets.entry(key).or_default().push(ep);
        }
    }
    buckets
}

/// Filter predicate — episode is "eligible" for templating:
/// - status == Success
/// - verification weight at or above the configured threshold
pub fn is_eligible(ep: &Episode, min_verification_weight: f32) -> bool {
    if ep.outcome.status != OutcomeStatus::Success {
        return false;
    }
    ep.outcome.verification.weight() >= min_verification_weight
}

/// Average verification weight across the cluster.
pub fn cluster_success_rate(cluster: &[&Episode]) -> f32 {
    if cluster.is_empty() {
        return 0.0;
    }
    let total: f32 = cluster
        .iter()
        .map(|e| e.outcome.verification.weight())
        .sum();
    total / cluster.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::{
        Context, Intent, Outcome, OutcomeStatus, TaskType, Utility, VerificationState,
    };

    fn mk_episode(
        id: &str,
        task_type: TaskType,
        domain: Vec<String>,
        status: OutcomeStatus,
        verification: VerificationState,
    ) -> Episode {
        Episode {
            schema_version: 5,
            id: id.to_string(),
            timestamp_start: Utc::now(),
            timestamp_end: Utc::now(),
            project: "p".into(),
            intent: Intent {
                raw_prompt: format!("do the thing in {id}"),
                extracted_intent: format!("did the thing in {id}"),
                task_type,
                domain,
                claim: None,
            },
            context: Context {
                files_read: vec![],
                files_modified: vec!["src/foo.rs".into()],
                tools_invoked: vec!["Read".into(), "Edit".into()],
                errors_encountered: vec![],
            },
            outcome: Outcome {
                status,
                tests_before: None,
                tests_after: None,
                commit_sha: None,
                pr_number: None,
                verification,
            },
            utility: Utility::default(),
            retrieval_history: vec![],
            session_id: None,
            related_episodes: vec![],
            alternatives_considered: vec![],
        }
    }

    #[test]
    fn group_buckets_by_task_and_domain() {
        let eps = vec![
            mk_episode(
                "a",
                TaskType::Bugfix,
                vec!["rust".into(), "async".into()],
                OutcomeStatus::Success,
                VerificationState::Untested,
            ),
            mk_episode(
                "b",
                TaskType::Bugfix,
                vec!["rust".into()],
                OutcomeStatus::Success,
                VerificationState::Untested,
            ),
            mk_episode(
                "c",
                TaskType::Feature,
                vec!["async".into()],
                OutcomeStatus::Success,
                VerificationState::Untested,
            ),
        ];
        let buckets = group_by_task_domain(&eps);
        let bugfix_rust = buckets
            .get(&TaskDomainKey {
                task_type: "bugfix".into(),
                domain: "rust".into(),
            })
            .unwrap();
        assert_eq!(bugfix_rust.len(), 2);
        let feature_async = buckets
            .get(&TaskDomainKey {
                task_type: "feature".into(),
                domain: "async".into(),
            })
            .unwrap();
        assert_eq!(feature_async.len(), 1);
    }

    #[test]
    fn episode_with_no_domain_contributes_to_no_bucket() {
        let eps = vec![mk_episode(
            "a",
            TaskType::Bugfix,
            vec![],
            OutcomeStatus::Success,
            VerificationState::Untested,
        )];
        let buckets = group_by_task_domain(&eps);
        assert!(buckets.is_empty());
    }

    #[test]
    fn is_eligible_requires_success_status() {
        let failure = mk_episode(
            "a",
            TaskType::Bugfix,
            vec!["x".into()],
            OutcomeStatus::Failure,
            VerificationState::Untested,
        );
        assert!(!is_eligible(&failure, 0.0));

        let partial = mk_episode(
            "a",
            TaskType::Bugfix,
            vec!["x".into()],
            OutcomeStatus::Partial,
            VerificationState::Untested,
        );
        assert!(!is_eligible(&partial, 0.0));

        let success = mk_episode(
            "a",
            TaskType::Bugfix,
            vec!["x".into()],
            OutcomeStatus::Success,
            VerificationState::Untested,
        );
        assert!(is_eligible(&success, 0.0));
    }

    #[test]
    fn is_eligible_respects_verification_threshold() {
        // Untested verification weight = 0.30.
        let untested = mk_episode(
            "a",
            TaskType::Bugfix,
            vec!["x".into()],
            OutcomeStatus::Success,
            VerificationState::Untested,
        );
        assert!(is_eligible(&untested, 0.20));
        assert!(!is_eligible(&untested, 0.50));
    }

    #[test]
    fn cluster_success_rate_averages_verification() {
        let a = mk_episode(
            "a",
            TaskType::Bugfix,
            vec!["x".into()],
            OutcomeStatus::Success,
            VerificationState::Untested,
        );
        let b = mk_episode(
            "b",
            TaskType::Bugfix,
            vec!["x".into()],
            OutcomeStatus::Success,
            VerificationState::Untested,
        );
        let cluster = vec![&a, &b];
        let rate = cluster_success_rate(&cluster);
        // Both Untested = 0.30 → avg = 0.30
        assert!((rate - 0.30).abs() < 1e-6);
    }

    #[tokio::test]
    async fn store_roundtrip() {
        let s = TemplateStore::open_in_memory().await.unwrap();
        let t = Template {
            id: "bugfix-rust-1".into(),
            task_type: "bugfix".into(),
            domain: "rust".into(),
            name: "spawn_blocking deadlock dance".into(),
            steps: vec![
                "Find every tokio::spawn_blocking site".into(),
                "Check Drop ordering for guards held across awaits".into(),
                "Confirm by running the test under loom".into(),
            ],
            evidence_episodes: vec!["a".into(), "b".into(), "c".into()],
            success_rate: 0.42,
            times_used: 0,
            model: "claude-sonnet-4-6".into(),
            created_at: Utc::now(),
            last_used: None,
        };
        s.put(&t).await.unwrap();
        let got = s.get_by_pair("bugfix", "rust").await.unwrap().unwrap();
        assert_eq!(got.steps.len(), 3);
        assert_eq!(got.evidence_episodes.len(), 3);
        assert!((got.success_rate - 0.42).abs() < 1e-6);
    }

    #[tokio::test]
    async fn store_upsert_by_pair() {
        let s = TemplateStore::open_in_memory().await.unwrap();
        let mk = |name: &str, sr: f32| Template {
            id: format!("bugfix-rust-{}", name),
            task_type: "bugfix".into(),
            domain: "rust".into(),
            name: name.into(),
            steps: vec!["step".into()],
            evidence_episodes: vec![],
            success_rate: sr,
            times_used: 0,
            model: "m".into(),
            created_at: Utc::now(),
            last_used: None,
        };
        s.put(&mk("v1", 0.3)).await.unwrap();
        s.put(&mk("v2", 0.7)).await.unwrap();
        let got = s.get_by_pair("bugfix", "rust").await.unwrap().unwrap();
        assert_eq!(got.name, "v2");
        let all = s.list_all().await.unwrap();
        assert_eq!(all.len(), 1);
    }

    #[tokio::test]
    async fn touch_used_increments_and_sets_last_used() {
        let s = TemplateStore::open_in_memory().await.unwrap();
        let t = Template {
            id: "x".into(),
            task_type: "bugfix".into(),
            domain: "rust".into(),
            name: "n".into(),
            steps: vec!["a".into()],
            evidence_episodes: vec![],
            success_rate: 0.5,
            times_used: 0,
            model: "m".into(),
            created_at: Utc::now(),
            last_used: None,
        };
        s.put(&t).await.unwrap();
        s.touch_used("bugfix", "rust").await.unwrap();
        let got = s.get_by_pair("bugfix", "rust").await.unwrap().unwrap();
        assert_eq!(got.times_used, 1);
        assert!(got.last_used.is_some());
    }
}
