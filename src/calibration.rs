// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-bucket calibration tracking (v0.8.1).
//!
//! Every coding agent has systematic biases — declared "success" doesn't
//! always survive contact with reality. This module tracks, per
//! `(task_type, project)` bucket:
//!
//!   - declared_success  — captured with outcome=success
//!   - verified_success  — advanced verification to StableNoRevert+
//!   - declared_failure  — captured with outcome=failure
//!   - refuted_success   — declared success that was later flipped
//!     (reserved for v0.8.x revert detection; always 0 today)
//!
//! The overconfidence rate per bucket lets future retrieval damp
//! "fresh-but-unverified" claims by their historical refute rate.
//! v0.8.1 only collects + surfaces; retrieval-time application is
//! v0.8.1.1.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::Path;

use crate::config::Config;
use crate::episode::{Episode, OutcomeStatus, TaskType, VerificationState};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBucket {
    pub task_type: String,
    pub project: String,
    pub declared_success: i64,
    pub verified_success: i64,
    pub declared_failure: i64,
    pub refuted_success: i64,
    pub last_updated: DateTime<Utc>,
}

impl CalibrationBucket {
    /// Overconfidence rate: fraction of declared-success captures that
    /// either never reached `StableNoRevert+` or got refuted later.
    ///
    /// Returns 0.0 when there's no declared-success data (no signal
    /// yet). At `declared_success == 0` calibration shouldn't move
    /// ranking.
    ///
    /// Note: low sample sizes inflate this metric. Callers should
    /// gate on a minimum count (e.g. require `declared_success >= 10`
    /// before applying the correction at retrieval time).
    pub fn overconfidence_rate(&self) -> f32 {
        if self.declared_success <= 0 {
            return 0.0;
        }
        let unverified = self.declared_success - self.verified_success + self.refuted_success;
        (unverified as f32 / self.declared_success as f32).clamp(0.0, 1.0)
    }

    /// Verified ratio — for surface metrics. 1.0 = perfectly
    /// calibrated.
    pub fn verified_ratio(&self) -> f32 {
        if self.declared_success <= 0 {
            return 1.0;
        }
        (self.verified_success as f32 / self.declared_success as f32).clamp(0.0, 1.0)
    }

    pub fn sample_size(&self) -> i64 {
        self.declared_success + self.declared_failure
    }
}

// ===== Store =====

#[derive(Clone)]
pub struct CalibrationStore {
    pool: SqlitePool,
}

impl CalibrationStore {
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
            .with_context(|| format!("Failed to open calibration DB at {}", path.display()))?;
        MIGRATOR
            .run(&pool)
            .await
            .context("Failed to run calibration migrations")?;
        Ok(Self { pool })
    }

    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn get(&self, task_type: &str, project: &str) -> Result<Option<CalibrationBucket>> {
        let row = sqlx::query(
            "SELECT task_type, project, declared_success, verified_success,
                    declared_failure, refuted_success, last_updated
             FROM calibration_buckets
             WHERE task_type = ?1 AND project = ?2",
        )
        .bind(task_type)
        .bind(project)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::row_to_bucket).transpose()
    }

    pub async fn list_all(&self) -> Result<Vec<CalibrationBucket>> {
        let rows = sqlx::query(
            "SELECT task_type, project, declared_success, verified_success,
                    declared_failure, refuted_success, last_updated
             FROM calibration_buckets
             ORDER BY (declared_success + declared_failure) DESC, last_updated DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_bucket).collect()
    }

    pub async fn list_by_project(&self, project: &str) -> Result<Vec<CalibrationBucket>> {
        let rows = sqlx::query(
            "SELECT task_type, project, declared_success, verified_success,
                    declared_failure, refuted_success, last_updated
             FROM calibration_buckets
             WHERE project = ?1
             ORDER BY (declared_success + declared_failure) DESC",
        )
        .bind(project)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(Self::row_to_bucket).collect()
    }

    /// Atomic bump of one or more counters. Creates the bucket if it
    /// doesn't exist. Use the convenience helpers `record_capture` /
    /// `record_verification` instead of calling this directly.
    pub async fn bump(
        &self,
        task_type: &str,
        project: &str,
        delta_declared_success: i64,
        delta_verified_success: i64,
        delta_declared_failure: i64,
        delta_refuted_success: i64,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO calibration_buckets
                (task_type, project, declared_success, verified_success,
                 declared_failure, refuted_success, last_updated)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(task_type, project) DO UPDATE SET
                declared_success = declared_success + ?3,
                verified_success = verified_success + ?4,
                declared_failure = declared_failure + ?5,
                refuted_success  = refuted_success + ?6,
                last_updated     = ?7",
        )
        .bind(task_type)
        .bind(project)
        .bind(delta_declared_success)
        .bind(delta_verified_success)
        .bind(delta_declared_failure)
        .bind(delta_refuted_success)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    fn row_to_bucket(row: &sqlx::sqlite::SqliteRow) -> Result<CalibrationBucket> {
        let ts: i64 = row.get("last_updated");
        Ok(CalibrationBucket {
            task_type: row.get("task_type"),
            project: row.get("project"),
            declared_success: row.get("declared_success"),
            verified_success: row.get("verified_success"),
            declared_failure: row.get("declared_failure"),
            refuted_success: row.get("refuted_success"),
            last_updated: DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_default(),
        })
    }
}

// ===== Recording =====

/// Bump declared_* counters from a new capture's outcome. Called from
/// the MCP capture handler and the CLI capture path after the episode
/// is saved.
pub async fn record_capture(store: &CalibrationStore, episode: &Episode) -> Result<()> {
    let task_type = episode.intent.task_type.to_string();
    let project = &episode.project;
    match episode.outcome.status {
        OutcomeStatus::Success => store.bump(&task_type, project, 1, 0, 0, 0).await,
        OutcomeStatus::Failure => store.bump(&task_type, project, 0, 0, 1, 0).await,
        // Partial captures count as neither — they're observations, not claims.
        OutcomeStatus::Partial => Ok(()),
    }
}

/// Bump verified_success when an episode reaches StableNoRevert or
/// ValidatedCrossProject. Called from advance-verification.
pub async fn record_verification_advance(
    store: &CalibrationStore,
    episode: &Episode,
) -> Result<()> {
    let should_count = matches!(
        episode.outcome.verification,
        VerificationState::StableNoRevert { .. } | VerificationState::ValidatedCrossProject { .. }
    );
    if !should_count {
        return Ok(());
    }
    // Only count verification of *successful* outcomes — verifying a
    // failure doesn't add calibration signal in the success direction.
    if episode.outcome.status != OutcomeStatus::Success {
        return Ok(());
    }
    let task_type = episode.intent.task_type.to_string();
    store.bump(&task_type, &episode.project, 0, 1, 0, 0).await
}

/// Stable identity for a `TaskType` — kept here so the calibration UI
/// keeps a consistent label even if `episode.rs`'s Display impl changes.
pub fn task_type_label(t: &TaskType) -> String {
    t.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::Episode;

    #[test]
    fn overconfidence_zero_when_no_data() {
        let b = CalibrationBucket {
            task_type: "bugfix".into(),
            project: "p".into(),
            declared_success: 0,
            verified_success: 0,
            declared_failure: 0,
            refuted_success: 0,
            last_updated: Utc::now(),
        };
        assert_eq!(b.overconfidence_rate(), 0.0);
        assert_eq!(b.verified_ratio(), 1.0);
        assert_eq!(b.sample_size(), 0);
    }

    #[test]
    fn overconfidence_high_when_no_verifications() {
        let b = CalibrationBucket {
            task_type: "bugfix".into(),
            project: "p".into(),
            declared_success: 10,
            verified_success: 0,
            declared_failure: 0,
            refuted_success: 0,
            last_updated: Utc::now(),
        };
        // 10 declared, 0 verified → 100% unverified
        assert!((b.overconfidence_rate() - 1.0).abs() < 1e-6);
        assert_eq!(b.verified_ratio(), 0.0);
    }

    #[test]
    fn overconfidence_decreases_with_verifications() {
        let b = CalibrationBucket {
            task_type: "bugfix".into(),
            project: "p".into(),
            declared_success: 10,
            verified_success: 7,
            declared_failure: 2,
            refuted_success: 0,
            last_updated: Utc::now(),
        };
        // 30% of declared successes are still unverified.
        assert!((b.overconfidence_rate() - 0.3).abs() < 1e-6);
        assert!((b.verified_ratio() - 0.7).abs() < 1e-6);
        assert_eq!(b.sample_size(), 12);
    }

    #[test]
    fn overconfidence_counts_refuted_against_declared() {
        let b = CalibrationBucket {
            task_type: "bugfix".into(),
            project: "p".into(),
            declared_success: 10,
            verified_success: 5,
            declared_failure: 0,
            refuted_success: 3,
            last_updated: Utc::now(),
        };
        // 10 declared, 5 verified, 3 refuted → (10 - 5 + 3) / 10 = 0.8
        assert!((b.overconfidence_rate() - 0.8).abs() < 1e-6);
    }

    #[tokio::test]
    async fn bump_creates_bucket_on_first_insert() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        s.bump("bugfix", "tempera", 1, 0, 0, 0).await.unwrap();
        let b = s.get("bugfix", "tempera").await.unwrap().expect("bucket");
        assert_eq!(b.declared_success, 1);
        assert_eq!(b.verified_success, 0);
    }

    #[tokio::test]
    async fn bump_accumulates_atomically() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        s.bump("bugfix", "tempera", 1, 0, 0, 0).await.unwrap();
        s.bump("bugfix", "tempera", 1, 0, 0, 0).await.unwrap();
        s.bump("bugfix", "tempera", 0, 1, 0, 0).await.unwrap();
        let b = s.get("bugfix", "tempera").await.unwrap().unwrap();
        assert_eq!(b.declared_success, 2);
        assert_eq!(b.verified_success, 1);
    }

    #[tokio::test]
    async fn bump_per_task_type_separately() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        s.bump("bugfix", "tempera", 1, 0, 0, 0).await.unwrap();
        s.bump("feature", "tempera", 1, 0, 0, 0).await.unwrap();
        assert_eq!(s.list_all().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn list_by_project_filters() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        s.bump("bugfix", "tempera", 1, 0, 0, 0).await.unwrap();
        s.bump("bugfix", "smelt", 1, 0, 0, 0).await.unwrap();
        let tempera_buckets = s.list_by_project("tempera").await.unwrap();
        assert_eq!(tempera_buckets.len(), 1);
        assert_eq!(tempera_buckets[0].project, "tempera");
    }

    #[tokio::test]
    async fn record_capture_success_bumps_declared() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        let mut ep = Episode::new("tempera".into(), "x".into());
        ep.intent.task_type = TaskType::Bugfix;
        ep.outcome.status = OutcomeStatus::Success;
        record_capture(&s, &ep).await.unwrap();
        let b = s.get("bugfix", "tempera").await.unwrap().unwrap();
        assert_eq!(b.declared_success, 1);
        assert_eq!(b.declared_failure, 0);
    }

    #[tokio::test]
    async fn record_capture_failure_bumps_declared_failure() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        let mut ep = Episode::new("tempera".into(), "x".into());
        ep.intent.task_type = TaskType::Feature;
        ep.outcome.status = OutcomeStatus::Failure;
        record_capture(&s, &ep).await.unwrap();
        let b = s.get("feature", "tempera").await.unwrap().unwrap();
        assert_eq!(b.declared_success, 0);
        assert_eq!(b.declared_failure, 1);
    }

    #[tokio::test]
    async fn record_capture_partial_does_not_bump() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        let mut ep = Episode::new("tempera".into(), "x".into());
        ep.intent.task_type = TaskType::Bugfix;
        ep.outcome.status = OutcomeStatus::Partial;
        record_capture(&s, &ep).await.unwrap();
        // No bucket created — Partial captures are observations, not claims.
        assert!(s.get("bugfix", "tempera").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn record_verification_only_counts_stable_or_above() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        let mut ep = Episode::new("tempera".into(), "x".into());
        ep.intent.task_type = TaskType::Bugfix;
        ep.outcome.status = OutcomeStatus::Success;

        // Untested → no count
        record_verification_advance(&s, &ep).await.unwrap();
        assert!(s.get("bugfix", "tempera").await.unwrap().is_none());

        // Merged → still no count (not yet stable)
        ep.outcome.verification = VerificationState::Merged {
            commit: "abc".into(),
            at: Utc::now(),
        };
        record_verification_advance(&s, &ep).await.unwrap();
        assert!(s.get("bugfix", "tempera").await.unwrap().is_none());

        // StableNoRevert → counts
        ep.outcome.verification = VerificationState::StableNoRevert {
            days: 30,
            since: Utc::now(),
        };
        record_verification_advance(&s, &ep).await.unwrap();
        let b = s.get("bugfix", "tempera").await.unwrap().unwrap();
        assert_eq!(b.verified_success, 1);

        // ValidatedCrossProject → also counts
        ep.outcome.verification = VerificationState::ValidatedCrossProject {
            evidence_episodes: vec!["a".into(), "b".into()],
        };
        record_verification_advance(&s, &ep).await.unwrap();
        let b = s.get("bugfix", "tempera").await.unwrap().unwrap();
        assert_eq!(b.verified_success, 2);
    }

    #[tokio::test]
    async fn record_verification_ignores_failure_outcomes() {
        let s = CalibrationStore::open_in_memory().await.unwrap();
        let mut ep = Episode::new("tempera".into(), "x".into());
        ep.intent.task_type = TaskType::Bugfix;
        ep.outcome.status = OutcomeStatus::Failure;
        ep.outcome.verification = VerificationState::StableNoRevert {
            days: 30,
            since: Utc::now(),
        };
        record_verification_advance(&s, &ep).await.unwrap();
        // Verified-failure doesn't add calibration signal.
        assert!(s.get("bugfix", "tempera").await.unwrap().is_none());
    }
}
