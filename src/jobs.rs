// Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Background job queue for tempera.
//!
//! SQLite-backed, single-writer (one daemon). Existing synchronous CLI
//! commands still work unchanged; the queue is opt-in for offloading work
//! that shouldn't block an MCP call (indexing, propagation, future dream
//! cycle phases).
//!
//! Lease semantics: leaser SELECTs a pending job, atomically updates it to
//! status='running' with `locked_until = now + LEASE_SECONDS`. If the worker
//! crashes mid-job, the lease expires and the job becomes leasable again.
//! Per-job retries with exponential backoff; after `max_attempts` failures,
//! the job is marked `dead` and surfaced by `list_dead()`.

#![allow(dead_code)]

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use crate::config::Config;

/// How long a leased job is "owned" by a worker before becoming leasable again.
pub const LEASE_SECONDS: i64 = 60;
/// Daemon polling cadence.
pub const POLL_INTERVAL_SECONDS: u64 = 5;
/// Per-job execution timeout. Must be < LEASE_SECONDS so the lease covers it.
pub const HANDLER_TIMEOUT_SECONDS: u64 = 55;
/// Default retry budget per job.
pub const DEFAULT_MAX_ATTEMPTS: i64 = 3;

/// Embedded migrator. `sqlx::migrate!()` reads `./migrations/` at compile
/// time, bundles them into the binary, and tracks applied versions in
/// `_sqlx_migrations`. Add a new migration by dropping a new file there;
/// no code change required.
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Dead,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Dead => "dead",
        }
    }
}

impl FromStr for JobStatus {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "pending" => Self::Pending,
            "running" => Self::Running,
            "completed" => Self::Completed,
            "dead" => Self::Dead,
            other => anyhow::bail!("unknown job status: {other}"),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: i64,
    pub kind: String,
    pub payload: serde_json::Value,
    pub status: JobStatus,
    pub attempts: i64,
    pub max_attempts: i64,
    pub locked_until: Option<i64>,
    pub last_error: Option<String>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
}

/// SQLite-backed job queue. Cheap to clone (`Pool` is internally `Arc`-shared).
#[derive(Clone)]
pub struct JobQueue {
    pool: SqlitePool,
}

impl JobQueue {
    /// Open or create the queue DB at `~/.tempera/jobs.sqlite`.
    pub async fn open_default() -> Result<Self> {
        let path = Config::data_dir()?.join("jobs.sqlite");
        Self::open(&path).await
    }

    /// Open or create the queue DB at the given path.
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts)
            .await
            .with_context(|| format!("Failed to open jobs DB at {}", path.display()))?;
        Self::run_migrations(&pool).await?;
        Ok(Self { pool })
    }

    /// In-memory queue for tests.
    pub async fn open_in_memory() -> Result<Self> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        Self::run_migrations(&pool).await?;
        Ok(Self { pool })
    }

    async fn run_migrations(pool: &SqlitePool) -> Result<()> {
        MIGRATOR
            .run(pool)
            .await
            .context("Failed to run SQLite migrations")?;
        Ok(())
    }

    /// Submit a new job. Returns the assigned id.
    pub async fn submit(&self, kind: &str, payload: serde_json::Value) -> Result<i64> {
        self.submit_with_max_attempts(kind, payload, DEFAULT_MAX_ATTEMPTS)
            .await
    }

    pub async fn submit_with_max_attempts(
        &self,
        kind: &str,
        payload: serde_json::Value,
        max_attempts: i64,
    ) -> Result<i64> {
        let now = Utc::now().timestamp();
        let row = sqlx::query(
            "INSERT INTO jobs (kind, payload, status, max_attempts, created_at)
             VALUES (?1, ?2, 'pending', ?3, ?4)
             RETURNING id",
        )
        .bind(kind)
        .bind(payload.to_string())
        .bind(max_attempts)
        .bind(now)
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert job")?;
        Ok(row.get(0))
    }

    /// Lease the oldest pending job. Atomic UPDATE under a transaction.
    /// Returns `Ok(None)` when no leasable job exists.
    pub async fn lease(&self) -> Result<Option<Job>> {
        let now = Utc::now().timestamp();
        let lease_until = now + LEASE_SECONDS;

        let mut tx = self.pool.begin().await?;

        let row = sqlx::query(
            "SELECT id FROM jobs
             WHERE status = 'pending'
               AND (locked_until IS NULL OR locked_until < ?1)
             ORDER BY created_at ASC
             LIMIT 1",
        )
        .bind(now)
        .fetch_optional(&mut *tx)
        .await?;

        let Some(row) = row else {
            tx.commit().await?;
            return Ok(None);
        };
        let id: i64 = row.get(0);

        sqlx::query(
            "UPDATE jobs
             SET status = 'running',
                 locked_until = ?1,
                 attempts = attempts + 1,
                 started_at = COALESCE(started_at, ?2)
             WHERE id = ?3",
        )
        .bind(lease_until)
        .bind(now)
        .bind(id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        let job = self
            .get_by_id(id)
            .await?
            .context("Job disappeared mid-lease")?;
        Ok(Some(job))
    }

    /// Mark a job complete.
    pub async fn complete(&self, id: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "UPDATE jobs
             SET status = 'completed', completed_at = ?1, locked_until = NULL
             WHERE id = ?2",
        )
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Mark a job failed. If attempts remain, reset to pending with backoff;
    /// otherwise mark dead.
    pub async fn fail(&self, id: i64, err: &str) -> Result<()> {
        let now = Utc::now().timestamp();
        let job = self.get_by_id(id).await?.context("Job disappeared")?;

        if job.attempts >= job.max_attempts {
            sqlx::query(
                "UPDATE jobs
                 SET status = 'dead', last_error = ?1, completed_at = ?2, locked_until = NULL
                 WHERE id = ?3",
            )
            .bind(err)
            .bind(now)
            .bind(id)
            .execute(&self.pool)
            .await?;
        } else {
            // Exponential backoff capped at ~3h (2^10 * 10s).
            let exponent = job.attempts.min(10) as u32;
            let backoff = 2_i64.pow(exponent).saturating_mul(10);
            let next_at = now.saturating_add(backoff);
            sqlx::query(
                "UPDATE jobs
                 SET status = 'pending', last_error = ?1, locked_until = ?2
                 WHERE id = ?3",
            )
            .bind(err)
            .bind(next_at)
            .bind(id)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn get_by_id(&self, id: i64) -> Result<Option<Job>> {
        let row = sqlx::query(
            "SELECT id, kind, payload, status, attempts, max_attempts, locked_until,
                    last_error, created_at, started_at, completed_at
             FROM jobs WHERE id = ?1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::row_to_job).transpose()
    }

    pub async fn list(&self, status: Option<JobStatus>, limit: i64) -> Result<Vec<Job>> {
        let rows = if let Some(s) = status {
            sqlx::query(
                "SELECT id, kind, payload, status, attempts, max_attempts, locked_until,
                        last_error, created_at, started_at, completed_at
                 FROM jobs
                 WHERE status = ?1
                 ORDER BY created_at DESC
                 LIMIT ?2",
            )
            .bind(s.as_str())
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, kind, payload, status, attempts, max_attempts, locked_until,
                        last_error, created_at, started_at, completed_at
                 FROM jobs
                 ORDER BY created_at DESC
                 LIMIT ?1",
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };
        rows.iter().map(Self::row_to_job).collect()
    }

    fn row_to_job(row: &sqlx::sqlite::SqliteRow) -> Result<Job> {
        let payload_str: String = row.get("payload");
        let payload: serde_json::Value = serde_json::from_str(&payload_str)
            .with_context(|| format!("invalid JSON payload in job {}", row.get::<i64, _>("id")))?;
        let status: String = row.get("status");
        Ok(Job {
            id: row.get("id"),
            kind: row.get("kind"),
            payload,
            status: status.parse()?,
            attempts: row.get("attempts"),
            max_attempts: row.get("max_attempts"),
            locked_until: row.get("locked_until"),
            last_error: row.get("last_error"),
            created_at: row.get("created_at"),
            started_at: row.get("started_at"),
            completed_at: row.get("completed_at"),
        })
    }
}

/// Context handed to job handlers.
pub struct JobContext<'a> {
    pub config: &'a Config,
}

/// Route a leased job to the right handler. New job kinds get a new arm here.
pub async fn dispatch_job(
    kind: &str,
    payload: &serde_json::Value,
    ctx: &JobContext<'_>,
) -> Result<()> {
    match kind {
        "index" => handle_index(payload, ctx).await,
        "propagate" => handle_propagate(payload, ctx).await,
        other => anyhow::bail!("unknown job kind: {other}"),
    }
}

async fn handle_index(payload: &serde_json::Value, _ctx: &JobContext<'_>) -> Result<()> {
    let reindex = payload
        .get("reindex")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut indexer = crate::indexer::EpisodeIndexer::new().await?;
    indexer.index_all(reindex).await?;
    Ok(())
}

async fn handle_propagate(_payload: &serde_json::Value, ctx: &JobContext<'_>) -> Result<()> {
    crate::utility::run_propagation(ctx.config).await?;
    Ok(())
}

/// Run the daemon's poll loop until Ctrl+C.
pub async fn run_daemon(queue: &JobQueue, config: &Config) -> Result<()> {
    println!(
        "tempera daemon started (poll every {}s, lease {}s, handler timeout {}s). Ctrl+C to stop.",
        POLL_INTERVAL_SECONDS, LEASE_SECONDS, HANDLER_TIMEOUT_SECONDS
    );

    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    let ctx = JobContext { config };

    loop {
        match queue.lease().await {
            Ok(Some(job)) => {
                let id = job.id;
                let kind = job.kind.clone();
                println!(
                    "[job {id}] running {kind} (attempt {}/{})",
                    job.attempts, job.max_attempts
                );

                let timeout = Duration::from_secs(HANDLER_TIMEOUT_SECONDS);
                let result =
                    tokio::time::timeout(timeout, dispatch_job(&kind, &job.payload, &ctx)).await;

                match result {
                    Ok(Ok(())) => {
                        if let Err(e) = queue.complete(id).await {
                            eprintln!("[job {id}] complete() failed: {e:#}");
                        } else {
                            println!("[job {id}] completed");
                        }
                    }
                    Ok(Err(e)) => {
                        let err = format!("{e:#}");
                        eprintln!("[job {id}] failed: {err}");
                        let _ = queue.fail(id, &err).await;
                    }
                    Err(_) => {
                        let err = format!("timeout after {HANDLER_TIMEOUT_SECONDS}s");
                        eprintln!("[job {id}] {err}");
                        let _ = queue.fail(id, &err).await;
                    }
                }
                // Loop again immediately to drain any backlog before sleeping.
                continue;
            }
            Ok(None) => {}
            Err(e) => eprintln!("lease error: {e:#}"),
        }

        tokio::select! {
            _ = &mut shutdown => {
                println!("\nShutdown signal received, exiting.");
                return Ok(());
            }
            _ = tokio::time::sleep(Duration::from_secs(POLL_INTERVAL_SECONDS)) => {}
        }
    }
}

/// Format a job for human display.
pub fn format_job_summary(job: &Job) -> String {
    use chrono::TimeZone;
    let created = Utc
        .timestamp_opt(job.created_at, 0)
        .single()
        .map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_default();
    let status_str = match job.status {
        JobStatus::Pending => "pending",
        JobStatus::Running => "running",
        JobStatus::Completed => "✓ done",
        JobStatus::Dead => "✗ dead",
    };
    let err_part = job
        .last_error
        .as_deref()
        .map(|e| format!(" — {}", truncate(e, 60)))
        .unwrap_or_default();
    format!(
        "[{id:>4}] {status:<8} {kind:<12} {attempts}/{max:<2} {created}{err}",
        id = job.id,
        status = status_str,
        kind = job.kind,
        attempts = job.attempts,
        max = job.max_attempts,
        created = created,
        err = err_part,
    )
}

pub fn fmt_path_for_data(name: &str) -> PathBuf {
    Config::data_dir()
        .map(|d| d.join(name))
        .unwrap_or_else(|_| PathBuf::from(name))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn new_queue() -> JobQueue {
        JobQueue::open_in_memory().await.unwrap()
    }

    #[tokio::test]
    async fn submit_and_get() {
        let q = new_queue().await;
        let id = q.submit("index", serde_json::json!({})).await.unwrap();
        let job = q.get_by_id(id).await.unwrap().expect("job exists");
        assert_eq!(job.kind, "index");
        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.attempts, 0);
    }

    #[tokio::test]
    async fn lease_moves_to_running_and_bumps_attempts() {
        let q = new_queue().await;
        let id = q.submit("index", serde_json::json!({})).await.unwrap();
        let leased = q.lease().await.unwrap().expect("got job");
        assert_eq!(leased.id, id);
        assert_eq!(leased.status, JobStatus::Running);
        assert_eq!(leased.attempts, 1);
        assert!(leased.locked_until.is_some());
    }

    #[tokio::test]
    async fn lease_returns_none_when_empty() {
        let q = new_queue().await;
        assert!(q.lease().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn lease_skips_running_jobs() {
        let q = new_queue().await;
        q.submit("index", serde_json::json!({})).await.unwrap();
        let _first = q.lease().await.unwrap().expect("first lease");
        // Second lease should find nothing because the only job is leased.
        assert!(q.lease().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn lease_orders_by_created_at() {
        let q = new_queue().await;
        let id1 = q.submit("index", serde_json::json!({})).await.unwrap();
        // Tiny delay so created_at differs
        tokio::time::sleep(Duration::from_millis(1100)).await;
        let _id2 = q.submit("propagate", serde_json::json!({})).await.unwrap();
        let leased = q.lease().await.unwrap().expect("got job");
        assert_eq!(leased.id, id1, "oldest job should lease first");
    }

    #[tokio::test]
    async fn complete_marks_completed() {
        let q = new_queue().await;
        let id = q.submit("index", serde_json::json!({})).await.unwrap();
        let _ = q.lease().await.unwrap();
        q.complete(id).await.unwrap();
        let job = q.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Completed);
        assert!(job.completed_at.is_some());
        assert!(job.locked_until.is_none());
    }

    #[tokio::test]
    async fn fail_with_retries_returns_to_pending_with_backoff() {
        let q = new_queue().await;
        let id = q
            .submit_with_max_attempts("index", serde_json::json!({}), 3)
            .await
            .unwrap();
        let _ = q.lease().await.unwrap();
        q.fail(id, "boom").await.unwrap();
        let job = q.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.last_error.as_deref(), Some("boom"));
        assert!(job.locked_until.is_some(), "backoff lease should be set");
        let now = Utc::now().timestamp();
        assert!(
            job.locked_until.unwrap() > now,
            "lease is in the future for backoff"
        );
    }

    #[tokio::test]
    async fn fail_after_max_attempts_marks_dead() {
        let q = new_queue().await;
        let id = q
            .submit_with_max_attempts("index", serde_json::json!({}), 1)
            .await
            .unwrap();
        let _ = q.lease().await.unwrap();
        q.fail(id, "permanent").await.unwrap();
        let job = q.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Dead);
        assert_eq!(job.last_error.as_deref(), Some("permanent"));
    }

    #[tokio::test]
    async fn list_filters_by_status() {
        let q = new_queue().await;
        q.submit("index", serde_json::json!({})).await.unwrap();
        let id2 = q.submit("propagate", serde_json::json!({})).await.unwrap();
        let _ = q.lease().await.unwrap();
        q.complete(id2 - 1).await.unwrap();
        // Now: id1 completed, id2 pending
        let pending = q.list(Some(JobStatus::Pending), 10).await.unwrap();
        assert_eq!(pending.len(), 1);
        let completed = q.list(Some(JobStatus::Completed), 10).await.unwrap();
        assert_eq!(completed.len(), 1);
        let all = q.list(None, 10).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn dispatch_unknown_kind_errors() {
        let cfg = Config::default();
        let ctx = JobContext { config: &cfg };
        let err = dispatch_job("does-not-exist", &serde_json::json!({}), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown job kind"));
    }

    #[tokio::test]
    async fn payload_roundtrips_as_json() {
        let q = new_queue().await;
        let payload = serde_json::json!({"reindex": true, "since": "2026-05-01"});
        let id = q.submit("index", payload.clone()).await.unwrap();
        let job = q.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(job.payload, payload);
    }

    #[tokio::test]
    async fn job_status_parse_roundtrip() {
        for s in &[
            JobStatus::Pending,
            JobStatus::Running,
            JobStatus::Completed,
            JobStatus::Dead,
        ] {
            let parsed: JobStatus = s.as_str().parse().unwrap();
            assert_eq!(*s, parsed);
        }
        assert!("bogus".parse::<JobStatus>().is_err());
    }
}
