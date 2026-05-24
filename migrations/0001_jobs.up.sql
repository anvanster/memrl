-- Background job queue. SQLite-backed, single-writer assumption (one daemon).
--
-- Status transitions:
--   pending → running   (worker leases)
--   running → completed (handler succeeds)
--   running → pending   (handler fails; attempts < max_attempts; backoff applies)
--   running → dead      (handler fails; attempts >= max_attempts)
--
-- Lease semantics: `locked_until` is a unix timestamp. A row is leasable when
-- status='pending' AND (locked_until IS NULL OR locked_until < now).

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'dead')),
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    locked_until INTEGER,
    last_error   TEXT,
    created_at   INTEGER NOT NULL,
    started_at   INTEGER,
    completed_at INTEGER
);

CREATE INDEX IF NOT EXISTS jobs_pending
    ON jobs(status, locked_until)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS jobs_by_kind ON jobs(kind, status, created_at);
