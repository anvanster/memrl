-- v0.8.1: per-bucket calibration tracking.
--
-- One row per (task_type, project) tuple. Counters bump on capture
-- (declared_success / declared_failure) and on verification advance
-- (verified_success when reaching StableNoRevert or above).
-- refuted_success is reserved for v0.8.x revert detection (no signal
-- source for it today; column stays 0).
--
-- The overconfidence rate per bucket is computed in code as
--   max(0, 1 - verified_success / max(declared_success, 1))
-- — used by future retrieval-time damping and by `tempera calibration`
-- for surfaced reporting.

CREATE TABLE IF NOT EXISTS calibration_buckets (
    task_type        TEXT NOT NULL,
    project          TEXT NOT NULL,
    declared_success INTEGER NOT NULL DEFAULT 0,
    verified_success INTEGER NOT NULL DEFAULT 0,
    declared_failure INTEGER NOT NULL DEFAULT 0,
    refuted_success  INTEGER NOT NULL DEFAULT 0,
    last_updated     INTEGER NOT NULL,
    PRIMARY KEY (task_type, project)
);

CREATE INDEX IF NOT EXISTS calibration_by_project
    ON calibration_buckets(project, last_updated DESC);
