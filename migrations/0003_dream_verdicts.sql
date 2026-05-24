-- v0.7.2: cache of daily triage verdicts.
--
-- Keyed by (date, captures_hash) so an unchanged set of episodes for a
-- given day never re-runs the LLM. captures_hash is a blake3 of the
-- sorted (episode_id, timestamp_end) tuples — any mutation invalidates
-- the cache automatically.
--
-- One verdict per day per content snapshot; if episodes get edited the
-- next triage will produce a new row (old verdicts stay for audit).

CREATE TABLE IF NOT EXISTS dream_verdicts (
    date          TEXT    NOT NULL,
    captures_hash TEXT    NOT NULL,
    verdict       TEXT    NOT NULL,
    created_at    INTEGER NOT NULL,
    PRIMARY KEY (date, captures_hash)
);

CREATE INDEX IF NOT EXISTS verdicts_by_date
    ON dream_verdicts(date, created_at DESC);
