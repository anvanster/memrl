-- v0.6.5: stack-trace fingerprint index.
--
-- One row per (fingerprint hash, episode_id) pair. occurrence_count tracks
-- how many times this fingerprint fired within the same episode (e.g. two
-- separate errors in one capture that normalize to the same fingerprint).
-- Different episodes with the same fingerprint = different rows.
--
-- Surfaces "you've seen this exact error N times before" in the
-- tempera_capture MCP response — like git blame but for bugs.

CREATE TABLE IF NOT EXISTS error_fingerprints (
    hash             TEXT    NOT NULL,
    episode_id       TEXT    NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen       INTEGER NOT NULL,
    last_seen        INTEGER NOT NULL,
    PRIMARY KEY (hash, episode_id)
);

CREATE INDEX IF NOT EXISTS fp_by_hash
    ON error_fingerprints(hash, last_seen DESC);
