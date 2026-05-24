-- v0.7.5: pairwise contradiction findings.
--
-- The probe selects pairs of frequently-retrieved episodes whose
-- embeddings are similar (cosine 0.6-0.95: related but not duplicate),
-- asks Haiku to judge whether they contradict on a factual claim, and
-- stores the verdicts here. Surfaced via `tempera contradict` and the
-- doctor's quality dimension (v0.7.6+).
--
-- (episode_a, episode_b) is the natural pair key — we record each
-- pair once and reuse the same row when the probe re-runs.

CREATE TABLE IF NOT EXISTS contradictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_a       TEXT    NOT NULL,
    episode_b       TEXT    NOT NULL,
    severity        TEXT    NOT NULL CHECK(severity IN ('low', 'medium', 'high')),
    confidence      REAL    NOT NULL,
    explanation     TEXT    NOT NULL,
    resolution_hint TEXT,
    similarity      REAL    NOT NULL,
    found_at        INTEGER NOT NULL,
    resolved_at     INTEGER,
    resolved_action TEXT,
    UNIQUE (episode_a, episode_b)
);

CREATE INDEX IF NOT EXISTS contradictions_by_severity
    ON contradictions(severity, found_at DESC)
    WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS contradictions_by_episode_a
    ON contradictions(episode_a)
    WHERE resolved_at IS NULL;
