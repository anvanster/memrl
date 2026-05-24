-- v0.7.4: cross-day patterns surfaced by clustering reflections.
--
-- The reflect phase writes one page per day. The patterns phase reads
-- the last N days of reflections, embeds them, clusters by cosine
-- similarity, and (when a cluster has >= min_evidence reflections)
-- asks Sonnet to name the shared theme. The result is one row + one
-- .md sidecar per pattern.
--
-- theme_slug is the natural key (one row per theme); occurrence_count
-- bumps each time a new reflection joins the cluster.

CREATE TABLE IF NOT EXISTS patterns (
    id                      TEXT    NOT NULL PRIMARY KEY,
    theme_slug              TEXT    NOT NULL UNIQUE,
    statement               TEXT    NOT NULL,
    evidence_reflection_ids TEXT    NOT NULL,   -- JSON array
    first_seen              TEXT    NOT NULL,   -- ISO date of earliest evidence
    last_reinforced         TEXT    NOT NULL,   -- ISO date of newest evidence
    occurrence_count        INTEGER NOT NULL,
    model                   TEXT    NOT NULL,
    created_at              INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS patterns_by_last_reinforced
    ON patterns(last_reinforced DESC);
