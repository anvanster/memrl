-- v0.7.3: daily reflection pages (Sonnet-authored).
--
-- One row per (date, project) — the source of truth is the .md sidecar
-- at ~/.tempera/reflections/<id>.md; this table is a queryable mirror
-- so the dream cycle can check "did we already reflect on this day?"
-- without scanning the filesystem.

CREATE TABLE IF NOT EXISTS reflections (
    id          TEXT    NOT NULL PRIMARY KEY,
    date        TEXT    NOT NULL,
    project     TEXT,
    body        TEXT    NOT NULL,
    citations   TEXT    NOT NULL,   -- JSON array of episode IDs
    signals     TEXT    NOT NULL,   -- JSON array; triage signals at author time
    triage_score REAL   NOT NULL,
    model       TEXT    NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS reflections_by_date
    ON reflections(date DESC);

CREATE INDEX IF NOT EXISTS reflections_by_project_date
    ON reflections(project, date DESC);
