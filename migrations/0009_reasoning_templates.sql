-- v0.8.3: reasoning templates extracted from successful clusters.
--
-- The templates dream phase groups verified-successful episodes by
-- (task_type, domain) and asks Sonnet to extract the reasoning steps
-- the agent followed. The result is a recallable template that future
-- Claude can pull at task start via the `tempera_template` MCP tool —
-- "for `bugfix` in `async-rust`, the template is: 1) check the
-- spawn_blocking sites, 2) check Drop ordering, 3) ...".
--
-- (task_type, domain) is the natural key — re-runs of the phase
-- refresh the row in place rather than creating duplicates.

CREATE TABLE IF NOT EXISTS reasoning_templates (
    id                TEXT    NOT NULL PRIMARY KEY,
    task_type         TEXT    NOT NULL,
    domain            TEXT    NOT NULL,
    name              TEXT    NOT NULL,
    steps             TEXT    NOT NULL,    -- JSON array of strings
    evidence_episodes TEXT    NOT NULL,    -- JSON array of episode ids
    success_rate      REAL    NOT NULL,
    times_used        INTEGER NOT NULL DEFAULT 0,
    model             TEXT    NOT NULL,
    created_at        INTEGER NOT NULL,
    last_used         INTEGER,
    UNIQUE (task_type, domain)
);

CREATE INDEX IF NOT EXISTS templates_by_pair
    ON reasoning_templates(task_type, domain);
