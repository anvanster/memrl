-- v0.8.2: anchored mistakes — categorized history of corrections
-- the user made to the agent's assumptions, decisions, or code.
--
-- The MCP tool `tempera_log_correction` writes here whenever the
-- agent recognizes that a user feedback turn was a correction (not
-- praise, not a continuation). Future `tempera_brief` (v0.9) surfaces
-- the top-N categories for files in the current directory so the
-- agent knows where it tends to be wrong before it acts.

CREATE TABLE IF NOT EXISTS mistakes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    episode_id  TEXT,
    files       TEXT    NOT NULL DEFAULT '[]',  -- JSON array
    description TEXT    NOT NULL,
    correction  TEXT    NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS mistakes_by_proj_cat
    ON mistakes(project, category, created_at DESC);

CREATE INDEX IF NOT EXISTS mistakes_by_episode
    ON mistakes(episode_id)
    WHERE episode_id IS NOT NULL;
