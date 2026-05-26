-- v0.8.4: should-have-asked log.
--
-- Mirrors the v0.8.2 mistakes index, but for the *opposite* failure
-- mode. Where mistakes records "you asserted X, the user corrected you
-- to Y," should_have_asked records "the user told you Y when you
-- finally asked — but the observable context Z should have made you
-- ask up front rather than guess."
--
-- The brief surface (v0.9) will look up rows by `trigger` for each
-- file/topic the agent is about to touch, so the agent learns *which
-- questions to ask first* in recurring contexts.

CREATE TABLE IF NOT EXISTS should_have_asked (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL,
    trigger     TEXT    NOT NULL,    -- normalized kebab/snake_case context signal
    question    TEXT    NOT NULL,    -- what to ask next time
    answer      TEXT    NOT NULL,    -- what the user said (fallback knowledge)
    episode_id  TEXT,                -- session where the realization happened
    files       TEXT    NOT NULL DEFAULT '[]',  -- JSON array
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS shas_by_proj_trigger
    ON should_have_asked(project, trigger, created_at DESC);
CREATE INDEX IF NOT EXISTS shas_by_episode
    ON should_have_asked(episode_id)
    WHERE episode_id IS NOT NULL;
