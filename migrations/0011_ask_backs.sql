-- v0.8.5: ask-backs queued at capture time.
--
-- Most subtle of the v0.8 surfaces: the system itself notices, at
-- capture time, when an episode's intent was vague AND its outcome was
-- failure/partial, and asks Haiku to draft one clarifying question.
-- That question gets queued as "pending" for the project, and the next
-- session in that project sees it via `tempera_session_start`.
--
-- Unlike mistakes (v0.8.2) and should-have-asked (v0.8.4) which are
-- agent-logged, ask-backs are *system-generated* — the agent doesn't
-- need to remember to record anything; the capture path does it for
-- the agent.
--
-- The partial unique index on `(project)` WHERE status='pending' is
-- load-bearing: it enforces at-most-one-pending-per-project at the DB
-- level, so the capture hook can call `record` unconditionally and
-- rely on a unique-violation as the natural debounce signal.

CREATE TABLE IF NOT EXISTS ask_backs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL,
    episode_id  TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | served | dismissed
    model       TEXT    NOT NULL,
    created_at  INTEGER NOT NULL,
    served_at   INTEGER
);

CREATE INDEX IF NOT EXISTS ask_backs_by_proj_status
    ON ask_backs(project, status, created_at DESC);

-- At most one pending ask-back per project. Capture relies on this for
-- debounce: the second qualifying failure in a project won't generate
-- a duplicate prompt until the agent acknowledges the first.
CREATE UNIQUE INDEX IF NOT EXISTS ask_backs_one_pending_per_proj
    ON ask_backs(project) WHERE status = 'pending';
