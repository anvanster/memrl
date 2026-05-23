# Tempera Roadmap: v0.5 → v1.0

Forward-looking design reference. Each release is a coherent layer that earns
its successor. Schemas, prompts, math, and file-level integration points are
spelled out here so an implementer can pick up any release and build it.

This document is descriptive, not prescriptive — when reality disagrees with
the plan, reality wins and this doc gets updated.

---

## Table of contents

1. [Philosophy](#philosophy)
2. [Release layering](#release-layering)
3. [v0.5 — Foundation](#v05--foundation)
4. [v0.6 — Grounded capture](#v06--grounded-capture)
5. [v0.7 — Dreaming](#v07--dreaming)
6. [v0.8 — Self-improvement](#v08--self-improvement)
7. [v0.9 — Pre-emption](#v09--pre-emption)
8. [v1.0 — World-change awareness](#v10--world-change-awareness)
9. [Cross-cutting](#cross-cutting)
10. [Appendix A — Consolidated schema](#appendix-a--consolidated-schema)
11. [Appendix B — MCP tool catalog](#appendix-b--mcp-tool-catalog)
12. [Appendix C — Configuration reference](#appendix-c--configuration-reference)

---

## Philosophy

Tempera today is a log-with-RL. After this roadmap it is an epistemic partner:
it grades its own claims, does generative work while idle, surfaces relevant
context before being asked, and re-evaluates beliefs when the world changes.

Five thematic areas drive the work. They don't ship as separate releases;
they layer. Foundation (V) opens the door for capture richness (IV), which
feeds dreaming (I), which produces material for self-improvement (III),
which enables pre-emption (II).

Each release answers one question:

| Release | Question |
|---------|----------|
| 0.5 | Can we measure ourselves? |
| 0.6 | What's missing from a capture? |
| 0.7 | What can memory do while we sleep? |
| 0.8 | How does memory learn about itself and about us? |
| 0.9 | Can memory speak before being spoken to? |
| 1.0 | Does memory know that the world changes? |

## Release layering

```
                  PRE-EMPTION (II)              v0.9
                          ▲
                          │
                  SELF-IMPROVEMENT (III)        v0.8
                          ▲
                          │
                       DREAMING (I)             v0.7
                          ▲
                          │
                    NEGATIVE SPACE (IV)         v0.6
                          ▲
                          │
                  OPERATIONAL SHAPE (V)         v0.5
                  WORLD-CHANGE (cross-cutting)  v1.0
```

Effort estimates are engineer-weeks for a single full-time engineer fluent
in the codebase.

---

## v0.5 — Foundation

**Theme:** Stop flying blind. Build the runway.
**Effort:** ≈8 weeks.
**Ships:** eval harness, hybrid retrieval, background job queue, schema
migrations, doctor v1.

### 5.1 Eval harness

**Fixture format** — JSONL, one query per line, stored in `evals/fixtures/`:

```jsonl
{"id": "q001", "query": "tokio spawn_blocking deadlock", "relevant": [{"id": "abc12345", "grade": 3}, {"id": "def67890", "grade": 2}], "tags": ["concurrency", "rust"]}
```

`grade` is 0-3 (irrelevant / partial / relevant / highly relevant), used for nDCG.

**Metrics**:
- `P@K = |relevant ∩ retrieved_topK| / K`
- `R@K = |relevant ∩ retrieved_topK| / |relevant|`
- `MRR = mean(1 / rank_of_first_relevant)`
- `nDCG@K = DCG@K / IDCG@K` where `DCG@K = Σ_i (2^grade_i - 1) / log2(i + 2)`

**Commands**:

```bash
tempera eval baseline --fixture evals/fixtures/general.jsonl
# Writes evals/baselines/<ISO8601>-<commit>.json

tempera eval run --fixture evals/fixtures/general.jsonl
# Loads most recent baseline, prints delta table

tempera eval diff --against HEAD~1 --fixture <path>
# Runs current + git-stashes-and-runs-prev, prints side-by-side
```

**Module:** `src/eval.rs`. Reuses existing `EpisodeIndexer` and `EpisodeStore`.

**CI:** GitHub Action `eval-bench.yml` runs on PR, posts a markdown table to
the PR comment. Regression > 2 points on P@5 fails the check (override via
PR label `eval-regression-ok`).

### 5.2 Hybrid retrieval (BM25 + vector + RRF)

**Architecture:**

```
query
  │
  ├──► vector_search (vectrust)        ──┐
  ├──► bm25_search   (tantivy)         ──┤
  │                                       │
  └──► reciprocal_rank_fusion(k=60) ◄────┘
                │
                ▼
        top_K candidates
```

**RRF math:**

```rust
fn rrf(rankings: &[Vec<EpisodeId>], k_rrf: f32) -> Vec<(EpisodeId, f32)> {
    let mut scores: HashMap<EpisodeId, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) +=
                1.0 / (k_rrf + rank as f32 + 1.0);
        }
    }
    let mut v: Vec<_> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v
}
```

`k_rrf = 60` is the value most of the IR literature lands on. Knob, not a constant.

**Tantivy schema** (new module `src/bm25.rs`):

```rust
fn build_schema() -> Schema {
    let mut b = Schema::builder();
    b.add_text_field("id", STRING | STORED);
    b.add_text_field("project", STRING | INDEXED);
    b.add_text_field("intent_raw", TEXT | STORED);
    b.add_text_field("intent_extracted", TEXT);
    b.add_text_field("summary", TEXT);
    b.add_text_field("files", TEXT);
    b.add_text_field("errors", TEXT);
    b.add_u64_field("timestamp", INDEXED | FAST);
    b.build()
}
```

**Integration point:** `src/retrieve.rs::try_vector_search` becomes
`try_hybrid_search`. The BM25 and vector calls run via `tokio::join!`. The
RRF-fused candidate list then passes through the existing
`combined_score` weighting.

**Config knobs** (`config.toml`):

```toml
[retrieval]
mode = "hybrid"          # "vector" | "bm25" | "hybrid"
rrf_k = 60
bm25_weight = 1.0        # multiplier for BM25 rankings in RRF input
vector_weight = 1.0
```

### 5.3 Background job queue

**Schema** (`migrations/0001_jobs.sql`):

```sql
CREATE TABLE jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,
    payload     TEXT    NOT NULL,                  -- JSON
    status      TEXT    NOT NULL CHECK(status IN
                  ('pending','running','completed','failed','dead')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    locked_until INTEGER,                          -- unix ts; NULL if free
    last_error  TEXT,
    created_at  INTEGER NOT NULL,
    started_at  INTEGER,
    completed_at INTEGER
);

CREATE INDEX jobs_pending ON jobs(status, locked_until)
    WHERE status = 'pending';
```

**Lease semantics:**

1. Daemon polls: `SELECT * FROM jobs WHERE status='pending' AND
   (locked_until IS NULL OR locked_until < ?) ORDER BY created_at LIMIT 1`
2. Atomically lease: `UPDATE jobs SET status='running', attempts=attempts+1,
   locked_until=? WHERE id=? AND status='pending'`
3. Run handler with `timeout = lease_duration - 5s`. Default lease 60s.
4. On success: `UPDATE jobs SET status='completed', completed_at=?`
5. On failure with `attempts < max_attempts`: clear lease, status back to
   `pending`, exponential backoff via `locked_until = now + 2^attempts * 10s`
6. On failure with `attempts >= max_attempts`: status `dead`, surfaced by doctor

**Handler trait** (`src/jobs/mod.rs`):

```rust
#[async_trait]
pub trait JobHandler: Send + Sync {
    type Payload: DeserializeOwned + Send;
    fn kind(&self) -> &'static str;
    async fn handle(&self, payload: Self::Payload, ctx: &JobContext) -> Result<()>;
}

pub struct JobContext {
    pub store: Arc<EpisodeStore>,
    pub config: Arc<Config>,
    pub budget: Arc<CostBudget>,
    pub job_id: i64,
}
```

**Initial handlers** (v0.5 set):

| Kind | Payload | Purpose |
|------|---------|---------|
| `index` | `{ since: Option<DateTime> }` | Re-embed episodes |
| `propagate` | `{ depth: u32, temporal: bool }` | Bellman propagation |
| `review` | `{ project: Option<String> }` | Consolidation pass |

**Daemon:**

```bash
tempera daemon              # runs in foreground
tempera daemon --detach     # forks, writes pid to ~/.tempera/daemon.pid
tempera daemon --stop       # SIGTERM via pid file
```

Idempotency: every handler must be safe to re-run. Use SQLite transactions
where possible; for vector writes, dedupe by episode_id.

### 5.4 Schema migrations

**Episode versioning** — add to `src/episode.rs`:

```rust
pub const CURRENT_EPISODE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct Episode {
    #[serde(default = "default_v1")]
    pub schema_version: u32,
    // ... existing fields
}

fn default_v1() -> u32 { 1 }

impl Episode {
    pub fn migrate(self) -> Result<Self> {
        let mut ep = self;
        while ep.schema_version < CURRENT_EPISODE_VERSION {
            ep = match ep.schema_version {
                1 => migrate_v1_to_v2(ep)?,
                2 => migrate_v2_to_v3(ep)?,
                v => bail!("episode at version {v} cannot migrate forward"),
            };
        }
        Ok(ep)
    }
}
```

Migration functions live in `src/episode/migrations.rs`. Every up has a down.
Round-trip tested in `#[cfg(test)] mod migration_tests`.

**SQLite migrations** — `sqlx::migrate!("./migrations")` runs at startup.
Files named `NNNN_description.sql`. Down files: `NNNN_description.down.sql`.

**Backup discipline:**

```bash
tempera backup                          # snapshot ~/.tempera/ → ~/.tempera/backups/<ts>/
tempera backup --restore <ts>           # restore from snapshot
```

Auto-runs before any migration that bumps `schema_version`. Refuses to run
when `--backup-disabled` is not explicit.

### 5.5 Doctor v1 (read-only)

**Scoring formula** (0-100):

```
score = round(
    20 × fresh_index_score
  + 20 × embedding_coverage
  + 15 × session_link_health
  + 10 × propagation_recency
  + 10 × bkm_consolidation_score
  + 10 × eval_fixture_pass_rate
  + 15 × placeholder_for_verification   // 0 until v0.6
)
```

**Dimensions:**

| Dimension | Calculation |
|-----------|-------------|
| `fresh_index_score` | `1.0` if index updated within 24h, linear decay to `0.0` over 7d |
| `embedding_coverage` | `episodes_with_embedding / total_episodes` |
| `session_link_health` | `1.0 - broken_session_link_count / total_sessions` |
| `propagation_recency` | `1.0` if propagated within 7d, decay |
| `bkm_consolidation_score` | `1.0 - near_duplicate_pairs(threshold=0.85) / total_pairs` |
| `eval_fixture_pass_rate` | latest P@5 against `evals/fixtures/general.jsonl` |

**Output:**

```
$ tempera doctor
Health: 73 / 100

✓ index            score 20/20   updated 2h ago
✗ embeddings       score 14/20   12 episodes missing embeddings
✓ session links    score 15/15   ok
✗ propagation      score  6/10   last run 12d ago
~ consolidation    score  8/10   3 near-duplicate pairs found
✗ eval             score  5/10   P@5 = 0.42

Run `tempera doctor --json` for machine-readable output.
v0.6+ adds verification dimension (auto-remediation in v0.7).
```

### 5.6 Known follow-ups

Surfaced during implementation, deferred to keep the initial release focused.

- **Shared embedder for eval runs.** `eval::run_eval` calls
  `retrieve::try_vector_search` once per query, which constructs a new
  `EpisodeIndexer` each time and reloads the BGE-Small model. Correct but
  slow — a 50-query fixture pays the model-load cost 50×. Fix: thread a
  single `EpisodeIndexer` (or a thin trait around it) through the loop, or
  add an `EpisodeIndexer::shared()` accessor that memoizes the model. Same
  refactor would also benefit `mcp_server.rs` if multiple retrievals happen
  per session.

---

## v0.6 — Grounded capture

**Theme:** Every capture becomes information-dense. Negative space, made first-class.
**Effort:** ≈10 weeks.
**Ships:** verification states, falsifiability scoring, alternatives_considered,
validity scope + decay math, stack-trace fingerprints.

### 6.1 Verification state machine

```rust
#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum VerificationState {
    Untested,
    TestsPass { run_id: String, at: DateTime<Utc> },
    Merged { commit: String, at: DateTime<Utc> },
    StableNoRevert { days: u32, since: DateTime<Utc> },
    ValidatedCrossProject { evidence_episodes: Vec<String> },
}

impl VerificationState {
    pub fn weight(&self) -> f32 {
        match self {
            Self::Untested                              => 0.30,
            Self::TestsPass { .. }                      => 0.60,
            Self::Merged { .. }                         => 0.80,
            Self::StableNoRevert { days, .. } if *days >= 30 => 0.95,
            Self::StableNoRevert { .. }                 => 0.85,
            Self::ValidatedCrossProject { .. }          => 1.00,
        }
    }
}
```

**Episode integration** — replaces the bool-ish `OutcomeStatus`:

```rust
pub struct Outcome {
    pub status: OutcomeStatus,            // Success | Partial | Failure (kept)
    pub verification: VerificationState,  // NEW
    // ... rest unchanged
}
```

**Transition sources:**

| Trigger | Implementation |
|---------|----------------|
| Test pass | post-test hook writes to `~/.tempera/verification_inbox/`; daemon consumes |
| Git merge | post-merge hook posts `{commit_sha, files, episode_refs}` to inbox |
| Time-based | daemon cron job: `Merged` + 30d clean → `StableNoRevert` |
| Cross-project | dream cycle finds N≥2 projects citing the episode → upgrade |
| Manual | `tempera advance-verification --episode <id> --to <state>` |

**Integration with retrieval ranking** — `combined_score` gains:

```rust
score = (sim_w × sim + util_w × utility × verification_weight + rec_w × recency)
        / (sim_w + util_w + rec_w)
```

### 6.2 Falsifiability scoring

**Prompt addition** — appended to existing `--extract-intent` call in `src/llm.rs`:

```
For this episode, also assess:

falsifiability (0.0-1.0):
  Can the central claim of this episode be checked against future reality?
  1.0 = specific, testable assertion ("function X always returns Y when Z")
  0.7 = strong directional claim
  0.5 = soft pattern ("usually", "tends to")
  0.0 = pure logistics ("bumped version", "ran migration")

claim_category: one of
  api_contract | performance | structural | conventional | workaround |
  logistics | other

Output as JSON keys alongside extracted_intent.
```

**Storage:**

```rust
pub struct Claim {
    pub falsifiability: f32,
    pub category: ClaimCategory,
    pub validity_scope: Option<ValidityScope>,  // populated in 6.4
}

pub enum ClaimCategory {
    ApiContract, Performance, Structural, Conventional,
    Workaround, Logistics, Other,
}
```

Added to `Intent`:

```rust
pub struct Intent {
    // ... existing fields
    pub claim: Option<Claim>,    // None for pre-v0.6 episodes
}
```

**BKM lane gate:** in `propagate.rs`, propagation only runs over episodes
where `claim.falsifiability >= 0.7`. Below-threshold episodes are stored,
indexed, and retrievable, but they don't propagate utility through the graph
and don't get promoted by `review`.

**Cost:** zero incremental tokens — piggybacks on the existing extract call.

### 6.3 Alternatives considered

```rust
pub struct Alternative {
    pub approach: String,        // "use Arc<Mutex<HashMap>>"
    pub why_not: String,         // "lock contention on hot path"
    pub how_close: HowClose,
    pub would_revisit_if: Option<String>,  // "single-writer becomes the case"
}

pub enum HowClose {
    NearMiss,    // would have worked except for one specific reason
    Plausible,   // could work, traded off against chosen approach
    LongShot,    // considered briefly, dismissed
}
```

Added to Episode:

```rust
pub struct Episode {
    // ... existing
    pub alternatives_considered: Vec<Alternative>,  // default empty
}
```

**MCP capture contract:** the `tempera_capture` tool description teaches
the agent to populate this. Sample prompt addition:

> When `falsifiability >= 0.7`, include `alternatives_considered` — list
> approaches you nearly took with the reason you ruled them out. This is the
> single highest-value field for future-you debugging.

**Retrieval surfacing:** when an episode with alternatives is returned, the
markdown formatter inlines them under a `## Alternatives considered` header.

### 6.4 Validity scope and decay math

```rust
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValidityScope {
    Forever,
    Language  { name: String },                          // "rust"
    Crate     { name: String, version: VersionConstraint }, // "tokio @ =1.43.0"
    Domain    { tag: String },                           // "async-rust"
    Workaround { ref_: String, expires: Option<DateTime<Utc>> },
    Project   { name: String },
}

pub enum VersionConstraint {
    Exact(String),
    Range { min: String, max: Option<String> },
    Any,
}
```

**Decay rates** (per day):

| Scope | Rate | Notes |
|-------|------|-------|
| `Forever` | 0.000 | language semantics |
| `Language` | 0.001 | 0.1%/day, ~3y half-life |
| `Domain` | 0.005 | 0.5%/day, ~140d half-life |
| `Project` | 0.010 | 1%/day (current default) |
| `Crate` | 0.020 | 2%/day; **full invalidation on version mismatch** |
| `Workaround` | 0.050 | 5%/day; **full invalidation on ref close/expires** |

**Decay function:**

```rust
fn utility_at_time(ep: &Episode, now: DateTime<Utc>) -> f32 {
    let scope = ep.intent.claim.as_ref()
        .and_then(|c| c.validity_scope.as_ref())
        .unwrap_or(&ValidityScope::Project { name: ep.project.clone() });
    let days = (now - ep.timestamp_end).num_days() as f32;
    let r = decay_rate_per_day(scope);
    ep.utility.score * (1.0 - r).powf(days.max(0.0))
}
```

**Invalidation triggers** — new daily job `validity_check`:

```rust
async fn validity_check(ctx: &JobContext) -> Result<()> {
    for ep in episodes_with_crate_scope(&ctx.store)? {
        if let ValidityScope::Crate { name, version } = &ep.claim.scope {
            if !crate_version_matches_current_locks(name, version) {
                soft_retire(&ep, utility_floor=0.10).await?;
            }
        }
    }
    for ep in episodes_with_workaround_scope(&ctx.store)? {
        if workaround_expired(&ep) {
            soft_retire(&ep, utility_floor=0.05).await?;
        }
    }
    Ok(())
}
```

`soft_retire` sets utility to the floor and marks the episode with an
`invalidation_event` for transparency.

### 6.5 Stack-trace fingerprint index

**Normalization function** — `src/fingerprint.rs`:

```rust
pub fn fingerprint_error(err: &ErrorRecord) -> String {
    let frames: Vec<String> = err.message.lines()
        .filter(|l| l.trim_start().starts_with("at "))
        .map(strip_hex_addresses)
        .map(strip_user_paths)
        .map(strip_line_numbers)  // optional; keeps fingerprint loose
        .collect();
    let canonical = format!(
        "{}|{}|{}",
        err.error_type,
        frames.join("\n"),
        active_crate_versions_signature(),
    );
    format!("{:x}", blake3::hash(canonical.as_bytes()))
}
```

`strip_hex_addresses`: regex `0x[0-9a-f]+` → `0xADDR`
`strip_user_paths`: home dir prefix → `/USER/`
`strip_line_numbers`: optional, behind config flag (default off)

**Schema** (`migrations/0002_fingerprints.sql`):

```sql
CREATE TABLE error_fingerprints (
    hash        TEXT    NOT NULL,
    episode_id  TEXT    NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen  INTEGER NOT NULL,
    last_seen   INTEGER NOT NULL,
    PRIMARY KEY (hash, episode_id)
);
CREATE INDEX fp_by_hash ON error_fingerprints(hash, last_seen DESC);
```

**Retrieval mode:**

```bash
tempera retrieve --by-error <hash>
# Returns episodes that share the fingerprint, with their resolutions

tempera_retrieve(error_fingerprint="abc123")  # MCP
```

Surfaced automatically: when `tempera_capture` includes errors, the response
includes "you've seen this exact fingerprint N times before, here are the
resolutions" — like git blame but for bugs.

---

## v0.7 — Dreaming

**Theme:** Tempera does work while idle. Authorship, not consolidation.
**Effort:** ≈14 weeks.
**Ships:** phased dream cycle, two-tier LLM gating, reflections, patterns,
contradiction probe, doctor v2 auto-remediation, salience ranking.

### 7.1 Phase orchestrator

**Phase trait:**

```rust
#[async_trait]
pub trait Phase: Send + Sync {
    fn name(&self) -> &'static str;
    fn depends_on(&self) -> &'static [&'static str];
    fn estimate_cost(&self, ctx: &PhaseContext) -> CostEstimate;
    async fn run(&self, ctx: &PhaseContext) -> Result<PhaseReport>;
    fn is_protected(&self) -> bool { false }  // MCP cannot invoke protected
}
```

**Phase set:**

```
verify      → []                       (advance verification states)
decay       → [verify]                 (apply validity-aware decay)
reflect     → [verify, decay]          (Sonnet authorship over today's captures)
patterns    → [reflect]                (detect themes across reflections)
contradict  → [reflect]                (Haiku judge over BKM pairs)
embed       → [reflect, patterns]      (index new artifacts)
investigate → [reflect]                (idle hypothesis falsification — v0.9)
invalidate  → [embed]                  (world-change response — v1.0)
```

**Orchestrator:**

```rust
pub struct DreamCycle {
    phases: BTreeMap<&'static str, Box<dyn Phase>>,
    budget: Arc<CostBudget>,
}

impl DreamCycle {
    pub async fn run(&self, requested: Option<&str>) -> Result<CycleReport> {
        let order = topo_sort(&self.phases, requested)?;
        let mut report = CycleReport::new();
        for name in order {
            let phase = &self.phases[name];
            let est = phase.estimate_cost(&self.ctx());
            if !self.budget.can_afford(est.usd) {
                report.skipped(name, "budget");
                continue;
            }
            match phase.run(&self.ctx()).await {
                Ok(r) => report.add(name, r),
                Err(e) => report.failed(name, e),
            }
        }
        Ok(report)
    }
}
```

**CLI:**

```bash
tempera dream                          # full cycle, respects daily cap
tempera dream --phase reflect          # single phase
tempera dream --dry-run --json         # plan + cost estimate
tempera dream --max-usd 0.50           # override budget
tempera dream --since 2026-05-20       # restrict capture window
```

**Cost budget:**

```rust
pub struct CostBudget {
    max_usd: f32,
    spent_usd: parking_lot::Mutex<f32>,
}

impl CostBudget {
    pub fn try_spend(&self, amount: f32) -> Result<(), BudgetExceeded> {
        let mut spent = self.spent_usd.lock();
        if *spent + amount > self.max_usd {
            return Err(BudgetExceeded { requested: amount, remaining: self.max_usd - *spent });
        }
        *spent += amount;
        Ok(())
    }
}
```

Per-day cap stored in `~/.tempera/budget_state.json`. Reset at local midnight.

### 7.2 Two-tier LLM gating

**Triage with Haiku** (cheap, gates Sonnet):

```rust
async fn triage_day(date: NaiveDate, captures: &[Episode]) -> Result<TriageVerdict> {
    if let Some(cached) = load_verdict(date, hash(captures)) {
        return Ok(cached);
    }
    let prompt = build_triage_prompt(captures);
    let response = llm_call("claude-haiku-4-5", prompt, max_tokens=200).await?;
    let verdict: TriageVerdict = serde_json::from_str(&response)?;
    cache_verdict(date, hash(captures), &verdict);
    Ok(verdict)
}

pub struct TriageVerdict {
    pub score: f32,             // 0.0-1.0, "worth synthesizing"
    pub signals: Vec<String>,   // ["multiple-projects", "novel-pattern"]
    pub reasoning: String,      // 1-2 sentence rationale
}
```

**Authorship with Sonnet** (only if triage > threshold):

```rust
async fn author_reflection(captures: &[Episode], verdict: &TriageVerdict) -> Result<Reflection> {
    ctx.budget.try_spend(0.05)?;  // Sonnet author call costs ~$0.05
    let prompt = build_reflection_prompt(captures, &verdict.signals);
    let body = llm_call("claude-sonnet-4-6", prompt, max_tokens=1500).await?;
    if !passes_quality_bar(&body, captures) {
        ctx.budget.try_spend(0.05)?;  // one regen
        body = regen_reflection_prompt(captures, &verdict.signals).await?;
    }
    if !passes_quality_bar(&body, captures) {
        body = template_fallback(captures);  // hand-written, no LLM cost
    }
    Ok(Reflection::new(captures, body))
}
```

**Quality bar:**
- At least 1 verbatim quote from a capture (regex: `r"`[^`]+`"`)
- At least 1 episode ID citation
- Body length 100-1500 words
- No phrases from the slop blacklist: `["we should consider", "perhaps", "in conclusion", "moving forward"]`

**Cache** (`migrations/0007_dream_cache.sql`):

```sql
CREATE TABLE dream_verdicts (
    date          TEXT NOT NULL,
    captures_hash TEXT NOT NULL,
    verdict       TEXT NOT NULL,   -- JSON
    created_at    INTEGER NOT NULL,
    PRIMARY KEY (date, captures_hash)
);
```

### 7.3 Reflection authorship

**Prompt skeleton** (`src/dream/prompts/reflect.txt`):

```
You are reading today's capture log from a developer's coding sessions.
The captures are factual records of what happened. Your job is to write
a short reflection (under 400 words) that says something the captures
themselves don't say — a pattern, a surprise, what this day teaches
about how this codebase wants to be touched.

Rules:
- Quote captures verbatim when citing them. Do not paraphrase memorable
  phrasings. Format: > `episode-id:` "exact quote".
- Every claim must reference at least one capture by ID.
- No "we should consider...", no "perhaps...", no "in conclusion".
  Concrete or silent.
- If today was logistics-only, write exactly: "no reflection — logistics day."

Captures from <date>, project <project>:
<captures_json>

Triage signals: <signals>
```

**Output schema:**

```rust
pub struct Reflection {
    pub id: String,                 // <date>-<project>-<hash>
    pub date: NaiveDate,
    pub project: Option<String>,
    pub body: String,
    pub citations: Vec<String>,     // episode IDs referenced
    pub extracted_claims: Vec<Claim>,  // for later promotion
    pub created_at: DateTime<Utc>,
}
```

**Storage:** markdown file at `~/.tempera/reflections/<date>-<project>.md`
with TOML frontmatter:

```markdown
+++
id = "2026-05-23-tempera-a1b2c3"
date = "2026-05-23"
project = "tempera"
citations = ["abc12345", "def67890"]
+++

# Reflection: 2026-05-23 / tempera

...body...
```

Mirror in SQLite for fast querying:

```sql
CREATE TABLE reflections (
    id           TEXT PRIMARY KEY,
    date         TEXT NOT NULL,
    project      TEXT,
    body         TEXT NOT NULL,
    citations    TEXT NOT NULL,    -- JSON array
    created_at   INTEGER NOT NULL
);
```

### 7.4 Patterns phase

**Algorithm:**

```rust
async fn patterns_phase(ctx: &PhaseContext) -> Result<PhaseReport> {
    let lookback_days = ctx.config.dream.patterns.lookback_days;  // default 30
    let min_evidence  = ctx.config.dream.patterns.min_evidence;   // default 3
    let cluster_thresh = ctx.config.dream.patterns.cluster_threshold;  // default 0.75

    let reflections = load_reflections_since(now - lookback_days)?;
    let embeddings  = embed_all(&reflections).await?;
    let clusters    = agglomerative_cluster(&embeddings, cluster_thresh);

    let mut new_patterns = vec![];
    for cluster in clusters {
        if cluster.size() < min_evidence { continue; }
        if pattern_already_covered(&cluster, &existing_patterns) { continue; }

        ctx.budget.try_spend(0.04)?;
        let pattern = author_pattern(&cluster).await?;
        save_pattern(&pattern)?;
        new_patterns.push(pattern);
    }
    Ok(PhaseReport::patterns_found(new_patterns.len()))
}
```

**Storage:**

```sql
CREATE TABLE patterns (
    id              TEXT PRIMARY KEY,
    theme_slug      TEXT NOT NULL UNIQUE,
    statement       TEXT NOT NULL,
    evidence_reflection_ids TEXT NOT NULL,  -- JSON
    first_seen      TEXT NOT NULL,          -- date
    last_reinforced TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL
);
```

Pattern pages mirror to `~/.tempera/patterns/<slug>.md`.

### 7.5 Contradiction probe

**Pair selection:**

```rust
async fn select_contradiction_pairs(ctx: &PhaseContext) -> Result<Vec<(Episode, Episode)>> {
    let candidates = top_retrieved_episodes(ctx, n=50, window_days=30).await?;
    let mut pairs = vec![];
    for (a, b) in candidates.iter().combinations(2) {
        let sim = cosine_similarity(a.embedding, b.embedding);
        if sim > 0.6 && sim < 0.95 {  // related but not duplicate
            pairs.push((a.clone(), b.clone()));
        }
    }
    pairs.truncate(ctx.config.dream.contradict.max_pairs);  // default 30
    Ok(pairs)
}
```

**Judge prompt** (Haiku, `src/dream/prompts/contradict.txt`):

```
Two episodes appear to answer questions about the same topic.
Do they CONTRADICT on a factual claim about code or behavior?

Episode A (id: <id_a>, captured <date_a>):
  Claim: <a.intent.extracted_intent>
  Outcome: <a.outcome>

Episode B (id: <id_b>, captured <date_b>):
  Claim: <b.intent.extracted_intent>
  Outcome: <b.outcome>

Output JSON only:
{
  "contradicts": bool,
  "confidence": 0.0-1.0,
  "severity": "low" | "medium" | "high",
  "explanation": "one sentence",
  "resolution_hint": "supersede" | "keep_both" | "needs_review"
}
```

**Wilson 95% CI:**

```rust
fn wilson_ci(positives: u32, total: u32) -> (f32, f32) {
    let n = total as f32;
    let p = positives as f32 / n;
    let z = 1.96;
    let denom = 1.0 + z*z/n;
    let centre = (p + z*z/(2.0*n)) / denom;
    let halfw = z * ((p*(1.0-p)/n + z*z/(4.0*n*n)).sqrt()) / denom;
    (centre - halfw, centre + halfw)
}
```

**Storage:**

```sql
CREATE TABLE contradictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_a       TEXT NOT NULL,
    episode_b       TEXT NOT NULL,
    severity        TEXT NOT NULL,   -- 'low' | 'medium' | 'high'
    confidence      REAL NOT NULL,
    explanation     TEXT NOT NULL,
    resolution_hint TEXT,
    found_at        INTEGER NOT NULL,
    resolved_at     INTEGER,
    resolved_action TEXT
);
```

Surfaced via `tempera_review --show-contradictions`.

### 7.6 Doctor v2 — auto-remediation

**Plan generation:**

```rust
pub struct RemediationStep {
    pub kind: &'static str,      // job kind
    pub payload: serde_json::Value,
    pub est_seconds: u32,
    pub est_usd: f32,
}

pub fn plan_remediation(score: &HealthScore, target: u32, budget_usd: f32)
    -> Result<Vec<RemediationStep>> {
    let mut candidates = vec![];

    if score.fresh_index_score < 0.9 {
        candidates.push(step("index", json!({}), 30, 0.0));
    }
    if score.embedding_coverage < 0.95 {
        candidates.push(step("embed", json!({}), 60, 0.0));
    }
    if score.session_link_health < 0.85 {
        candidates.push(step("session_repair", json!({}), 20, 0.0));
    }
    if score.verification_progress < 0.7 {
        candidates.push(step("verify_advance", json!({}), 15, 0.0));
    }
    if score.propagation_recency < 0.9 {
        candidates.push(step("propagate", json!({}), 30, 0.0));
    }
    if score.bkm_consolidation_score < 0.8 {
        candidates.push(step("review_consolidate", json!({}), 120, 0.20));
    }

    let sorted = topo_sort_by_dep(&candidates, &JOB_DEPS);
    let mut chosen = vec![];
    let mut spent = 0.0;
    for s in sorted {
        if spent + s.est_usd > budget_usd { break; }
        spent += s.est_usd;
        chosen.push(s);
    }
    Ok(chosen)
}
```

**Execution:**

```bash
tempera doctor --remediation-plan --json          # preview
tempera doctor --remediate --yes --target-score 90 --max-usd 2
```

Walks the plan: submit step → wait for job completion → re-check score →
stop if target hit OR budget exhausted OR a step fails repeatedly.

**Protected jobs:** `review_consolidate`, `patterns`, `contradict` cannot be
submitted via MCP — only local CLI. Mirrors gbrain's `PROTECTED_JOB_NAMES`.

### 7.7 Salience ranking

**Formula:**

```rust
fn salience(ep: &Episode, now: DateTime<Utc>, cfg: &Config) -> f32 {
    let utility    = ep.utility.score;
    let verification = ep.outcome.verification.weight();
    let recency    = recency_decay(ep.timestamp_end, now, cfg.salience.halflife_days);
    let inv_freq   = 1.0 / (1.0 + ep.retrieval_history.len() as f32 / cfg.salience.freq_normalizer);
    utility * verification * recency * inv_freq
}
```

Defaults: `halflife_days = 30`, `freq_normalizer = 100`.

**Integration** — replaces utility term in `combined_score`:

```rust
fn combined_score(sim: f32, ep: &Episode, cfg: &Config) -> f32 {
    let sal = salience(ep, Utc::now(), cfg);
    let w = &cfg.retrieval;
    let total_w = w.similarity_weight + w.salience_weight;
    (w.similarity_weight * sim + w.salience_weight * sal) / total_w
}
```

---

## v0.8 — Self-improvement

**Theme:** Tempera learns about me and about itself.
**Effort:** ≈8 weeks.
**Ships:** calibration profile, anchored-mistakes index, reasoning templates,
should-have-asked log, ask-back at capture.

### 8.1 Calibration profile

**Per-bucket tracking** (task_type × project):

```sql
CREATE TABLE calibration_buckets (
    task_type   TEXT NOT NULL,
    project     TEXT NOT NULL,
    declared_success INTEGER NOT NULL DEFAULT 0,
    verified_success INTEGER NOT NULL DEFAULT 0,
    refuted_success  INTEGER NOT NULL DEFAULT 0,
    declared_failure INTEGER NOT NULL DEFAULT 0,
    last_updated     INTEGER NOT NULL,
    PRIMARY KEY (task_type, project)
);
```

**Update flow:**

- On capture with `outcome.status = Success`: `declared_success += 1`
- On verification advance to `StableNoRevert`: `verified_success += 1`
- On verification regression (revert detected, episode reopened):
  `refuted_success += 1`

**Calibration correction:**

```rust
fn overconfidence_rate(b: &CalibrationBucket) -> f32 {
    let total = b.declared_success.max(1) as f32;
    let refuted = b.refuted_success as f32;
    (refuted / total).clamp(0.0, 0.5)  // cap correction at 50%
}

fn apply_calibration(ep: &Episode, score: f32) -> f32 {
    if matches!(ep.outcome.status, OutcomeStatus::Success)
       && matches!(ep.outcome.verification, VerificationState::Untested | VerificationState::TestsPass{..}) {
        let bucket = load_bucket(&ep.intent.task_type, &ep.project);
        if bucket.declared_success >= 10 {  // need sample size
            let correction = overconfidence_rate(&bucket);
            return score * (1.0 - correction);
        }
    }
    score
}
```

**MCP surface:**

```
tempera_calibration(task_type: Option, project: Option) -> CalibrationReport
```

Returns the agent's overconfidence rates so it can self-correct phrasing
("I'm 80% sure" → actually-right 65% of the time on `bugfix` in this project).

### 8.2 Anchored mistakes

**Schema:**

```sql
CREATE TABLE mistakes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    category    TEXT NOT NULL,         -- "lifetime_annotations", "test_setup", ...
    episode_id  TEXT,
    files       TEXT,                  -- JSON array
    description TEXT NOT NULL,
    correction  TEXT,
    created_at  INTEGER NOT NULL
);
CREATE INDEX mistakes_by_proj_cat ON mistakes(project, category, created_at DESC);
```

**Detection** — new MCP tool:

```
tempera_log_correction(
    category: String,
    description: String,
    correction: String,
    files: Vec<String> = []
)
```

The agent calls this when the user corrects an assumption. The agent's
system prompt is updated to teach this behavior.

**Surfacing:** `tempera_brief` (v0.9) includes the top-3 mistake categories
for files in the current directory.

### 8.3 Reasoning templates

**Detection during dream cycle:**

```rust
async fn detect_templates(ctx: &PhaseContext) -> Vec<Template> {
    let clusters = cluster_successful_episodes(
        ctx,
        by: ["task_type", "domain"],
        min_size: 3,
        min_verification: VerificationState::Merged{..},
    );
    let mut templates = vec![];
    for cluster in clusters {
        ctx.budget.try_spend(0.03)?;
        let t = author_template(&cluster).await?;
        if t.steps.len() >= 2 {
            templates.push(t);
        }
    }
    templates
}
```

**Schema:**

```sql
CREATE TABLE reasoning_templates (
    id              TEXT PRIMARY KEY,
    task_type       TEXT NOT NULL,
    domain          TEXT NOT NULL,
    name            TEXT NOT NULL,
    steps           TEXT NOT NULL,   -- JSON array of strings
    evidence_episodes TEXT NOT NULL, -- JSON array
    success_rate    REAL NOT NULL,
    times_used      INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    last_used       INTEGER
);
```

**MCP surface:**

```
tempera_template(task_type: String, domain: String) -> Option<Template>
```

Called by agent at task start: "for `bugfix` in `async-rust`, the template
is: 1) check the spawn_blocking sites, 2) check Drop ordering, 3) ..."

### 8.4 Should-have-asked log

```sql
CREATE TABLE assumptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project         TEXT NOT NULL,
    assumption      TEXT NOT NULL,
    was_correct     INTEGER NOT NULL,  -- 0 | 1
    should_have_asked TEXT,
    category        TEXT,
    episode_id      TEXT,
    created_at      INTEGER NOT NULL
);
```

**MCP tool:**

```
tempera_log_assumption(
    assumption: String,
    was_correct: bool,
    should_have_asked: Option<String> = None,
    category: Option<String> = None
)
```

Aggregated per project: which categories of decision had high incorrect-assumption
rates? Surfaced in `tempera_brief`.

### 8.5 Ask-back at capture time

Modified `tempera_capture` response:

```rust
pub struct CaptureResponse {
    pub episode_id: String,
    pub similar_existing: Vec<SimilarEpisode>,
    pub ask: Option<String>,  // human-readable prompt to the agent
}

pub struct SimilarEpisode {
    pub id: String,
    pub similarity: f32,
    pub summary: String,
}
```

Behavior: when capture embedding's max similarity to existing episodes is
≥ `config.capture.ask_back_threshold` (default 0.85), the response includes:

```json
{
  "episode_id": "abc12345",
  "similar_existing": [
    {"id": "def67890", "similarity": 0.89, "summary": "Fixed login redirect..."}
  ],
  "ask": "This is 89% similar to def67890. If it's the same problem, call tempera_consolidate(['abc12345','def67890']). If distinct, do nothing."
}
```

The agent decides — no automatic merging.

---

## v0.9 — Pre-emption

**Theme:** Memory speaks before being spoken to.
**Effort:** ≈14 weeks.
**Ships:** spatial index, pre-edit brief, co-edit Markov, hypothesis tracking
+ idle investigation, continuity snapshots, cross-project meta-synthesis.

### 9.1 Spatial / file-locality index

**Schema:**

```sql
CREATE TABLE file_episode_index (
    project       TEXT NOT NULL,
    file_path     TEXT NOT NULL,    -- relative to project root
    episode_id    TEXT NOT NULL,
    relevance     REAL NOT NULL,    -- 0.0-1.0
    last_touched  INTEGER NOT NULL,
    PRIMARY KEY (project, file_path, episode_id)
);
CREATE INDEX fei_by_path ON file_episode_index(project, file_path, relevance DESC);
```

**Relevance computation:**

```rust
fn compute_relevance(ep: &Episode, file: &str) -> f32 {
    let mut r = 0.0;
    if ep.context.files_modified.contains(file) { r += 0.50; }
    if ep.context.files_read.contains(file)     { r += 0.20; }
    if mentioned_in_intent(file, &ep.intent)     { r += 0.10; }
    if file_in_same_module(file, &ep.context)    { r += 0.10; }
    let age_days = (Utc::now() - ep.timestamp_end).num_days() as f32;
    r * (1.0 - 0.005).powf(age_days)  // soft recency decay
}
```

**Maintenance:** triggered by capture (insert/update rows for the episode's
files) and by daemon job `rebuild_spatial_index` (full rebuild weekly).

### 9.2 Pre-edit brief

**MCP tool:**

```
tempera_brief(file: String, project: Option<String> = None) -> Brief
```

**Brief structure:**

```rust
pub struct Brief {
    pub file: String,
    pub project: String,
    pub top_episodes: Vec<EpisodeSummary>,        // top 5 by relevance
    pub failure_density: FailureDensity,
    pub active_hypotheses: Vec<HypothesisRef>,
    pub mistake_categories: Vec<(String, u32)>,   // (category, count)
    pub conventions: Vec<String>,                 // distilled from CLAUDE.md + reflections
}

pub struct FailureDensity {
    pub failure_episodes_90d: u32,
    pub partial_episodes_90d: u32,
    pub common_cause: Option<String>,             // extracted by dream cycle
}
```

**Output format** — compact markdown, ≤200 lines, designed to be cheap to read:

```markdown
# Brief: src/auth.rs (tempera)

## Recent context (5 episodes)
- `abc12345` ✅ Fixed login redirect by validating return URLs (utility 0.85)
- `def67890` ⚠️ Partial: token refresh race condition still present
- ...

## Trouble in this region (90d)
- 5 failures, 2 partials
- Common cause: misordered match arms in parse_token()

## Active hypotheses
- H42 "auth middleware ordering matters" (open, evidence: 3 supporting, 1 refuting)

## Mistakes you've made here
- lifetime_annotations: 4 corrections
- test_setup: 2 corrections

## Conventions
- Always use VerifyMiddleware before AuthMiddleware (from CLAUDE.md)
```

**Performance budget:** p50 < 100ms, p99 < 300ms. Cache invalidated on
new captures touching the file.

### 9.3 Co-edit Markov chains

**Schema:**

```sql
CREATE TABLE file_transitions (
    project    TEXT NOT NULL,
    from_file  TEXT NOT NULL,
    to_file    TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 1,
    last_seen  INTEGER NOT NULL,
    PRIMARY KEY (project, from_file, to_file)
);
CREATE INDEX ft_by_from ON file_transitions(project, from_file, count DESC);
```

**Tracking:** new MCP tool `tempera_track_file_open(file)`. The agent calls
this whenever it opens a file. Within a session window (default 30 min),
consecutive opens insert/increment the transition.

**Prefetch:**

```rust
async fn prefetch_on_open(file: &str, project: &str) {
    let next_likely = top_transitions_from(project, file, n=3);
    for nf in next_likely {
        tokio::spawn(async move {
            let _ = warm_brief_cache(&nf.to_file, &project).await;
        });
    }
}
```

Cache: `LruCache<(file, project), Brief>` with TTL 5 min.

### 9.4 Hypothesis tracking + idle investigation

**Schema:**

```sql
CREATE TABLE hypotheses (
    id            TEXT PRIMARY KEY,
    statement     TEXT NOT NULL,
    project       TEXT,
    tagged_files  TEXT,              -- JSON array
    status        TEXT NOT NULL,     -- 'open' | 'confirmed' | 'refuted' | 'inconclusive'
    confidence    REAL NOT NULL DEFAULT 0.5,
    evidence_for  TEXT NOT NULL DEFAULT '[]',
    evidence_against TEXT NOT NULL DEFAULT '[]',
    created_at    INTEGER NOT NULL,
    verify_by     INTEGER,
    last_investigated INTEGER,
    closed_at     INTEGER
);
```

**MCP tools:**

```
tempera_hypothesize(
    statement: String,
    project: Option<String>,
    tagged_files: Vec<String> = [],
    verify_by: Option<DateTime> = None
) -> HypothesisId

tempera_hypothesis_evidence(
    id: String,
    supports: bool,
    note: String,
    episode_id: Option<String>
)

tempera_hypothesis_close(
    id: String,
    resolution: "confirmed" | "refuted" | "inconclusive",
    note: String
)
```

**Investigation phase** (during dream):

```rust
async fn investigate_phase(ctx: &PhaseContext) -> Result<PhaseReport> {
    let open = load_open_hypotheses(ctx)?;
    let mut findings = vec![];
    for h in open {
        if h.last_investigated > now - 24h { continue; }
        let candidates = if let Some(files) = &h.tagged_files {
            ast_grep_in_files(files, &h.statement).await?
        } else {
            semantic_search_codebase(&h.statement, n=20).await?
        };
        let classified = classify_evidence(&candidates, &h).await?;
        update_hypothesis_evidence(&h, &classified).await?;
        if should_auto_close(&h, &classified) {
            close_hypothesis(&h, derive_resolution(&classified));
        }
        findings.push(InvestigationFinding { hypothesis_id: h.id, ... });
    }
    Ok(PhaseReport::investigation(findings))
}
```

Reported in the next wake-up note.

### 9.5 Continuity snapshots

**Schema:**

```sql
CREATE TABLE continuity_snapshots (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    project      TEXT,
    snapshot     TEXT NOT NULL,      -- JSON
    created_at   INTEGER NOT NULL,
    consumed_at  INTEGER,
    consumed_by_session TEXT
);
CREATE INDEX cs_unconsumed ON continuity_snapshots(consumed_at)
    WHERE consumed_at IS NULL;
```

**Structure:**

```rust
pub struct ContinuitySnapshot {
    pub active_hypotheses: Vec<HypothesisRef>,
    pub mental_model: Vec<Belief>,
    pub pending_verifications: Vec<EpisodeId>,
    pub open_decisions: Vec<Decision>,
    pub current_focus: Option<String>,
    pub user_open_requests: Vec<String>,
}

pub struct Belief {
    pub statement: String,
    pub confidence: f32,
    pub evidence_episodes: Vec<EpisodeId>,
}

pub struct Decision {
    pub what: String,
    pub because: String,
    pub when: DateTime<Utc>,
    pub revisit_if: Option<String>,
}
```

**MCP tools:**

```
tempera_snapshot(session_id: String, snapshot: ContinuitySnapshot) -> SnapshotId

tempera_retrieve_continuity(project: Option<String>, max_age_hours: u32 = 24)
    -> Option<ContinuitySnapshot>
```

**Capture moments:** agent calls `tempera_snapshot` when:
- User says "let's pick this up later"
- Context-compression event detected (heuristic: very long session)
- Major branch reached (PR opened, design decided)

### 9.6 Cross-project meta-synthesis

**Phase `meta_patterns`**, runs weekly:

```rust
async fn meta_patterns_phase(ctx: &PhaseContext) -> Result<PhaseReport> {
    let all_reflections = load_reflections_all_projects(since=now - 90d)?;
    let by_project = group_by_project(&all_reflections);
    if by_project.len() < 2 { return Ok(PhaseReport::skipped("need 2+ projects")); }

    let embeddings = embed_all(&all_reflections).await?;
    let clusters = agglomerative_cluster(&embeddings, threshold=0.70);

    let mut meta_patterns = vec![];
    for cluster in clusters {
        let projects: HashSet<_> = cluster.items.iter()
            .map(|r| r.project.as_deref()).collect();
        if projects.len() < 2 { continue; }
        if cluster.size() < 4 { continue; }

        ctx.budget.try_spend(0.06)?;
        let mp = author_meta_pattern(&cluster).await?;
        meta_patterns.push(mp);
    }
    Ok(PhaseReport::meta_patterns(meta_patterns.len()))
}
```

Stored at `~/.tempera/meta-patterns/<slug>.md` with cross-project evidence.

---

## v1.0 — World-change awareness

**Theme:** Tempera knows the world changes.
**Effort:** ≈8 weeks.
**Ships:** git diff subscription, mental-model invalidation, cross-session
co-authorship awareness, public benchmark.

### 10.1 Git diff subscription

**Hook installation** (`tempera init --hooks <repo_path>`):

Creates `.git/hooks/post-commit`:

```bash
#!/bin/sh
git rev-parse HEAD | tempera ingest-commit \
    --project "$(basename "$(pwd)")" \
    --stdin
```

`tempera ingest-commit` parses the diff and writes to:

```sql
CREATE TABLE world_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    kind        TEXT NOT NULL,    -- 'commit' | 'merge' | 'cargo_update' | ...
    commit_sha  TEXT,
    payload     TEXT NOT NULL,    -- JSON: { files_changed, hunks, message, ... }
    processed   INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
);
CREATE INDEX we_unprocessed ON world_events(processed, created_at);
```

`cargo_update` event detected by watching `Cargo.lock` change events.

### 10.2 Mental-model invalidation

**Phase `invalidate`:**

```rust
async fn invalidate_phase(ctx: &PhaseContext) -> Result<PhaseReport> {
    let events = unprocessed_world_events(ctx, limit=100)?;
    let mut invalidations = vec![];

    for event in events {
        let touched_files = parse_touched_files(&event.payload);
        let candidates = episodes_referencing_files(&touched_files, &event.project)?;

        for ep in candidates {
            if should_invalidate(&ep, &event) {
                let inv = apply_invalidation(&ep, &event)?;
                invalidations.push(inv);
            }
        }
        mark_processed(&event)?;
    }
    Ok(PhaseReport::invalidations(invalidations))
}

fn should_invalidate(ep: &Episode, event: &WorldEvent) -> bool {
    let verif_weight = ep.outcome.verification.weight();
    if verif_weight >= 0.85 { return false; }     // long-stable, leave alone

    let claim_cat = ep.intent.claim.as_ref().map(|c| &c.category);
    matches!(claim_cat, Some(ClaimCategory::Structural | ClaimCategory::ApiContract))
}

fn apply_invalidation(ep: &Episode, event: &WorldEvent) -> Result<Invalidation> {
    let new_utility = ep.utility.score * 0.85;     // 15% haircut
    update_utility(&ep.id, new_utility)?;
    add_invalidation_record(&ep.id, event.id, "diff_overlap")?;
    Ok(Invalidation::new(ep.id.clone(), event.id, new_utility))
}
```

**Cargo.lock specialization:** when `cargo_update` event detected, look up
episodes with `validity_scope = Crate(name, ...)`. If version changed in
the lock file, soft-retire (utility floor 0.10) per v0.6's `validity_check`.

### 10.3 Cross-session co-authorship

**Detection:**

```rust
async fn detect_collisions(my_session: &str, my_focus: &Focus) -> Vec<Collision> {
    let recent_snapshots = unconsumed_snapshots(within=1h)?;
    let mut collisions = vec![];
    for snap in recent_snapshots {
        if snap.session_id == my_session { continue; }
        if files_overlap(&snap.snapshot.current_focus, &my_focus) {
            collisions.push(Collision {
                other_session: snap.session_id,
                other_focus: snap.snapshot.current_focus.clone(),
                overlap_files: compute_overlap(&snap, my_focus),
            });
        }
    }
    collisions
}
```

**MCP surface:** `tempera_collisions()` returns active conflicts. Agent
typically calls this at session start.

### 10.4 Public benchmark

**Synthetic corpus** (`tempera-bench` repo):

- 200 coding-episode templates parameterized over 5 task types × 8 domains
  × 5 project shapes
- 50 query-answer pairs hand-labeled with relevance grades
- Reproducibility recipe: `tempera-bench run --version 1.0`

**Methodology doc** covers:
- Corpus construction
- Query labeling protocol
- Metric definitions
- A/B testing protocol
- How to add new fixtures

---

## Cross-cutting

### Cost discipline

All LLM calls route through a single `llm.rs::call_with_budget()` helper:

```rust
pub async fn call_with_budget(
    model: &str,
    prompt: &str,
    budget: &CostBudget,
    cfg: &LlmConfig,
) -> Result<String> {
    let est_cost = estimate_cost(model, prompt.len(), cfg.max_tokens);
    budget.try_spend(est_cost)?;
    let resp = call_llm(model, prompt, cfg).await?;
    record_actual_cost(model, resp.usage);
    Ok(resp.text)
}
```

Per-day, per-phase, per-call caps in `config.toml`:

```toml
[budget]
daily_max_usd        = 0.50
per_phase_max_usd    = 0.20
per_call_max_usd     = 0.10

[budget.alerts]
warn_at_pct          = 0.75
```

Hard refusal, not warning. Dry-run mode prints estimated spend without
calling.

### Self-referential safety

Tempera builds itself. Releases that change MCP behavior follow this protocol:

1. Pin the running tempera-mcp binary to a known-good version during the build
2. Run `cargo build --release`
3. Atomic binary swap with `mv target/release/tempera-mcp ~/.local/bin/tempera-mcp.new`
   then `mv tempera-mcp.new tempera-mcp` (rename is atomic on same filesystem)
4. The next MCP call uses the new binary; the running one finishes its call
5. Hold an in-process upgrade lock in SQLite during indexing operations

### Trust posture

Every MCP tool gets a scope and local-only flag:

```rust
pub struct ToolMetadata {
    pub scope: Scope,                  // Read | Write | Admin
    pub local_only: bool,              // refuse over HTTP
    pub idempotent: bool,
}

pub enum Scope { Read, Write, Admin }
```

`tempera_capture` → Write. `tempera_retrieve` → Read. `tempera dream` →
Admin + LocalOnly (subprocess-only invocation). Protected job kinds (`patterns`,
`contradict`, `review_consolidate`) cannot be submitted via MCP.

### Migrations are forever

For every up there must be a down. Round-trip tests in
`#[cfg(test)] mod migration_tests` exercise `v_n → v_{n+1} → v_n` equality.

`tempera backup` auto-runs before any schema bump.

### Telemetry

From v0.5 onward, `tempera stats --detailed` shows:

- Retrieval: P@5, R@5, MRR over time
- Cost: daily LLM spend, projection to month
- Calibration: overconfidence by bucket (v0.8+)
- Salience: distribution histogram
- Health: doctor score over time

Stored in `~/.tempera/telemetry.sqlite`. Never sent off-machine.

---

## Appendix A — Consolidated schema

```sql
-- v0.5
CREATE TABLE jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed','dead')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    locked_until INTEGER,
    last_error  TEXT,
    created_at  INTEGER NOT NULL,
    started_at  INTEGER,
    completed_at INTEGER
);

-- v0.6
CREATE TABLE error_fingerprints (
    hash        TEXT NOT NULL,
    episode_id  TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen  INTEGER NOT NULL,
    last_seen   INTEGER NOT NULL,
    PRIMARY KEY (hash, episode_id)
);

-- v0.7
CREATE TABLE dream_verdicts (
    date          TEXT NOT NULL,
    captures_hash TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    PRIMARY KEY (date, captures_hash)
);

CREATE TABLE reflections (
    id           TEXT PRIMARY KEY,
    date         TEXT NOT NULL,
    project      TEXT,
    body         TEXT NOT NULL,
    citations    TEXT NOT NULL,
    created_at   INTEGER NOT NULL
);

CREATE TABLE patterns (
    id              TEXT PRIMARY KEY,
    theme_slug      TEXT NOT NULL UNIQUE,
    statement       TEXT NOT NULL,
    evidence_reflection_ids TEXT NOT NULL,
    first_seen      TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL
);

CREATE TABLE contradictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_a       TEXT NOT NULL,
    episode_b       TEXT NOT NULL,
    severity        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    explanation     TEXT NOT NULL,
    resolution_hint TEXT,
    found_at        INTEGER NOT NULL,
    resolved_at     INTEGER,
    resolved_action TEXT
);

-- v0.8
CREATE TABLE calibration_buckets (
    task_type   TEXT NOT NULL,
    project     TEXT NOT NULL,
    declared_success INTEGER NOT NULL DEFAULT 0,
    verified_success INTEGER NOT NULL DEFAULT 0,
    refuted_success  INTEGER NOT NULL DEFAULT 0,
    declared_failure INTEGER NOT NULL DEFAULT 0,
    last_updated     INTEGER NOT NULL,
    PRIMARY KEY (task_type, project)
);

CREATE TABLE mistakes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    category    TEXT NOT NULL,
    episode_id  TEXT,
    files       TEXT,
    description TEXT NOT NULL,
    correction  TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE reasoning_templates (
    id              TEXT PRIMARY KEY,
    task_type       TEXT NOT NULL,
    domain          TEXT NOT NULL,
    name            TEXT NOT NULL,
    steps           TEXT NOT NULL,
    evidence_episodes TEXT NOT NULL,
    success_rate    REAL NOT NULL,
    times_used      INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    last_used       INTEGER
);

CREATE TABLE assumptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project         TEXT NOT NULL,
    assumption      TEXT NOT NULL,
    was_correct     INTEGER NOT NULL,
    should_have_asked TEXT,
    category        TEXT,
    episode_id      TEXT,
    created_at      INTEGER NOT NULL
);

-- v0.9
CREATE TABLE file_episode_index (
    project       TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    episode_id    TEXT NOT NULL,
    relevance     REAL NOT NULL,
    last_touched  INTEGER NOT NULL,
    PRIMARY KEY (project, file_path, episode_id)
);

CREATE TABLE file_transitions (
    project    TEXT NOT NULL,
    from_file  TEXT NOT NULL,
    to_file    TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 1,
    last_seen  INTEGER NOT NULL,
    PRIMARY KEY (project, from_file, to_file)
);

CREATE TABLE hypotheses (
    id            TEXT PRIMARY KEY,
    statement     TEXT NOT NULL,
    project       TEXT,
    tagged_files  TEXT,
    status        TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.5,
    evidence_for  TEXT NOT NULL DEFAULT '[]',
    evidence_against TEXT NOT NULL DEFAULT '[]',
    created_at    INTEGER NOT NULL,
    verify_by     INTEGER,
    last_investigated INTEGER,
    closed_at     INTEGER
);

CREATE TABLE continuity_snapshots (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    project      TEXT,
    snapshot     TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    consumed_at  INTEGER,
    consumed_by_session TEXT
);

-- v1.0
CREATE TABLE world_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    commit_sha  TEXT,
    payload     TEXT NOT NULL,
    processed   INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
);
```

## Appendix B — MCP tool catalog

Final tool surface after v1.0:

| Tool | Scope | Local-only | First in |
|------|-------|------------|----------|
| `tempera_retrieve` | Read | No | existing |
| `tempera_capture` | Write | No | existing |
| `tempera_feedback` | Write | No | existing |
| `tempera_status` | Read | No | existing |
| `tempera_stats` | Read | No | existing |
| `tempera_propagate` | Admin | Yes | existing |
| `tempera_review` | Admin | Yes | existing |
| `tempera_calibration` | Read | No | v0.8 |
| `tempera_log_correction` | Write | No | v0.8 |
| `tempera_log_assumption` | Write | No | v0.8 |
| `tempera_template` | Read | No | v0.8 |
| `tempera_brief` | Read | No | v0.9 |
| `tempera_track_file_open` | Write | No | v0.9 |
| `tempera_hypothesize` | Write | No | v0.9 |
| `tempera_hypothesis_evidence` | Write | No | v0.9 |
| `tempera_hypothesis_close` | Write | No | v0.9 |
| `tempera_snapshot` | Write | No | v0.9 |
| `tempera_retrieve_continuity` | Read | No | v0.9 |
| `tempera_collisions` | Read | No | v1.0 |

CLI-only (never MCP):
- `tempera dream` and all phases
- `tempera doctor --remediate`
- `tempera backup` / `tempera backup --restore`
- `tempera ingest-commit`
- `tempera eval *`

## Appendix C — Configuration reference

Full `config.toml` after v1.0:

```toml
[retrieval]
mode                  = "hybrid"         # vector | bm25 | hybrid (v0.5)
rrf_k                 = 60                # v0.5
bm25_weight           = 1.0               # v0.5
vector_weight         = 1.0               # v0.5
similarity_weight     = 0.3
salience_weight       = 0.7               # v0.7 (replaces utility_weight)
recency_weight        = 0.0
mmr_lambda            = 0.7
min_similarity        = 0.5

[salience]                                # v0.7
halflife_days         = 30.0
freq_normalizer       = 100.0

[bellman]
gamma                 = 0.9
alpha                 = 0.1
decay_rate            = 0.01
propagation_threshold = 0.5
max_propagation_depth = 2
temporal_credit_window_hours = 1

[storage]
max_age_days          = 180
min_utility_threshold = 0.05
min_retrievals        = 2
consolidation_threshold = 0.85
cluster_threshold     = 0.85

[capture]                                 # v0.6
extract_falsifiability = true
extract_alternatives  = true
ask_back_threshold    = 0.85              # v0.8

[validity]                                # v0.6
language_decay_per_day = 0.001
domain_decay_per_day   = 0.005
project_decay_per_day  = 0.010
crate_decay_per_day    = 0.020
workaround_decay_per_day = 0.050

[budget]
daily_max_usd         = 0.50
per_phase_max_usd     = 0.20
per_call_max_usd      = 0.10

[budget.alerts]
warn_at_pct           = 0.75

[dream]                                   # v0.7
enabled               = true
cooldown_hours        = 12
daily_cap_usd         = 0.30
exclude_projects      = []

[dream.triage]
model                 = "claude-haiku-4-5"
score_threshold       = 0.5
cache_ttl_days        = 7

[dream.reflect]
model                 = "claude-sonnet-4-6"
max_tokens            = 1500
max_regens            = 1

[dream.patterns]
lookback_days         = 30
min_evidence          = 3
cluster_threshold     = 0.75

[dream.contradict]
max_pairs             = 30
min_pair_similarity   = 0.60
max_pair_similarity   = 0.95

[dream.meta_patterns]                     # v0.9
cadence_days          = 7
min_projects          = 2
min_cluster_size      = 4

[spatial]                                 # v0.9
brief_cache_ttl_seconds = 300
prefetch_top_n        = 3
file_mention_weight   = 0.10

[continuity]                              # v0.9
max_snapshot_age_hours = 24
auto_consume_on_match  = true

[invalidate]                              # v1.0
verification_floor    = 0.85              # don't invalidate above this
utility_haircut       = 0.15
hard_invalidation_floor = 0.10

[collisions]                              # v1.0
window_minutes        = 60
```

---

## Cut order under pressure

If capacity halves: ship v0.5 → v0.6 → v0.7 in full and stop. That's a
transformed product.

If capacity is severe (one engineer, six months): ship v0.5, plus from v0.6
just `verification_states` + `alternatives_considered`, plus from v0.7 just
the `reflect` phase. That's a coherent v0.5.1 — meaningfully better than
today's tempera, with a clear path forward.
