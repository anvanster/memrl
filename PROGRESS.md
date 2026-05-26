# Tempera — Current State Snapshot

Tempera is a persistent episodic memory system for AI coding assistants. This document describes what the system *currently does* — the moving parts, the data flow, and where each piece lives. For *why* it's structured this way and what's coming next, see [docs/ROADMAP.md](docs/ROADMAP.md). For user-facing instructions, see [README.md](README.md).

Tagged versions and headline features are visible via `git log --oneline` and `git tag -l`. The version in `Cargo.toml` is the current release.

---

## At a glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Tempera                                     │
├───────────────────────────────┬─────────────────────────────────────────┤
│  WRITE PATH                   │  READ PATH                              │
│  ────────────                 │  ─────────                              │
│  capture                       │  retrieve            (vector + BM25)    │
│  log_correction                │  brief               (joins everything) │
│  log_should_have_asked         │  session_start       (pending ask-back) │
│  feedback                      │  template            (reasoning steps)  │
│  advance-verification          │  stats / trends      (analytics)        │
├───────────────────────────────┴─────────────────────────────────────────┤
│  BACKGROUND PATH (dream cycle)                                          │
│  ─────────────────────────────                                          │
│  verify_advance → decay → reflect → patterns → contradict → templates   │
├─────────────────────────────────────────────────────────────────────────┤
│  STORAGE                                                                │
│  ───────                                                                │
│  ~/.tempera/episodes/  — episode JSON (one file per capture)            │
│  ~/.tempera/jobs.sqlite — SQLite for everything indexable               │
│  ~/.tempera/vectors/   — vector index (vectrust)                        │
│  ~/.tempera/models/    — embedding model cache                          │
│  ~/.tempera/reflections/, /patterns/, /templates/ — markdown sidecars   │
└─────────────────────────────────────────────────────────────────────────┘
```

Two binaries from one crate: `tempera` (CLI) and `tempera-mcp` (MCP server for Claude Code). See [CLAUDE.md](CLAUDE.md) for the module-level breakdown.

---

## The data model

The atom is an **Episode** (`src/episode.rs`) — one captured coding session. Each episode carries:

| Field | Notes |
|-------|-------|
| `intent` | raw prompt + LLM-extracted summary + `task_type` + domain tags + optional **Claim** with falsifiability, category, and **ValidityScope** |
| `context` | files read, files modified, tools invoked, errors encountered |
| `outcome` | status (success / partial / failure), test results, commit SHA, and a **VerificationState** that advances over time |
| `utility` | retrieval count, helpful count, calculated score (Wilson lower bound) |
| `session_id` | episodes in the same project within 2 hours auto-link into a session |
| `related_episodes` | similarity-graph edges for utility propagation |
| `alternatives_considered` | what other approaches were tried and why they weren't picked |
| `schema_version` | currently 5; migrations live in `src/episode.rs::migrate()` |

### Claim and ValidityScope (v0.6.2 + v0.6.4 + v0.10.3)

A **Claim** is the central insight of an episode — what about reality is being asserted. Two key fields:

- `falsifiability` (0.0–1.0): can this be checked against future code? Pure logistics ("bumped a version") sits at 0.0; specific assertions ("function X always returns Y when Z") at 1.0.
- `validity_scope`: where in the world the claim is true. Drives both **decay rate** and **cross-project retrieval** (v0.10).

| Scope | Decay/day | Cross-project? |
|-------|-----------|----------------|
| `Forever` | 0.000 | yes |
| `Language { name }` | 0.001 | yes |
| `Domain { tag }` | 0.005 | yes |
| `Project { name }` | 0.010 | **no** (project-bound) |
| `Crate { name, version }` | 0.020 | yes |
| `Workaround { ref_, expires }` | 0.050 | yes (while open) |

The intent-extraction LLM call suggests a scope automatically (v0.10.3); the agent can override via the MCP `validity_scope` parameter on capture.

### VerificationState (v0.6.1)

How well-verified is the episode's success claim?

```
Untested → TestsPass{run_id} → Merged{commit} → StableNoRevert{days}
                                                 → ValidatedCrossProject{evidence}
```

Each state has a `weight()` that multiplies into salience at retrieval time. The **dream cycle's `verify_advance` phase** promotes `Merged` → `StableNoRevert` after `stable_threshold_days`. Captures default to `Untested`; the agent or CI moves them forward.

---

## Storage layout

| Path | Contents |
|------|----------|
| `~/.tempera/config.toml` | All knobs (decay rates, retrieval weights, dream budget) |
| `~/.tempera/episodes/<date>/<id>.json` | Canonical episode storage, one file per capture |
| `~/.tempera/jobs.sqlite` | SQLite for everything indexable (see below) |
| `~/.tempera/vectors/` | vectrust embeddings for semantic search |
| `~/.tempera/models/` | BGE-Small embedding model (~128MB, downloaded once) |
| `~/.tempera/reflections/<date>.md` | Daily reflection pages (v0.7.3) |
| `~/.tempera/patterns/<slug>.md` | Cross-day pattern pages (v0.7.4) |
| `~/.tempera/templates/<task>__<domain>.md` | Reasoning templates (v0.8.3) |

### SQLite tables (in `jobs.sqlite`)

All migrations live in `migrations/` and run automatically on each store's `open()`.

| Table | Migration | Owner | Purpose |
|-------|-----------|-------|---------|
| `jobs` | 0001 | `jobs.rs` | Background job queue (lease semantics) |
| `error_fingerprints` | 0002 | `fingerprint.rs` | blake3-hashed normalized error text |
| `dream_verdicts` | 0003 | `triage.rs` | Day-level Haiku triage cache |
| `reflections` | 0004 | `reflect.rs` | Daily reflection records |
| `patterns` | 0005 | `patterns.rs` | Cross-day theme clusters |
| `contradictions` | 0006 | `contradict.rs` | Episode-pair disagreements + Wilson CI |
| `calibration_buckets` | 0007 | `calibration.rs` | (task_type, project) declared vs verified counts |
| `mistakes` | 0008 | `mistakes.rs` | Anchored correction log |
| `reasoning_templates` | 0009 | `templates.rs` | Extracted reasoning step sequences |
| `should_have_asked` | 0010 | `asks.rs` | Questions the agent should have asked up front |
| `ask_backs` | 0011 | `ask_backs.rs` | System-drafted clarifying questions queued for next session |

`ask_backs` has a load-bearing partial unique index on `(project) WHERE status='pending'` — enforces at-most-one-pending-per-project at the DB layer (debounce by design).

---

## How the parts fit together

### 1. Capture writes everything

```
session transcript ──► capture ──► extract_intent (LLM)
                          │              │
                          │              └─► validity_scope suggestion (v0.10.3)
                          │
                          ├─► store.save(&episode)                    [episodes/]
                          ├─► EpisodeIndexer.index(&episode)          [vectors/, BM25]
                          ├─► calibration.record_capture(...)         [calibration_buckets]
                          └─► ask_back_gen.maybe_generate_and_record(…) [ask_backs]
                                  (if outcome ∈ {Failure,Partial} AND intent vague)
```

The capture path is best-effort: the ask-back generator never fails the capture if it errors.

### 2. Retrieve is the read path

```
query ──► EpisodeIndexer.search() ──► vector hits + BM25 hits ──► hybrid fuse
                                                                       │
                                                                       ▼
                                                        salience score (utility + 
                                                        recency + verification weight)
                                                                       │
                                                                       ▼
                                                              MMR diversity
                                                                       │
                                                                       ▼
                                                            scoped filter (v0.10.1)
                                                          (Project | CrossProject)
                                                                       │
                                                                       ▼
                                                                  top-K episodes
```

Cross-project mode (v0.10.1) keeps the current project's episodes AND transferable episodes from elsewhere — see ValidityScope table above.

### 3. Dream cycle runs in the background

`tempera dream` (or scheduled) walks the phase pipeline:

```
verify_advance  →  decay  →  reflect  →  patterns  →  contradict  →  templates
   (free)        (free)   (Sonnet)    (Sonnet)    (Haiku)        (Sonnet)
                          ↓             ↓            ↓             ↓
                  reflections/  patterns/   contradictions  templates/
```

Each phase is gated by a shared `CostBudget`. Free phases ignore it; paid phases call `try_spend()` before each LLM call. The full cycle's worst-case is configured by `dream.default_max_usd` (default $0.50).

### 4. Brief is the join surface (v0.9)

`tempera_brief(files, task_type?, domain?, cross_project?)` opens every store independently and assembles:

| Section | Source | Project-scoped? |
|---------|--------|-----------------|
| Pending ask-back | `ask_backs` | yes (per-project debounce) |
| Reasoning template | `reasoning_templates` | global by `(task_type, domain)` |
| Top correction categories | `mistakes` | + cross-project rows tagged `[from <project>]` |
| Should-have-asked triggers | `should_have_asked` | + cross-project rows tagged |
| Calibration warning | `calibration_buckets` | yes |

Each section is independently optional — a store-open failure produces an empty section, not a brief-wide error. Empty sections are silently omitted from the rendered output so the response stays signal-rich.

---

## MCP tool surface (12 tools)

| Tool | Path | Read/Write |
|------|------|------------|
| `tempera_retrieve` | episodic memory search (project / cross-project) | R |
| `tempera_capture` | persist a session as an episode | W |
| `tempera_feedback` | mark episodes helpful / not helpful | W |
| `tempera_status` | per-project memory health | R |
| `tempera_stats` | analytics + trends | R |
| `tempera_propagate` | multi-hop Bellman propagation (maintenance) | W |
| `tempera_review` | consolidate similar BKMs (maintenance) | W |
| `tempera_log_correction` | agent records a user correction | W |
| `tempera_log_should_have_asked` | agent records a question it should have asked | W |
| `tempera_session_start` | check for pending ask-back at session start | R |
| `tempera_template` | pull the reasoning template for `(task_type, domain)` | R |
| `tempera_brief` | one-call join of every v0.8 surface against the working set | R |

Standard session warmup pattern:
```
tempera_session_start          # is anything queued from last time?
tempera_brief(files, ...)      # what does tempera know about this exact change?
tempera_retrieve(query)        # then deep-dive into specific episodes
```

---

## Test coverage

As of v0.4.27: **470 unit tests** across both binaries (`cargo test --workspace`), no integration suite yet. Coverage emphasises:

- Pure logic: scoring formulas, salience math, decay rates, scope parsing, fingerprint normalization
- Store roundtrips for every table (SQLite in-memory)
- Phase pipelines exercised via in-memory budgets without live LLM calls

Retrieval quality is tracked by a fixed-fixture eval (`tempera eval run --fixture evals/fixtures/real.jsonl --mode hybrid`). Baseline locked at P@5=0.264, R@5=0.893, MRR=0.895, nDCG=0.877 since v0.4.19.

---

## What this snapshot omits

- Performance characteristics and benchmarks — see `cargo bench` and the `--full` CI check.
- Configuration reference — see `~/.tempera/config.toml` and `src/config.rs` for every knob.
- The forward plan — see [docs/ROADMAP.md](docs/ROADMAP.md).
- Installation and Claude Code setup — see [README.md](README.md).
