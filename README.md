# Tempera - Persistent Memory for Claude Code

Tempera gives Claude Code a persistent memory that learns from experience. Instead of starting fresh each session, Claude can recall past solutions, learn what works, and get smarter over time.

## Why Tempera?

**The Problem**: Claude Code forgets everything between sessions. You solve the same problems repeatedly, and Claude can't learn from past successes or failures.

**The Solution**: Tempera captures coding sessions as "episodes", indexes them for semantic search, and uses reinforcement learning to surface the most valuable memories when relevant.

```
Without Tempera:                    With Tempera:
┌─────────────┐                  ┌─────────────┐
│  Session 1  │ ──forgotten──>   │  Session 1  │ ──captured──┐
└─────────────┘                  └─────────────┘             │
┌─────────────┐                  ┌─────────────┐             ▼
│  Session 2  │ ──forgotten──>   │  Session 2  │ ◄──recalls──┤
└─────────────┘                  └─────────────┘             │
┌─────────────┐                  ┌─────────────┐             │
│  Session 3  │ ──forgotten──>   │  Session 3  │ ◄──recalls──┘
└─────────────┘                  └─────────────┘
     │                                 │
     ▼                                 ▼
  No learning                    Continuous improvement
```

## How It Works

### The Learning Loop

```
┌────────────────────────────────────────────────────────────────┐
│  1. START TASK                                                 │
│     User: "Fix the login redirect bug"                         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  2. RETRIEVE MEMORIES                                          │
│     Claude searches: "login redirect bug"                      │
│     Finds: "Fixed similar issue by sanitizing return URLs"     │
│     + Session context: related episodes from the same task     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  3. SOLVE FASTER                                               │
│     Claude uses past experience to solve the problem           │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  4. CAPTURE SESSION                                            │
│     Claude saves: what was done, what worked, what failed      │
│     Auto-links to current session for multi-step tasks         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  5. LEARN FROM FEEDBACK                                        │
│     User: "That memory was helpful!"                           │
│     → Episode utility increases                                │
│     → Multi-hop Bellman propagation spreads value               │
│     → Session-linked episodes get boosted                      │
│     → Unhelpful memories fade over time                        │
└────────────────────────────────────────────────────────────────┘
```

### What Makes It "Learn"

| Mechanism | What It Does |
|-----------|--------------|
| **Feedback** | Helpful episodes gain utility score |
| **Multi-hop Bellman Propagation** | Value spreads through the similarity graph across multiple hops |
| **Session Chaining** | Related episodes in multi-step tasks are linked and boost each other |
| **Temporal Credit** | Episodes before successes get credit (even across session boundaries) |
| **Recency Boost** | Fresh episodes can be weighted higher in retrieval (opt-in) |
| **Scope-aware Decay** | Project-bound claims fade in ~70 days; language-level facts last ~3 years; universal truths never decay |
| **Verification State** | Captures advance from `Untested` → `TestsPass` → `Merged` → `StableNoRevert`; later states weigh more |
| **Calibration** | Per-(task, project) verified vs. declared ratio surfaces overconfidence |
| **Dream Cycle** | Nightly reflection, pattern detection, contradiction probing, and template extraction |
| **Self-Improvement Log** | Tracks corrections, missed questions, and queues clarifying questions for next session |
| **Cross-Project Transfer** | Claims marked language / crate / domain / forever-scoped surface across projects |

Over time, frequently helpful knowledge rises to the top, while stale or unhelpful memories fade away — and the system itself accumulates a per-project picture of where it tends to be wrong.

### The bigger surfaces (v0.6 onward)

Beyond the basic capture/retrieve loop, Tempera ships several higher-order surfaces. Each is opt-in but all flow through the same MCP tools — Claude can use them without any custom client code.

- **Grounded capture** (v0.6): Every captured claim carries a `falsifiability` score, a `category`, and a `ValidityScope` (Forever / Language / Crate / Domain / Workaround / Project). Decay rates are per-scope — universal truths never expire, project-specific conventions fade in months, workarounds expire when the underlying issue closes.
- **Dream cycle** (v0.7): A budgeted nightly pipeline that runs `verify_advance → decay → reflect → patterns → contradict → templates`. Reflections turn high-signal days into prose; patterns surface themes that keep recurring; contradict probes pairs of frequently-retrieved episodes for factual disagreements; templates extract reusable step sequences from successful task clusters.
- **Self-improvement** (v0.8): Calibration tracks the ratio of declared vs. verified successes per (task, project). Mistakes log records corrections the agent made. Should-have-asked log records questions it realized it should have asked first. Ask-backs are clarifying questions the *system itself* drafts via Haiku when a capture ends in failure with vague intent — queued for the next session in that project.
- **Brief surface** (v0.9): One MCP call joins all of the above against the file set the agent is about to touch. `tempera_brief(files, task_type?, domain?)` returns pending ask-backs, the matching reasoning template, top correction categories for those files, should-have-asked triggers, and a calibration warning if the agent's track record on this kind of task is shaky.
- **Cross-project learning** (v0.10): `tempera_retrieve` and `tempera_brief` both accept `cross_project=true`. Transferable claims (anything not project-scoped) surface across projects; Project-scoped knowledge stays bound to its codebase. Legacy captures default to non-transferable until reclassified.

## Installation

### Build from Source

```bash
# Clone and build
git clone https://github.com/anvanster/tempera.git
cd tempera
cargo build --release

# Two binaries are created:
# - target/release/tempera      (CLI tool)
# - target/release/tempera-mcp  (MCP server for Claude Code)
```

### Install from crates.io

```bash
cargo install tempera
```

### First Run - Model Download

On first use, Tempera downloads the BGE-Small embedding model (~128MB) for semantic search. This happens automatically and only once:

```bash
# Initialize and trigger model download
tempera init

# Output:
# 🔄 Loading embedding model (this may download the model on first run)...
# ✅ Embedding model loaded
```

The model is cached globally at `~/.tempera/models/` and shared across all projects.

## Setup with Claude Code

### 1. Add the MCP Server

```bash
claude mcp add tempera --scope user -- /path/to/Tempera/target/release/tempera-mcp
```

The `--scope user` flag makes it available across all your projects.

### 2. Restart Claude Code

Exit and restart Claude Code to load the new MCP server.

### 3. Verify

Run `/mcp` in Claude Code. You should see `tempera` with 12 tools.

## MCP Tools

Once connected, Claude has access to these 12 tools, grouped by purpose:

### Session warmup (call at task start)

| Tool | When to Use |
|------|-------------|
| `tempera_session_start` | Call ONCE at the very start. Returns any clarifying question tempera drafted after a previous failed/partial session in this project. |
| `tempera_brief` | Call once the file set is known. Joins pending ask-back, reasoning template, top correction categories for these files, should-have-asked triggers, and calibration warning into one response. Pass `task_type` + `domain` for richer output. Set `cross_project=true` to supplement with rows from other projects. |
| `tempera_retrieve` | Search for similar past episodes. Set `scope="cross-project"` to include transferable claims from other projects. |
| `tempera_template` | Pull the reasoning template stored for a `(task_type, domain)` pair. The step sequence past wins followed. |

### During task

| Tool | When to Use |
|------|-------------|
| `tempera_log_correction` | When the user corrects an assumption / decision / piece of code. Categorized log; the brief surface uses it. |
| `tempera_log_should_have_asked` | When you realize mid-task you should have asked a question up front. Records the trigger context, the question, and the eventual answer. |

### End of task

| Tool | When to Use |
|------|-------------|
| `tempera_capture` | Save session as an episode. Auto-detects session links and runs propagation. The intent-extraction LLM call also suggests a `ValidityScope` for cross-project routing. |
| `tempera_feedback` | Mark retrieved episodes as helpful or not. Drives the utility-learning loop. |

### Diagnostics + maintenance

| Tool | When to Use |
|------|-------------|
| `tempera_status` | Per-project memory health snapshot. |
| `tempera_stats` | Statistics + trend analytics (helpfulness over time, domain growth, learning curve). |
| `tempera_propagate` | Multi-hop Bellman propagation with convergence tracking. Periodic maintenance. |
| `tempera_review` | Consolidate similar BKMs, cleanup. Run after related task series. |

### Standard warmup pattern

```
tempera_session_start          # is anything queued from last session?
tempera_brief(files, ...)      # what does tempera know about this exact change?
tempera_retrieve(query)        # then deep-dive into specific episodes
```

### Key Lifecycle Behaviors

**Start of session**: Call `tempera_session_start` first to consume any pending ask-back, then `tempera_brief` once the file set is known.

**End of task**: Capture successful sessions with `tempera_capture` — don't wait to be asked. Capture also drafts an ask-back via Haiku if the outcome was Failure/Partial with vague intent, so the next session in this project sees the gap.

**Session chaining**: Episodes captured within 2 hours of each other in the same project are automatically linked. Multi-step tasks share utility and context.

**Cross-project routing**: New captures get a `ValidityScope` suggestion from the intent-extraction LLM (v0.10.3). Project-scoped claims stay bound to one codebase; Language / Crate / Domain / Workaround / Forever claims surface across projects when retrieval opts in.

### tempera_retrieve Modes

```
Search:           tempera_retrieve(query: "authentication bug")
List all:         tempera_retrieve(all: true)
Show details:     tempera_retrieve(query: "abc12345")        # episode ID
Cross-project:    tempera_retrieve(query: "sqlx ordering", scope: "cross-project")
```

## Example Conversation

```
User: Fix the login redirect vulnerability

Claude: [Calls tempera_session_start()]
        No pending ask-back — proceeding.

        [Calls tempera_brief(files: ["src/auth.rs", "src/router.rs"],
                             task_type: "bugfix", domain: "auth-middleware")]

        🔴 PENDING ASK-BACK
          Ask the user first: "Should auth run before or after request logging?"

        📋 REASONING TEMPLATE (bugfix / auth-middleware)
          url-redirect validation
            1. Find every redirect target read from request input
            2. Validate against allow-list or treat as untrusted
            3. Add regression test with adversarial input

        ⚠️  TOP CORRECTION CATEGORIES for these files
          - validation_skipped (3×)
          - host_header_trust (2×)

        Tempera says I've been wrong here before in specific ways.
        Let me ask: should auth run before or after logging?

User: Before — we need an auth'd subject for log lines.

Claude: [Implements with URL allow-list, following the template steps]

        Done. [Calls tempera_capture(
          summary: "Fixed login redirect by validating return URL against allow-list",
          outcome: "success",
          validity_scope: "domain:auth-middleware"
        )]
        ✅ Episode saved. (Cross-project — this is an auth-middleware pattern.)

User: That memory was helpful!

Claude: [Calls tempera_feedback(helpful: true, episodes: ["..."])]
        Marked helpful — future retrievals for similar tasks will rank this higher.
```

## CLI Commands

The CLI mirrors the MCP tool surface so you can drive everything Claude does from a shell.

### Basics

```bash
# Initialize Tempera
tempera init

# Capture an episode (from a session transcript or interactively)
tempera capture --session /path/to/transcript.md

# Index episodes for semantic search (or re-index)
tempera index [--reindex]

# Search memories — project-scoped by default
tempera retrieve "database connection issues"
tempera retrieve "sqlx pattern" --cross-project       # v0.10.1 — pull from other projects

# Provide feedback
tempera feedback helpful --episodes abc123,def456
```

### The brief surface (v0.9)

```bash
# Joint summary of every self-improvement signal for these files
tempera brief --files src/auth.rs,src/router.rs \
              --task-type bugfix --domain auth-middleware

# Include rows from other projects (foreign rows are tagged [from <project>])
tempera brief --files src/store.rs --cross-project
```

### Session warmup (v0.8.5)

```bash
# Show + clear the pending ask-back for this project (if any)
tempera session-start

# History of system-drafted clarifying questions
tempera ask-backs [--pending] [--project P]
```

### Self-improvement surfaces (v0.8)

```bash
# Log a correction the user made
tempera log-correction --category "lifetime annotations" \
                       --description "I assumed &str when &'a str was needed" \
                       --correction "use named lifetime to match trait"

# View the correction log
tempera mistakes [--top 5]              # top categories
tempera mistakes --project tempera      # raw list filtered

# Log a question you should have asked up front
tempera log-should-have-asked --trigger "edit auth middleware" \
                              --question "Which auth provider is wired up?" \
                              --answer "No auth — internal-only service."

# View the should-have-asked log
tempera asks --top 5
```

### Reasoning templates (v0.8.3)

```bash
# List stored templates
tempera templates list

# Fetch a specific template
tempera templates get --task-type bugfix --domain async-rust

# Manually trigger extraction (otherwise runs in dream cycle)
tempera templates extract --max-usd 0.20
```

### Calibration (v0.8.1)

```bash
# Per-(task_type, project) verified vs declared rates
tempera calibration --project tempera --task-type bugfix
```

### Dream cycle (v0.7)

```bash
# Run the full cycle with a budget cap (default $0.50)
tempera dream --max-usd 0.50

# Run one phase, or list available phases
tempera dream --phase reflect
tempera dream --list

# Plan only — show what would happen without making LLM calls
tempera dream --dry-run

# Author yesterday's reflection (Haiku triage + Sonnet authorship if score >= 0.5)
tempera reflect [--date 2026-05-26] [--dry-run]

# Surface active factual contradictions found during dream
tempera contradict --list
```

### Verification (v0.6.1)

```bash
# Move an episode forward in the verification chain
tempera advance-verification --episode abc123 --to tests_pass --run-id <id>
tempera advance-verification --episode abc123 --to merged --commit <sha>
tempera advance-verification --episode abc123 --to stable_no_revert --days 30
```

### Maintenance + analytics

```bash
# Multi-hop Bellman propagation (run weekly)
tempera propagate --temporal

# Prune old / low-value episodes
tempera prune --older-than 90 --min-utility 0.2 --execute

# Stats + trends
tempera stats
tempera trends --project tempera --bucket weekly

# Health check + remediation
tempera doctor [--remediate --yes --target-score 90]

# Eval harness (P@5, R@5, MRR, nDCG@5 against a fixture)
tempera eval run --fixture evals/fixtures/real.jsonl --mode hybrid

# Snapshot / restore the data dir
tempera backup
tempera backup --list
tempera backup --restore 20260524T123456Z
```

## Data Storage

Tempera stores everything locally in `~/.tempera/` (shared across all projects). One memory pool serves every project; the project filter is applied at query time.

```
~/.tempera/
├── config.toml              # Configuration (all RL params configurable)
├── episodes/                # Canonical episode JSON
│   └── 2026-01-25/
│       └── <id>.json
├── jobs.sqlite              # SQLite for everything indexable (see below)
├── vectors/                 # Vector index (vectrust embeddings)
├── models/                  # BGE-Small embedding model (~128MB)
├── reflections/             # Daily reflection markdown (v0.7.3)
├── patterns/                # Cross-day pattern pages (v0.7.4)
└── templates/               # Reasoning templates (v0.8.3)
```

### SQLite tables (in `jobs.sqlite`)

Everything that needs SQL lives here. Each store opens the DB on first use and runs its migration; migrations are in `migrations/` and run in order.

| Migration | Table | Purpose |
|-----------|-------|---------|
| 0001 | `jobs` | Background job queue with lease semantics |
| 0002 | `error_fingerprints` | blake3-hashed normalized error text |
| 0003 | `dream_verdicts` | Day-level Haiku triage cache |
| 0004 | `reflections` | Daily reflection records |
| 0005 | `patterns` | Cross-day theme clusters |
| 0006 | `contradictions` | Episode-pair disagreements + Wilson CI |
| 0007 | `calibration_buckets` | (task_type, project) declared vs verified counts |
| 0008 | `mistakes` | Anchored correction log |
| 0009 | `reasoning_templates` | Extracted reasoning step sequences |
| 0010 | `should_have_asked` | Questions the agent should have asked up front |
| 0011 | `ask_backs` | System-drafted clarifying questions for next session |

All projects share the same pool. Cross-project routing is controlled by each episode's `ValidityScope` (see below) — not by separate storage.

## Configuration

All knobs live in `~/.tempera/config.toml`. The defaults are tuned to be useful out of the box; you only need to touch this if you want to change retrieval ranking, dream-cycle behavior, or per-phase budgets.

### Retrieval + ranking

```toml
[retrieval]
mode = "hybrid"                  # vector | keyword | hybrid (BM25 + vector fusion)
similarity_weight = 0.3          # Weight for semantic similarity (project mode)
utility_weight = 0.7             # Weight for learned utility (project mode)
hybrid_similarity_weight = 0.85  # RRF-normalized retrieval (hybrid mode)
hybrid_utility_weight = 0.15
recency_weight = 0.0             # Recency (0 = off, opt-in)
recency_halflife_days = 30.0
mmr_lambda = 0.7                 # MMR diversity (0=diverse, 1=relevant)
min_similarity = 0.5             # Filter threshold

[bellman]
gamma = 0.9                      # Discount factor for Bellman updates
alpha = 0.1                      # Learning rate
propagation_threshold = 0.5      # Min similarity for propagation
max_propagation_depth = 2        # Multi-hop depth (hops)
temporal_credit_window_hours = 1
```

### Capture + verification

```toml
[capture]
auto_capture = true
extract_intent_llm = true        # Use LLM to extract intent + claim + scope
capture_diffs = true
ask_back_on_failure = true       # Draft a clarifying question on Failure/Partial captures (v0.8.5)
```

### Dream cycle (v0.7)

```toml
[dream]
default_max_usd = 0.50           # Per-cycle budget cap
stable_threshold_days = 30       # Days before Merged → StableNoRevert
triage_model = "claude-haiku-4-5-20251001"
reflect_model = "claude-sonnet-4-6"

# Patterns phase
patterns_lookback_days = 30
patterns_min_evidence = 3
patterns_cluster_threshold = 0.75

# Contradict phase
contradict_top_n = 50
contradict_min_similarity = 0.6
contradict_max_similarity = 0.95
contradict_max_pairs = 30
contradict_min_confidence = 0.7

# Templates phase (v0.8.3)
templates_min_evidence = 3
templates_min_verification_weight = 0.30  # 0.30 = Untested (lenient); 0.60 = Merged
```

### Storage + maintenance

```toml
[storage]
max_age_days = 180               # Max episode age for pruning
min_utility_threshold = 0.05     # Min utility to keep
min_retrievals = 2               # Min retrievals before pruning allowed
consolidation_threshold = 0.85   # BKM merge threshold
cluster_threshold = 0.85
stale_age_days = 30
stale_utility_threshold = 0.2
```

Decay rates are **scope-aware** (per the `ValidityScope` on each episode's claim):

| Scope | Decay/day | Half-life |
|-------|-----------|-----------|
| `Forever` | 0.000 | ∞ |
| `Language { name }` | 0.001 | ~3 years |
| `Domain { tag }` | 0.005 | ~140 days |
| `Project { name }` | 0.010 | ~70 days |
| `Crate { name, version }` | 0.020 | ~35 days |
| `Workaround { ref, expires }` | 0.050 | ~14 days |
| (no scope set, legacy) | 0.010 | ~70 days |

## Under the Hood

### Multi-hop Bellman Propagation

Value from helpful episodes spreads through the similarity graph in multiple hops:

```
Hop 0: Source episodes (high helpfulness, ≥2 retrievals)
  │
  ▼  γ¹ discount
Hop 1: Similar episodes updated
  │
  ▼  γ² discount
Hop 2: Episodes similar to hop-1 updated
  │
  ▼  Converges when no updates occur
```

### Session Chaining

Episodes captured within 2 hours of each other in the same project are automatically linked:

```
Session abc123:
  ├── Episode 1: "Investigated auth bug" (debug)
  ├── Episode 2: "Found root cause in token validation" (research)
  └── Episode 3: "Fixed token expiry check" (bugfix, success)
       ↓
  Temporal credit flows back to episodes 1 & 2
  Session-linked propagation boosts all 3
```

### The Dream Cycle (v0.7)

A budgeted background pipeline that runs nightly (or on demand). Each phase shares a `CostBudget`; free phases ignore it, paid phases check `try_spend()` before each LLM call.

```
verify_advance  →  decay  →  reflect  →  patterns  →  contradict  →  templates
   (free)         (free)   (Sonnet)    (Sonnet)    (Haiku)        (Sonnet)
                          ↓             ↓            ↓             ↓
                  reflections/  patterns/   contradictions  templates/
```

- **verify_advance**: bumps episodes from `Merged` to `StableNoRevert` after `stable_threshold_days`.
- **decay**: scope-aware utility decay (see table above).
- **reflect**: Haiku triage gates Sonnet authorship; high-signal days get a reflection page.
- **patterns**: agglomerative clustering on reflection embeddings → cross-day themes.
- **contradict**: pairs frequently-retrieved BKM episodes and asks Haiku whether they disagree on a factual claim; surfaces a Wilson 95% CI on the contradiction rate.
- **templates**: groups successful verified episodes by `(task_type, domain)`, extracts reusable step sequences via Sonnet.

Worst case per full cycle: roughly $0.50 with default settings.

### Scoring Formula

Retrieval ranking combines three signals with normalized weights:

```
score = (sim_w × similarity + util_w × utility + rec_w × recency) / (sim_w + util_w + rec_w)
```

Default in hybrid mode: 85% similarity (RRF-normalized over vector + BM25), 15% utility, 0% recency. The `VerificationState` of each episode multiplies into salience — well-verified successes weigh more.

### Cross-project routing (v0.10)

Every claim carries a `ValidityScope` that determines:

- **Decay rate** (table above).
- **Transferability**: `is_transferable()` returns true for everything except `Project { name }`. The retrieve and brief surfaces use this to decide what surfaces when the agent opts into `cross_project=true`.

Legacy episodes captured before v0.6.4 don't have a scope set, so they stay project-bound by default. New captures (v0.10.3+) get a scope suggested automatically by the intent-extraction LLM call — using a colon-encoded format like `language:rust`, `crate:sqlx@0.8`, `domain:async-rust`, `workaround:repo#123`, or `project`. The default when in doubt is `project`, keeping the system conservative.

## Maintenance

Run periodically to keep memory healthy:

```bash
# Nightly: dream cycle (verify_advance + decay + reflect + patterns + contradict + templates)
tempera dream --max-usd 0.50

# Weekly: Propagate utility values (multi-hop with convergence)
tempera propagate --temporal

# Monthly: Clean up old/useless episodes
tempera prune --older-than 90 --min-utility 0.2 --execute

# As needed: Check trends
tempera trends

# As needed: Review and consolidate
# (via MCP) tempera_review(action: "consolidate")

# As needed: health check + auto-remediate
tempera doctor --remediate --yes
```

The dream cycle is the load-bearing piece for long-running memory hygiene. It uses Haiku for cheap gating and Sonnet for authorship — the default $0.50 cap is the worst case across every phase.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | For LLM-based intent extraction (`--extract-intent`) |
| `TEMPERA_DATA_DIR` | Override default data directory |
| `FASTEMBED_CACHE_DIR` | Override embedding model cache location |

## Troubleshooting

### MCP server not loading
1. Check path: `ls /path/to/tempera-mcp`
2. Check config: `cat ~/.claude.json`
3. Restart Claude Code completely
4. Run `/mcp` to verify

### Embeddings slow on first run
The BGE-Small model (~128MB) downloads on first use from HuggingFace. This requires internet access. After download, the model is cached at `~/.tempera/models/` and works offline.

### Vector search not finding anything
Run `tempera index` to create/update the vector database.

### Model download fails
If behind a firewall or proxy, ensure access to `huggingface.co`. The model files are downloaded via HTTPS.

### `tempera_brief` returns "nothing to surface"
This is normal early on — the brief joins against signal data (mistakes, asks, templates, calibration) that accrues over time. Specifically:
- The mistakes / should-have-asked sections only fire when the **files** you pass overlap with previously-logged rows.
- The template section only fires when at least 3 successful verified episodes share the `(task_type, domain)` pair (templates accrue during the dream cycle).
- The calibration warning needs ≥5 declared-success captures in the bucket before it surfaces.

Fall back to `tempera_retrieve` for episode-level recall.

### `tempera retrieve --cross-project` finds nothing
Episodes captured before v0.6.4 don't have a `ValidityScope` set, and v0.10's cross-project filter treats unscoped claims as project-bound (conservative default). Either (a) capture new episodes with v0.10.3+, which auto-suggests a scope, or (b) manually classify legacy episodes via the MCP `validity_scope` parameter on capture.

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or PR.
