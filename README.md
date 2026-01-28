# MemRL - Memory-Augmented Reinforcement Learning for Claude Code

MemRL gives Claude Code a persistent memory that learns from experience. Instead of starting fresh each session, Claude can recall past solutions, learn what works, and get smarter over time.

## Why MemRL?

**The Problem**: Claude Code forgets everything between sessions. You solve the same problems repeatedly, and Claude can't learn from past successes or failures.

**The Solution**: MemRL captures coding sessions as "episodes", indexes them for semantic search, and uses reinforcement learning to surface the most valuable memories when relevant.

```
Without MemRL:                    With MemRL:
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
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  5. LEARN FROM FEEDBACK                                        │
│     User: "That memory was helpful!"                           │
│     → Episode utility increases                                │
│     → Similar episodes get boosted (Bellman propagation)       │
│     → Unhelpful memories fade over time                        │
└────────────────────────────────────────────────────────────────┘
```

### What Makes It "Learn"

| Mechanism | What It Does |
|-----------|--------------|
| **Feedback** | Helpful episodes gain utility score |
| **Bellman Propagation** | Value spreads to semantically similar episodes |
| **Temporal Credit** | Episodes before successes get credit |
| **Decay** | Unused memories fade (1% per day) |
| **Retrieval Ranking** | High-utility episodes surface first |

Over time, frequently helpful knowledge rises to the top, while stale or unhelpful memories fade away.

## Installation

### Prerequisites

LanceDB requires the Protocol Buffers compiler (`protoc`) to build:

**macOS:**
```bash
brew install protobuf
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install protobuf-compiler
```

**Windows:**
```powershell
# Using Chocolatey
choco install protoc

# Or download from: https://github.com/protocolbuffers/protobuf/releases
# Add to PATH after extraction
```

### Build from Source

```bash
# Clone and build
git clone https://github.com/anvanster/memrl.git
cd memrl
cargo build --release

# Two binaries are created:
# - target/release/memrl      (CLI tool)
# - target/release/memrl-mcp  (MCP server for Claude Code)
```

### Install from crates.io

```bash
cargo install memrl
```

### First Run - Model Download

On first use, MemRL downloads the BGE-Small embedding model (~128MB) for semantic search. This happens automatically and only once:

```bash
# Initialize and trigger model download
memrl init

# Output:
# 🔄 Loading embedding model (this may download the model on first run)...
# ✅ Embedding model loaded
```

The model is cached globally at `~/.memrl/models/` and shared across all projects.

## Setup with Claude Code

### 1. Add the MCP Server

```bash
claude mcp add memrl --scope user -- /path/to/MemRL/target/release/memrl-mcp
```

The `--scope user` flag makes it available across all your projects.

### 2. Restart Claude Code

Exit and restart Claude Code to load the new MCP server.

### 3. Verify

Run `/mcp` in Claude Code. You should see `memrl` with 7 tools.

## MCP Tools

Once connected, Claude has access to these tools:

| Tool | Description | When to Use |
|------|-------------|-------------|
| `memrl_retrieve` | Search memories by query, list all, or show details | **Start of session** - always check first |
| `memrl_capture` | Save session as episode (auto-propagates utility) | **End of task** - capture successes proactively |
| `memrl_feedback` | Mark episodes as helpful/not helpful | After using retrieved memories |
| `memrl_status` | Check memory health for current project | Understand memory state |
| `memrl_stats` | View overall memory statistics | Analytics and monitoring |
| `memrl_propagate` | Spread value to similar episodes | Periodic maintenance |
| `memrl_review` | Consolidate and cleanup memories | After related task series |

### Key Lifecycle Behaviors

**Start of session**: Claude should automatically check for relevant memories using `memrl_retrieve` before starting non-trivial tasks.

**End of task**: Claude should proactively capture successful sessions using `memrl_capture` - don't wait to be asked. The capture tool automatically runs utility propagation.

**Cross-project insights**: Use `memrl_capture(project: "other-project")` to tag memories for a different project.

### memrl_retrieve Modes

```
Search:       memrl_retrieve(query: "authentication bug")
List all:     memrl_retrieve(all: true)
Show details: memrl_retrieve(query: "abc12345")  # episode ID
```

## Example Conversation

```
User: Fix the login redirect vulnerability

Claude: Let me check if we've solved similar problems before...
        [Calls memrl_retrieve("login redirect vulnerability")]

        Found 2 relevant past experiences:
        1. "Fixed open redirect by validating return URLs against whitelist"
           - 78% similarity, 85% utility
        2. "Auth redirect issue - added URL sanitization"
           - 65% similarity, 72% utility

        Based on past experience, I'll implement URL validation...
        [Works on the fix]

        Done! Let me save this session.
        [Calls memrl_capture(summary: "Fixed login redirect...", outcome: "success")]

User: That retrieved memory was really helpful!

Claude: [Calls memrl_feedback(helpful: true, episodes: ["abc123"])]
        Marked as helpful - this will improve future retrievals!
```

## CLI Commands

```bash
# Initialize MemRL
memrl init

# Capture an episode manually
memrl capture --prompt "Fixed the authentication bug"

# Index episodes for semantic search
memrl index

# Search memories
memrl retrieve "database connection issues"

# Provide feedback
memrl feedback helpful --episodes abc123,def456

# Run utility propagation
memrl propagate --temporal

# Prune old/low-value episodes
memrl prune --older-than 90 --min-utility 0.2 --execute

# View statistics
memrl stats
```

## Data Storage

MemRL stores everything locally in `~/.memrl/` (shared across all projects):

```
~/.memrl/
├── config.toml              # Configuration
├── episodes/                # Episode JSON files
│   └── 2026-01-25/
│       └── session-abc123.json
├── vectors/                 # Vector database
│   └── episodes.lance/      # LanceDB embeddings
└── models/                  # Embedding model cache (~128MB)
    └── models--Xenova--bge-small-en-v1.5/
```

All projects share the same memory database, enabling cross-project learning.

## The RL Behind the Scenes

MemRL uses reinforcement learning concepts:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `decay_rate` | 0.01 | 1% utility decay per day |
| `discount_factor` | 0.9 | RL gamma for Bellman updates |
| `learning_rate` | 0.1 | Conservative alpha for updates |
| `propagation_threshold` | 0.5 | Min similarity for propagation |

**Episode Lifecycle**:
```
Captured → Indexed → Retrieved → Feedback → Utility Updated → Propagated
                                                    ↓
                                            [Low utility + old]
                                                    ↓
                                                 Pruned
```

## Maintenance

Run periodically to keep memory healthy:

```bash
# Weekly: Propagate utility values
memrl propagate --temporal

# Monthly: Clean up old/useless episodes
memrl prune --older-than 90 --min-utility 0.2 --execute

# As needed: Check health
memrl stats
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | For LLM-based intent extraction (`--extract-intent`) |
| `MEMRL_DATA_DIR` | Override default data directory |

## Troubleshooting

### MCP server not loading
1. Check path: `ls /path/to/memrl-mcp`
2. Check config: `cat ~/.claude.json`
3. Restart Claude Code completely
4. Run `/mcp` to verify

### Embeddings slow on first run
The BGE-Small model (~128MB) downloads on first use from HuggingFace. This requires internet access. After download, the model is cached at `~/.memrl/models/` and works offline.

### Vector search not finding anything
Run `memrl index` to create/update the vector database.

### Model download fails
If behind a firewall or proxy, ensure access to `huggingface.co`. The model files are downloaded via HTTPS.

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or PR.
