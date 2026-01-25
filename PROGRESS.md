# MemRL Implementation Progress

## Overview

MemRL is a memory-augmented reinforcement learning system for Claude Code that learns from past coding sessions to improve future assistance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MemRL System                              │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Session Capture    │  Phase 2: Semantic Indexing      │
│  ✅ COMPLETE                 │  ✅ COMPLETE                      │
│  - Episode data model        │  - LanceDB vector storage        │
│  - Session parsing           │  - BGE-Small embeddings (384d)   │
│  - Git integration           │  - Semantic similarity search    │
│  - JSON/Markdown storage     │  - Automatic fallback            │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Utility Learning   │  Phase 4: Advanced Features      │
│  ✅ COMPLETE                 │  ✅ COMPLETE                      │
│  - Feedback collection       │  - MCP server integration        │
│  - Wilson score calculation  │  - Claude Code hooks             │
│  - Utility decay over time   │  - LLM intent extraction         │
│  - Bellman propagation       │  - Anthropic API integration     │
│  - Temporal credit assign.   │                                  │
│  - Episode pruning           │                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Completed Features

### Phase 1: Session Capture ✅

**Episode Data Model** (`src/episode.rs`)
- `Episode` struct with intent, context, outcome, utility
- `TaskType` classification (bugfix, feature, refactor, test, docs, research, debug, setup)
- `OutcomeStatus` tracking (success, partial, failure)
- Wilson score interval for utility calculation
- Markdown export for human-readable storage

**Session Capture** (`src/capture.rs`)
- Extract first user prompt from session transcripts
- Classify task types using keyword patterns
- Detect files read/modified from session content
- Identify tools invoked during session
- Determine outcome based on session indicators

**Episode Storage** (`src/store.rs`)
- JSON-based episode persistence
- Date-organized directory structure (`~/.memrl/episodes/YYYY-MM-DD/`)
- List, load, and query episodes
- Project-based filtering

**Configuration** (`src/config.rs`)
- TOML configuration file support
- Configurable data directory, retrieval limits, utility thresholds
- Default configuration generation

**Statistics** (`src/stats.rs`)
- Episode count by project, task type, outcome
- Success rate calculations
- Retrieval and feedback statistics

### Phase 2: Semantic Indexing ✅

**Vector Embeddings** (`src/indexer.rs`)
- LanceDB embedded vector database (no server required)
- fastembed with BGE-Small-EN-v1.5 model (384 dimensions)
- Local embedding generation (no API calls)
- Episode-to-embedding text conversion including:
  - Raw prompt and extracted intent
  - Task type and domain tags
  - Files modified and tools used
  - Errors encountered

**Semantic Search**
- Vector similarity search using LanceDB
- Project-based filtering support
- Similarity score calculation (L2 distance → similarity)
- Utility score integration in results

**Retrieval System** (`src/retrieve.rs`)
- Automatic vector search when index exists
- Graceful fallback to text-based search
- Combined relevance scoring (similarity × utility)
- Formatted output with episode details

### Phase 3: Utility Learning ✅

**Utility Module** (`src/utility.rs`)
- Utility decay: Exponential decay based on time since last retrieval (1% per day default)
- Bellman propagation: Spreads utility from high-value episodes to similar ones
- Tag-based propagation: Fallback when vector index unavailable
- Temporal credit assignment: Credits episodes that preceded successful outcomes

**Utility Parameters**
```rust
decay_rate: 0.01          // 1% decay per day
discount_factor: 0.9      // Standard RL discount (gamma)
learning_rate: 0.1        // Conservative update rate (alpha)
propagation_threshold: 0.5 // 50% similarity minimum for propagation
```

**Episode Pruning**
- Age-based pruning (--older-than N days)
- Utility-based pruning (--min-utility threshold)
- Dry-run mode by default (use --execute to delete)
- Protects episodes with helpful feedback

**Feedback Integration**
- Mark episodes as helpful/not-helpful/mixed
- Automatic utility score recalculation
- Retrieval history tracking
- Feedback logging for analysis

### Phase 4: Advanced Features ✅

**MCP Server** (`src/mcp_server.rs`)
- Full MCP (Model Context Protocol) server implementation
- JSON-RPC 2.0 over stdio
- Five MCP tools exposed:
  - `memrl_retrieve`: Multi-mode retrieval tool
    - Semantic search: `query: "search term"`
    - List all episodes: `all: true`
    - Show episode details: `query: "episode_id"`
  - `memrl_capture`: Capture current session as an episode
  - `memrl_feedback`: Record whether retrieved episodes were helpful
  - `memrl_stats`: Get memory statistics
  - `memrl_propagate`: Run utility propagation with optional temporal credit
- Automatic vector search with fallback
- Claude Code integration via mcp-config.json

**LLM Integration** (`src/llm.rs`)
- Anthropic API client for Claude
- Intent extraction from prompts
- Full session analysis including:
  - Summary generation
  - Task type classification
  - Outcome determination
  - Tag extraction
  - Error/resolution pairing
  - Key learnings extraction
- Graceful fallback to simple extraction when API unavailable

**Claude Code Hooks** (`hooks/`)
- `post-session.sh`: Automatic session capture after Claude Code sessions
- `pre-task.sh`: Retrieve relevant episodes before starting tasks
- Easy integration with Claude Code's hook system

### CLI Commands

| Command | Status | Description |
|---------|--------|-------------|
| `memrl init` | ✅ | Initialize memrl in current project |
| `memrl capture` | ✅ | Capture a coding session as an episode |
| `memrl capture --extract-intent` | ✅ | Capture with LLM-based intent extraction |
| `memrl list` | ✅ | List all episodes with filtering |
| `memrl show <id>` | ✅ | Show detailed episode information |
| `memrl stats` | ✅ | Display statistics and metrics |
| `memrl retrieve <query>` | ✅ | Semantic search for relevant episodes |
| `memrl feedback` | ✅ | Record feedback on retrieved episodes |
| `memrl index` | ✅ | Create/update vector embeddings |
| `memrl propagate` | ✅ | Run Bellman utility propagation |
| `memrl prune` | ✅ | Remove old/low-utility episodes |

### Binaries

| Binary | Description |
|--------|-------------|
| `memrl` | Main CLI application |
| `memrl-mcp` | MCP server for Claude Code integration |

## Test Results

```
running 17 tests (memrl)
test stats::tests::test_percentage ... ok
test feedback::tests::test_parse_feedback_type ... ok
test config::tests::test_default_config ... ok
test episode::tests::test_utility_score_calculation ... ok
test capture::tests::test_classify_task_type ... ok
test capture::tests::test_determine_outcome ... ok
test stats::tests::test_truncate ... ok
test indexer::tests::test_episode_to_embedding_text ... ok
test retrieve::tests::test_calculate_text_similarity ... ok
test utility::tests::test_utility_params_default ... ok
test utility::tests::test_decay_calculation ... ok
test llm::tests::test_parse_task_type ... ok
test llm::tests::test_parse_outcome ... ok
test config::tests::test_config_serialization ... ok
test store::tests::test_save_and_load ... ok
test store::tests::test_list_all ... ok
test capture::tests::test_extract_first_prompt ... ok

test result: ok. 17 passed; 0 failed

running 12 tests (memrl-mcp)
... all pass
```

## Dependencies

```toml
# Core
clap = "4.5"           # CLI framework
tokio = "1.43"         # Async runtime
serde = "1.0"          # Serialization
anyhow = "1.0"         # Error handling

# Storage
sqlx = "0.8"           # SQLite (future use)
lancedb = "0.23"       # Vector database
lance-arrow = "1.0"    # Arrow extensions

# Embeddings
fastembed = "4"        # Local embeddings
arrow-array = "56"     # Arrow data structures
arrow-schema = "56"    # Arrow schema

# HTTP/API
reqwest = "0.12"       # HTTP client for Anthropic API

# Utilities
chrono = "0.4"         # Date/time
uuid = "1.11"          # Unique IDs
regex = "1.11"         # Pattern matching
git2 = "0.19"          # Git operations
```

## Directory Structure

```
~/.memrl/
├── config.toml              # Configuration file
├── feedback.log             # Feedback history
├── episodes/                # Episode storage
│   └── YYYY-MM-DD/         # Date-organized
│       └── session-*.json  # Episode files
└── vectors/                 # Vector database
    └── episodes.lance/     # LanceDB storage
```

## MCP Server Setup

1. Build the MCP server:
```bash
cargo build --release --bin memrl-mcp
```

2. Add to Claude Code using the CLI (recommended):
```bash
claude mcp add memrl --scope user -- /path/to/memrl-mcp
```

The `--scope user` flag makes it available across all projects. Configuration is stored in `~/.claude.json`.

Alternatively, edit `~/.claude.json` directly:
```json
{
  "mcpServers": {
    "memrl": {
      "command": "/path/to/memrl-mcp",
      "args": []
    }
  }
}
```

3. Restart Claude Code and verify with `/mcp` command.

4. The following tools become available to Claude:
   - `memrl_retrieve` - Search, list, or show episode details
   - `memrl_capture` - Save current session to memory
   - `memrl_feedback` - Mark episodes as helpful/unhelpful
   - `memrl_stats` - View memory statistics
   - `memrl_propagate` - Run utility propagation

## Hook Integration

Install hooks for automatic session capture:

```bash
# Copy hooks to your hooks directory
cp hooks/*.sh ~/.claude/hooks/

# Or configure via Claude Code CLI:
claude config set hooks.post-session "/path/to/memrl/hooks/post-session.sh"
claude config set hooks.pre-task "/path/to/memrl/hooks/pre-task.sh"
```

## Future Enhancements

- [ ] Episode clustering and pattern detection
- [ ] Export/import functionality
- [ ] Web dashboard for visualization
- [ ] Multi-project memory sharing
- [ ] Custom embedding models

## Usage Examples

```bash
# Initialize in a project
memrl init

# Capture a session with LLM analysis
memrl capture --session transcript.txt --extract-intent

# Index for semantic search
memrl index

# Find relevant past experiences
memrl retrieve "fix authentication bug"

# Provide feedback
memrl feedback helpful --episodes abc123,def456

# Run utility propagation
memrl propagate --temporal

# Prune old episodes (dry run)
memrl prune --older-than 90 --min-utility 0.3

# View statistics
memrl stats
```

## MCP Tool Example

When Claude Code has MemRL integrated via MCP:

```
User: Fix the login redirect bug

Claude: Let me check if we've solved similar problems before...
[Calls memrl_retrieve with query "login redirect bug"]

Found 2 relevant past experiences:
1. fix the authentication bug in the login flow
   - Relevance: 72% similarity, 85% utility
   - Resolution: Added proper URL sanitization

Let me apply a similar approach...
[Works on the fix]

[Calls memrl_capture to save this session]
Episode captured: abc12345
```

---

## Working Example

MemRL is actively used in this project! Current memory stats:
- 12+ episodes captured
- Semantic search working with BGE-Small embeddings
- 5 MCP tools operational
- Utility propagation and temporal credit working

---

*Last updated: 2026-01-25*
*All 4 phases complete*
