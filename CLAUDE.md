# Tempera

Persistent episodic memory system for AI coding assistants. Single Rust crate, two binaries.

## Build & Test
```bash
./scripts/ci-checks.sh          # clippy + fmt + tests
./scripts/ci-checks.sh --full   # + benchmarks, docs, coverage
cargo test --workspace           # tests only
cargo test -- <test_name>        # specific test
cargo build --release            # release build (LTO, slow)
```

## Architecture
Single crate (`src/`) with two binaries:
- `tempera` (CLI): `src/main.rs`
- `tempera-mcp` (MCP server): `src/mcp_server.rs`

Key modules (functional grouping):

| Group | Modules |
|-------|---------|
| Core data | `episode.rs`, `store.rs`, `indexer.rs`, `keyword.rs`, `fingerprint.rs` |
| Capture + retrieve | `capture.rs`, `retrieve.rs`, `utility.rs`, `llm.rs`, `config.rs` |
| Diagnostics | `stats.rs`, `feedback.rs`, `doctor.rs`, `eval.rs` |
| Operations | `jobs.rs`, `backup.rs` |
| Dream cycle (v0.7) | `dream.rs`, `triage.rs`, `reflect.rs`, `patterns.rs`, `contradict.rs` |
| Self-improvement (v0.8) | `calibration.rs`, `mistakes.rs`, `templates.rs`, `templates_phase.rs`, `asks.rs`, `ask_backs.rs`, `ask_back_gen.rs` |
| Brief surface (v0.9) | `brief.rs` |

`templates.rs` / `ask_backs.rs` / `templates_phase.rs` / `ask_back_gen.rs` split: the lean store types are shared between both binaries, the LLM-bearing pipelines are only registered in the `tempera` (CLI) binary.

Error handling: `anyhow::Result` throughout, `thiserror` for typed errors.

## MCP Tool Surface (12 tools)
- **Read surfaces**: `tempera_retrieve`, `tempera_status`, `tempera_stats`, `tempera_brief`, `tempera_session_start`, `tempera_template`
- **Write surfaces**: `tempera_capture`, `tempera_feedback`, `tempera_log_correction`, `tempera_log_should_have_asked`
- **Maintenance**: `tempera_propagate`, `tempera_review`

`tempera_brief(files, task_type?, domain?)` is the recommended task-start call once the file set is known. `tempera_session_start` is the recommended call even earlier — before files are known.

## Self-Referential Warning
This project IS the tempera MCP server. When rebuilding the binary, do NOT call tempera MCP tools simultaneously — the binary may be locked or replaced mid-execution. Use smelt and stellarion MCP tools instead while working on this project.

## Version
Version is in `Cargo.toml` (currently `0.4.27`). Tagging `v*` triggers GitHub Actions release for 5 platforms + npm publish.
