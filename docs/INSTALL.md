# Tempera Installation Guide

Install instructions for Tempera on Windows, macOS, and Linux. For day-to-day usage see [../README.md](../README.md); for the current architecture see [../PROGRESS.md](../PROGRESS.md).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation paths](#installation-paths)
  - [Option A — `cargo install` (recommended for Rust users)](#option-a--cargo-install-recommended-for-rust-users)
  - [Option B — Build from source](#option-b--build-from-source)
  - [Option C — Pre-built binary](#option-c--pre-built-binary)
- [Set up with Claude Code](#set-up-with-claude-code)
- [First run](#first-run)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Uninstall](#uninstall)

---

## Prerequisites

Tempera has no external runtime dependencies beyond the embedding model it downloads on first use. The build-time prerequisites depend on which install path you take:

| Path | Needs |
|------|-------|
| `cargo install` | Rust toolchain ([rustup.rs](https://rustup.rs/)). 1-2 GB of disk and ~5 min for the first build. |
| Build from source | Same as above, plus a git checkout. |
| Pre-built binary | Nothing — just `unzip`/`tar`. |

**Note**: Tempera does **not** require `protoc`, LanceDB, or any other external binary. Earlier releases (pre-v0.4.x) needed protoc for LanceDB; the current build uses [vectrust](https://crates.io/crates/vectrust) which builds without it.

Network access is needed once on first run to download the [BGE-Small](https://huggingface.co/BAAI/bge-small-en-v1.5) embedding model (~128 MB) from HuggingFace. The model is cached locally afterward and Tempera works offline.

---

## Installation paths

### Option A — `cargo install` (recommended for Rust users)

```bash
cargo install tempera
```

This builds both binaries (`tempera` and `tempera-mcp`) and installs them into your Cargo bin directory:

- **Linux/macOS**: `~/.cargo/bin/`
- **Windows**: `%USERPROFILE%\.cargo\bin\`

`~/.cargo/bin` is on PATH by default after running `rustup`. If not, add it manually.

### Option B — Build from source

```bash
git clone https://github.com/anvanster/tempera.git
cd tempera
cargo build --release
```

Both binaries land in `target/release/`:

```
target/release/tempera         # CLI
target/release/tempera-mcp     # MCP server for Claude Code
```

Copy them somewhere on PATH:

```bash
# Linux/macOS
sudo cp target/release/{tempera,tempera-mcp} /usr/local/bin/

# macOS without sudo (using ~/.local/bin if it's on PATH)
mkdir -p ~/.local/bin
cp target/release/{tempera,tempera-mcp} ~/.local/bin/

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\tempera"
Copy-Item target\release\tempera.exe,target\release\tempera-mcp.exe "$env:LOCALAPPDATA\tempera\"
# Add $env:LOCALAPPDATA\tempera to PATH — see step under Option C below.
```

The release build uses LTO + strip — slow (~5 minutes), but the resulting binaries are smaller and faster.

### Option C — Pre-built binary

Releases are published at [github.com/anvanster/tempera/releases](https://github.com/anvanster/tempera/releases). Each release ships archives named by Rust target triple:

| Platform | Asset |
|----------|-------|
| Intel/AMD Linux | `tempera-x86_64-unknown-linux-gnu.tar.gz` |
| ARM64 Linux | `tempera-aarch64-unknown-linux-gnu.tar.gz` |
| Intel macOS | `tempera-x86_64-apple-darwin.tar.gz` |
| Apple Silicon macOS | `tempera-aarch64-apple-darwin.tar.gz` |
| Windows | `tempera-x86_64-pc-windows-msvc.zip` |

#### Linux / macOS

```bash
# Pick the right archive for your platform.
TAG=$(curl -s https://api.github.com/repos/anvanster/tempera/releases/latest | jq -r .tag_name)
TARGET=x86_64-unknown-linux-gnu   # or aarch64-apple-darwin, etc.

curl -LO "https://github.com/anvanster/tempera/releases/download/${TAG}/tempera-${TARGET}.tar.gz"
tar -xzf "tempera-${TARGET}.tar.gz"

# Install
sudo mv tempera tempera-mcp /usr/local/bin/
sudo chmod +x /usr/local/bin/tempera /usr/local/bin/tempera-mcp
```

#### Windows (PowerShell)

```powershell
# Download the latest release archive (replace v0.4.27 with the actual tag).
$tag = "v0.4.27"
$url = "https://github.com/anvanster/tempera/releases/download/$tag/tempera-x86_64-pc-windows-msvc.zip"
Invoke-WebRequest -Uri $url -OutFile "$env:USERPROFILE\Downloads\tempera.zip"

# Extract
Expand-Archive -Path "$env:USERPROFILE\Downloads\tempera.zip" -DestinationPath "$env:LOCALAPPDATA\tempera" -Force

# Add to user PATH (permanent)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$env:LOCALAPPDATA\tempera*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$env:LOCALAPPDATA\tempera", "User")
}

# Refresh PATH in current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

#### Verify

```bash
tempera --version
tempera-mcp --version
```

Both should print the same version (matches `Cargo.toml`).

---

## Set up with Claude Code

Tempera is consumed by Claude Code as an MCP server. The `tempera-mcp` binary speaks the [Model Context Protocol](https://modelcontextprotocol.io/) over stdio.

### Recommended: `claude mcp add`

The simplest path is to register the server with the Claude CLI:

```bash
# User-scoped (available across all your projects).
claude mcp add tempera --scope user -- $(which tempera-mcp)

# Or with an explicit path:
claude mcp add tempera --scope user -- /usr/local/bin/tempera-mcp
```

Restart Claude Code, then run `/mcp` inside Claude Code. You should see `tempera` listed with 12 tools.

### Alternative: VS Code mcp.json

If you want project-scoped configuration instead of user-scoped, create or edit `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "tempera": {
      "command": "/usr/local/bin/tempera-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

On Windows the `command` looks like `"C:\\Users\\You\\AppData\\Local\\tempera\\tempera-mcp.exe"` (note the doubled backslashes — required by JSON).

Find the exact path Tempera installed to:

```bash
# Linux/macOS
which tempera-mcp

# Windows (PowerShell)
(Get-Command tempera-mcp).Source
```

Restart VS Code to load the configuration.

---

## First run

The first time `tempera-mcp` (or any retrieval/index command) runs, Tempera downloads the BGE-Small embedding model (~128 MB) from HuggingFace. This is a one-time cost:

```bash
# Trigger the download manually from the CLI:
tempera init
# 🔄 Loading embedding model (this may download the model on first run)...
# ✅ Embedding model loaded
```

The model — and everything else Tempera persists — is cached under a single directory:

| OS | Path |
|----|------|
| Linux / macOS | `~/.tempera/` |
| Windows | `%USERPROFILE%\.tempera\` |

Layout after first run:

```
~/.tempera/
├── config.toml         # Configuration (auto-created with defaults)
├── episodes/           # Canonical JSON per captured session
├── jobs.sqlite         # SQLite for everything indexable (11 tables)
├── vectors/            # vectrust embedding index
├── models/             # BGE-Small model files (~128 MB)
├── reflections/        # Daily reflection markdown (v0.7.3+)
├── patterns/           # Cross-day pattern pages (v0.7.4+)
└── templates/          # Reasoning templates (v0.8.3+)
```

Override the data directory with the `TEMPERA_DATA_DIR` environment variable; override the model cache with `FASTEMBED_CACHE_DIR`.

---

## Configuration

All knobs live in `~/.tempera/config.toml`. Defaults are tuned to be useful out of the box — you only need to touch the file if you want to change retrieval ranking, dream-cycle behaviour, or per-phase budgets.

See the [Configuration section in the README](../README.md#configuration) for the full reference. A minimal example to override the dream-cycle budget:

```toml
[dream]
default_max_usd = 1.00       # Default is 0.50
templates_min_evidence = 5   # Default is 3 — require denser evidence per template
```

---

## Troubleshooting

### `tempera --version` reports "command not found"

PATH isn't set up. Check where the binary actually landed:

```bash
# cargo install
ls ~/.cargo/bin/tempera*

# Source build
ls target/release/tempera*

# Pre-built binary, Linux/macOS
ls /usr/local/bin/tempera*

# Pre-built binary, Windows
Get-ChildItem "$env:LOCALAPPDATA\tempera\"
```

Then make sure the containing directory is on PATH. On Linux/macOS, add to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # or wherever tempera lives
```

### MCP server doesn't appear in Claude Code

1. **Verify the path** — Claude Code can't auto-detect the binary. Check what's registered:
   ```bash
   claude mcp list
   ```
   If `tempera-mcp` isn't there or points to the wrong path, re-add it.

2. **Restart Claude Code completely** — closing and reopening the VS Code panel isn't enough; the CLI needs a fresh start to load MCP servers.

3. **Check for errors** — in VS Code, open the developer tools (`Help → Toggle Developer Tools`) and look for errors mentioning "tempera" in the console.

4. **Test the binary directly** — `tempera-mcp` is interactive, but launching it should produce no output (it waits on stdin):
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | tempera-mcp
   ```
   Expect a JSON response listing 12 tools.

### Embedding model download fails

The download requires HTTPS access to `huggingface.co`. If you're behind a corporate proxy, set `HTTPS_PROXY` and `HTTP_PROXY` before invoking Tempera:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
tempera init
```

If the download succeeded once and you want to share the model between machines, copy the entire `~/.tempera/models/` directory.

### `tempera retrieve` finds nothing despite captures

The vector index may be empty or out of date. Re-index everything:

```bash
tempera index --reindex
```

Then verify:

```bash
tempera status
# Look for "Vector index: N episodes" — should match capture count.
```

### `tempera_brief` returns "nothing to surface"

Normal early on — the brief joins against signal data (mistakes, asks, templates, calibration) that accrues over time. See the [README troubleshooting section](../README.md#troubleshooting) for the per-section gates.

### `tempera retrieve --cross-project` returns nothing

Episodes captured before v0.6.4 don't have an explicit `ValidityScope`, so v0.10's cross-project filter treats them as project-bound (conservative default). To enable cross-project surfacing for an existing capture, re-capture with `validity_scope` set (e.g. `language:rust`) or wait for the LLM-suggested scope from new v0.10.3+ captures to accrue.

### macOS: "cannot be opened because the developer cannot be verified"

The pre-built binaries aren't notarized. Either:

```bash
# Remove the quarantine attribute (per-binary)
xattr -d com.apple.quarantine /usr/local/bin/tempera
xattr -d com.apple.quarantine /usr/local/bin/tempera-mcp
```

Or build from source — Apple's Gatekeeper trusts locally-compiled binaries.

### Windows: PowerShell blocks the binary

Add the install directory to Windows Defender exclusions (Settings → Update & Security → Virus & Threat Protection → Add an exclusion → Folder), or build from source.

### Database errors at startup

If `jobs.sqlite` got corrupted, the cleanest recovery is to snapshot the data dir, delete the SQLite file, and let Tempera recreate it. Episodes themselves live in `~/.tempera/episodes/` as JSON; they re-populate the SQLite tables when you next call any indexing command.

```bash
# Back up first
tempera backup

# Then remove the SQLite store
rm ~/.tempera/jobs.sqlite

# Re-create on next call (which also re-runs migrations)
tempera status
tempera index --reindex
```

---

## Uninstall

Tempera leaves nothing outside `~/.tempera/` and the binaries themselves. To remove cleanly:

```bash
# Delete the binaries
# (cargo install)
cargo uninstall tempera

# (source/pre-built install on Linux/macOS)
sudo rm /usr/local/bin/tempera /usr/local/bin/tempera-mcp

# (Windows)
Remove-Item "$env:LOCALAPPDATA\tempera" -Recurse -Force

# Remove the MCP registration
claude mcp remove tempera

# Optionally, wipe the data dir (this deletes ALL captured memory)
rm -rf ~/.tempera/
```

Back up `~/.tempera/episodes/` first if you want to keep your memory.

---

## Support

- **Issues**: [github.com/anvanster/tempera/issues](https://github.com/anvanster/tempera/issues)
- **Repository**: [github.com/anvanster/tempera](https://github.com/anvanster/tempera)
- **License**: Apache-2.0
