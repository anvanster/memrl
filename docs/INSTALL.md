# MemRL Installation Guide

Complete installation instructions for MemRL on Windows, macOS, and Linux.

## Table of Contents

- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Build from Source](#build-from-source)
- [Verification](#verification)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Windows Installation

### Option 1: Download Pre-built Binary (Recommended)

#### Step 1: Download Release Package

1. Go to the [MemRL Releases page](https://github.com/anvanster/memrl/releases)
2. Download the latest `memrl-vX.X.X-windows-x64.zip` file
3. Download the corresponding `.sha256` checksum file (optional, for verification)

#### Step 2: Verify Checksum (Optional but Recommended)

Open PowerShell and run:

```powershell
# Navigate to your downloads folder
cd $env:USERPROFILE\Downloads

# Verify the checksum
$hash = (Get-FileHash -Path "memrl-v0.1.3-windows-x64.zip" -Algorithm SHA256).Hash
$expected = (Get-Content "memrl-v0.1.3-windows-x64.sha256" -Raw).Split()[0]

if ($hash -eq $expected) {
    Write-Host "✅ Checksum verified!" -ForegroundColor Green
} else {
    Write-Host "❌ Checksum mismatch!" -ForegroundColor Red
}
```

#### Step 3: Extract Archive

```powershell
# Extract to Program Files
Expand-Archive -Path "memrl-v0.1.3-windows-x64.zip" -DestinationPath "$env:ProgramFiles\memrl"

# Or extract to a user directory
Expand-Archive -Path "memrl-v0.1.3-windows-x64.zip" -DestinationPath "$env:LOCALAPPDATA\memrl"
```

#### Step 4: Add to PATH

##### Using PowerShell (Temporary - Current Session Only):

```powershell
$env:Path += ";$env:LOCALAPPDATA\memrl"
```

##### Using PowerShell (Permanent - Recommended):

```powershell
# Add to user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;$env:LOCALAPPDATA\memrl", "User")

# Refresh current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

##### Using GUI:

1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", select "Path" and click "Edit"
5. Click "New" and add: `C:\Users\YourUsername\AppData\Local\memrl`
6. Click "OK" on all dialogs
7. Restart your terminal

#### Step 5: Verify Installation

Open a new PowerShell or Command Prompt window:

```powershell
# Check memrl CLI
memrl --version

# Check MCP server
memrl-mcp --version
```

You should see version information for both executables.

---

### Option 2: Install from crates.io

If you have Rust installed:

```powershell
cargo install memrl
```

This compiles from source and installs both `memrl.exe` and `memrl-mcp.exe` to your Cargo bin directory (usually `C:\Users\YourUsername\.cargo\bin`).

---

## macOS Installation

### Option 1: Download Pre-built Binary

#### Step 1: Download and Extract

```bash
# Download the latest release
cd ~/Downloads
curl -LO https://github.com/anvanster/memrl/releases/download/v0.1.3/memrl-v0.1.3-macos-x64.zip
curl -LO https://github.com/anvanster/memrl/releases/download/v0.1.3/memrl-v0.1.3-macos-x64.sha256

# Verify checksum
shasum -a 256 -c memrl-v0.1.3-macos-x64.sha256

# Extract
unzip memrl-v0.1.3-macos-x64.zip -d memrl
```

#### Step 2: Install

```bash
# Move to local bin
sudo mv memrl/memrl /usr/local/bin/
sudo mv memrl/memrl-mcp /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/memrl
sudo chmod +x /usr/local/bin/memrl-mcp
```

#### Step 3: Verify

```bash
memrl --version
memrl-mcp --version
```

### Option 2: Install from crates.io

```bash
cargo install memrl
```

---

## Linux Installation

### Option 1: Download Pre-built Binary

#### Step 1: Download and Extract

```bash
# Download the latest release
cd ~/Downloads
wget https://github.com/anvanster/memrl/releases/download/v0.1.3/memrl-v0.1.3-linux-x64.zip
wget https://github.com/anvanster/memrl/releases/download/v0.1.3/memrl-v0.1.3-linux-x64.sha256

# Verify checksum
sha256sum -c memrl-v0.1.3-linux-x64.sha256

# Extract
unzip memrl-v0.1.3-linux-x64.zip -d memrl
```

#### Step 2: Install

```bash
# Move to local bin
sudo mv memrl/memrl /usr/local/bin/
sudo mv memrl/memrl-mcp /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/memrl
sudo chmod +x /usr/local/bin/memrl-mcp
```

#### Step 3: Verify

```bash
memrl --version
memrl-mcp --version
```

### Option 2: Install from crates.io

```bash
cargo install memrl
```

---

## Build from Source

### Prerequisites

- **Rust**: Install from [rustup.rs](https://rustup.rs/)
  - Windows: Download and run `rustup-init.exe`
  - macOS/Linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

- **Protocol Buffers Compiler (protoc)**: Required by LanceDB dependencies

  **Windows:**
  1. Download [protoc-34.0-rc-1-win64.zip](https://github.com/protocolbuffers/protobuf/releases/download/v34.0-rc1.1/protoc-34.0-rc-1-win64.zip)
  2. Extract to a directory (e.g., `C:\tools\protoc`)
  3. Set the environment variable:
     ```powershell
     $env:PROTOC = "C:\tools\protoc\bin\protoc.exe"
     ```

  **Linux:**
  ```bash
  # Download and extract protoc
  wget https://github.com/protocolbuffers/protobuf/releases/download/v34.0-rc1.1/protoc-34.0-rc-1-linux-x86_64.zip
  unzip protoc-34.0-rc-1-linux-x86_64.zip -d protoc

  # Add to bashrc for persistence
  echo "export PROTOC=$(pwd)/protoc/bin/protoc" >> ~/.bashrc
  source ~/.bashrc
  ```

### Build Steps

```bash
# Clone repository
git clone https://github.com/anvanster/memrl.git
cd memrl

# Build release binaries
cargo build --release

# Binaries are created in:
# - target/release/memrl      (CLI tool)
# - target/release/memrl-mcp  (MCP server)
```

### Install Built Binaries

**Windows:**
```powershell
# Copy to local bin directory
New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\memrl"
Copy-Item target\release\memrl.exe "$env:LOCALAPPDATA\memrl\"
Copy-Item target\release\memrl-mcp.exe "$env:LOCALAPPDATA\memrl\"

# Add to PATH (see Windows installation steps above)
```

**macOS/Linux:**
```bash
sudo cp target/release/memrl /usr/local/bin/
sudo cp target/release/memrl-mcp /usr/local/bin/
```

---

## Verification

### Test CLI Tool

```bash
# Check version
memrl --version

# View help
memrl --help

# Check memory status
memrl status
```

### Test MCP Server

```bash
# Check version
memrl-mcp --version
```

---

## Configuration

### Configure Claude Desktop (VS Code)

1. Open your VS Code workspace
2. Create or edit `.vscode/mcp.json`:

```json
{
  "servers": {
    "memrl": {
      "command": "C:\\Users\\YourUsername\\AppData\\Local\\memrl\\memrl-mcp.exe",
      "args": [],
      "env": {}
    }
  }
}
```

**Important**: Replace `C:\\Users\\YourUsername\\AppData\\Local\\memrl\\memrl-mcp.exe` with the actual path where you installed memrl-mcp.exe.

To find the exact path:

```powershell
# Windows
(Get-Command memrl-mcp).Source

# macOS/Linux
which memrl-mcp
```

3. Restart VS Code
4. The memrl MCP server will be available to Claude Code

### First Run - Model Download

The first time you use memrl, it will download the embedding model (~90MB):

```bash
memrl status
```

Output:
```
🔄 Downloading embedding model (one-time, ~90MB)...
✅ Model downloaded successfully
📊 Memory Status for 'your-project'
...
```

The model is cached in:
- **Windows**: `C:\Users\YourUsername\.cache\memrl\`
- **macOS**: `~/Library/Caches/memrl/`
- **Linux**: `~/.cache/memrl/`

### Optional: Custom Configuration

Create `~/.memrl/config.toml` to customize settings:

```toml
# Memory settings
max_episodes = 10000
retrieval_limit = 10

# Utility parameters
learning_rate = 0.3
discount_factor = 0.95
decay_rate = 0.01

# Vector search
vector_enabled = true
similarity_threshold = 0.7
```

---

## Troubleshooting

### Windows: "Command not found"

**Problem**: PowerShell doesn't recognize `memrl` command.

**Solutions**:
1. Verify PATH was updated: `$env:Path -split ';' | Select-String memrl`
2. Restart your terminal completely
3. Try using the full path: `C:\Users\YourUsername\AppData\Local\memrl\memrl.exe --version`

### Windows: "Cannot be loaded because running scripts is disabled"

**Problem**: PowerShell execution policy blocks scripts.

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows: Antivirus Blocks Executable

**Problem**: Windows Defender or antivirus quarantines memrl.exe.

**Solution**:
1. Add memrl installation directory to antivirus exclusions
2. Or build from source yourself (antivirus trusts self-built binaries)

### macOS: "Cannot be opened because the developer cannot be verified"

**Problem**: macOS Gatekeeper blocks unsigned binary.

**Solution**:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine /usr/local/bin/memrl
xattr -d com.apple.quarantine /usr/local/bin/memrl-mcp
```

### All Platforms: "Error: Failed to initialize database"

**Problem**: SQLite database initialization failed.

**Solution**:
```bash
# Check if memrl data directory exists
# Windows: %APPDATA%\memrl\
# macOS/Linux: ~/.memrl/

# If corrupted, delete and reinitialize
rm -rf ~/.memrl/episodes.db
memrl status
```

### Build Error: "protoc not found" or "PROTOC environment variable not set"

**Problem**: Building from source fails because protoc is missing.

**Solution**: See [Prerequisites](#prerequisites) section for protoc installation instructions.

### MCP Server Not Appearing in Claude Code

**Problem**: Claude Code doesn't show memrl tools.

**Solutions**:
1. Verify `.vscode/mcp.json` syntax is valid JSON
2. Check the path to `memrl-mcp.exe` is correct
3. Restart VS Code completely
4. Check VS Code Developer Console for errors: `Help > Toggle Developer Tools`

---

## Next Steps

After installation:

1. **Read the README**: [README.md](../README.md) for usage examples
2. **View CLI help**: `memrl --help`
3. **Check memory status**: `memrl status`
4. **Start using with Claude**: Ask Claude to capture your first episode!

---

## Support

- **Issues**: [GitHub Issues](https://github.com/anvanster/memrl/issues)
- **Repository**: [github.com/anvanster/memrl](https://github.com/anvanster/memrl)
- **License**: Apache-2.0
