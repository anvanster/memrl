#!/bin/bash
set -e

# Prepare binaries for npm package distribution
# Builds native binary and cross-compiles for darwin-x64 if on darwin-arm64
# Run from project root: ./scripts/prepare-binaries.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PREBUILT_DIR="$PROJECT_ROOT/prebuilt"

echo "Preparing memrl binaries..."

mkdir -p "$PREBUILT_DIR"

# Expected binary names (two binaries per platform)
PLATFORMS=(
    "darwin-arm64"
    "darwin-x64"
    "linux-x64"
    "win32-x64"
)

# Detect current platform
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$PLATFORM" in
    darwin) PLATFORM_STR="darwin" ;;
    linux) PLATFORM_STR="linux" ;;
    mingw*|msys*|cygwin*) PLATFORM_STR="win32" ;;
    *) PLATFORM_STR="" ;;
esac

case "$ARCH" in
    arm64|aarch64) ARCH_STR="arm64" ;;
    x86_64) ARCH_STR="x64" ;;
    *) ARCH_STR="" ;;
esac

CURRENT_PLATFORM="${PLATFORM_STR}-${ARCH_STR}"

# Build for current platform
echo ""
echo "Building for current platform ($CURRENT_PLATFORM)..."
cargo build --release

# Copy binaries
if [ "$PLATFORM_STR" = "win32" ]; then
    cp "$PROJECT_ROOT/target/release/memrl.exe" "$PREBUILT_DIR/memrl-${CURRENT_PLATFORM}.exe"
    cp "$PROJECT_ROOT/target/release/memrl-mcp.exe" "$PREBUILT_DIR/memrl-mcp-${CURRENT_PLATFORM}.exe"
else
    cp "$PROJECT_ROOT/target/release/memrl" "$PREBUILT_DIR/memrl-${CURRENT_PLATFORM}"
    cp "$PROJECT_ROOT/target/release/memrl-mcp" "$PREBUILT_DIR/memrl-mcp-${CURRENT_PLATFORM}"
    chmod +x "$PREBUILT_DIR/memrl-${CURRENT_PLATFORM}"
    chmod +x "$PREBUILT_DIR/memrl-mcp-${CURRENT_PLATFORM}"
fi
echo "Saved: prebuilt/memrl-${CURRENT_PLATFORM}"
echo "Saved: prebuilt/memrl-mcp-${CURRENT_PLATFORM}"

# Cross-compile for Mac x64 if on Mac ARM
if [ "$PLATFORM_STR" = "darwin" ] && [ "$ARCH_STR" = "arm64" ]; then
    echo ""
    echo "Cross-compiling for darwin-x64..."
    if rustup target list --installed | grep -q "x86_64-apple-darwin"; then
        if cargo build --release --target x86_64-apple-darwin 2>&1; then
            cp "$PROJECT_ROOT/target/x86_64-apple-darwin/release/memrl" "$PREBUILT_DIR/memrl-darwin-x64"
            cp "$PROJECT_ROOT/target/x86_64-apple-darwin/release/memrl-mcp" "$PREBUILT_DIR/memrl-mcp-darwin-x64"
            chmod +x "$PREBUILT_DIR/memrl-darwin-x64"
            chmod +x "$PREBUILT_DIR/memrl-mcp-darwin-x64"
            echo "Saved: prebuilt/memrl-darwin-x64"
            echo "Saved: prebuilt/memrl-mcp-darwin-x64"
        else
            echo "  ⚠ Cross-compilation failed (OpenSSL cross-compile setup required)"
            echo "    Build natively on x64 Mac instead, or set up OpenSSL for cross-compilation"
        fi
    else
        echo "  ⚠ x86_64-apple-darwin target not installed. Run:"
        echo "    rustup target add x86_64-apple-darwin"
    fi
fi

# Summary
echo ""
echo "========================================"
echo "prebuilt/ contents:"
ls -lh "$PREBUILT_DIR/" 2>/dev/null | grep memrl || echo "  (empty)"

echo ""
echo "Status by platform:"
for PLAT in "${PLATFORMS[@]}"; do
    EXT=""
    [ "$PLAT" = "win32-x64" ] && EXT=".exe"

    if [ -f "$PREBUILT_DIR/memrl-${PLAT}${EXT}" ] && [ -f "$PREBUILT_DIR/memrl-mcp-${PLAT}${EXT}" ]; then
        echo "  ✓ $PLAT"
    else
        echo "  ✗ $PLAT (missing)"
    fi
done

echo ""
echo "To build for other platforms, run this script on each platform"
echo "and copy the binaries to prebuilt/"
