#!/bin/bash
set -e

# Build script for @anvanster/memrl npm package
# Run from project root: ./scripts/build-npm-package.sh
#
# Prerequisites: Run ./scripts/prepare-binaries.sh first on each platform

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NPM_PACKAGE="$PROJECT_ROOT/npm"
PREBUILT_DIR="$PROJECT_ROOT/prebuilt"

echo "Building @anvanster/memrl npm package..."

# Optionally run prepare-binaries first to ensure we have latest builds
# Skip with --skip-build flag
if [ "$1" != "--skip-build" ]; then
    echo ""
    echo "Step 1: Preparing binaries (use --skip-build to skip)..."
    "$SCRIPT_DIR/prepare-binaries.sh"
else
    echo ""
    echo "Step 1: Skipping binary build (--skip-build)"
fi

# Ensure npm package bin directory exists
mkdir -p "$NPM_PACKAGE/bin"

# Expected platforms
PLATFORMS=(
    "darwin-arm64"
    "darwin-x64"
    "linux-x64"
    "win32-x64"
)

# Copy binaries from prebuilt to npm package
echo ""
echo "Step 2: Copying binaries to npm/bin/..."
FOUND_COUNT=0
MISSING=()

for PLAT in "${PLATFORMS[@]}"; do
    EXT=""
    [ "$PLAT" = "win32-x64" ] && EXT=".exe"

    MEMRL_BIN="memrl-${PLAT}${EXT}"
    MCP_BIN="memrl-mcp-${PLAT}${EXT}"

    if [ -f "$PREBUILT_DIR/$MEMRL_BIN" ] && [ -f "$PREBUILT_DIR/$MCP_BIN" ]; then
        cp "$PREBUILT_DIR/$MEMRL_BIN" "$NPM_PACKAGE/bin/"
        cp "$PREBUILT_DIR/$MCP_BIN" "$NPM_PACKAGE/bin/"
        chmod +x "$NPM_PACKAGE/bin/$MEMRL_BIN" 2>/dev/null || true
        chmod +x "$NPM_PACKAGE/bin/$MCP_BIN" 2>/dev/null || true
        echo "  ✓ $PLAT"
        ((FOUND_COUNT++))
    else
        MISSING+=("$PLAT")
        echo "  ✗ $PLAT (not found)"
    fi
done

# Summary
echo ""
echo "========================================"
echo "Package contents:"
ls -lh "$NPM_PACKAGE/bin/" 2>/dev/null | grep -E "memrl" || echo "  (no binaries)"

echo ""
echo "Found: $FOUND_COUNT/${#PLATFORMS[@]} platforms"

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "Missing platforms (build on respective systems):"
    for M in "${MISSING[@]}"; do
        echo "  - $M"
    done
    echo ""
    echo "Copy built binaries to: $PREBUILT_DIR/"
fi

echo ""
echo "Package size:"
du -sh "$NPM_PACKAGE"
du -sh "$NPM_PACKAGE/bin" 2>/dev/null || true

echo ""
echo "========================================"
if [ $FOUND_COUNT -eq ${#PLATFORMS[@]} ]; then
    echo "✓ All platforms present. Ready to publish!"
    echo ""
    echo "  cd npm"
    echo "  npm publish --access public"
else
    echo "⚠ Missing platforms. Build on other systems first."
fi
echo ""
echo "To test locally:"
echo "  cd npm && npm link"
echo "  memrl --help"
echo "  memrl-mcp --help"
