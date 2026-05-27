#!/bin/bash
# Copyright 2024-2026 Andrey Vasilevsky <anvanster@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Package the combined multi-platform MCP server distribution.
# Run from the repo root after all platform binaries are in dist/.
#
# Usage:
#   ./scripts/package-mcp.sh               # assemble + npm pack
#   ./scripts/package-mcp.sh --publish      # also publish to npmjs.com

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$REPO_ROOT/mcp-package"
BIN_DIR="$PKG_DIR/bin"
DIST_DIR="$REPO_ROOT/dist"

echo "=== Tempera MCP package builder ==="
echo ""

# ── Step 1: Verify dist/ archives exist ──────────────────────────────

ARCHIVES=(
  "tempera-aarch64-apple-darwin.tar.gz"
  "tempera-x86_64-apple-darwin.tar.gz"
  "tempera-x86_64-unknown-linux-gnu.tar.gz"
  "tempera-x86_64-pc-windows-msvc.zip"
)

MISSING=0
for archive in "${ARCHIVES[@]}"; do
  if [ ! -f "$DIST_DIR/$archive" ]; then
    echo "  ✗ Missing: dist/$archive"
    MISSING=1
  else
    SIZE=$(du -h "$DIST_DIR/$archive" | cut -f1)
    echo "  ✓ Found: dist/$archive ($SIZE)"
  fi
done

if [ "$MISSING" -eq 1 ]; then
  echo ""
  echo "ERROR: Not all platform archives are present in dist/"
  echo "Build missing platforms first."
  exit 1
fi

# ── Step 2: Extract binaries into bin/ with platform suffixes ────────

echo ""
echo "Extracting and renaming binaries..."

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# macOS ARM
tar -xzf "$DIST_DIR/tempera-aarch64-apple-darwin.tar.gz" -C "$TMPDIR"
cp "$TMPDIR/tempera-mcp" "$BIN_DIR/tempera-mcp-darwin-arm64"
cp "$TMPDIR/tempera"     "$BIN_DIR/tempera-darwin-arm64"
chmod +x "$BIN_DIR/tempera-mcp-darwin-arm64" "$BIN_DIR/tempera-darwin-arm64"
echo "  ✓ darwin-arm64"

# macOS x86
rm -f "$TMPDIR/tempera" "$TMPDIR/tempera-mcp"
tar -xzf "$DIST_DIR/tempera-x86_64-apple-darwin.tar.gz" -C "$TMPDIR"
cp "$TMPDIR/tempera-mcp" "$BIN_DIR/tempera-mcp-darwin-x64"
cp "$TMPDIR/tempera"     "$BIN_DIR/tempera-darwin-x64"
chmod +x "$BIN_DIR/tempera-mcp-darwin-x64" "$BIN_DIR/tempera-darwin-x64"
echo "  ✓ darwin-x64"

# Linux x86
rm -f "$TMPDIR/tempera" "$TMPDIR/tempera-mcp"
tar -xzf "$DIST_DIR/tempera-x86_64-unknown-linux-gnu.tar.gz" -C "$TMPDIR"
cp "$TMPDIR/tempera-mcp" "$BIN_DIR/tempera-mcp-linux-x64"
cp "$TMPDIR/tempera"     "$BIN_DIR/tempera-linux-x64"
chmod +x "$BIN_DIR/tempera-mcp-linux-x64" "$BIN_DIR/tempera-linux-x64"
echo "  ✓ linux-x64"

# Windows x86
rm -f "$TMPDIR/tempera.exe" "$TMPDIR/tempera-mcp.exe"
unzip -o -q "$DIST_DIR/tempera-x86_64-pc-windows-msvc.zip" -d "$TMPDIR"
cp "$TMPDIR/tempera-mcp.exe" "$BIN_DIR/tempera-mcp-win32-x64.exe"
cp "$TMPDIR/tempera.exe"     "$BIN_DIR/tempera-win32-x64.exe"
echo "  ✓ win32-x64"

# ── Step 3: Bundle onnxruntime.dll for Windows ───────────────────────

ONNX_DLL=""
# Check common locations
for candidate in \
  "$REPO_ROOT/dist/onnxruntime.dll" \
  "$HOME/projects/codegraph/vscode/bin/onnxruntime.dll" \
  "$HOME/projects/codegraph/mcp-package/bin/onnxruntime.dll"; do
  if [ -f "$candidate" ]; then
    ONNX_DLL="$candidate"
    break
  fi
done

if [ -n "$ONNX_DLL" ]; then
  cp "$ONNX_DLL" "$BIN_DIR/onnxruntime.dll"
  echo "  ✓ Copied onnxruntime.dll ($(du -h "$ONNX_DLL" | cut -f1))"
else
  echo "  ⚠ WARNING: onnxruntime.dll not found!"
  echo "    Windows users will fail at runtime without this DLL."
  echo "    Try: scp administrator@192.168.254.103:C:/path/to/onnxruntime.dll dist/"
fi

# ── Step 4: Verify launcher exists ───────────────────────────────────

if [ ! -f "$BIN_DIR/tempera-mcp.js" ]; then
  echo ""
  echo "ERROR: mcp-package/bin/tempera-mcp.js not found"
  exit 1
fi
chmod +x "$BIN_DIR/tempera-mcp.js"

# ── Step 5: Show package contents ────────────────────────────────────

echo ""
echo "Package contents:"
ls -lh "$BIN_DIR/"

# Version from package.json
PKG_VERSION=$(node -e "console.log(require('$PKG_DIR/package.json').version)")
echo ""
echo "package.json version: $PKG_VERSION"

# ── Step 6: npm pack ─────────────────────────────────────────────────

echo ""
echo "Packing..."
cd "$PKG_DIR"
npm pack 2>&1

TARBALL=$(ls -t *.tgz 2>/dev/null | head -1)
if [ -n "$TARBALL" ]; then
  SIZE=$(du -h "$TARBALL" | cut -f1)
  echo ""
  echo "✓ Created: mcp-package/$TARBALL ($SIZE)"
fi

# ── Step 7: Publish if requested ─────────────────────────────────────

if [ "${1:-}" = "--publish" ]; then
  echo ""
  echo "Publishing to npmjs.com..."
  npm publish --access public
  echo "✓ Published @astudioplus/tempera-mcp@$PKG_VERSION"
else
  echo ""
  echo "To publish: cd mcp-package && npm publish --access public"
fi
