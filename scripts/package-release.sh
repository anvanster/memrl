#!/bin/bash
# Package memrl release for GitHub (Linux)

set -e

VERSION="${1:-0.1.3}"
OUTPUT_DIR="${2:-releases}"

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "📦 Packaging memrl v$VERSION for release..."

# Ensure release builds exist
RELEASE_PATH="$PROJECT_ROOT/target/release"
if [ ! -d "$RELEASE_PATH" ]; then
    echo "❌ Release directory not found. Run 'cargo build --release' first."
    exit 1
fi

# Check for executables
MEMRL_BIN="$RELEASE_PATH/memrl"
MCP_BIN="$RELEASE_PATH/memrl-mcp"

if [ ! -f "$MEMRL_BIN" ]; then
    echo "❌ memrl not found in target/release"
    exit 1
fi

if [ ! -f "$MCP_BIN" ]; then
    echo "❌ memrl-mcp not found in target/release"
    exit 1
fi

# Create output directory
OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$OUTPUT_PATH"

# Create temp staging directory
STAGING_DIR=$(mktemp -d)
trap "rm -rf $STAGING_DIR" EXIT

echo "📋 Copying files to staging..."

# Copy executables
cp "$MEMRL_BIN" "$STAGING_DIR/"
cp "$MCP_BIN" "$STAGING_DIR/"
echo "  ✓ Copied executables"

# Copy documentation and license
for file in README.md LICENSE default_config.toml; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        cp "$PROJECT_ROOT/$file" "$STAGING_DIR/"
        echo "  ✓ Copied $file"
    fi
done

# Detect platform
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH_NAME="x64" ;;
    aarch64) ARCH_NAME="arm64" ;;
    *) ARCH_NAME="$ARCH" ;;
esac

PLATFORM="linux-$ARCH_NAME"

# Create archive name
ARCHIVE_NAME="memrl-v$VERSION-$PLATFORM"
ZIP_PATH="$OUTPUT_PATH/$ARCHIVE_NAME.zip"

echo "🗜️  Creating archive: $ARCHIVE_NAME.zip"

# Remove existing archive if present
rm -f "$ZIP_PATH"

# Create zip archive
(cd "$STAGING_DIR" && zip -9 "$ZIP_PATH" *)

# Calculate checksum
echo "🔐 Calculating SHA256 checksum..."
HASH=$(sha256sum "$ZIP_PATH" | cut -d' ' -f1)
CHECKSUM_PATH="$OUTPUT_PATH/$ARCHIVE_NAME.sha256"
echo "$HASH  $ARCHIVE_NAME.zip" > "$CHECKSUM_PATH"

# Get file size in MB
SIZE=$(du -m "$ZIP_PATH" | cut -f1)

# Display results
echo ""
echo "✅ Release package created successfully!"
echo ""
echo "📦 Archive: $ZIP_PATH"
echo "📏 Size: ${SIZE} MB"
echo "🔐 SHA256: $HASH"
echo ""
echo "Contents:"
unzip -l "$ZIP_PATH" | tail -n +4 | head -n -2 | awk '{print "  - " $4}'
echo ""
echo "📤 Ready to upload to GitHub release!"
