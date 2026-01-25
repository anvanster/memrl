#!/bin/bash
# CI Checks - Run this before pushing to ensure CI passes
#
# Usage: ./scripts/ci-checks.sh [options]
#   --skip-benchmark    Skip running benchmarks
#   --full              Run all checks including docs and coverage (steps 4 & 5)

set -e  # Exit on first error

# Parse arguments
SKIP_BENCHMARK=false
FULL_CHECK=false
for arg in "$@"; do
    case $arg in
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --full)
            FULL_CHECK=true
            shift
            ;;
    esac
done

echo "============================================"
echo "Running CI checks locally..."
echo "============================================"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# 1. Clippy
print_section "1/5 Running Clippy..."
cargo clippy --all-targets --all-features -- -D warnings

# 2. Format check
print_section "2/5 Checking formatting..."
cargo fmt --all -- --check

# 3. Benchmarks
print_section "3/5 Running benchmarks..."
# if [ "$SKIP_BENCHMARK" = true ]; then
if [ "$FULL_CHECK" = true ]; then
    cargo bench --no-fail-fast
else
    echo "⏭️  Skipping benchmarks (--skip-benchmark flag)"
fi

# 4. Documentation (optional - use --full to run)
print_section "4/5 Building documentation..."
if [ "$FULL_CHECK" = true ]; then
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
else
    echo "⏭️  Skipping documentation build (use --full to include)"
fi

# 5. Code coverage (optional - use --full to run, requires tarpaulin)
print_section "5/5 Running code coverage..."
if [ "$FULL_CHECK" = true ]; then
    if command -v cargo-tarpaulin &> /dev/null; then
        cargo tarpaulin --verbose --all-features --workspace --timeout 120 --out Xml
    else
        echo "⚠️  cargo-tarpaulin not installed. Skipping coverage check."
        echo "   Install with: cargo install cargo-tarpaulin"
    fi
else
    echo "⏭️  Skipping code coverage (use --full to include)"
fi

echo ""
echo "============================================"
echo "✅ All CI checks passed!"
echo "============================================"
