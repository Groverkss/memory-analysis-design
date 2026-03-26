#!/bin/bash
# Run coalescing benchmark on the current machine.
# Auto-detects GPU arch. Saves results to <hostname>_<gpu>.txt
#
# Usage:
#   ./run_bench.sh              # auto-detect GPU
#   ./run_bench.sh gfx1201      # override arch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Detect GPU arch ---
if [[ $# -ge 1 ]]; then
  GPU_ARCH="$1"
else
  GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || true)
  if [[ -z "$GPU_ARCH" ]]; then
    echo "ERROR: Could not detect GPU arch. Pass it as argument: ./run_bench.sh gfx1201"
    exit 1
  fi
fi

# --- Gather system info ---
HOSTNAME=$(hostname)
GPU_NAME=$(rocminfo 2>/dev/null | grep -A2 "Name:.*$GPU_ARCH" | grep "Marketing Name" | head -1 | sed 's/.*: *//' | xargs || echo "unknown")
HIPCC=${HIPCC:-hipcc}
HIPCC_VERSION=$($HIPCC --version 2>&1 | grep "HIP version" | head -1 || echo "unknown")
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")

OUTFILE="${HOSTNAME}_${GPU_ARCH}.txt"

echo "=== Coalescing Benchmark ==="
echo "Host:       $HOSTNAME"
echo "GPU:        $GPU_NAME ($GPU_ARCH)"
echo "ROCm:       $ROCM_VERSION"
echo "hipcc:      $HIPCC_VERSION"
echo "Config:     50 iterations per kernel, median reported, cache flushed between iterations"
echo "Output:     $OUTFILE"
echo ""

# --- Build ---
echo "Building for $GPU_ARCH..."
make clean >/dev/null 2>&1 || true
make GPU_ARCH="$GPU_ARCH" HIPCC="$HIPCC" 2>&1
echo ""

# --- Run ---
echo "Running benchmark..."
TMPFILE=$(mktemp)
./coalesce_bench > "$TMPFILE" 2>&1

# --- Write output ---
{
  echo "=== Coalescing Benchmark Results ==="
  echo "Date:       $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "Host:       $HOSTNAME"
  echo "GPU:        $GPU_NAME ($GPU_ARCH)"
  echo "ROCm:       $ROCM_VERSION"
  echo "hipcc:      $HIPCC_VERSION"
  echo "Config:     50 iterations per kernel, median reported, cache flushed between iterations"
  echo ""
  cat "$TMPFILE"
} > "$OUTFILE"

rm -f "$TMPFILE"

echo ""
echo "Results written to $OUTFILE"
echo ""
cat "$OUTFILE"
