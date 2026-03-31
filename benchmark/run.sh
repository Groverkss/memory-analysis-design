#!/usr/bin/env bash
# Benchmark all library kernels on a HIP GPU.
#
# Usage:
#   ./run.sh [--target gfx942] [--build-dir /path/to/iree/build]
#
# Outputs:
#   results/<target>/report.txt   — human-readable report
#   results/<target>/raw.json     — raw benchmark data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="${SCRIPT_DIR}/../library_kernels"
BUILD_DIR="${SCRIPT_DIR}/../../"   # default: iree/build
TARGET=""
MIN_TIME="1s"

# Parse args.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --min-time) MIN_TIME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

IREE_COMPILE="${BUILD_DIR}/tools/iree-compile"
IREE_BENCHMARK="${BUILD_DIR}/tools/iree-benchmark-module"
CONVERTER="${SCRIPT_DIR}/convert_dispatch_to_module.py"

# Validate tools exist.
for tool in "$IREE_COMPILE" "$IREE_BENCHMARK"; do
  if [[ ! -x "$tool" ]]; then
    echo "ERROR: $tool not found. Build it first."
    exit 1
  fi
done

# ── Auto-detect GPU target ──────────────────────────────────────────────────
if [[ -z "$TARGET" ]]; then
  if command -v rocminfo &>/dev/null; then
    TARGET=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || true)
  fi
  if [[ -z "$TARGET" ]]; then
    echo "ERROR: Could not detect GPU target. Pass --target gfxNNNN"
    exit 1
  fi
  echo "Auto-detected GPU target: ${TARGET}"
fi

# ── Peak bandwidth measurement ──────────────────────────────────────────────
PEAK_BW_BINARY="${SCRIPT_DIR}/peak_bandwidth_${TARGET}"
PEAK_BW_GBPS=""

if command -v hipcc &>/dev/null; then
  echo "=== Building peak bandwidth test ==="
  hipcc --offload-arch="${TARGET}" -O3 \
    "${SCRIPT_DIR}/peak_bandwidth.hip" -o "${PEAK_BW_BINARY}" 2>&1

  echo "=== Measuring peak bandwidth ==="
  PEAK_OUTPUT=$("${PEAK_BW_BINARY}" 2>&1)
  echo "$PEAK_OUTPUT"
  PEAK_BW_GBPS=$(echo "$PEAK_OUTPUT" | grep -oP 'PEAK_BW_GBPS=\K[\d.]+')
  echo ""
else
  echo "WARNING: hipcc not found, skipping peak bandwidth measurement"
fi

# ── Prepare output directories ──────────────────────────────────────────────
RESULTS_DIR="${SCRIPT_DIR}/results/${TARGET}"
WORK_DIR="${SCRIPT_DIR}/.work/${TARGET}"
mkdir -p "${RESULTS_DIR}" "${WORK_DIR}/modules" "${WORK_DIR}/vmfb"

# ── Convert and compile all kernels ─────────────────────────────────────────
echo "=== Converting and compiling kernels for ${TARGET} ==="

KERNEL_LIST=()
FAILED_CONVERT=()
FAILED_COMPILE=()

for mlir in "${KERNEL_DIR}"/*.mlir; do
  name=$(basename "$mlir" .mlir)

  # Convert dispatch IR to module IR.
  module="${WORK_DIR}/modules/${name}.mlir"
  meta="${WORK_DIR}/modules/${name}.json"

  if ! python3 "${CONVERTER}" "$mlir" "$module" "$meta" 2>/dev/null; then
    FAILED_CONVERT+=("$name")
    continue
  fi

  # Compile to vmfb.
  vmfb="${WORK_DIR}/vmfb/${name}.vmfb"
  if ! "${IREE_COMPILE}" \
      --iree-hal-target-backends=rocm \
      --iree-rocm-target="${TARGET}" \
      "$module" -o "$vmfb" 2>/dev/null; then
    FAILED_COMPILE+=("$name")
    continue
  fi

  KERNEL_LIST+=("$name")
  printf "  %-50s OK\n" "$name"
done

echo ""
echo "Converted: ${#KERNEL_LIST[@]}, skipped: ${#FAILED_CONVERT[@]}, compile failed: ${#FAILED_COMPILE[@]}"
if [[ ${#FAILED_COMPILE[@]} -gt 0 ]]; then
  echo "  Compile failures: ${FAILED_COMPILE[*]}"
fi
echo ""

# ── Benchmark each kernel ───────────────────────────────────────────────────
echo "=== Running benchmarks ==="

RAW_JSON="${RESULTS_DIR}/raw.json"
echo "[" > "$RAW_JSON"
FIRST=true

for name in "${KERNEL_LIST[@]}"; do
  vmfb="${WORK_DIR}/vmfb/${name}.vmfb"
  meta="${WORK_DIR}/modules/${name}.json"
  func_name=$(python3 -c "import json; print(json.load(open('$meta'))['name'])")
  bytes_moved=$(python3 -c "import json; print(json.load(open('$meta'))['bytes_moved'])")

  # Build --input flags from metadata.
  input_flags=$(python3 -c "
import json
meta = json.load(open('$meta'))
for inp in meta['inputs']:
    print(f'--input={inp[\"shape\"]}')
")

  # Run benchmark.
  bench_output=$("${IREE_BENCHMARK}" \
    --module="${vmfb}" \
    --function="${func_name}" \
    ${input_flags} \
    --device=hip \
    --benchmark_format=json \
    --benchmark_min_time="${MIN_TIME}" \
    2>/dev/null) || {
    printf "  %-50s FAILED\n" "$name"
    continue
  }

  # Extract median time in ns from benchmark JSON.
  time_ns=$(echo "$bench_output" | python3 -c "
import json, sys
data = json.load(sys.stdin)
bms = data.get('benchmarks', [])
# Prefer the median aggregate if available.
median = [b for b in bms if b.get('aggregate_name') == 'median']
if median:
    print(int(median[0]['real_time']))
elif bms:
    print(int(bms[0]['real_time']))
else:
    print(0)
" 2>/dev/null) || time_ns=0

  if [[ "$time_ns" -eq 0 ]]; then
    printf "  %-50s NO DATA\n" "$name"
    continue
  fi

  time_us=$(python3 -c "print(f'{$time_ns / 1000:.1f}')")
  bw_gbps=$(python3 -c "print(f'{$bytes_moved / ($time_ns / 1e9) / 1e9:.1f}')")

  pct=""
  if [[ -n "$PEAK_BW_GBPS" ]]; then
    pct=$(python3 -c "print(f'{$bw_gbps / $PEAK_BW_GBPS * 100:.0f}')")
    printf "  %-50s %8s us  %8s GB/s  %3s%%\n" "$name" "$time_us" "$bw_gbps" "$pct"
  else
    printf "  %-50s %8s us  %8s GB/s\n" "$name" "$time_us" "$bw_gbps"
  fi

  # Write JSON record.
  if [[ "$FIRST" == "true" ]]; then
    FIRST=false
  else
    echo "," >> "$RAW_JSON"
  fi
  cat >> "$RAW_JSON" <<ENTRY
  {"name": "$name", "time_ns": $time_ns, "bytes_moved": $bytes_moved, "bw_gbps": $bw_gbps, "pct_peak": ${pct:-null}}
ENTRY

done

echo "]" >> "$RAW_JSON"

# ── Generate report ─────────────────────────────────────────────────────────
echo ""
echo "=== Generating report ==="

python3 "${SCRIPT_DIR}/generate_report.py" \
  --raw "${RAW_JSON}" \
  --target "${TARGET}" \
  --peak-bw "${PEAK_BW_GBPS:-0}" \
  --output "${RESULTS_DIR}/report.txt"

echo "Report: ${RESULTS_DIR}/report.txt"
echo "Raw data: ${RAW_JSON}"
