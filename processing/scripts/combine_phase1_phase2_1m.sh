#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

PHASE1_DIR="${PHASE1_DIR:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
PHASE2_DIR="${PHASE2_DIR:-processing/second_phase_web_crawling/outputs/stage2_all4_t60k}"
OUTPUT_DIR="${OUTPUT_DIR:-processing/outputs/combined_phase1_phase2_1m}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/merge_summary.json}"

TOTAL_DOCS="${TOTAL_DOCS:-1000000}"
MIN_PHASE2_SHARE="${MIN_PHASE2_SHARE:-0.40}"
SEED="${SEED:-42}"

TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

DISABLE_DEDUP="${DISABLE_DEDUP:-0}"
ALLOW_LOWER_PHASE2_SHARE="${ALLOW_LOWER_PHASE2_SHARE:-0}"

CMD=(
  "$PYTHON_BIN" -m processing.combine_phase_benchmarks
  --phase1-dir "$PHASE1_DIR"
  --phase2-dir "$PHASE2_DIR"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --total-docs "$TOTAL_DOCS"
  --min-phase2-share "$MIN_PHASE2_SHARE"
  --seed "$SEED"
  --train-ratio "$TRAIN_RATIO"
  --dev-ratio "$DEV_RATIO"
  --test-ratio "$TEST_RATIO"
  --log-level INFO
)

if [[ "$DISABLE_DEDUP" == "1" ]]; then
  CMD+=(--disable-dedup)
fi

if [[ "$ALLOW_LOWER_PHASE2_SHARE" == "1" ]]; then
  CMD+=(--allow-lower-phase2-share)
fi

"${CMD[@]}"

echo "Done."
echo "Output: $OUTPUT_DIR"
echo "Report: $REPORT_PATH"
