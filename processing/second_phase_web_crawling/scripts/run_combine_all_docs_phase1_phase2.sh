#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

PHASE1_DIR="${PHASE1_DIR:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
PHASE2_DIR="${PHASE2_DIR:-processing/second_phase_web_crawling/outputs/pipeline_all4_t300k_cap10M}"

RUN_TAG="${RUN_TAG:-phase1_official_plus_phase2_all4_all_docs}"
OUTPUT_DIR="${OUTPUT_DIR:-processing/outputs/combined_${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/merge_summary.json}"

SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

# Combine all loaded documents while still using rebalance/sampling logic by default.
TAKE_ALL_DOCS="${TAKE_ALL_DOCS:-0}"
DISABLE_DEDUP="${DISABLE_DEDUP:-1}"
DISABLE_CROSS_PHASE_OVERLAP_REMOVAL="${DISABLE_CROSS_PHASE_OVERLAP_REMOVAL:-1}"
MIN_PHASE2_SHARE="${MIN_PHASE2_SHARE:-0.5}"
ALLOW_LOWER_PHASE2_SHARE="${ALLOW_LOWER_PHASE2_SHARE:-1}"

# Optional hard cap. Leave empty to keep all docs.
TOTAL_DOCS="${TOTAL_DOCS:-}"
EXACT_50_50="${EXACT_50_50:-1}"

if [[ "$EXACT_50_50" == "1" && "$TAKE_ALL_DOCS" != "1" && -z "$TOTAL_DOCS" ]]; then
  TOTAL_DOCS="$("$PYTHON_BIN" - "$PHASE1_DIR" "$PHASE2_DIR" <<'PY'
from pathlib import Path
import sys
from processing.postprocess import _read_stage_documents

p1 = Path(sys.argv[1])
p2 = Path(sys.argv[2])
n1 = len(_read_stage_documents(p1))
n2 = len(_read_stage_documents(p2))
print(2 * min(n1, n2))
PY
)"
  echo "EXACT_50_50 enabled: auto-set TOTAL_DOCS=${TOTAL_DOCS} (2 * min(phase1, phase2))."
fi

CMD=(
  "$PYTHON_BIN" -m processing.combine_phase_benchmarks
  --phase1-dir "$PHASE1_DIR"
  --phase2-dir "$PHASE2_DIR"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --seed "$SEED"
  --train-ratio "$TRAIN_RATIO"
  --dev-ratio "$DEV_RATIO"
  --test-ratio "$TEST_RATIO"
  --min-phase2-share "$MIN_PHASE2_SHARE"
  --log-level INFO
)

if [[ "$TAKE_ALL_DOCS" == "1" ]]; then
  CMD+=(--take-all-docs)
fi
if [[ "$DISABLE_DEDUP" == "1" ]]; then
  CMD+=(--disable-dedup)
fi
if [[ "$DISABLE_CROSS_PHASE_OVERLAP_REMOVAL" == "1" ]]; then
  CMD+=(--disable-cross-phase-overlap-removal)
fi
if [[ "$ALLOW_LOWER_PHASE2_SHARE" == "1" ]]; then
  CMD+=(--allow-lower-phase2-share)
fi
if [[ -n "$TOTAL_DOCS" ]]; then
  CMD+=(--total-docs "$TOTAL_DOCS")
fi

"${CMD[@]}"

echo "Done."
echo "Output: $OUTPUT_DIR"
echo "Report: $REPORT_PATH"

if command -v jq >/dev/null 2>&1 && [[ -f "$REPORT_PATH" ]]; then
  echo ""
  echo "Merge summary:"
  jq '{effective_total_docs: .inputs.effective_total_docs, phase1_selected: .stage_counts.phase1_selected, phase2_selected: .stage_counts.phase2_selected, phase2_share_final: .stage_counts.phase2_share_final, split_documents: {train: .splits.train.documents, dev: .splits.dev.documents, test: .splits.test.documents}}' "$REPORT_PATH"
fi
