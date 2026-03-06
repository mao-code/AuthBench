#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

PHASE1_DIR="${PHASE1_DIR:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
PHASE2_DIR="${PHASE2_DIR:-processing/second_phase_web_crawling/outputs/pipeline_all4_t300k_cap10M}"

RUN_TAG="${RUN_TAG:-retag_combine_600k_50_50}"
OUTPUT_DIR="${OUTPUT_DIR:-processing/outputs/combined_${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/merge_summary.json}"

SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

TARGET_PER_PHASE_DOCS="${TARGET_PER_PHASE_DOCS:-300000}"
STRICT_EXPECTED_COUNTS="${STRICT_EXPECTED_COUNTS:-1}"

# Keep all docs from each side up to TARGET_PER_PHASE_DOCS and disable drop-heavy steps.
DISABLE_DEDUP="${DISABLE_DEDUP:-1}"
DISABLE_CROSS_PHASE_OVERLAP_REMOVAL="${DISABLE_CROSS_PHASE_OVERLAP_REMOVAL:-1}"
ALLOW_LOWER_PHASE2_SHARE="${ALLOW_LOWER_PHASE2_SHARE:-0}"

# Language retag controls.
RETAG_LANGUAGES="${RETAG_LANGUAGES:-1}"
LANG_AUDIT_MIN_DETECT_CHARS="${LANG_AUDIT_MIN_DETECT_CHARS:-80}"
LANG_AUDIT_MIN_CONFIDENCE="${LANG_AUDIT_MIN_CONFIDENCE:-0.85}"
LANG_AUDIT_MAX_DETECT_DOCS="${LANG_AUDIT_MAX_DETECT_DOCS:-50000}"

TOTAL_DOCS=$(( TARGET_PER_PHASE_DOCS * 2 ))

read -r PHASE1_AVAILABLE PHASE2_AVAILABLE < <("$PYTHON_BIN" - "$PHASE1_DIR" "$PHASE2_DIR" <<'PY'
from pathlib import Path
import sys
from processing.postprocess import _read_stage_documents

p1 = Path(sys.argv[1])
p2 = Path(sys.argv[2])
print(len(_read_stage_documents(p1)), len(_read_stage_documents(p2)))
PY
)

echo "Available docs: phase1=${PHASE1_AVAILABLE}, phase2=${PHASE2_AVAILABLE}"

if [[ "$STRICT_EXPECTED_COUNTS" == "1" ]]; then
  if (( PHASE1_AVAILABLE < TARGET_PER_PHASE_DOCS )); then
    echo "Error: phase1 has ${PHASE1_AVAILABLE} docs, but ${TARGET_PER_PHASE_DOCS} required for 50/50 ${TOTAL_DOCS}."
    exit 1
  fi
  if (( PHASE2_AVAILABLE < TARGET_PER_PHASE_DOCS )); then
    echo "Error: phase2 has ${PHASE2_AVAILABLE} docs, but ${TARGET_PER_PHASE_DOCS} required for 50/50 ${TOTAL_DOCS}."
    exit 1
  fi
fi

# Best-effort fallback if strict mode is off: still enforce exact 50/50 from the smaller pool.
if [[ "$STRICT_EXPECTED_COUNTS" != "1" ]]; then
  MIN_POOL="$PHASE1_AVAILABLE"
  if (( PHASE2_AVAILABLE < MIN_POOL )); then
    MIN_POOL="$PHASE2_AVAILABLE"
  fi
  if (( MIN_POOL < TARGET_PER_PHASE_DOCS )); then
    TARGET_PER_PHASE_DOCS="$MIN_POOL"
    TOTAL_DOCS=$(( TARGET_PER_PHASE_DOCS * 2 ))
    echo "Adjusted TARGET_PER_PHASE_DOCS to ${TARGET_PER_PHASE_DOCS}; TOTAL_DOCS=${TOTAL_DOCS} for exact 50/50."
  fi
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
  --total-docs "$TOTAL_DOCS"
  --min-phase2-share 0.5
  --log-level INFO
)

if [[ "$DISABLE_DEDUP" == "1" ]]; then
  CMD+=(--disable-dedup)
fi
if [[ "$DISABLE_CROSS_PHASE_OVERLAP_REMOVAL" == "1" ]]; then
  CMD+=(--disable-cross-phase-overlap-removal)
fi
if [[ "$ALLOW_LOWER_PHASE2_SHARE" == "1" ]]; then
  CMD+=(--allow-lower-phase2-share)
fi
if [[ "$RETAG_LANGUAGES" == "1" ]]; then
  CMD+=(
    --retag-languages
    --lang-audit-min-detect-chars "$LANG_AUDIT_MIN_DETECT_CHARS"
    --lang-audit-min-confidence "$LANG_AUDIT_MIN_CONFIDENCE"
    --lang-audit-max-detect-docs "$LANG_AUDIT_MAX_DETECT_DOCS"
  )
fi

"${CMD[@]}"

echo "Done."
echo "Output: $OUTPUT_DIR"
echo "Report: $REPORT_PATH"

if command -v jq >/dev/null 2>&1 && [[ -f "$REPORT_PATH" ]]; then
  echo ""
  echo "Merge summary:"
  jq '{effective_total_docs: .inputs.effective_total_docs, phase1_selected: .stage_counts.phase1_selected, phase2_selected: .stage_counts.phase2_selected, phase2_share_final: .stage_counts.phase2_share_final, phase1_retagged: .phase1_language_audit.retagged_docs, phase2_retagged: .phase2_language_audit.retagged_docs}' "$REPORT_PATH"
fi
