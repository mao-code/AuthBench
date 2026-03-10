#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-phase1_official_plus_phase2_all4_all_docs}"
DATASET_DIR="${DATASET_DIR:-processing/outputs/combined_${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-post_analysis/outputs/${RUN_TAG}}"

"$PYTHON_BIN" post_analysis/analyze_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "${OUTPUT_ROOT}/statistics" \
  --splits all

"$PYTHON_BIN" post_analysis/qualitative_analysis.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "${OUTPUT_ROOT}/qualitative" \
  --splits all

echo "Analysis complete."
echo "Dataset: $DATASET_DIR"
echo "Statistics: ${OUTPUT_ROOT}/statistics"
echo "Qualitative: ${OUTPUT_ROOT}/qualitative"
