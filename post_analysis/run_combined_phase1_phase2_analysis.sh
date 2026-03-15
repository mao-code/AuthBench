#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-combined_phase1_phase2}"
DATASET_DIR="${DATASET_DIR:-processing/outputs/${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-post_analysis/outputs/${RUN_NAME}}"

"$PYTHON_BIN" -m post_analysis.analyze_dataset \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "${OUTPUT_ROOT}/statistics" \
  --splits all

"$PYTHON_BIN" -m post_analysis.qualitative_analysis \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "${OUTPUT_ROOT}/qualitative" \
  --splits all

"$PYTHON_BIN" -m post_analysis.authorship_benchmark_analysis \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "${OUTPUT_ROOT}/benchmark_profile" \
  --splits all

echo "Analysis complete."
echo "Dataset: $DATASET_DIR"
echo "Statistics: ${OUTPUT_ROOT}/statistics"
echo "Qualitative: ${OUTPUT_ROOT}/qualitative"
echo "Benchmark profile: ${OUTPUT_ROOT}/benchmark_profile"
