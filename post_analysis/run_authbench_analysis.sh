#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${DATASET_DIR:-processing/outputs/authbench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-post_analysis/outputs/authbench}"
SPLITS="${SPLITS:-all}"
PHASE1_DIR="${PHASE1_DIR:-processing/outputs/pipeline_phase1_official}"
PHASE2_DIR="${PHASE2_DIR:-processing/second_phase_web_crawling/outputs/pipeline_phase2_official}"

read -r -a SPLIT_ARGS <<<"${SPLITS}"

echo "Running AuthBench post-analysis"
echo "  dataset: ${DATASET_DIR}"
echo "  output:  ${OUTPUT_ROOT}"
echo "  splits:  ${SPLITS}"

"${PYTHON_BIN}" -m post_analysis.analyze_dataset \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_ROOT}/statistics" \
  --splits "${SPLIT_ARGS[@]}"

"${PYTHON_BIN}" -m post_analysis.qualitative_analysis \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_ROOT}/qualitative" \
  --splits "${SPLIT_ARGS[@]}"

"${PYTHON_BIN}" -m post_analysis.authorship_benchmark_analysis \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_ROOT}/benchmark_profile" \
  --splits "${SPLIT_ARGS[@]}" \
  --phase1-dir "${PHASE1_DIR}" \
  --phase2-dir "${PHASE2_DIR}"

echo "AuthBench analysis complete."
echo "Artifacts written to ${OUTPUT_ROOT}"
