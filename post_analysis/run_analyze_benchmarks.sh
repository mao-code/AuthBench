#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_A_DIR="${DATASET_A_DIR:-processing/outputs/pipeline_phase1_official}"
DATASET_B_DIR="${DATASET_B_DIR:-processing/second_phase_web_crawling/outputs/pipeline_phase2_official}"
DATASET_A_NAME="${DATASET_A_NAME:-pipeline_phase1_official}"
DATASET_B_NAME="${DATASET_B_NAME:-pipeline_phase2_official}"
OUTPUT_DIR="${OUTPUT_DIR:-post_analysis/outputs/phase1_vs_phase2}"
SEED="${SEED:-42}"

"$PYTHON_BIN" -m post_analysis.analyze_benchmarks \
  --dataset-a-dir "$DATASET_A_DIR" \
  --dataset-b-dir "$DATASET_B_DIR" \
  --dataset-a-name "$DATASET_A_NAME" \
  --dataset-b-name "$DATASET_B_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  "$@"

echo "Benchmark analysis complete."
echo "Outputs: $OUTPUT_DIR"
