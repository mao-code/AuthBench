#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-processing/datasets_manifest.json}"
RUN_TAG="${RUN_TAG:-phase1_official}"

TOTAL_DOCS="${TOTAL_DOCS:-300000}"
POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-}"
SEED="${SEED:-42}"

MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-10000000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
CHUNK_PROBABILITY="${CHUNK_PROBABILITY:-0.7}"
TRUNCATE_TO_TOKENS="${TRUNCATE_TO_TOKENS:-2000}"
ALLOW_OTHER_LANGUAGES="${ALLOW_OTHER_LANGUAGES:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-processing/outputs/pipeline_${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/pipeline_dynamics.json}"

DATASET_MAX_DOCS="${DATASET_MAX_DOCS:-}"
NO_SHUFFLE_DATASETS="${NO_SHUFFLE_DATASETS:-}"

CMD=(
  "$PYTHON_BIN" -m processing.construct_benchmark
  --manifest "$MANIFEST_PATH"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --overwrite-report
  --total-docs "$TOTAL_DOCS"
  --seed "$SEED"
  --max-documents-per-dataset "$MAX_DOCUMENTS_PER_DATASET"
  --shuffle-buffer-size "$SHUFFLE_BUFFER_SIZE"
  --chunk-probability "$CHUNK_PROBABILITY"
  --truncate-to-tokens "$TRUNCATE_TO_TOKENS"
  --log-level INFO
)

if [[ "$ALLOW_OTHER_LANGUAGES" == "1" ]]; then
  CMD+=(--allow-other-languages)
fi

if [[ -n "$POST_TARGET_TOTAL" ]]; then
  CMD+=(--post-target-total "$POST_TARGET_TOTAL")
fi

if [[ -n "$DATASET_MAX_DOCS" ]]; then
  read -r -a _CAPS <<<"$DATASET_MAX_DOCS"
  CMD+=(--dataset-max-docs "${_CAPS[@]}")
fi

if [[ -n "$NO_SHUFFLE_DATASETS" ]]; then
  read -r -a _NOSHUFFLE <<<"$NO_SHUFFLE_DATASETS"
  CMD+=(--no-shuffle-datasets "${_NOSHUFFLE[@]}")
fi

"${CMD[@]}"

echo "Done."
echo "Output: $OUTPUT_DIR"
echo "Report: $REPORT_PATH"
