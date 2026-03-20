#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${ROOT_DIR}"
# sbatch -p rush --nodelist=rush-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G -t 720:00:00 processing/scripts/run_phase1_construction.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-processing/datasets_manifest.json}"
RUN_TAG="${RUN_TAG:-phase1_official}"

TOTAL_DOCS="${TOTAL_DOCS:-600000}"
POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-$TOTAL_DOCS}"
SEED="${SEED:-42}"
SANITY_CHECK="${SANITY_CHECK:-0}"
SANITY_LIMIT="${SANITY_LIMIT:-2000}"

MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-10000000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
CHUNK_PROBABILITY="${CHUNK_PROBABILITY:-0.7}"
TRUNCATE_TO_TOKENS="${TRUNCATE_TO_TOKENS:-2000}"
ALLOW_OTHER_LANGUAGES="${ALLOW_OTHER_LANGUAGES:-1}"
DISABLE_LANG_AUDIT="${DISABLE_LANG_AUDIT:-0}"
LANG_AUDIT_DROP_DETECTED_MISMATCHES="${LANG_AUDIT_DROP_DETECTED_MISMATCHES:-0}"
LANG_AUDIT_MAX_DETECT_DOCS="${LANG_AUDIT_MAX_DETECT_DOCS:-50000}"
LANG_AUDIT_MAX_SUSPECTS="${LANG_AUDIT_MAX_SUSPECTS:-5000}"

OUTPUT_DIR="${OUTPUT_DIR:-processing/outputs/pipeline_${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/pipeline_dynamics.json}"

DEFAULT_PD_BOOK_DATASETS="french_pd_books german_pd russian_pd spanish_pd_books"
DEFAULT_PD_BOOK_CAPS="french_pd_books=10000 german_pd=10000 russian_pd=10000 spanish_pd_books=10000"

DATASET_MAX_DOCS="${DATASET_MAX_DOCS:-$DEFAULT_PD_BOOK_CAPS}"
NO_SHUFFLE_DATASETS="${NO_SHUFFLE_DATASETS:-$DEFAULT_PD_BOOK_DATASETS}"

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
  --lang-audit-max-detect-docs "$LANG_AUDIT_MAX_DETECT_DOCS"
  --lang-audit-max-suspects "$LANG_AUDIT_MAX_SUSPECTS"
  --log-level INFO
)

if [[ "$ALLOW_OTHER_LANGUAGES" == "1" ]]; then
  CMD+=(--allow-other-languages)
fi

if [[ "$SANITY_CHECK" == "1" ]]; then
  CMD+=(--sanity-check --sanity-limit "$SANITY_LIMIT")
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

if [[ "$DISABLE_LANG_AUDIT" == "1" ]]; then
  CMD+=(--disable-lang-audit)
fi

if [[ "$LANG_AUDIT_DROP_DETECTED_MISMATCHES" == "1" ]]; then
  CMD+=(--lang-audit-drop-detected-mismatches)
fi

"${CMD[@]}"

echo "Done."
echo "Output: $OUTPUT_DIR"
echo "Report: $REPORT_PATH"
