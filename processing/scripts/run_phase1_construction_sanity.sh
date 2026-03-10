#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}"

export RUN_TAG="${RUN_TAG:-phase1_sanity}"
export TOTAL_DOCS="${TOTAL_DOCS:-10000}"
export POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-$TOTAL_DOCS}"
export SANITY_CHECK="${SANITY_CHECK:-1}"
export SANITY_LIMIT="${SANITY_LIMIT:-1000}"
export MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-5000}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2000}"

bash "$ROOT_DIR/processing/scripts/run_phase1_construction.sh"
