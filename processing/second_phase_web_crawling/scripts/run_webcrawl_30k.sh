#!/usr/bin/env bash
set -euo pipefail

# Run second-phase web crawling + monitor/build/postprocess at 30K target.
# This keeps the same core logic as your original AuthBench pipeline:
# - processing.monitor_pipeline
# - processing.build_benchmark
# - processing.postprocess

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

TARGET_TOTAL="${TARGET_TOTAL:-30000}"
POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-30000}"
SEED="${SEED:-42}"

# StackExchange mode:
# - api: uses Stack Exchange API (good for smoke/medium runs, quota-limited)
# - archive: expects/local-downloads legacy dump archives under downloads/stackexchange
STACKEXCHANGE_MODE="${STACKEXCHANGE_MODE:-api}"
STACKEXCHANGE_SITES="${STACKEXCHANGE_SITES:-stackoverflow.com,es.stackoverflow.com,ru.stackoverflow.com,pt.stackoverflow.com,superuser.com,serverfault.com,mathoverflow.net}"
STACKEXCHANGE_MAX_POSTS_PER_SITE="${STACKEXCHANGE_MAX_POSTS_PER_SITE:-20000}"
STACKEXCHANGE_SKIP_COMMENTS="${STACKEXCHANGE_SKIP_COMMENTS:-1}"

# Gutenberg and Wikisource crawling caps (raw corpus caps before AuthBench filtering/sampling)
GUTENBERG_MAX_DOCS="${GUTENBERG_MAX_DOCS:-180000}"
GUTENBERG_LANGUAGES="${GUTENBERG_LANGUAGES:-en,zh,es,ar,fr,ru,de,ja,ko,hi,it,pt,nl,sv,fi,pl,la,el,cs,hu,da,no,ro,tr,ca,eo}"
WIKISOURCE_WIKIS="${WIKISOURCE_WIKIS:-enwikisource,zhwikisource,eswikisource,arwikisource,frwikisource,ruwikisource,dewikisource,jawikisource,kowikisource,hiwikisource}"
WIKISOURCE_MAX_DOCS_PER_WIKI="${WIKISOURCE_MAX_DOCS_PER_WIKI:-20000}"
WIKISOURCE_MAX_TOTAL_DOCS="${WIKISOURCE_MAX_TOTAL_DOCS:-220000}"

# Keep these aligned with your original pipeline behavior.
MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-10000000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
CHUNK_PROBABILITY="${CHUNK_PROBABILITY:-0.7}"
TRUNCATE_TO_TOKENS="${TRUNCATE_TO_TOKENS:-2000}"

MANIFEST_PATH="${MANIFEST_PATH:-processing/second_phase_web_crawling/datasets_manifest.json}"
RUN_TAG="${RUN_TAG:-all3_t30k}"
MONITOR_REPORT_PATH="${MONITOR_REPORT_PATH:-processing/second_phase_web_crawling/outputs/monitoring/pipeline_dynamics_${RUN_TAG}.json}"
BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-processing/second_phase_web_crawling/outputs/stage1_${RUN_TAG}}"
POSTPROCESS_OUTPUT_DIR="${POSTPROCESS_OUTPUT_DIR:-processing/second_phase_web_crawling/outputs/stage2_${RUN_TAG}}"

echo "[1/4] Preparing virtual environment at ${VENV_DIR}"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip >/dev/null
python -m pip install -r requirements.txt >/dev/null
python -m pip install py7zr >/dev/null

echo "[2/4] Running crawl + monitor + build + postprocess (target=${TARGET_TOTAL})"

CMD=(
  python -m processing.second_phase_web_crawling.run_pipeline
  --stages crawl monitor build postprocess
  --manifest-path "$MANIFEST_PATH"
  --stackexchange-download-mode "$STACKEXCHANGE_MODE"
  --stackexchange-sites "$STACKEXCHANGE_SITES"
  --stackexchange-max-posts-per-site "$STACKEXCHANGE_MAX_POSTS_PER_SITE"
  --gutenberg-max-docs "$GUTENBERG_MAX_DOCS"
  --gutenberg-languages "$GUTENBERG_LANGUAGES"
  --wikisource-wikis "$WIKISOURCE_WIKIS"
  --wikisource-max-docs-per-wiki "$WIKISOURCE_MAX_DOCS_PER_WIKI"
  --wikisource-max-total-docs "$WIKISOURCE_MAX_TOTAL_DOCS"
  --monitor-report-path "$MONITOR_REPORT_PATH"
  --monitor-overwrite
  --build-output-dir "$BUILD_OUTPUT_DIR"
  --postprocess-output-dir "$POSTPROCESS_OUTPUT_DIR"
  --total-docs "$TARGET_TOTAL"
  --post-target-total "$POST_TARGET_TOTAL"
  --allow-other-languages
  --max-documents-per-dataset "$MAX_DOCUMENTS_PER_DATASET"
  --shuffle-buffer-size "$SHUFFLE_BUFFER_SIZE"
  --chunk-probability "$CHUNK_PROBABILITY"
  --truncate-to-tokens "$TRUNCATE_TO_TOKENS"
  --seed "$SEED"
  --log-level INFO
)

if [[ "$STACKEXCHANGE_SKIP_COMMENTS" == "1" ]]; then
  CMD+=(--stackexchange-skip-comments)
fi

"${CMD[@]}"

echo "[3/4] Summaries"
echo "Monitor:      $MONITOR_REPORT_PATH"
echo "Stage1:       $BUILD_OUTPUT_DIR"
echo "Stage2:       $POSTPROCESS_OUTPUT_DIR"

if [[ -f "$BUILD_OUTPUT_DIR/processing_summary.json" ]]; then
  echo ""
  echo "Stage1 processing summary:"
  jq '{after_author_filter: .after_author_filter.total, after_sampling: .after_sampling.total, splits: {train: .splits.train.total, dev: .splits.dev.total, test: .splits.test.total}}' "$BUILD_OUTPUT_DIR/processing_summary.json"
fi

if [[ -f "$POSTPROCESS_OUTPUT_DIR/postprocessing_summary.json" ]]; then
  echo ""
  echo "Stage2 postprocessing summary:"
  jq '{before_filter: .before_filter.total, after_filter: .after_filter.total, after_sampling: .after_sampling.total, split_candidates: {train: .splits.train.candidates, dev: .splits.dev.candidates, test: .splits.test.candidates}}' "$POSTPROCESS_OUTPUT_DIR/postprocessing_summary.json"
fi

echo "[4/4] Done"

