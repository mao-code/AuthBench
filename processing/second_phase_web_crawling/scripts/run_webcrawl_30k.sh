#!/usr/bin/env bash
set -euo pipefail

# Run second-phase web crawling + unified construct pipeline at 30K target.
# This keeps the same core logic as your original AuthBench pipeline:
# - processing.construct_benchmark (build + postprocess + dedup + report)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Load local StackExchange key if present and not already exported.
if [[ -z "${STACKEXCHANGE_API_KEY:-}" && -f "$ROOT_DIR/.env.stackexchange" ]]; then
  set -a
  source "$ROOT_DIR/.env.stackexchange"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

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

# Restart toggles (useful if one source is already complete):
# - set SKIP_STACKEXCHANGE=1 to reuse existing stackexchange.jsonl and only rerun others.
# - set SKIP_GUTENBERG=1 or SKIP_WIKISOURCE=1 similarly.
SKIP_STACKEXCHANGE="${SKIP_STACKEXCHANGE:-0}"
SKIP_GUTENBERG="${SKIP_GUTENBERG:-0}"
SKIP_WIKISOURCE="${SKIP_WIKISOURCE:-0}"
SKIP_YTCOMMENTS="${SKIP_YTCOMMENTS:-1}"

# Auto-ramp crawl caps until stage1 after_sampling reaches target.
# Set AUTO_RAMP=0 to run a single round.
AUTO_RAMP="${AUTO_RAMP:-1}"
RAMP_MAX_ROUNDS="${RAMP_MAX_ROUNDS:-6}"
RAMP_FACTOR="${RAMP_FACTOR:-2}"
STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX="${STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX:-200000}"
GUTENBERG_MAX_DOCS_MAX="${GUTENBERG_MAX_DOCS_MAX:-600000}"
WIKISOURCE_MAX_DOCS_PER_WIKI_MAX="${WIKISOURCE_MAX_DOCS_PER_WIKI_MAX:-120000}"
WIKISOURCE_MAX_TOTAL_DOCS_MAX="${WIKISOURCE_MAX_TOTAL_DOCS_MAX:-1200000}"

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

if ! "$PYTHON_BIN" -c "import langdetect" >/dev/null 2>&1; then
  echo "Installing missing dependency: langdetect"
  "$PYTHON_BIN" -m pip install langdetect
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required by this script for summary/ramp parsing."
  exit 1
fi

bump_cap() {
  local current="$1"
  local max_cap="$2"
  local next=$(( current * RAMP_FACTOR ))
  if (( next > max_cap )); then
    next="$max_cap"
  fi
  echo "$next"
}

SE_CAP="$STACKEXCHANGE_MAX_POSTS_PER_SITE"
GB_CAP="$GUTENBERG_MAX_DOCS"
WS_CAP_PER_WIKI="$WIKISOURCE_MAX_DOCS_PER_WIKI"
WS_CAP_TOTAL="$WIKISOURCE_MAX_TOTAL_DOCS"

ROUND=1
FINAL_AFTER_SAMPLING=0

echo "[1/3] Running crawl + construct (target=${TARGET_TOTAL})"
echo "skip flags: stackexchange=${SKIP_STACKEXCHANGE} gutenberg=${SKIP_GUTENBERG} wikisource=${SKIP_WIKISOURCE} ytcomments=${SKIP_YTCOMMENTS}"

while true; do
  echo ""
  echo "=== Round ${ROUND} ==="
  echo "caps: stackexchange_max_posts_per_site=${SE_CAP} gutenberg_max_docs=${GB_CAP} wikisource_max_docs_per_wiki=${WS_CAP_PER_WIKI} wikisource_max_total_docs=${WS_CAP_TOTAL}"

  CMD=(
    "$PYTHON_BIN" -m processing.second_phase_web_crawling.run_pipeline
    --stages crawl construct
    --manifest-path "$MANIFEST_PATH"
    --stackexchange-download-mode "$STACKEXCHANGE_MODE"
    --stackexchange-sites "$STACKEXCHANGE_SITES"
    --stackexchange-max-posts-per-site "$SE_CAP"
    --gutenberg-max-docs "$GB_CAP"
    --gutenberg-languages "$GUTENBERG_LANGUAGES"
    --wikisource-wikis "$WIKISOURCE_WIKIS"
    --wikisource-max-docs-per-wiki "$WS_CAP_PER_WIKI"
    --wikisource-max-total-docs "$WS_CAP_TOTAL"
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
  if [[ "$SKIP_STACKEXCHANGE" == "1" ]]; then
    CMD+=(--skip-stackexchange)
  fi
  if [[ "$SKIP_GUTENBERG" == "1" ]]; then
    CMD+=(--skip-gutenberg)
  fi
  if [[ "$SKIP_WIKISOURCE" == "1" ]]; then
    CMD+=(--skip-wikisource)
  fi
  if [[ "$SKIP_YTCOMMENTS" == "1" ]]; then
    CMD+=(--skip-ytcomments)
  fi

  "${CMD[@]}"

  if [[ ! -f "$BUILD_OUTPUT_DIR/processing_summary.json" ]]; then
    echo "Error: missing $BUILD_OUTPUT_DIR/processing_summary.json after round ${ROUND}"
    exit 1
  fi

  FINAL_AFTER_SAMPLING="$(jq -r '.after_sampling.total // 0' "$BUILD_OUTPUT_DIR/processing_summary.json")"
  echo "round=${ROUND} after_sampling=${FINAL_AFTER_SAMPLING} target=${TARGET_TOTAL}"

  if [[ "$AUTO_RAMP" != "1" ]]; then
    break
  fi

  if (( FINAL_AFTER_SAMPLING >= TARGET_TOTAL )); then
    echo "Target reached in round ${ROUND}."
    break
  fi

  if (( ROUND >= RAMP_MAX_ROUNDS )); then
    echo "Reached RAMP_MAX_ROUNDS=${RAMP_MAX_ROUNDS} without hitting target."
    break
  fi

  OLD_SE="$SE_CAP"
  OLD_GB="$GB_CAP"
  OLD_WS_PER="$WS_CAP_PER_WIKI"
  OLD_WS_TOTAL="$WS_CAP_TOTAL"

  SE_CAP="$(bump_cap "$SE_CAP" "$STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX")"
  GB_CAP="$(bump_cap "$GB_CAP" "$GUTENBERG_MAX_DOCS_MAX")"
  WS_CAP_PER_WIKI="$(bump_cap "$WS_CAP_PER_WIKI" "$WIKISOURCE_MAX_DOCS_PER_WIKI_MAX")"
  WS_CAP_TOTAL="$(bump_cap "$WS_CAP_TOTAL" "$WIKISOURCE_MAX_TOTAL_DOCS_MAX")"

  if (( SE_CAP == OLD_SE && GB_CAP == OLD_GB && WS_CAP_PER_WIKI == OLD_WS_PER && WS_CAP_TOTAL == OLD_WS_TOTAL )); then
    echo "All caps are at max limits; cannot ramp further."
    break
  fi

  ROUND=$(( ROUND + 1 ))
done

echo "[2/3] Summaries"
echo "Monitor:      $MONITOR_REPORT_PATH"
echo "Stage1:       $BUILD_OUTPUT_DIR"
echo "Stage2:       $POSTPROCESS_OUTPUT_DIR"
echo "Final after_sampling (stage1): $FINAL_AFTER_SAMPLING"
echo "Final caps: stackexchange_max_posts_per_site=$SE_CAP gutenberg_max_docs=$GB_CAP wikisource_max_docs_per_wiki=$WS_CAP_PER_WIKI wikisource_max_total_docs=$WS_CAP_TOTAL"

if [[ -f "$BUILD_OUTPUT_DIR/processing_summary.json" ]]; then
  echo ""
  echo "Stage1 processing summary:"
  jq '{after_author_filter: .after_author_filter.total, after_sampling: .after_sampling.total, splits: {train: .splits.train.total, dev: .splits.dev.total, test: .splits.test.total}}' "$BUILD_OUTPUT_DIR/processing_summary.json"
fi

if [[ -f "$POSTPROCESS_OUTPUT_DIR/postprocessing_summary.json" ]]; then
  echo ""
  echo "Stage2 postprocessing summary:"
  jq '{before_filter: .before_filter.total, after_filter: .after_filter.total, after_dedup: .after_dedup.total, after_sampling: .after_sampling.total, split_candidates: {train: .splits.train.candidates, dev: .splits.dev.candidates, test: .splits.test.candidates}}' "$POSTPROCESS_OUTPUT_DIR/postprocessing_summary.json"
fi

echo "[3/3] Done"
