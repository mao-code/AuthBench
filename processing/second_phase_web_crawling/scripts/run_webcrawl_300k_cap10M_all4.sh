#!/usr/bin/env bash
set -euo pipefail

# Run second-phase web crawling + unified construct pipeline at 300K target
# using 4 web datasets:
# - StackExchange
# - Project Gutenberg
# - Wikisource
# - YTComments (YouTube Data API)

# ROOT_DIR can be overridden by environment.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}"
# sbatch -p rush --nodelist=rush-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G -t 720:00:00 processing/second_phase_web_crawling/scripts/run_webcrawl_300k_cap10M_all4.sh

# Load .env first (YouTube API key), then optional StackExchange env file.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi
if [[ -z "${STACKEXCHANGE_API_KEY:-}" && -f "$ROOT_DIR/.env.stackexchange" ]]; then
  set -a
  source "$ROOT_DIR/.env.stackexchange"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_TAG="${RUN_TAG:-all4_t300k_cap10M}"
TARGET_TOTAL="${TARGET_TOTAL:-300000}"
POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-300000}"
SEED="${SEED:-42}"
PIPELINE_STAGES="${PIPELINE_STAGES:-crawl construct}"

TARGET_LANGUAGES="${TARGET_LANGUAGES:-en,zh,hi,es,fr,ar,ru,de,ja,ko}"

# StackExchange
STACKEXCHANGE_MODE="${STACKEXCHANGE_MODE:-api}"
STACKEXCHANGE_SITES="${STACKEXCHANGE_SITES:-stackoverflow.com,es.stackoverflow.com,ru.stackoverflow.com,ja.stackoverflow.com,spanish.stackexchange.com,french.stackexchange.com,german.stackexchange.com,arabic.stackexchange.com,chinese.stackexchange.com,korean.stackexchange.com,hindi.stackexchange.com}"
STACKEXCHANGE_MAX_POSTS_PER_SITE="${STACKEXCHANGE_MAX_POSTS_PER_SITE:-80000}"
STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX="${STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX:-1200000}"
STACKEXCHANGE_MAX_COMMENTS_PER_SITE="${STACKEXCHANGE_MAX_COMMENTS_PER_SITE:-4000}"
STACKEXCHANGE_SKIP_COMMENTS="${STACKEXCHANGE_SKIP_COMMENTS:-1}"

# Gutenberg
GUTENBERG_MAX_DOCS="${GUTENBERG_MAX_DOCS:-400000}"
GUTENBERG_MAX_DOCS_MAX="${GUTENBERG_MAX_DOCS_MAX:-2400000}"
GUTENBERG_LANGUAGES="${GUTENBERG_LANGUAGES:-$TARGET_LANGUAGES}"

# Wikisource
WIKISOURCE_WIKIS="${WIKISOURCE_WIKIS:-enwikisource,zhwikisource,hiwikisource,eswikisource,frwikisource,arwikisource,ruwikisource,dewikisource,jawikisource,kowikisource}"
WIKISOURCE_MAX_DOCS_PER_WIKI="${WIKISOURCE_MAX_DOCS_PER_WIKI:-50000}"
WIKISOURCE_MAX_DOCS_PER_WIKI_MAX="${WIKISOURCE_MAX_DOCS_PER_WIKI_MAX:-300000}"
WIKISOURCE_MAX_TOTAL_DOCS="${WIKISOURCE_MAX_TOTAL_DOCS:-500000}"
WIKISOURCE_MAX_TOTAL_DOCS_MAX="${WIKISOURCE_MAX_TOTAL_DOCS_MAX:-3000000}"

# YTComments
YTCOMMENTS_MAX_DOCS="${YTCOMMENTS_MAX_DOCS:-500000}"
YTCOMMENTS_MAX_DOCS_MAX="${YTCOMMENTS_MAX_DOCS_MAX:-3000000}"
YTCOMMENTS_LANGUAGES="${YTCOMMENTS_LANGUAGES:-$TARGET_LANGUAGES}"
YTCOMMENTS_REGION_MAP="${YTCOMMENTS_REGION_MAP:-en:US,zh:TW,hi:IN,es:ES,fr:FR,ar:EG,ru:RU,de:DE,ja:JP,ko:KR}"
YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG="${YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG:-80}"
YTCOMMENTS_MAX_COMMENTS_PER_VIDEO="${YTCOMMENTS_MAX_COMMENTS_PER_VIDEO:-300}"
YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO="${YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO:-10}"
YTCOMMENTS_MAX_EMPTY_PAGES_PER_VIDEO="${YTCOMMENTS_MAX_EMPTY_PAGES_PER_VIDEO:-2}"
YTCOMMENTS_MIN_CHARS="${YTCOMMENTS_MIN_CHARS:-20}"
YTCOMMENTS_SKIP_LANGDETECT="${YTCOMMENTS_SKIP_LANGDETECT:-0}"
YTCOMMENTS_RESUME="${YTCOMMENTS_RESUME:-1}"
YTCOMMENTS_TIMEOUT="${YTCOMMENTS_TIMEOUT:-60}"
YTCOMMENTS_RETRIES="${YTCOMMENTS_RETRIES:-6}"
YTCOMMENTS_RETRY_BACKOFF_SEC="${YTCOMMENTS_RETRY_BACKOFF_SEC:-2.0}"
YTCOMMENTS_SLEEP_SECONDS="${YTCOMMENTS_SLEEP_SECONDS:-0.15}"

# Restart toggles
SKIP_STACKEXCHANGE="${SKIP_STACKEXCHANGE:-0}"
SKIP_GUTENBERG="${SKIP_GUTENBERG:-0}"
SKIP_WIKISOURCE="${SKIP_WIKISOURCE:-0}"
SKIP_YTCOMMENTS="${SKIP_YTCOMMENTS:-0}"
YT_ONLY="${YT_ONLY:-0}"

AUTO_RAMP="${AUTO_RAMP:-1}"
RAMP_MAX_ROUNDS="${RAMP_MAX_ROUNDS:-8}"
RAMP_FACTOR="${RAMP_FACTOR:-2}"
STACKEXCHANGE_API_SAFE_MODE="${STACKEXCHANGE_API_SAFE_MODE:-1}"
STACKEXCHANGE_API_SAFE_MAX_REQUESTS="${STACKEXCHANGE_API_SAFE_MAX_REQUESTS:-250}"
STACKEXCHANGE_API_SAFE_MAX_SITES_NO_KEY="${STACKEXCHANGE_API_SAFE_MAX_SITES_NO_KEY:-2}"
STACKEXCHANGE_API_SAFE_MAX_SITES_WITH_KEY="${STACKEXCHANGE_API_SAFE_MAX_SITES_WITH_KEY:-5}"
STACKEXCHANGE_API_SAFE_MAX_POSTS_PER_SITE_NO_KEY="${STACKEXCHANGE_API_SAFE_MAX_POSTS_PER_SITE_NO_KEY:-3000}"
STACKEXCHANGE_API_SAFE_MAX_POSTS_PER_SITE_WITH_KEY="${STACKEXCHANGE_API_SAFE_MAX_POSTS_PER_SITE_WITH_KEY:-15000}"
STACKEXCHANGE_API_SAFE_MAX_COMMENTS_PER_SITE_NO_KEY="${STACKEXCHANGE_API_SAFE_MAX_COMMENTS_PER_SITE_NO_KEY:-0}"
STACKEXCHANGE_API_SAFE_MAX_COMMENTS_PER_SITE_WITH_KEY="${STACKEXCHANGE_API_SAFE_MAX_COMMENTS_PER_SITE_WITH_KEY:-1000}"

# Processing controls
MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-10000000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
CHUNK_PROBABILITY="${CHUNK_PROBABILITY:-0.7}"
TRUNCATE_TO_TOKENS="${TRUNCATE_TO_TOKENS:-2000}"

MANIFEST_PATH="${MANIFEST_PATH:-processing/second_phase_web_crawling/datasets_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-processing/second_phase_web_crawling/outputs/pipeline_${RUN_TAG}}"
MONITOR_REPORT_PATH="${MONITOR_REPORT_PATH:-${OUTPUT_DIR}/pipeline_dynamics.json}"

if [[ "$YT_ONLY" == "1" ]]; then
  SKIP_STACKEXCHANGE=1
  SKIP_GUTENBERG=1
  SKIP_WIKISOURCE=1
  SKIP_YTCOMMENTS=0
fi

read -r -a STAGE_ARGS <<<"$PIPELINE_STAGES"
RUN_CONSTRUCT=0
for stage in "${STAGE_ARGS[@]}"; do
  if [[ "$stage" == "construct" ]]; then
    RUN_CONSTRUCT=1
    break
  fi
done

if [[ "$SKIP_YTCOMMENTS" != "1" && -z "${YOUTUBE_API_KEY:-}" ]]; then
  echo "Error: YOUTUBE_API_KEY is required for YTComments. Add it to .env or export it."
  exit 1
fi

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

apply_stackexchange_api_guardrails() {
  if [[ "$STACKEXCHANGE_MODE" != "api" || "$STACKEXCHANGE_API_SAFE_MODE" != "1" || "$SKIP_STACKEXCHANGE" == "1" ]]; then
    return
  fi

  local safe_max_sites
  local safe_max_posts_per_site
  local safe_max_comments_per_site
  if [[ -n "${STACKEXCHANGE_API_KEY:-}" ]]; then
    safe_max_sites="$STACKEXCHANGE_API_SAFE_MAX_SITES_WITH_KEY"
    safe_max_posts_per_site="$STACKEXCHANGE_API_SAFE_MAX_POSTS_PER_SITE_WITH_KEY"
    safe_max_comments_per_site="$STACKEXCHANGE_API_SAFE_MAX_COMMENTS_PER_SITE_WITH_KEY"
  else
    safe_max_sites="$STACKEXCHANGE_API_SAFE_MAX_SITES_NO_KEY"
    safe_max_posts_per_site="$STACKEXCHANGE_API_SAFE_MAX_POSTS_PER_SITE_NO_KEY"
    safe_max_comments_per_site="$STACKEXCHANGE_API_SAFE_MAX_COMMENTS_PER_SITE_NO_KEY"
  fi

  IFS=',' read -r -a _se_sites <<<"$STACKEXCHANGE_SITES"
  if (( ${#_se_sites[@]} > safe_max_sites )); then
    STACKEXCHANGE_SITES="$(IFS=,; echo "${_se_sites[*]:0:${safe_max_sites}}")"
    echo "StackExchange API safe mode: limiting sites to first ${safe_max_sites}."
  fi

  if (( STACKEXCHANGE_MAX_POSTS_PER_SITE > safe_max_posts_per_site )); then
    STACKEXCHANGE_MAX_POSTS_PER_SITE="$safe_max_posts_per_site"
    echo "StackExchange API safe mode: capping posts/site to ${STACKEXCHANGE_MAX_POSTS_PER_SITE}."
  fi
  if (( STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX > safe_max_posts_per_site )); then
    STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX="$safe_max_posts_per_site"
  fi
  if (( STACKEXCHANGE_MAX_COMMENTS_PER_SITE > safe_max_comments_per_site )); then
    STACKEXCHANGE_MAX_COMMENTS_PER_SITE="$safe_max_comments_per_site"
    echo "StackExchange API safe mode: capping comments/site to ${STACKEXCHANGE_MAX_COMMENTS_PER_SITE}."
  fi
  if (( STACKEXCHANGE_MAX_COMMENTS_PER_SITE == 0 )); then
    STACKEXCHANGE_SKIP_COMMENTS=1
  fi
  if [[ "$AUTO_RAMP" == "1" ]]; then
    AUTO_RAMP=0
    echo "StackExchange API safe mode: AUTO_RAMP disabled to avoid repeated quota burn."
  fi

  IFS=',' read -r -a _se_sites <<<"$STACKEXCHANGE_SITES"
  local site_count="${#_se_sites[@]}"
  if (( site_count == 0 )); then
    return
  fi

  local requests_per_site=$(( ((STACKEXCHANGE_MAX_POSTS_PER_SITE + 99) / 100) * 2 ))
  if [[ "$STACKEXCHANGE_SKIP_COMMENTS" != "1" ]]; then
    requests_per_site=$(( requests_per_site + ((STACKEXCHANGE_MAX_COMMENTS_PER_SITE + 99) / 100) ))
  fi
  local estimated_total_requests=$(( requests_per_site * site_count ))

  if (( estimated_total_requests > STACKEXCHANGE_API_SAFE_MAX_REQUESTS )); then
    local per_site_budget=$(( STACKEXCHANGE_API_SAFE_MAX_REQUESTS / site_count ))
    if (( per_site_budget < 2 )); then
      per_site_budget=2
    fi
    local budget_posts=$(( (per_site_budget / 2) * 100 ))
    if (( budget_posts < 100 )); then
      budget_posts=100
    fi
    if (( STACKEXCHANGE_MAX_POSTS_PER_SITE > budget_posts )); then
      STACKEXCHANGE_MAX_POSTS_PER_SITE="$budget_posts"
      STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX="$budget_posts"
      echo "StackExchange API safe mode: further reducing posts/site to ${budget_posts} for request budget."
    fi
  fi
}

apply_stackexchange_api_guardrails

SE_CAP="$STACKEXCHANGE_MAX_POSTS_PER_SITE"
GB_CAP="$GUTENBERG_MAX_DOCS"
WS_CAP_PER_WIKI="$WIKISOURCE_MAX_DOCS_PER_WIKI"
WS_CAP_TOTAL="$WIKISOURCE_MAX_TOTAL_DOCS"
YT_CAP="$YTCOMMENTS_MAX_DOCS"

ROUND=1
FINAL_AFTER_SAMPLING=0

echo "[1/3] Running crawl + unified pipeline (target=${TARGET_TOTAL})"
echo "skip flags: stackexchange=${SKIP_STACKEXCHANGE} gutenberg=${SKIP_GUTENBERG} wikisource=${SKIP_WIKISOURCE} ytcomments=${SKIP_YTCOMMENTS}"
echo "stages: ${PIPELINE_STAGES}"

while true; do
  echo ""
  echo "=== Round ${ROUND} ==="
  echo "caps: stackexchange_max_posts_per_site=${SE_CAP} gutenberg_max_docs=${GB_CAP} wikisource_max_docs_per_wiki=${WS_CAP_PER_WIKI} wikisource_max_total_docs=${WS_CAP_TOTAL} ytcomments_max_docs=${YT_CAP}"

  DATASET_CAP_ARGS=(
    "stackexchange_web_crawl=${MAX_DOCUMENTS_PER_DATASET}"
    "project_gutenberg_web_crawl=${MAX_DOCUMENTS_PER_DATASET}"
    "wikisource_web_crawl=${MAX_DOCUMENTS_PER_DATASET}"
    "ytcomments_web_crawl=${MAX_DOCUMENTS_PER_DATASET}"
  )

  CMD=(
    "$PYTHON_BIN" -m processing.second_phase_web_crawling.run_pipeline
    --stages "${STAGE_ARGS[@]}"
    --manifest-path "$MANIFEST_PATH"
    --stackexchange-download-mode "$STACKEXCHANGE_MODE"
    --stackexchange-sites "$STACKEXCHANGE_SITES"
    --stackexchange-max-posts-per-site "$SE_CAP"
    --stackexchange-max-comments-per-site "$STACKEXCHANGE_MAX_COMMENTS_PER_SITE"
    --gutenberg-max-docs "$GB_CAP"
    --gutenberg-languages "$GUTENBERG_LANGUAGES"
    --wikisource-wikis "$WIKISOURCE_WIKIS"
    --wikisource-max-docs-per-wiki "$WS_CAP_PER_WIKI"
    --wikisource-max-total-docs "$WS_CAP_TOTAL"
    --ytcomments-max-docs "$YT_CAP"
    --ytcomments-languages "$YTCOMMENTS_LANGUAGES"
    --ytcomments-region-map "$YTCOMMENTS_REGION_MAP"
    --ytcomments-max-video-pages-per-lang "$YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG"
    --ytcomments-max-comments-per-video "$YTCOMMENTS_MAX_COMMENTS_PER_VIDEO"
    --ytcomments-max-comment-pages-per-video "$YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO"
    --ytcomments-max-empty-pages-per-video "$YTCOMMENTS_MAX_EMPTY_PAGES_PER_VIDEO"
    --ytcomments-min-chars "$YTCOMMENTS_MIN_CHARS"
    --ytcomments-timeout "$YTCOMMENTS_TIMEOUT"
    --ytcomments-retries "$YTCOMMENTS_RETRIES"
    --ytcomments-retry-backoff-sec "$YTCOMMENTS_RETRY_BACKOFF_SEC"
    --ytcomments-sleep-seconds "$YTCOMMENTS_SLEEP_SECONDS"
    --monitor-report-path "$MONITOR_REPORT_PATH"
    --monitor-overwrite
    --output-dir "$OUTPUT_DIR"
    --total-docs "$TARGET_TOTAL"
    --post-target-total "$POST_TARGET_TOTAL"
    --max-documents-per-dataset "$MAX_DOCUMENTS_PER_DATASET"
    --dataset-max-docs "${DATASET_CAP_ARGS[@]}"
    --shuffle-buffer-size "$SHUFFLE_BUFFER_SIZE"
    --chunk-probability "$CHUNK_PROBABILITY"
    --truncate-to-tokens "$TRUNCATE_TO_TOKENS"
    --seed "$SEED"
    --log-level INFO
  )

  if [[ "$STACKEXCHANGE_SKIP_COMMENTS" == "1" ]]; then
    CMD+=(--stackexchange-skip-comments)
  fi
  if [[ "$YTCOMMENTS_SKIP_LANGDETECT" == "1" ]]; then
    CMD+=(--ytcomments-skip-langdetect)
  fi
  if [[ "$YTCOMMENTS_RESUME" == "1" ]]; then
    CMD+=(--ytcomments-resume)
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

  if [[ "$RUN_CONSTRUCT" != "1" ]]; then
    echo "Crawl-only run finished (no construct stage requested)."
    break
  fi

  if [[ ! -f "$MONITOR_REPORT_PATH" ]]; then
    echo "Error: missing monitoring report at $MONITOR_REPORT_PATH after round ${ROUND}"
    exit 1
  fi

  FINAL_AFTER_SAMPLING="$(jq -r '.stage_transitions.build_after_sampling_total // 0' "$MONITOR_REPORT_PATH")"
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
  OLD_YT="$YT_CAP"

  SE_CAP="$(bump_cap "$SE_CAP" "$STACKEXCHANGE_MAX_POSTS_PER_SITE_MAX")"
  GB_CAP="$(bump_cap "$GB_CAP" "$GUTENBERG_MAX_DOCS_MAX")"
  WS_CAP_PER_WIKI="$(bump_cap "$WS_CAP_PER_WIKI" "$WIKISOURCE_MAX_DOCS_PER_WIKI_MAX")"
  WS_CAP_TOTAL="$(bump_cap "$WS_CAP_TOTAL" "$WIKISOURCE_MAX_TOTAL_DOCS_MAX")"
  YT_CAP="$(bump_cap "$YT_CAP" "$YTCOMMENTS_MAX_DOCS_MAX")"

  if (( SE_CAP == OLD_SE && GB_CAP == OLD_GB && WS_CAP_PER_WIKI == OLD_WS_PER && WS_CAP_TOTAL == OLD_WS_TOTAL && YT_CAP == OLD_YT )); then
    echo "All caps are at max limits; cannot ramp further."
    break
  fi

  ROUND=$(( ROUND + 1 ))
done

echo "[2/3] Summaries"
echo "Monitor:      $MONITOR_REPORT_PATH"
echo "Output:       $OUTPUT_DIR"
echo "Build-stage after_sampling: $FINAL_AFTER_SAMPLING"
echo "Final caps: stackexchange_max_posts_per_site=$SE_CAP gutenberg_max_docs=$GB_CAP wikisource_max_docs_per_wiki=$WS_CAP_PER_WIKI wikisource_max_total_docs=$WS_CAP_TOTAL ytcomments_max_docs=$YT_CAP"

if [[ -f "$OUTPUT_DIR/pipeline_summary.json" ]]; then
  echo ""
  echo "Unified pipeline summary:"
  jq '{build: {after_author_filter: .build.summary.after_author_filter.total, after_sampling: .build.summary.after_sampling.total}, final: {before_filter: .finalize.input_docs.total, after_filter: .finalize.after_filter.total, after_dedup: .finalize.after_dedup.total, after_sampling: .finalize.after_sampling.total, split_documents: {train: .finalize.splits.train.documents, dev: .finalize.splits.dev.documents, test: .finalize.splits.test.documents}, split_candidates: {train: .finalize.splits.train.candidates, dev: .finalize.splits.dev.candidates, test: .finalize.splits.test.candidates}}}' "$OUTPUT_DIR/pipeline_summary.json"
fi

echo "[3/3] Done"

# YT_ONLY=1 \
# PIPELINE_STAGES="crawl" \
# AUTO_RAMP=0 \
# YTCOMMENTS_RESUME=1 \
# YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG=12 \
# YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO=2 \
# YTCOMMENTS_MAX_EMPTY_PAGES_PER_VIDEO=1 \
# YTCOMMENTS_SLEEP_SECONDS=0.2 \
# bash processing/second_phase_web_crawling/scripts/run_webcrawl_300k_cap10M_all4.sh


# PIPELINE_STAGES="construct" AUTO_RAMP=0 \
# bash processing/second_phase_web_crawling/scripts/run_webcrawl_300k_cap10M_all4.sh
