#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

needs_kaggle=0

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

path_ready() {
  local target="$1"
  if [[ -f "$target" ]]; then
    return 0
  fi
  if [[ -d "$target" ]] && find "$target" -mindepth 1 -print -quit >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

run_fetch() {
  local label="$1"
  local target="$2"
  shift 2

  if [[ "$FORCE_DOWNLOAD" != "1" ]] && path_ready "$target"; then
    echo "[skip] $label already present at $target"
    return 0
  fi

  echo "[fetch] $label -> $target"
  "$PYTHON_BIN" "$@"
}

echo "Checking prerequisites..."
require_command "$PYTHON_BIN"

if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
  needs_kaggle=1
else
  for target in \
    "raw_analysis/outputs/amazon_reviews_multi_raw" \
    "raw_analysis/outputs/arxiv_raw/arxiv-metadata-oai-snapshot.json" \
    "raw_analysis/outputs/xiaohongshu_raw" \
    "raw_analysis/outputs/douban_raw" \
    "raw_analysis/outputs/arabic_poetry_raw"
  do
    if ! path_ready "$target"; then
      needs_kaggle=1
      break
    fi
  done
fi

if [[ "$needs_kaggle" == "1" ]]; then
  require_command kaggle
fi

run_fetch \
  "amazon_reviews_multi" \
  "raw_analysis/outputs/amazon_reviews_multi_raw" \
  raw_analysis/amazon_reviews_multi_analysis.py

run_fetch \
  "arxiv_snapshot" \
  "raw_analysis/outputs/arxiv_raw/arxiv-metadata-oai-snapshot.json" \
  raw_analysis/arxiv_metadata_analysis.py

run_fetch \
  "xiaohongshu" \
  "raw_analysis/outputs/xiaohongshu_raw" \
  raw_analysis/xiaohongshu_analysis.py

run_fetch \
  "douban_reviews" \
  "raw_analysis/outputs/douban_raw" \
  raw_analysis/douban_reviews_analysis.py

run_fetch \
  "hindi_discourse" \
  "raw_analysis/outputs/hindi_discourse/discourse_dataset.json" \
  raw_analysis/hindi_discourse_analysis.py

run_fetch \
  "arabic_poetry" \
  "raw_analysis/outputs/arabic_poetry_raw" \
  raw_analysis/arabic_poetry_analysis.py

echo ""
echo "Manifest-backed local phase1 sources:"
for target in \
  "raw_analysis/outputs/amazon_reviews_multi_raw" \
  "raw_analysis/outputs/arxiv_raw/arxiv-metadata-oai-snapshot.json" \
  "raw_analysis/outputs/xiaohongshu_raw" \
  "raw_analysis/outputs/douban_raw" \
  "raw_analysis/outputs/hindi_discourse/discourse_dataset.json" \
  "raw_analysis/outputs/arabic_poetry_raw"
do
  if path_ready "$target"; then
    echo "  [ok] $target"
  else
    echo "  [missing] $target"
  fi
done

echo ""
echo "Done."
