#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/combined_phase1_phase2}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
SPLIT="${SPLIT:-test}"
TASK="${TASK:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/baselines}"
BASELINES="${BASELINES:-tfidf ngram ppm}"

MAX_QUERIES="${MAX_QUERIES:-}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"
CANDIDATE_POOL="${CANDIDATE_POOL:-all}"
MAX_TOPIC_CANDIDATES="${MAX_TOPIC_CANDIDATES:-}"
TOPIC_SEED="${TOPIC_SEED:-13}"
NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-50}"
NEGATIVE_STRATEGY="${NEGATIVE_STRATEGY:-all}"
SEED="${SEED:-13}"

NGRAM_MAX_TRAIN_EXAMPLES="${NGRAM_MAX_TRAIN_EXAMPLES:-120000}"
PPM_MAX_TRAIN_EXAMPLES="${PPM_MAX_TRAIN_EXAMPLES:-80000}"
PPM_ORDER="${PPM_ORDER:-4}"
PPM_HASH_FEATURES="${PPM_HASH_FEATURES:-8192}"
PPM_ALPHA="${PPM_ALPHA:-0.5}"

mkdir -p "${OUTPUT_DIR}"

read -r -a BASELINE_LIST <<<"${BASELINES}"

COMMON_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --train-split "${TRAIN_SPLIT}"
  --split "${SPLIT}"
  --task "${TASK}"
  --candidate-pool "${CANDIDATE_POOL}"
  --topic-seed "${TOPIC_SEED}"
  --negatives-per-query "${NEGATIVES_PER_QUERY}"
  --negative-strategy "${NEGATIVE_STRATEGY}"
  --seed "${SEED}"
  --ngram-max-train-examples "${NGRAM_MAX_TRAIN_EXAMPLES}"
  --ppm-max-train-examples "${PPM_MAX_TRAIN_EXAMPLES}"
  --ppm-order "${PPM_ORDER}"
  --ppm-hash-features "${PPM_HASH_FEATURES}"
  --ppm-alpha "${PPM_ALPHA}"
)

if [[ -n "${MAX_QUERIES}" ]]; then
  COMMON_ARGS+=(--max-queries "${MAX_QUERIES}")
fi
if [[ -n "${MAX_CANDIDATES}" ]]; then
  COMMON_ARGS+=(--max-candidates "${MAX_CANDIDATES}")
fi
if [[ -n "${MAX_TOPIC_CANDIDATES}" ]]; then
  COMMON_ARGS+=(--max-topic-candidates "${MAX_TOPIC_CANDIDATES}")
fi

for BASELINE in "${BASELINE_LIST[@]}"; do
  echo ">>> Evaluating baseline=${BASELINE} split=${SPLIT}"
  python -m eval.baseline_runner \
    "${COMMON_ARGS[@]}" \
    --baselines "${BASELINE}" \
    --output-json "${OUTPUT_DIR}/${BASELINE}.json"
done
