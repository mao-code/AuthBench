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
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-1800}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

run_with_heartbeat() {
  local log_path="$1"
  shift

  : >"${log_path}"
  "$@" >>"${log_path}" 2>&1 &
  local cmd_pid=$!
  local start_time
  start_time=$(date +%s)

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    sleep "${HEARTBEAT_INTERVAL}"
    if kill -0 "${cmd_pid}" 2>/dev/null; then
      local now elapsed
      now=$(date +%s)
      elapsed=$((now - start_time))
      log "Still running (pid=${cmd_pid}, elapsed=${elapsed}s). Current log: ${log_path}"
    fi
  done

  wait "${cmd_pid}"
}

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

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

TOTAL_BASELINES="${#BASELINE_LIST[@]}"
log "Starting baseline evaluation run."
log "Baselines=${TOTAL_BASELINES} split=${SPLIT} task=${TASK} dataset_root=${DATASET_ROOT} output_dir=${OUTPUT_DIR} log_dir=${LOG_DIR}"

BASELINE_INDEX=0
for BASELINE in "${BASELINE_LIST[@]}"; do
  BASELINE_INDEX=$((BASELINE_INDEX + 1))
  OUTPUT_PATH="${OUTPUT_DIR}/${BASELINE}.json"
  LOG_PATH="${LOG_DIR}/${BASELINE}.log"
  START_TS=$(date +%s)
  log "[${BASELINE_INDEX}/${TOTAL_BASELINES}] Evaluating baseline=${BASELINE} split=${SPLIT}"
  log "Writing baseline log to ${LOG_PATH}"
  run_with_heartbeat "${LOG_PATH}" \
    python -m eval.baseline_runner \
      "${COMMON_ARGS[@]}" \
      --baselines "${BASELINE}" \
      --output-json "${OUTPUT_PATH}"
  END_TS=$(date +%s)
  log "Finished baseline=${BASELINE} in $((END_TS - START_TS))s. Metrics saved to ${OUTPUT_PATH}"
done

log "Completed baseline evaluation run."
