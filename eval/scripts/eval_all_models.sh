#!/usr/bin/env bash
# Evaluate a wide range of embedding models and store per-model metrics (including fine-grained breakdowns).
# Run from the repository root (AuthBench).
set -euo pipefail

# sbatch -p nlplarge-sasha-highpri --nodelist=nlplarge-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G -t 720:00:00 eval/scripts/eval_all_models.sh

# sbatch -p rush --nodelist=rush-compute-03 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G -t 720:00:00 eval/scripts/eval_all_models.sh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/authbench}"
SPLIT="${SPLIT:-test}"
TASK="${TASK:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results}"
BATCH_SIZE="${BATCH_SIZE:-4}"
# MAX_LENGTH="${MAX_LENGTH:-512}"
NO_TRUNCATION="${NO_TRUNCATION:-1}"
POOLING="${POOLING:-mean}"
NEG_PER_QUERY="${NEG_PER_QUERY:-50}"
NEGATIVE_STRATEGY="${NEGATIVE_STRATEGY:-all}"
CANDIDATE_CHUNK_SIZE="${CANDIDATE_CHUNK_SIZE:-128}"

# Weights & Biases defaults (override via env).
WANDB_PROJECT="${WANDB_PROJECT:-AuthBench}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-eval-all}"
WANDB_TAGS="${WANDB_TAGS:-AuthBench eval-all}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-1800}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

warn() {
  echo "[$(timestamp)] [WARN] $*" >&2
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

if [[ -n "${MODELS:-}" ]]; then
  # Allow overriding via `MODELS="m1 m2 ..."`.
  read -r -a MODEL_LIST <<<"${MODELS}"
else
  MODEL_LIST=(
    # BGE family
    bge-m3
    bge-large-en-v1.5
    bge-base-en-v1.5
    bge-small-en-v1.5
    bge-large-zh-v1.5
    bge-base-zh-v1.5
    # E5 (base/large + multilingual) and Mistral-instruct variant
    e5-large-v2
    e5-base-v2
    e5-small-v2
    multilingual-e5-large
    multilingual-e5-base
    e5-mistral-7b-instruct
    # GTE variants
    gte-large-en-v1.5
    gte-qwen2-7b-instruct
    gte-base
    gte-large
    # Jina, Snowflake Arctic, NVIDIA, Mixedbread
    jina-embeddings-v2-base-en
    jina-embeddings-v2-small-en
    snowflake-arctic-embed-l-v2
    snowflake-arctic-embed-m-v2
    nv-embed-v1
    mxbai-embed-large-v1
    # Nomic + Salesforce Mistral
    nomic-embed-text-v1.5
    nomic-embed-text-v1
    sfr-embedding-mistral
    Sentence-Transformers classics
    all-roberta-large-v1
    all-mpnet-base-v2
    all-minilm-l12-v2
    all-minilm-l6-v2
    # multi-qa-mpnet-base-dot-v1
    paraphrase-mpnet-base-v2
    paraphrase-multilingual-mpnet-base-v2
    distiluse-base-multilingual-cased-v2
    msmarco-distilbert-base-v4
    allenai-specter
    bert-base-uncased
    facebook-contriever
    facebook-contriever-msmarco
    # Qwen3 embeddings and Qwen2.5 base/instruct
    qwen3-embedding-0.6b
    qwen3-embedding-4b
    qwen3-embedding-8b
    qwen3-4b
    qwen3-4b-instruct
    qwen2.5-3b
    qwen2.5-3b-instruct
    qwen2.5-7b-instruct
    # Qwen3.5 (image-text-to-text)
    
    # LLaMA base/instruct (≤8B)
    llama3.1-8b
    llama3.1-8b-instruct
    llama3-8b
    llama3-8b-instruct
    llama2-7b
    llama2-7b-chat
    # DeepSeek base/chat/coder
    deepseek-llm-7b-base
    deepseek-llm-7b-chat
  )
fi

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --task "${TASK}"
  --split "${SPLIT}"
  --dataset-root "${DATASET_ROOT}"
  --batch-size "${BATCH_SIZE}"
  # --max-length "${MAX_LENGTH}"
  --pooling "${POOLING}"
  --negatives-per-query "${NEG_PER_QUERY}"
  --negative-strategy "${NEGATIVE_STRATEGY}"
  --candidate-chunk-size "${CANDIDATE_CHUNK_SIZE}"
)

if [[ -n "${TORCH_DTYPE:-}" ]]; then
  COMMON_ARGS+=(--torch-dtype "${TORCH_DTYPE}")
fi
if [[ -n "${MAX_QUERIES:-}" ]]; then
  COMMON_ARGS+=(--max-queries "${MAX_QUERIES}")
fi
if [[ -n "${MAX_CANDIDATES:-}" ]]; then
  COMMON_ARGS+=(--max-candidates "${MAX_CANDIDATES}")
fi
if [[ "${NO_TRUNCATION}" != "0" ]]; then
  COMMON_ARGS+=(--no-truncation)
fi
if [[ "${LATE_INTERACTION:-0}" != "0" ]]; then
  COMMON_ARGS+=(--late-interaction)
fi
if [[ -n "${QUERY_PREFIX:-}" ]]; then
  COMMON_ARGS+=(--query-prefix "${QUERY_PREFIX}")
fi
if [[ -n "${DOC_PREFIX:-}" ]]; then
  COMMON_ARGS+=(--doc-prefix "${DOC_PREFIX}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" != "0" ]]; then
  COMMON_ARGS+=(--trust-remote-code)
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  COMMON_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_TAGS:-}" ]]; then
    for tag in ${WANDB_TAGS}; do
      COMMON_ARGS+=(--wandb-tags "${tag}")
    done
  fi
fi

TOTAL_MODELS="${#MODEL_LIST[@]}"
log "Starting model evaluation run."
log "Models=${TOTAL_MODELS} split=${SPLIT} task=${TASK} dataset_root=${DATASET_ROOT} output_dir=${OUTPUT_DIR} log_dir=${LOG_DIR}"

MODEL_INDEX=0
for MODEL in "${MODEL_LIST[@]}"; do
  MODEL_INDEX=$((MODEL_INDEX + 1))
  log "[${MODEL_INDEX}/${TOTAL_MODELS}] Evaluating model=${MODEL} split=${SPLIT}"
  OUTPUT_PATH="${OUTPUT_DIR}/${MODEL//\//_}.json"
  LOG_PATH="${LOG_DIR}/${MODEL//\//_}.log"
  RUN_ARGS=("${COMMON_ARGS[@]}")
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    RUN_ARGS+=(--wandb-run-name "${WANDB_RUN_PREFIX}-${MODEL}")
  fi
  START_TS=$(date +%s)
  log "Writing model log to ${LOG_PATH}"
  if ! run_with_heartbeat "${LOG_PATH}" \
    python -m eval.runner \
      "${RUN_ARGS[@]}" \
      --models "${MODEL}" \
      --output-json "${OUTPUT_PATH}"
  then
    warn "Evaluation failed for ${MODEL}; skipping. Last log lines:"
    tail -n 20 "${LOG_PATH}" >&2 || true
    continue
  fi
  END_TS=$(date +%s)
  log "Finished model=${MODEL} in $((END_TS - START_TS))s. Metrics saved to ${OUTPUT_PATH}"
done

log "Completed model evaluation run."
