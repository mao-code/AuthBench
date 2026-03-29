#!/usr/bin/env bash
# Train and evaluate PART, LUAR, and STEL on a single base model.
# Run from the repository root (AuthBench).
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/authbench}"
BASE_MODEL="${BASE_MODEL:-qwen3-emb-4b}"
METHODS_STR="${METHODS:-part luar stel}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/authorship_methods}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-1}"
EVAL_FRACTION="${EVAL_FRACTION:-0.25}"
EVAL_KS="${EVAL_KS:-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

LORA_RANK="${LORA_RANK:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

PART_HIDDEN_SIZE="${PART_HIDDEN_SIZE:-512}"
LUAR_WINDOW_SIZE="${LUAR_WINDOW_SIZE:-32}"
LUAR_MAX_EPISODE_DOCS="${LUAR_MAX_EPISODE_DOCS:-4}"
LUAR_EVAL_EPISODE_DOCS="${LUAR_EVAL_EPISODE_DOCS:-4}"
LUAR_EMBEDDING_SIZE="${LUAR_EMBEDDING_SIZE:-512}"
LUAR_TEMPERATURE="${LUAR_TEMPERATURE:-0.01}"
STEL_MARGIN="${STEL_MARGIN:-0.5}"
STEL_CONTROL_KEYS="${STEL_CONTROL_KEYS:-genre source}"

SKIP_CHECKPOINT="${SKIP_CHECKPOINT:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

WANDB_PROJECT="${WANDB_PROJECT:-AuthBench}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-authorship-methods}"
WANDB_TAGS="${WANDB_TAGS:-AuthBench authorship-methods}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
read -r -a METHOD_LIST <<<"${METHODS_STR}"

for METHOD in "${METHOD_LIST[@]}"; do
  METHOD_OUTPUT_DIR="${OUTPUT_DIR}/${METHOD}"
  RUN_DIR="${METHOD_OUTPUT_DIR}/${BASE_MODEL}"
  LOG_PATH="${LOG_DIR}/${BASE_MODEL}-${METHOD}.log"
  mkdir -p "${METHOD_OUTPUT_DIR}"

  log "Training/evaluating method=${METHOD} base_model=${BASE_MODEL}"

  CMD=(
    python -m eval.train
    --dataset-root "${DATASET_ROOT}"
    --model "${BASE_MODEL}"
    --authorship-method "${METHOD}"
    --output-dir "${METHOD_OUTPUT_DIR}"
    --log-file "${RUN_DIR}/eval_history.jsonl"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --eval-fraction-epoch "${EVAL_FRACTION}"
    --eval-every-epoch
    --eval-ks "${EVAL_KS}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --max-length "${MAX_LENGTH}"
    --pooling mean
    --lora-rank "${LORA_RANK}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-target-modules "${LORA_TARGET_MODULES}"
    --part-hidden-size "${PART_HIDDEN_SIZE}"
    --luar-window-size "${LUAR_WINDOW_SIZE}"
    --luar-max-episode-docs "${LUAR_MAX_EPISODE_DOCS}"
    --luar-eval-episode-docs "${LUAR_EVAL_EPISODE_DOCS}"
    --luar-embedding-size "${LUAR_EMBEDDING_SIZE}"
    --luar-temperature "${LUAR_TEMPERATURE}"
    --stel-margin "${STEL_MARGIN}"
    --stel-control-keys ${STEL_CONTROL_KEYS}
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${WANDB_RUN_PREFIX}-${BASE_MODEL}-${METHOD}"
  )

  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  for TAG in ${WANDB_TAGS}; do
    CMD+=(--wandb-tags "${TAG}")
  done
  if [[ "${SKIP_CHECKPOINT}" != "0" ]]; then
    CMD+=(--skip-checkpoint)
  fi
  if [[ "${TRUST_REMOTE_CODE}" != "0" ]]; then
    CMD+=(--trust-remote-code)
  fi
  if [[ -n "${LORA_ALPHA:-}" ]]; then
    CMD+=(--lora-alpha "${LORA_ALPHA}")
  fi
  if [[ -n "${LORA_BIAS:-}" ]]; then
    CMD+=(--lora-bias "${LORA_BIAS}")
  fi
  if [[ -n "${MAX_STEPS:-}" ]]; then
    CMD+=(--max-steps "${MAX_STEPS}")
  fi
  if [[ -n "${MAX_TRAIN_AUTHORS:-}" ]]; then
    CMD+=(--max-train-authors "${MAX_TRAIN_AUTHORS}")
  fi
  if [[ -n "${MAX_EVAL_QUERIES:-}" ]]; then
    CMD+=(--max-eval-queries "${MAX_EVAL_QUERIES}")
  fi
  if [[ -n "${MAX_EVAL_CANDIDATES:-}" ]]; then
    CMD+=(--max-eval-candidates "${MAX_EVAL_CANDIDATES}")
  fi

  "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
  log "Finished method=${METHOD}. Summary: ${RUN_DIR}/training_summary.json"
done

log "Completed authorship-method training/evaluation sweep."
