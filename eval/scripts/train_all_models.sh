#!/usr/bin/env bash
# Train selected models for 1 epoch with mid-epoch evaluation.
# Run from the repository root (AuthBench).
set -euo pipefail

# sbatch -p nlplarge-sasha-highpri --nodelist=nlplarge-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=128G -t 720:00:00 eval/scripts/train_all_models.sh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Weights & Biases defaults (override via env).
WANDB_PROJECT="${WANDB_PROJECT:-AuthBench}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-train-all}"
WANDB_TAGS="${WANDB_TAGS:-AuthBench train-all}"

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/combined_phase1_phase2}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/training_summary}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_FRACTION="${EVAL_FRACTION:-0.5}"
EVAL_KS="${EVAL_KS:-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

if [[ -n "${MODELS:-}" ]]; then
  # Override with `MODELS="m1 m2 ..."` if desired.
  read -r -a MODELS <<<"${MODELS}"
else
  MODELS=(
    # LLM-instruct
    "qwen2.5-3b-instruct"
    "llama3-8b-instruct"
    # LLM-base
    "qwen2.5-3b"
    "llama3-8b"
    # Embedding-instruct
    "gte-qwen2-7b-instruct"
    "sfr-embedding-mistral"
    # Embedding
    "multilingual-e5-large"
    "qwen3-embedding-4b"
  )
fi

COMMON_ARGS=(
  --epochs 2
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --eval-fraction-epoch "$EVAL_FRACTION"
  --eval-every-epoch
  --eval-ks $EVAL_KS
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --skip-checkpoint
  --lora-rank "$LORA_RANK"
  --lora-dropout "$LORA_DROPOUT"
  --lora-target-modules "$LORA_TARGET_MODULES"
)

if [[ -n "${LORA_ALPHA:-}" ]]; then
  COMMON_ARGS+=(--lora-alpha "$LORA_ALPHA")
fi
if [[ -n "${LORA_BIAS:-}" ]]; then
  COMMON_ARGS+=(--lora-bias "$LORA_BIAS")
fi

for MODEL in "${MODELS[@]}"; do
  echo ">>> Training ${MODEL} ..."
  WANDB_ARGS=(
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "${WANDB_RUN_PREFIX}-${MODEL}"
  )
  # Only include entity flag if set.
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  # Split space-delimited tags into separate args.
  for TAG in $WANDB_TAGS; do
    WANDB_ARGS+=(--wandb-tags "$TAG")
  done

  python eval/train.py \
    --model "$MODEL" \
    --log-file "$OUTPUT_DIR/$MODEL/eval_history.jsonl" \
    "${COMMON_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
done
