#!/usr/bin/env bash
# Train a single registry model for 1 epoch with mid-epoch evaluation.
# Run from the repository root (AuthBench).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-name> [extra-args]" >&2
  exit 1
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL="$1"
shift || true

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/training_summary}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_FRACTION="${EVAL_FRACTION:-0.5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

echo ">>> Training ${MODEL} ..."
python eval/train.py \
  --model "$MODEL" \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --eval-fraction-epoch "$EVAL_FRACTION" \
  --eval-every-epoch \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --log-file "$OUTPUT_DIR/$MODEL/eval_history.jsonl" \
  --skip-checkpoint \
  --lora-rank "$LORA_RANK" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-target-modules "$LORA_TARGET_MODULES" \
  "$@"
