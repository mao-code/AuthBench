#!/usr/bin/env bash
# Sweep causal LLMs with baseline + self-consistency evaluation and write comparison summaries.
# Run from the repository root (AuthBench).
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# sbatch -p nlplarge-sasha-highpri --nodelist=nlplarge-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=128G -t 720:00:00 eval/scripts/self-consistency/eval_all_llms.sh

OUTPUT_DIR="${OUTPUT_DIR:-eval/results/self_consistency}"
mkdir -p "${OUTPUT_DIR}"

# Define a fixed default list here to avoid sweeping every supported causal LLM.
# Leave this empty to fall back to the registry-wide self-consistency model list.
DEFAULT_MODEL_LIST=(
  qwen2.5-3b-instruct
  qwen2.5-7b-instruct
  llama3-8b
  llama3-8b-instruct
  deepseek-llm-7b-base
  deepseek-coder-6.7b-instruct
)

if [[ $# -gt 0 ]]; then
  MODEL_LIST=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<<"${MODELS}"
elif [[ -n "${MODEL_LIST_FILE:-}" ]]; then
  mapfile -t MODEL_LIST < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "${MODEL_LIST_FILE}")
elif [[ ${#DEFAULT_MODEL_LIST[@]} -gt 0 ]]; then
  MODEL_LIST=("${DEFAULT_MODEL_LIST[@]}")
else
  mapfile -t MODEL_LIST < <(
    python - <<'PY'
from utilities import model_registry
for name in model_registry.self_consistency_model_names():
    print(name)
PY
  )
fi

echo ">>> Running self-consistency sweep for ${#MODEL_LIST[@]} model(s): ${MODEL_LIST[*]}"

for MODEL in "${MODEL_LIST[@]}"; do
  if ! eval/scripts/self-consistency/eval_model.sh "${MODEL}"; then
    echo "[WARN] Comparison run failed for ${MODEL}; skipping." >&2
    continue
  fi
done

python eval/scripts/self-consistency/aggregate_comparisons.py \
  --comparison-dir "${OUTPUT_DIR}/comparison" \
  --output-json "${OUTPUT_DIR}/comparison_summary.json" \
  --output-csv "${OUTPUT_DIR}/comparison_summary.csv"
