#!/usr/bin/env bash
# Sweep causal LLMs with baseline + self-consistency evaluation and write comparison summaries.
# Run from the repository root (AuthBench).
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-eval/results/self_consistency}"
mkdir -p "${OUTPUT_DIR}"

if [[ $# -gt 0 ]]; then
  MODEL_LIST=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<<"${MODELS}"
else
  mapfile -t MODEL_LIST < <(
    python - <<'PY'
from utilities import model_registry
for name in model_registry.self_consistency_model_names():
    print(name)
PY
  )
fi

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
