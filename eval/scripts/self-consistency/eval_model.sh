#!/usr/bin/env bash
# Run baseline and score-aggregated self-consistency evaluation for a single causal LLM model.
# Usage: eval/scripts/self-consistency/eval_model.sh <model-name>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-name>" >&2
  exit 1
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL="$1"
DEFAULT_DATASET_CANDIDATES=(
  "processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced"
  "processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed"
  "processing/outputs/combined_phase1_official_plus_phase2_all4_all_docs"
  "processing/outputs/pipeline_phase1_official"
)

resolve_dataset_root() {
  if [[ -n "${DATASET_ROOT:-}" ]]; then
    if [[ -d "${DATASET_ROOT}/${SPLIT:-test}" ]]; then
      printf '%s\n' "${DATASET_ROOT}"
      return 0
    fi
    echo "[ERROR] DATASET_ROOT does not contain split '${SPLIT:-test}': ${DATASET_ROOT}" >&2
    return 1
  fi

  local candidate
  for candidate in "${DEFAULT_DATASET_CANDIDATES[@]}"; do
    if [[ -d "${candidate}/${SPLIT:-test}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "[ERROR] Could not find a dataset root with split '${SPLIT:-test}'." >&2
  echo "[ERROR] Checked:" >&2
  for candidate in "${DEFAULT_DATASET_CANDIDATES[@]}"; do
    echo "  - ${candidate}" >&2
  done
  return 1
}

SPLIT="${SPLIT:-test}"
DATASET_ROOT="$(resolve_dataset_root)"
TASK="${TASK:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/self_consistency}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
POOLING="${POOLING:-mean}"
NEG_PER_QUERY="${NEG_PER_QUERY:-50}"
SELF_CONSISTENCY_SAMPLES="${SELF_CONSISTENCY_SAMPLES:-4}"
SELF_CONSISTENCY_TOP_K="${SELF_CONSISTENCY_TOP_K:-50}"
SELF_CONSISTENCY_TEMPERATURE="${SELF_CONSISTENCY_TEMPERATURE:-0.8}"
SELF_CONSISTENCY_MAX_NEW_TOKENS="${SELF_CONSISTENCY_MAX_NEW_TOKENS:-96}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_SELF_CONSISTENCY="${RUN_SELF_CONSISTENCY:-1}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-self-consistency-compare}"

echo ">>> Using dataset root: ${DATASET_ROOT}"

BASELINE_OUTPUT_DIR="${OUTPUT_DIR}/baseline"
SELF_CONSISTENCY_OUTPUT_DIR="${OUTPUT_DIR}/self_consistency"
COMPARISON_OUTPUT_DIR="${OUTPUT_DIR}/comparison"
mkdir -p "${BASELINE_OUTPUT_DIR}" "${SELF_CONSISTENCY_OUTPUT_DIR}" "${COMPARISON_OUTPUT_DIR}"

COMMON_ARGS=(
  --task "${TASK}"
  --split "${SPLIT}"
  --dataset-root "${DATASET_ROOT}"
  --models "${MODEL}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --pooling "${POOLING}"
  --negatives-per-query "${NEG_PER_QUERY}"
)

if [[ "${NO_TRUNCATION:-1}" != "0" ]]; then
  COMMON_ARGS+=(--no-truncation)
fi
if [[ -n "${TORCH_DTYPE:-}" ]]; then
  COMMON_ARGS+=(--torch-dtype "${TORCH_DTYPE}")
fi
if [[ -n "${MAX_QUERIES:-}" ]]; then
  COMMON_ARGS+=(--max-queries "${MAX_QUERIES}")
fi
if [[ -n "${MAX_CANDIDATES:-}" ]]; then
  COMMON_ARGS+=(--max-candidates "${MAX_CANDIDATES}")
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

BASELINE_OUTPUT_PATH="${BASELINE_OUTPUT_DIR}/${MODEL//\//_}.json"
SELF_CONSISTENCY_OUTPUT_PATH="${SELF_CONSISTENCY_OUTPUT_DIR}/${MODEL//\//_}.json"
COMPARISON_OUTPUT_PATH="${COMPARISON_OUTPUT_DIR}/${MODEL//\//_}.json"

run_eval() {
  local mode="$1"
  local output_path="$2"
  shift 2

  local -a run_args=("${COMMON_ARGS[@]}" "$@" --output-json "${output_path}")
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    run_args+=(--wandb-project "${WANDB_PROJECT}")
    run_args+=(--wandb-run-name "${WANDB_RUN_PREFIX}-${mode}-${MODEL}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      run_args+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    if [[ -n "${WANDB_TAGS:-}" ]]; then
      for tag in ${WANDB_TAGS}; do
        run_args+=(--wandb-tags "${tag}")
      done
    fi
  fi

  echo ">>> Running ${mode} evaluation for ${MODEL} ..."
  python -m eval.runner "${run_args[@]}"
}

if [[ "${RUN_BASELINE}" != "0" ]]; then
  run_eval "baseline" "${BASELINE_OUTPUT_PATH}"
fi

if [[ "${RUN_SELF_CONSISTENCY}" != "0" ]]; then
  SELF_CONSISTENCY_ARGS=(
    --self-consistency
    --self-consistency-samples "${SELF_CONSISTENCY_SAMPLES}"
    --self-consistency-top-k "${SELF_CONSISTENCY_TOP_K}"
    --self-consistency-temperature "${SELF_CONSISTENCY_TEMPERATURE}"
    --self-consistency-max-new-tokens "${SELF_CONSISTENCY_MAX_NEW_TOKENS}"
  )
  if [[ "${SELF_CONSISTENCY_INCLUDE_ORIGINAL:-0}" != "0" ]]; then
    SELF_CONSISTENCY_ARGS+=(--self-consistency-include-original)
  fi
  if [[ -n "${SELF_CONSISTENCY_PROMPT:-}" ]]; then
    SELF_CONSISTENCY_ARGS+=(--self-consistency-prompt "${SELF_CONSISTENCY_PROMPT}")
  fi
  run_eval "self-consistency" "${SELF_CONSISTENCY_OUTPUT_PATH}" "${SELF_CONSISTENCY_ARGS[@]}"
fi

if [[ -f "${BASELINE_OUTPUT_PATH}" && -f "${SELF_CONSISTENCY_OUTPUT_PATH}" ]]; then
  python eval/scripts/self-consistency/compare_results.py \
    --baseline "${BASELINE_OUTPUT_PATH}" \
    --self-consistency "${SELF_CONSISTENCY_OUTPUT_PATH}" \
    --output "${COMPARISON_OUTPUT_PATH}"
fi
