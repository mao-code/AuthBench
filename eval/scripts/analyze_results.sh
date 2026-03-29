#!/usr/bin/env bash
# Analyze zero-shot evaluation results, merge in baselines, export organized tables,
# generate bar-chart visualizations, and write a fine-grained markdown report.
# Usage (from repo root): eval/scripts/analyze_results.sh
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

RESULTS_DIR="${RESULTS_DIR:-eval/results}"
BASELINES_DIR="${BASELINES_DIR:-eval/results/baselines}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/analysis}"
DATASET_ROOT="${DATASET_ROOT:-processing/outputs/authbench}"
SPLIT="${SPLIT:-test}"
CANDIDATE_POOL="${CANDIDATE_POOL:-all}"
MAX_TOPIC_CANDIDATES="${MAX_TOPIC_CANDIDATES:-}"
MAX_QUERIES="${MAX_QUERIES:-}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"
SEED="${SEED:-13}"
TOPIC_SEED="${TOPIC_SEED:-13}"
BASELINE_MODELS="${BASELINE_MODELS:-tfidf ngram ppm}"
PLOT_FORMATS="${PLOT_FORMATS:-png pdf}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Results directory does not exist: ${RESULTS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${BASELINES_DIR}" ]]; then
  echo "Baselines directory does not exist: ${BASELINES_DIR}" >&2
  exit 1
fi

read -r -a BASELINE_MODELS_ARR <<<"${BASELINE_MODELS}"
read -r -a PLOT_FORMATS_ARR <<<"${PLOT_FORMATS}"

ANALYZE_ARGS=(
  --results-dir "${RESULTS_DIR}"
  --baselines-dir "${BASELINES_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-root "${DATASET_ROOT}"
  --split "${SPLIT}"
  --candidate-pool "${CANDIDATE_POOL}"
  --seed "${SEED}"
  --topic-seed "${TOPIC_SEED}"
  --baseline-models "${BASELINE_MODELS_ARR[@]}"
  --plot-formats "${PLOT_FORMATS_ARR[@]}"
)

if [[ -n "${MAX_TOPIC_CANDIDATES}" ]]; then
  ANALYZE_ARGS+=(--max-topic-candidates "${MAX_TOPIC_CANDIDATES}")
fi
if [[ -n "${MAX_QUERIES}" ]]; then
  ANALYZE_ARGS+=(--max-queries "${MAX_QUERIES}")
fi
if [[ -n "${MAX_CANDIDATES}" ]]; then
  ANALYZE_ARGS+=(--max-candidates "${MAX_CANDIDATES}")
fi
if [[ -n "${METRICS:-}" ]]; then
  read -r -a METRICS_ARR <<<"${METRICS}"
  ANALYZE_ARGS+=(--metrics "${METRICS_ARR[@]}")
fi
if [[ "${SKIP_PLOTS:-0}" != "0" ]]; then
  ANALYZE_ARGS+=(--skip-plots)
fi

echo "Analyzing model results from ${RESULTS_DIR}"
echo "Using baselines from ${BASELINES_DIR}"
echo "Benchmark root=${DATASET_ROOT} split=${SPLIT} candidate_pool=${CANDIDATE_POOL}"
echo "Writing analysis outputs to ${OUTPUT_DIR}"

python -m eval.analyze_results "${ANALYZE_ARGS[@]}"

echo "Done. Check ${OUTPUT_DIR}/tables, ${OUTPUT_DIR}/plots, ${OUTPUT_DIR}/reports, and ${OUTPUT_DIR}/metadata."
