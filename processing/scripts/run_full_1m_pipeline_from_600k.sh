#!/usr/bin/env bash
set -euo pipefail

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "$ROOT_DIR"

# Master pipeline:
# 1) phase1 build at 600K
# 2) phase2 webcrawl/build at 600K
# 3) combine phase1 + phase2 into 1M benchmark
#
# Default behavior keeps language/genre/length targeting logic from repo:
# - phase1 wrapper defaults ALLOW_OTHER_LANGUAGES=0
# - phase2 path does not pass --allow-other-languages

REPO_ROOT="/home/mh2653/AuthBench"
export PYTHONPATH="${REPO_ROOT}"
# sbatch -p rush --nodelist=rush-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G -t 720:00:00 processing/scripts/run_full_1m_pipeline_from_600k.sh

RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_COMBINE="${RUN_COMBINE:-1}"

PHASE1_RUN_TAG="${PHASE1_RUN_TAG:-phase1_t600k}"
PHASE2_RUN_TAG="${PHASE2_RUN_TAG:-all4_t600k}"
PHASE1_TOTAL_DOCS="${PHASE1_TOTAL_DOCS:-600000}"
PHASE2_TOTAL_DOCS="${PHASE2_TOTAL_DOCS:-600000}"
PHASE1_ALLOW_OTHER_LANGUAGES="${PHASE1_ALLOW_OTHER_LANGUAGES:-0}"

PHASE1_DIR="${PHASE1_DIR:-processing/outputs/stage2_${PHASE1_RUN_TAG}_2}"
PHASE2_DIR="${PHASE2_DIR:-processing/second_phase_web_crawling/outputs/stage2_${PHASE2_RUN_TAG}_2}"

COMBINE_TOTAL_DOCS="${COMBINE_TOTAL_DOCS:-1000000}"
COMBINE_MIN_PHASE2_SHARE="${COMBINE_MIN_PHASE2_SHARE:-0.40}"
COMBINE_ALLOW_LOWER_PHASE2_SHARE="${COMBINE_ALLOW_LOWER_PHASE2_SHARE:-0}"
COMBINE_DISABLE_DEDUP="${COMBINE_DISABLE_DEDUP:-0}"

COMBINED_OUTPUT_DIR="${COMBINED_OUTPUT_DIR:-processing/outputs/combined_phase1_phase2_1m_from600k_2}"
COMBINED_REPORT_PATH="${COMBINED_REPORT_PATH:-${COMBINED_OUTPUT_DIR}/merge_summary.json}"

echo "=== Full 1M pipeline from 600K + 600K ==="
echo "RUN_PHASE1=${RUN_PHASE1} RUN_PHASE2=${RUN_PHASE2} RUN_COMBINE=${RUN_COMBINE}"
echo "phase1_dir=${PHASE1_DIR}"
echo "phase2_dir=${PHASE2_DIR}"
echo "combined_output_dir=${COMBINED_OUTPUT_DIR}"
echo ""

if [[ "${RUN_PHASE1}" == "1" ]]; then
  echo "[1/3] Running phase1 600K construction..."
  RUN_TAG="${PHASE1_RUN_TAG}" \
  TOTAL_DOCS="${PHASE1_TOTAL_DOCS}" \
  POST_TARGET_TOTAL="${PHASE1_TOTAL_DOCS}" \
  ALLOW_OTHER_LANGUAGES="${PHASE1_ALLOW_OTHER_LANGUAGES}" \
  bash ${REPO_ROOT}/processing/scripts/run_phase1_construction_600k.sh
  echo ""
fi

if [[ "${RUN_PHASE2}" == "1" ]]; then
  echo "[2/3] Running phase2 600K webcrawl + construction..."
  RUN_TAG="${PHASE2_RUN_TAG}" \
  TARGET_TOTAL="${PHASE2_TOTAL_DOCS}" \
  POST_TARGET_TOTAL="${PHASE2_TOTAL_DOCS}" \
  bash ${REPO_ROOT}/processing/second_phase_web_crawling/scripts/run_webcrawl_600k_all4.sh
  echo ""
fi

if [[ "${RUN_COMBINE}" == "1" ]]; then
  echo "[3/3] Combining phase1 + phase2 into 1M..."

  if [[ ! -d "${PHASE1_DIR}" ]]; then
    echo "Error: phase1 dir not found: ${PHASE1_DIR}"
    exit 1
  fi
  if [[ ! -d "${PHASE2_DIR}" ]]; then
    echo "Error: phase2 dir not found: ${PHASE2_DIR}"
    exit 1
  fi

  PHASE1_DIR="${PHASE1_DIR}" \
  PHASE2_DIR="${PHASE2_DIR}" \
  OUTPUT_DIR="${COMBINED_OUTPUT_DIR}" \
  REPORT_PATH="${COMBINED_REPORT_PATH}" \
  TOTAL_DOCS="${COMBINE_TOTAL_DOCS}" \
  MIN_PHASE2_SHARE="${COMBINE_MIN_PHASE2_SHARE}" \
  ALLOW_LOWER_PHASE2_SHARE="${COMBINE_ALLOW_LOWER_PHASE2_SHARE}" \
  DISABLE_DEDUP="${COMBINE_DISABLE_DEDUP}" \
  bash ${REPO_ROOT}/processing/scripts/combine_phase1_phase2_1m.sh

  COMBINED_REPORT_PATH="${COMBINED_REPORT_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

report = Path(os.environ["COMBINED_REPORT_PATH"])
if not report.exists():
    print("combine report not found:", report)
    raise SystemExit(0)
obj = json.loads(report.read_text(encoding="utf-8"))
sc = obj.get("stage_counts", {})
print("")
print("Combined stage counts:")
for k in [
    "phase1_loaded",
    "phase2_loaded",
    "phase1_after_internal_dedup",
    "phase2_after_internal_dedup",
    "phase1_selected",
    "phase2_selected",
    "combined_selected",
    "phase2_share_final",
]:
    print(f"- {k}: {sc.get(k)}")
print("splits:")
for s, v in obj.get("splits", {}).items():
    print(
        f"  {s}: documents={v.get('documents')} candidates={v.get('candidates')} "
        f"queries={v.get('queries')} ground_truth={v.get('ground_truth')}"
    )
PY
fi

echo ""
echo "Pipeline complete."
