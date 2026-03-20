#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-processing/datasets_manifest.json}"
RUN_TAG="${RUN_TAG:-phase1_sanity}"

TOTAL_DOCS="${TOTAL_DOCS:-100}"
POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-$TOTAL_DOCS}"
SEED="${SEED:-42}"
SANITY_LIMIT="${SANITY_LIMIT:-100}"
PREFLIGHT_LIMIT="${PREFLIGHT_LIMIT:-1}"

MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-$SANITY_LIMIT}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100}"
CHUNK_PROBABILITY="${CHUNK_PROBABILITY:-0.7}"
TRUNCATE_TO_TOKENS="${TRUNCATE_TO_TOKENS:-2000}"
ALLOW_OTHER_LANGUAGES="${ALLOW_OTHER_LANGUAGES:-1}"
DISABLE_LANG_AUDIT="${DISABLE_LANG_AUDIT:-0}"
LANG_AUDIT_DROP_DETECTED_MISMATCHES="${LANG_AUDIT_DROP_DETECTED_MISMATCHES:-0}"
LANG_AUDIT_MAX_DETECT_DOCS="${LANG_AUDIT_MAX_DETECT_DOCS:-1000}"
LANG_AUDIT_MAX_SUSPECTS="${LANG_AUDIT_MAX_SUSPECTS:-500}"

OUTPUT_DIR="${OUTPUT_DIR:-processing/outputs/pipeline_${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-${OUTPUT_DIR}/pipeline_dynamics.json}"

AUTO_FETCH_LOCAL_SOURCES="${AUTO_FETCH_LOCAL_SOURCES:-0}"
DATASET_MAX_DOCS="${DATASET_MAX_DOCS:-}"
NO_SHUFFLE_DATASETS="${NO_SHUFFLE_DATASETS:-}"

if [[ "$AUTO_FETCH_LOCAL_SOURCES" == "1" ]]; then
  echo "Fetching any missing local phase-1 sources before sanity run..."
  bash processing/scripts/download_missing_phase1_sources.sh
fi

echo "Preflight: verifying every dataset in ${MANIFEST_PATH} is reachable..."
"$PYTHON_BIN" - "$MANIFEST_PATH" "$PREFLIGHT_LIMIT" <<'PY'
import sys
from pathlib import Path

from processing.datasets import iter_dataset, load_manifest

manifest_path = Path(sys.argv[1])
preflight_limit = int(sys.argv[2])

configs = load_manifest(manifest_path)
local_configs = [cfg for cfg in configs if cfg.loader in {"jsonl", "csv", "tsv"}]
remote_configs = [cfg for cfg in configs if cfg.loader not in {"jsonl", "csv", "tsv"}]


def check_dataset(cfg):
    dataset_label = f"{cfg.name} (loader={cfg.loader}, source={cfg.source})"
    location = cfg.path.as_posix() if cfg.path else cfg.extra.get("hf_dataset", cfg.extra.get("hf_repo", "<none>"))
    try:
        iterator = iter(iter_dataset(cfg, preflight_limit))
        first_doc = next(iterator)
    except StopIteration:
        return f"{cfg.name}: yielded zero documents from {location}", (
            f"[empty] {dataset_label} target={location}"
        )
    except Exception as exc:
        return f"{cfg.name}: {exc}", (
            f"[error] {dataset_label} target={location} error={exc}"
        )

    return None, (
        f"[ok] {dataset_label} target={location} sample_raw_id={first_doc.raw_id} "
        f"lang={first_doc.lang}"
    )


local_failures = []
for cfg in local_configs:
    failure, message = check_dataset(cfg)
    print(message, file=sys.stderr if failure else sys.stdout, flush=True)
    if failure:
        local_failures.append(failure)

if local_failures:
    print("", file=sys.stderr, flush=True)
    print(
        "Phase-1 sanity preflight failed on local datasets. Fix those paths, then rerun.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

remote_failures = []
for cfg in remote_configs:
    failure, message = check_dataset(cfg)
    print(message, file=sys.stderr if failure else sys.stdout, flush=True)
    if failure:
        remote_failures.append(failure)

if remote_failures:
    print("", file=sys.stderr, flush=True)
    print("Phase-1 sanity preflight failed. Fix the datasets above, then rerun.", file=sys.stderr, flush=True)
    sys.exit(1)
PY

echo "Preflight passed. Running the full pipeline on a tiny sample..."

CMD=(
  "$PYTHON_BIN" -m processing.construct_benchmark
  --manifest "$MANIFEST_PATH"
  --output-dir "$OUTPUT_DIR"
  --report-path "$REPORT_PATH"
  --overwrite-report
  --total-docs "$TOTAL_DOCS"
  --post-target-total "$POST_TARGET_TOTAL"
  --seed "$SEED"
  --sanity-check
  --sanity-limit "$SANITY_LIMIT"
  --max-documents-per-dataset "$MAX_DOCUMENTS_PER_DATASET"
  --shuffle-buffer-size "$SHUFFLE_BUFFER_SIZE"
  --chunk-probability "$CHUNK_PROBABILITY"
  --truncate-to-tokens "$TRUNCATE_TO_TOKENS"
  --lang-audit-max-detect-docs "$LANG_AUDIT_MAX_DETECT_DOCS"
  --lang-audit-max-suspects "$LANG_AUDIT_MAX_SUSPECTS"
  --log-level INFO
)

if [[ "$ALLOW_OTHER_LANGUAGES" == "1" ]]; then
  CMD+=(--allow-other-languages)
fi

if [[ -n "$DATASET_MAX_DOCS" ]]; then
  read -r -a _CAPS <<<"$DATASET_MAX_DOCS"
  CMD+=(--dataset-max-docs "${_CAPS[@]}")
fi

if [[ -n "$NO_SHUFFLE_DATASETS" ]]; then
  read -r -a _NOSHUFFLE <<<"$NO_SHUFFLE_DATASETS"
  CMD+=(--no-shuffle-datasets "${_NOSHUFFLE[@]}")
fi

if [[ "$DISABLE_LANG_AUDIT" == "1" ]]; then
  CMD+=(--disable-lang-audit)
fi

if [[ "$LANG_AUDIT_DROP_DETECTED_MISMATCHES" == "1" ]]; then
  CMD+=(--lang-audit-drop-detected-mismatches)
fi

"${CMD[@]}"

echo "Validating pipeline outputs..."
"$PYTHON_BIN" - "$OUTPUT_DIR" "$REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
report_path = Path(sys.argv[2])

required_paths = [
    report_path,
    output_dir / "pipeline_summary.json",
    output_dir / "train" / "candidates.jsonl",
    output_dir / "train" / "queries.jsonl",
    output_dir / "train" / "ground_truth.jsonl",
    output_dir / "dev" / "candidates.jsonl",
    output_dir / "dev" / "queries.jsonl",
    output_dir / "dev" / "ground_truth.jsonl",
    output_dir / "test" / "candidates.jsonl",
    output_dir / "test" / "queries.jsonl",
    output_dir / "test" / "ground_truth.jsonl",
]

missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    print("Missing expected pipeline outputs:", file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    sys.exit(1)

summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
finalize = summary.get("finalize", {})
split_summary = finalize.get("splits", {})
print(
    "Sanity run complete: "
    f"after_language_audit={finalize.get('after_language_audit', {}).get('total', 0)} "
    f"after_sampling={finalize.get('after_sampling', {}).get('total', 0)} "
    f"train_docs={split_summary.get('train', {}).get('documents', 0)} "
    f"dev_docs={split_summary.get('dev', {}).get('documents', 0)} "
    f"test_docs={split_summary.get('test', {}).get('documents', 0)}"
, flush=True)
PY

echo "Done."
echo "Output: $OUTPUT_DIR"
echo "Report: $REPORT_PATH"
