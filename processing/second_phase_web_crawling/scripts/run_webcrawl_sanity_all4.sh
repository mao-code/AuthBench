#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

TARGET_LANGS="${TARGET_LANGS:-en,zh,hi,es,fr,ar,ru,de,ja,ko}"
RUN_TAG="${RUN_TAG:-sanity_all4_t100}"

# Fast sanity run:
# - Reuse existing StackExchange/Gutenberg/Wikisource corpora by default
# - Crawl fresh YTComments samples to validate new integration
SKIP_STACKEXCHANGE="${SKIP_STACKEXCHANGE:-1}"
SKIP_GUTENBERG="${SKIP_GUTENBERG:-1}"
SKIP_WIKISOURCE="${SKIP_WIKISOURCE:-1}"
SKIP_YTCOMMENTS="${SKIP_YTCOMMENTS:-0}"

TARGET_TOTAL="${TARGET_TOTAL:-100}"
POST_TARGET_TOTAL="${POST_TARGET_TOTAL:-100}"
MAX_DOCUMENTS_PER_DATASET="${MAX_DOCUMENTS_PER_DATASET:-100}"

YTCOMMENTS_MAX_DOCS="${YTCOMMENTS_MAX_DOCS:-100}"
YTCOMMENTS_LANGUAGES="${YTCOMMENTS_LANGUAGES:-$TARGET_LANGS}"
YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG="${YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG:-10}"
YTCOMMENTS_MAX_COMMENTS_PER_VIDEO="${YTCOMMENTS_MAX_COMMENTS_PER_VIDEO:-30}"
YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO="${YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO:-2}"
YTCOMMENTS_MIN_CHARS="${YTCOMMENTS_MIN_CHARS:-8}"
YTCOMMENTS_SKIP_LANGDETECT="${YTCOMMENTS_SKIP_LANGDETECT:-1}"

AUTO_RAMP=0 \
RAMP_MAX_ROUNDS=1 \
RUN_TAG="$RUN_TAG" \
TARGET_TOTAL="$TARGET_TOTAL" \
POST_TARGET_TOTAL="$POST_TARGET_TOTAL" \
MAX_DOCUMENTS_PER_DATASET="$MAX_DOCUMENTS_PER_DATASET" \
SKIP_STACKEXCHANGE="$SKIP_STACKEXCHANGE" \
SKIP_GUTENBERG="$SKIP_GUTENBERG" \
SKIP_WIKISOURCE="$SKIP_WIKISOURCE" \
SKIP_YTCOMMENTS="$SKIP_YTCOMMENTS" \
YTCOMMENTS_MAX_DOCS="$YTCOMMENTS_MAX_DOCS" \
YTCOMMENTS_LANGUAGES="$YTCOMMENTS_LANGUAGES" \
YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG="$YTCOMMENTS_MAX_VIDEO_PAGES_PER_LANG" \
YTCOMMENTS_MAX_COMMENTS_PER_VIDEO="$YTCOMMENTS_MAX_COMMENTS_PER_VIDEO" \
YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO="$YTCOMMENTS_MAX_COMMENT_PAGES_PER_VIDEO" \
YTCOMMENTS_MIN_CHARS="$YTCOMMENTS_MIN_CHARS" \
YTCOMMENTS_SKIP_LANGDETECT="$YTCOMMENTS_SKIP_LANGDETECT" \
bash processing/second_phase_web_crawling/scripts/run_webcrawl_60k_all4.sh

STAGE2_DIR="processing/second_phase_web_crawling/outputs/stage2_${RUN_TAG}"

python3 - <<PY
import json
from collections import Counter
from pathlib import Path

langs = [x.strip() for x in """${TARGET_LANGS}""".split(",") if x.strip()]
corpora = {
    "stackexchange": Path("processing/second_phase_web_crawling/corpora/stackexchange/stackexchange.jsonl"),
    "gutenberg": Path("processing/second_phase_web_crawling/corpora/gutenberg/gutenberg.jsonl"),
    "wikisource": Path("processing/second_phase_web_crawling/corpora/wikisource/wikisource.jsonl"),
    "ytcomments": Path("processing/second_phase_web_crawling/corpora/ytcomments/ytcomments.jsonl"),
}

print("\\n[Sanity] Language coverage in first 100 docs per dataset:")
union = Counter()
for name, path in corpora.items():
    c = Counter()
    if not path.exists():
        print(f"- {name}: MISSING ({path})")
        continue
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 100:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            lang = str(row.get("lang", "")).strip().lower()
            if lang:
                c[lang] += 1
                union[lang] += 1
    print(f"- {name}: {dict(c)}")

missing = [l for l in langs if union.get(l, 0) == 0]
print(f"- union(first100x4): {dict(union)}")
if missing:
    raise SystemExit(f"ERROR: missing languages in collected sanity samples: {missing}")
print("OK: all target languages present in collected sanity samples.")

stage2_dir = Path("""${STAGE2_DIR}""")
cand_counter = Counter()
for split in ("train", "dev", "test"):
    p = stage2_dir / split / "candidates.jsonl"
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            lang = str(row.get("lang", "")).strip().lower()
            if lang:
                cand_counter[lang] += 1
print(f"[Sanity] Stage2 candidate language counts: {dict(cand_counter)}")
PY

printf "\nSanity run complete. Outputs:\n"
printf "%s\n" "- stage1: processing/second_phase_web_crawling/outputs/stage1_${RUN_TAG}"
printf "%s\n" "- stage2: ${STAGE2_DIR}"
