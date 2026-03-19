# AuthBench Processing Pipeline

This package turns the raw datasets listed in `DATASET.md` into the unified benchmark described in `PROCESSING.md`.

Recommended entrypoint:

```
python -m processing.construct_benchmark ...
```

It runs one unified pipeline with five stages: Build & Normalization, Quality Filtering, Redundancy Reduction, Language Audit, and Bucket Balanced Sampling, plus monitoring.

## CLI

Run the module from the repo parent, or set `PYTHONPATH` to the repo parent so
`AuthBench.*` imports resolve.

```
python -m processing.construct_benchmark \
  --manifest processing/datasets_manifest.json \
  --output-dir processing/outputs/pipeline_example \
  --report-path processing/outputs/pipeline_example/pipeline_dynamics.json \
  --overwrite-report \
  --total-docs 100000 \
  --allow-other-languages \
  --max-chunk-tokens 500 \
  --target-chunk-tokens 350 \
  --min-chunk-tokens 50 \
  --chunk-probability 0.8 \
  --truncate-to-tokens 2000 \
  --post-target-total 100000 \
  --sanity-check --sanity-limit 500
```

Key flags:
- `--manifest`: JSON manifest describing where each raw dataset lives and which fields contain text/author/lang/genre.
- `--output-dir`: unified pipeline output directory.
- `--work-dir`: optional persistent intermediate-work directory (otherwise temporary and auto-cleaned).
- `--report-path`: unified monitoring report path.
- `--sanity-check` + `--sanity-limit`: cap records per dataset for quick validation.
- `--total-docs`: final benchmark target when `--post-target-total` is not provided.
- `--post-target-total`: final target after quality filtering and redundancy reduction.
- `--train-ratio/--dev-ratio/--test-ratio`: split ratios (default 0.8/0.1/0.1).
- `--allow-other-languages`: fill leftover budget with non-target languages.
- `--max-chunk-tokens` / `--target-chunk-tokens` / `--min-chunk-tokens`: chunking controls.
- `--chunk-probability`: probability to chunk over-limit documents.
- `--truncate-to-tokens`: punctuation-aware post-chunk truncation cap.
- `--dedup-*`: controls exact/near-text and author-similarity dedup behavior.
- `--lang-audit-*`: controls automated language-tag audit and optional mismatch dropping.

Outputs per split (`train|dev|test`):
- `candidates.jsonl`: full documents with `candidate_id`, `author_id`, `lang`, `genre`, `content`, `source`, `token_length`.
- `queries.jsonl`: one query per eligible author in the split (no `author_id` field).
- `ground_truth.jsonl`: `query_id` → positive candidate ids + `author_id`.
- Logs and summaries:
  - pipeline summary: `pipeline_summary.json`
  - quality drops: `quality_filter_drops.log` (if drops exist)
  - language-audit suspects: `language_audit_suspects.jsonl` (if suspicious rows exist)
  - monitoring report: `pipeline_dynamics*.json`

## Dataset manifest

See `processing/datasets_manifest.json` and
`processing/second_phase_web_crawling/datasets_manifest.json` for the current
manifest structure. Each entry needs:
- `loader`: one of `jsonl`, `csv`, `tsv`, `hf_streaming`.
  The current code also supports `blog_authorship`.
- `path`: local file path (for tabular/jsonl) or HF dataset name via `extra.hf_dataset`.
- `split`: HF split when using `hf_streaming`.
- `text_field`, `author_field`, `lang_field` (or `static_lang`), optional `genre_field`, `raw_id_field`.
- `preprocess_row`: optional built-in helpers (e.g., `arxiv_first_author`).

## Processing steps implemented

- Stage 1: Build & Normalization standardizes source rows into the shared schema, tokenizes with `tiktoken` (`cl100k_base`), chunks long docs, applies first-pass dirty filtering, and enforces 3-5 docs per author with a fallback down to 2 when data are scarce.
- Stage 2: Quality Filtering cleans spacing and script artifacts, recomputes token statistics, and removes low-information or mismatched text.
- Stage 3: Redundancy Reduction performs exact/near-text dedup plus optional near-author dedup.
- Stage 4: Language Audit retags or drops high-confidence language mismatches and emits manual-review suspects.
- Stage 5: Bucket Balanced Sampling applies language, genre, and length-bucket targets, then writes deterministic train/dev/test retrieval files.
- Standardizes genres using the mapping in `processing/config.py`.
- Writes one unified monitoring report so stage-by-stage stats do not require a separate monitor run.
