# AuthBench Processing Pipeline

This package turns the raw datasets listed in `DATASET.md` into the unified benchmark described in `PROCESSING.md`.

Recommended entrypoint:

```
python -m processing.construct_benchmark ...
```

It runs one unified pipeline with multiple processing stages (build, quality filtering, dedup, final sampling/split) plus monitoring.

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
- `--total-docs`: global target size for the build stage (defaults to 100k).
- `--post-target-total`: final target after quality filtering/dedup.
- `--train-ratio/--dev-ratio/--test-ratio`: split ratios (default 0.8/0.1/0.1).
- `--allow-other-languages`: fill leftover budget with non-target languages.
- `--max-chunk-tokens` / `--target-chunk-tokens` / `--min-chunk-tokens`: chunking controls.
- `--chunk-probability`: probability to chunk over-limit documents.
- `--truncate-to-tokens`: punctuation-aware post-chunk truncation cap.
- `--dedup-*`: controls exact/near-text and author-similarity dedup behavior.

Outputs per split (`train|dev|test`):
- `documents.jsonl`: full split documents with `doc_id`, `author_id`, `lang`, `genre`, `content`, `source`, `token_length`.
- `candidates.jsonl`: full documents with `candidate_id`, `author_id`, `lang`, `genre`, `content`, `source`, `token_length`.
- `queries.jsonl`: one query per eligible author in the split (no `author_id` field).
- `ground_truth.jsonl`: `query_id` → positive candidate ids + `author_id`.
- Logs and summaries:
  - pipeline summary: `pipeline_summary.json`
  - quality drops: `quality_filter_drops.log` (if drops exist)
  - monitoring report: `pipeline_dynamics*.json`

## Dataset manifest

See `datasets_manifest.example.json` for a template. Each entry needs:
- `loader`: one of `jsonl`, `csv`, `tsv`, `hf_streaming`.
- `path`: local file path (for tabular/jsonl) or HF dataset name via `extra.hf_dataset`.
- `split`: HF split when using `hf_streaming`.
- `text_field`, `author_field`, `lang_field` (or `static_lang`), optional `genre_field`, `raw_id_field`.
- `preprocess_row`: optional built-in helpers (e.g., `arxiv_first_author`).

## Processing steps implemented

- Standardizes genres using the mapping in `PROCESSING.md` (Section 6).
- Tokenizes with `tiktoken` (`cl100k_base`) when available; falls back to whitespace.
- Splits long docs (>500 tokens) into 100–500 token chunks, preserving author/source/genre.
- Dirty data filters (unique token ratio, symbol ratio, dominant token ratio, zero-length) with logging to `dirty_docs.log`.
- Enforces 3–5 docs per author (Section 9) with a fallback down to 2 when data are scarce.
- Post-filters noisy text (spacing/script/language heuristics), then performs exact/near-text and near-author dedup.
- Samples to language, genre, and length-bucket targets (Sections 5, 8, 12) up to the global doc budget.
- Deterministic train/dev/test split per language (Section 4) and IR files for queries/candidates/ground truth (Section 15).
- Writes one unified monitoring report so stage-by-stage stats do not require a separate monitor run.
