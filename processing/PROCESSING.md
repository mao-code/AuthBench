# AuthBench Processing Specification (Implemented Pipeline)

This document describes the current, runnable data construction pipeline in `processing/`.

The authoritative pipeline entrypoint is:

```bash
python -m processing.construct_benchmark ...
```

It combines:
- build (ingest/chunk/filter/author-caps/sampling/split)
- quality filtering + dedup + language audit + final resampling/split
- integrated monitoring report

You no longer need to run a separate monitor command to obtain stage-by-stage stats.

---

## 1. Target Output Schema

Final `candidates.jsonl` rows:

```json
{
  "candidate_id": "doc_000123",
  "author_id": "<sha256(source:author)>",
  "lang": "en",
  "genre": "social_media/news",
  "content": "<text>",
  "source": "exorde",
  "token_length": 120
}
```

Final `queries.jsonl` rows:

```json
{
  "query_id": "doc_000456",
  "lang": "en",
  "genre": "social_media/news",
  "content": "<text>",
  "source": "exorde",
  "token_length": 115
}
```

Final `ground_truth.jsonl` rows:

```json
{
  "query_id": "doc_000456",
  "positive_ids": ["doc_000123", "doc_000321"],
  "author_id": "<hashed_author>"
}
```

---

## 2. Build Stage Logic

Build stage behavior is equivalent to the former `build_benchmark.py` behavior:

1. Load datasets from manifest (`processing/datasets.py`).
2. Optional bounded stream shuffle per dataset.
3. Chunk long docs (`processing/chunker.py`).
4. Optional token truncation.
5. Dirty filtering (`processing/dirty.py`):
   - low unique token ratio
   - high symbol ratio
   - long symbol runs (PD-focused)
   - long repeated char runs
6. Author constraints:
   - target 3-5 docs per author (fallback supports 2 docs)
7. Sampling to language/genre/length targets (`processing/sampling.py`).
8. Stratified split by language (`train/dev/test`).
9. Retrieval set materialization (`candidates/queries/ground_truth`).

Build artifacts are internal (temporary by default, or persisted if `--work-dir` is set).

---

## 3. Quality Filter + Dedup + Language Audit + Final Sampling

Input: build-stage documents from the internal pipeline working set.

### 3.1 Post filtering

Applied per document:
- collapse spaced letter artifacts
- dirty re-check
- script/language consistency heuristics
- low-information/untranslatable heuristics

### 3.2 Dedup stage (implemented)

Dedup is now an explicit stage in real code (`processing/deduplication.py`):

1. Exact normalized-text dedup.
2. Near-text dedup:
   - SimHash (64-bit) on unigram+bigram features
   - LSH bucket candidate generation
   - Hamming threshold from similarity parameter
3. Near-author dedup:
   - author profile SimHash over representative docs
   - configurable cross-source-only constraint
   - drops weaker duplicate author profile

Default behavior is enabled in the unified constructor and can be configured by CLI flags.

### 3.3 Language audit (automated + manual-review support)

After dedup:
- checks `lang` tags via script consistency and optional `langdetect` verification
- retags high-confidence `langdetect` mismatches instead of dropping by default
- writes suspicious rows to `<output_dir>/language_audit_suspects.jsonl`
- records metrics in `pipeline_summary.json` and `pipeline_dynamics.json`
- can optionally drop high-confidence detected mismatches via:
  - `--lang-audit-drop-detected-mismatches`

### 3.4 Final sampling and split

After filter+dedup:
- compute language targets
- sample toward `--post-target-total` (or all remaining docs)
- split to train/dev/test
- write final retrieval files

Outputs:
- `<output_dir>/{train,dev,test}/*.jsonl`
- `<output_dir>/pipeline_summary.json`
- `<output_dir>/quality_filter_drops.log` (if any drop records)
- `<output_dir>/language_audit_suspects.jsonl` (if suspicious rows exist)

Quick manual check:

```bash
jq -c '.' <output_dir>/language_audit_suspects.jsonl | head -n 20
```

---

## 4. Integrated Monitoring

`processing.construct_benchmark` writes a unified report:

- pipeline configuration
- build-stage summary and shortfall
- quality-filter + dedup + sampling stats
- transition counts across stages

Default report path:
- `<output_dir>/pipeline_dynamics.json`

For phase2 runs with the wrapper script, this defaults to:
- `<output_dir>/pipeline_dynamics.json`

---

## 5. Main Commands

## 5.1 Phase1 construction (single command)

```bash
python -m processing.construct_benchmark \
  --manifest processing/datasets_manifest.json \
  --output-dir processing/outputs/pipeline_phase1_official \
  --report-path processing/outputs/pipeline_phase1_official/pipeline_dynamics.json \
  --overwrite-report \
  --total-docs 300000 \
  --allow-other-languages \
  --max-documents-per-dataset 10000000 \
  --shuffle-buffer-size 10000 \
  --chunk-probability 0.7 \
  --truncate-to-tokens 2000 \
  --seed 42
```

Script wrapper:
- `processing/scripts/run_phase1_construction.sh`

---

## 5.2 Phase2 web crawl + construction

Unified phase2 runner:

```bash
python -m processing.second_phase_web_crawling.run_pipeline \
  --stages crawl construct \
  --output-dir processing/second_phase_web_crawling/outputs/pipeline_all4_t300k_cap10M \
  --total-docs 300000 \
  --post-target-total 300000 \
  --monitor-overwrite
```

Recommended script:
- `processing/second_phase_web_crawling/scripts/run_webcrawl_300k_cap10M_all4.sh`

---

## 5.3 Merge phase1 + phase2 (official defaults)

```bash
python -m processing.combine_phase_benchmarks \
  --output-dir processing/outputs/combined_phase1_official_phase2_webcrawl \
  --report-path processing/outputs/combined_phase1_official_phase2_webcrawl/merge_summary.json \
  --min-phase2-share 0.50 \
  --seed 42
```

Notes:
- Defaults already point to:
  - `processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced`
  - `processing/second_phase_web_crawling/outputs/pipeline_all4_t300k_cap10M`
- If `--total-docs` is omitted, the combiner uses all available docs after dedup + overlap removal.

---

## 6. Important CLI Flags

Build/ingest:
- `--total-docs`
- `--max-documents-per-dataset`
- `--dataset-max-docs`
- `--shuffle-buffer-size`
- `--chunk-probability`
- `--truncate-to-tokens`

Post-filter:
- `--post-target-total`
- `--post-spacing-collapse-ratio`
- `--post-max-single-letter-ratio`
- `--post-min-alpha-ratio`
- `--post-skip-langdetect`

Dedup:
- `--disable-dedup`
- `--dedup-near-similarity-threshold`
- `--dedup-author-similarity-threshold`
- `--dedup-min-tokens-for-near`
- `--dedup-lsh-bands`

Language audit:
- `--disable-lang-audit`
- `--lang-audit-max-detect-docs`
- `--lang-audit-min-confidence`
- `--lang-audit-drop-detected-mismatches`

Monitoring:
- `--report-path`
- `--overwrite-report`

---

## 7. Legacy Modules

Legacy standalone modules still exist for backward compatibility and debugging:
- `processing.build_benchmark`
- `processing.postprocess`
- `processing.monitor_pipeline`

For routine production construction, prefer:
- `processing.construct_benchmark`
