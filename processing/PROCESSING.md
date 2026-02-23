# AuthBench Processing Specification (Implemented Pipeline)

This document describes the current, runnable data construction pipeline in `processing/`.

The authoritative pipeline entrypoint is:

```bash
python -m processing.construct_benchmark ...
```

It combines:
- stage-1 build (ingest/chunk/filter/author-caps/sampling/split)
- stage-2 post-filter + dedup + final resampling/split
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

## 2. Stage-1 Build Logic

Stage-1 is equivalent to the former `build_benchmark.py` behavior:

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

Outputs:
- `<stage1_output_dir>/{train,dev,test}/*.jsonl`
- `<stage1_output_dir>/processing_summary.json`
- `<stage1_output_dir>/sampling_shortfall.json`

---

## 3. Stage-2 Postprocess + Dedup Logic

Input: stage-1 candidates from all splits.

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

### 3.3 Final sampling and split

After filter+dedup:
- compute language targets
- sample toward `--post-target-total` (or all remaining docs)
- split to train/dev/test
- write final retrieval files

Outputs:
- `<output_dir>/{train,dev,test}/*.jsonl`
- `<output_dir>/postprocessing_summary.json`
- `<output_dir>/postprocess_dirty.log` (if any drop records)

---

## 4. Integrated Monitoring

`processing.construct_benchmark` writes a unified report:

- pipeline configuration
- stage-1 summary and shortfall
- stage-2 filter + dedup + sampling stats
- transition counts across stages

Default report path:
- `processing/outputs/monitoring/pipeline_dynamics.json`

For phase2 runs this is usually set to:
- `processing/second_phase_web_crawling/outputs/monitoring/pipeline_dynamics_<run_tag>.json`

---

## 5. Main Commands

## 5.1 Phase1 construction (single command)

```bash
python -m processing.construct_benchmark \
  --manifest processing/datasets_manifest.json \
  --stage1-output-dir processing/outputs/stage1_phase1_official \
  --output-dir processing/outputs/stage2_phase1_official \
  --report-path processing/outputs/monitoring/pipeline_dynamics_phase1_official.json \
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
  --total-docs 60000 \
  --post-target-total 60000 \
  --monitor-overwrite
```

Recommended script:
- `processing/second_phase_web_crawling/scripts/run_webcrawl_60k_all4.sh`

---

## 5.3 Merge phase1 + phase2 to 1M (phase2 >= 40%)

```bash
python -m processing.combine_phase_benchmarks \
  --phase1-dir processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced \
  --phase2-dir processing/second_phase_web_crawling/outputs/stage2_all4_t60k \
  --output-dir processing/outputs/combined_phase1_phase2_1m \
  --report-path processing/outputs/combined_phase1_phase2_1m/merge_summary.json \
  --total-docs 1000000 \
  --min-phase2-share 0.40 \
  --seed 42
```

Script wrapper:
- `processing/scripts/combine_phase1_phase2_1m.sh`

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
