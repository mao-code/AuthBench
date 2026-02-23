# AuthBench Construction Flow (Current Implementation)

This file describes the actual processing flow implemented in `processing/` as of the unified constructor.

Primary entrypoint:

```bash
python -m processing.construct_benchmark ...
```

The constructor executes all stages in one run and writes:
- stage-1 outputs (`processing_summary.json`, `sampling_shortfall.json`, split JSONL files)
- final outputs (`postprocessing_summary.json`, split JSONL files)
- one unified monitoring report (`pipeline_dynamics*.json`)

---

## Stage 0: Inputs

Input sources are declared in a manifest JSON (`processing/datasets_manifest.json` for phase1, or second-phase manifest).

Each record is normalized into:
- `raw_id`
- `author`
- `text`
- `lang`
- `source`
- `genre`

---

## Stage 1: Build Candidates (former `build_benchmark.py`)

1. Stream records from each dataset.
2. Optionally bounded-shuffle each dataset stream.
3. Chunk long records (`processing/chunker.py`).
4. Optional token truncation.
5. Dirty filtering (`processing/dirty.py`).
6. Hash author ids (`source + raw_author`) and apply author caps:
   - keep author with 3-5 docs (fallback supports 2 docs).
7. Language/genre/length sampling (`processing/sampling.py`) toward `--total-docs`.
8. Stratified split (`train/dev/test`) by language.
9. Build retrieval files (`candidates.jsonl`, `queries.jsonl`, `ground_truth.jsonl`).

Stage-1 outputs:
- `<stage1_output_dir>/{train,dev,test}/*.jsonl`
- `<stage1_output_dir>/processing_summary.json`
- `<stage1_output_dir>/sampling_shortfall.json`

---

## Stage 2: Post-Filter + Dedup + Final Sampling

Input: stage-1 `candidates.jsonl` from all splits.

### 2.1 Post-filter

Per-doc normalization and quality filtering:
- collapse letter-by-letter spacing artifacts
- dirty heuristics re-check
- language/script sanity checks
- untranslatable-text heuristics

### 2.2 Deduplication (new real stage)

Implemented in `processing/deduplication.py`:

1. Exact-text dedup
   - normalized text hash
2. Near-text dedup
   - SimHash over unigram+bigram features
   - LSH bucket candidate generation
   - Hamming-threshold filtering
3. Cross-source author-similarity dedup
   - author profile SimHash
   - optional same-language and cross-source constraints
   - drop weaker duplicate author profile

All dedup stats are written into `postprocessing_summary.json` and the unified report.

### 2.3 Final sampling and split

After filtering+dedup:
- compute language targets
- sample documents
- split into `train/dev/test`
- write retrieval files

Stage-2 outputs:
- `<output_dir>/{train,dev,test}/*.jsonl`
- `<output_dir>/postprocessing_summary.json`
- `<output_dir>/postprocess_dirty.log` (if any dropped docs)

---

## Stage 3: Unified Monitoring Report (integrated)

No separate `monitor_pipeline` run is required.

The constructor writes one report including:
- pipeline inputs and knobs
- stage-1 summary and sampling shortfall
- stage-2 filtering, dedup, and split stats
- stage transitions (`build -> filter -> dedup -> final sampling`)

Default report path:
- `processing/outputs/monitoring/pipeline_dynamics.json`

---

## Phase Combination Flow (1M final benchmark)

Use:

```bash
python -m processing.combine_phase_benchmarks ...
```

Flow:
1. Load phase1 and phase2 final candidates.
2. Optional per-phase dedup.
3. Cross-phase exact overlap removal (phase2 priority).
4. Enforce total target (default 1,000,000) and minimum phase2 share (default 40%).
5. Sample phase1 and phase2 pools separately.
6. Merge, assign new IDs, split, and emit retrieval files.
7. Write merge monitoring report.

---

## Practical Entry Scripts

- Phase1 full construction:
  - `processing/scripts/run_phase1_construction.sh`
- Phase2 crawl + construction:
  - `processing/second_phase_web_crawling/scripts/run_webcrawl_60k_all4.sh`
- Phase1 + Phase2 merge to 1M:
  - `processing/scripts/combine_phase1_phase2_1m.sh`
