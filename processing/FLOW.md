# AuthBench Construction Flow (Current Implementation)

This file describes the actual processing flow implemented in `processing/` as of the unified constructor.

Primary entrypoint:

```bash
python -m processing.construct_benchmark ...
```

The constructor executes all stages in one run and writes:
- one unified output directory (split JSONL files + pipeline summary)
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

## Stage 1: Build Corpus (former `build_benchmark.py`)

1. Stream records from each dataset.
2. Optionally bounded-shuffle each dataset stream.
3. Chunk long records (`processing/chunker.py`).
4. Optional token truncation.
5. Dirty filtering (`processing/dirty.py`).
6. Hash author ids (`source + raw_author`) and apply author caps:
   - keep author with 3-5 docs (fallback supports 2 docs).
7. Language/genre/length sampling (`processing/sampling.py`) toward `--total-docs`.
8. Stratified split (`train/dev/test`) by language.
9. Write split retrieval files (`candidates.jsonl`, `queries.jsonl`, `ground_truth.jsonl`).

Build artifacts are written to an internal working directory.
- default: temporary directory (auto-cleaned)
- optional: persistent with `--work-dir`

---

## Stage 2-5: Quality Filter + Dedup + Language Audit + Final Sampling

Input: build-stage documents in memory/working-dir artifacts.

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

All dedup stats are written into `pipeline_summary.json` and the unified report.

### 2.3 Language audit (automated + manual-review artifacts)

After dedup:
- run language-audit checks for `lang` tag consistency:
  - script-to-language consistency checks
  - optional `langdetect` verification on a configurable sample
- emit suspicious rows for manual review:
  - `<output_dir>/language_audit_suspects.jsonl`
  - `<output_dir>/pipeline_summary.json` language-audit metrics
- optional strict mode:
  - drop high-confidence language mismatches with `--lang-audit-drop-detected-mismatches`

### 2.4 Final sampling and split

After filtering+dedup:
- compute language targets
- sample documents
- split into `train/dev/test`
- write retrieval files

Pipeline outputs:
- `<output_dir>/{train,dev,test}/*.jsonl` (retrieval files)
- `<output_dir>/pipeline_summary.json`
- `<output_dir>/quality_filter_drops.log` (if any dropped docs)
- `<output_dir>/language_audit_suspects.jsonl` (if any suspicious rows)

Manual review quick command:

```bash
jq -c '.' <output_dir>/language_audit_suspects.jsonl | head -n 20
```

---

## Stage 3: Unified Monitoring Report (integrated)

No separate `monitor_pipeline` run is required.

The constructor writes one report including:
- pipeline inputs and knobs
- build-stage summary and sampling shortfall
- quality-filter/dedup/split stats
- stage transitions (`build -> filter -> dedup -> language_audit -> final sampling`)

Default report path:
- `<output_dir>/pipeline_dynamics.json`

---

## Phase Combination Flow

Use:

```bash
python -m processing.combine_phase_benchmarks ...
```

Flow:
1. Load phase1 and phase2 final stage docs from retrieval files (`candidates.jsonl` + `queries.jsonl`).
2. Optional per-phase dedup.
3. Cross-phase exact overlap removal (phase2 priority).
4. Enforce total target and minimum phase2 share.
   Defaults are wired to:
   - `processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced`
   - `processing/second_phase_web_crawling/outputs/pipeline_all4_t300k_cap10M`
   If `--total-docs` is not provided, all available docs are merged.
5. Sample phase1 and phase2 pools separately.
6. Merge, assign new IDs, split, and emit retrieval files.
7. Write merge monitoring report.

---

## Practical Entry Scripts

- Phase1 full construction:
  - `processing/scripts/run_phase1_construction.sh`
- Phase2 crawl + construction:
  - `processing/second_phase_web_crawling/scripts/run_webcrawl_300k_cap10M_all4.sh`
- Phase1 + Phase2 merge:
  - use `python -m processing.combine_phase_benchmarks`
