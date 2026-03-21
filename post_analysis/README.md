## Post-analysis for AuthBench

This directory contains the analysis scripts, generated artifacts, and usage instructions for inspecting AuthBench benchmark construction and evaluation outputs.

### Canonical benchmark roots

- Combined benchmark: `processing/outputs/combined_phase1_phase2`
- Phase 1 benchmark: `processing/outputs/pipeline_phase1_official`
- Phase 2 benchmark: `processing/second_phase_web_crawling/outputs/pipeline_phase2_official`

### Directory layout

```text
post_analysis/
├── README.md
├── OUTPUT_LAYOUT.md
├── requirements.txt
├── analyze_dataset.py
├── qualitative_analysis.py
├── analyze_benchmarks.py
├── authorship_benchmark_analysis.py
├── leakage_audit.py
├── plot_results.py
├── run_authbench_analysis.sh
├── run_combined_phase1_phase2_analysis.sh
├── run_analyze_benchmarks.sh
└── outputs/
    ├── combined_phase1_phase2/                  # recommended output root for the merged benchmark
    ├── phase1_vs_phase2/                        # phase1 vs phase2 comparison outputs
    └── phase1_official_plus_phase2_all4_all_docs/  # legacy combined run already present in the repo
```

`analyze_dataset.py`, `qualitative_analysis.py`, `authorship_benchmark_analysis.py`, and `leakage_audit.py` are the main benchmark-analysis entry points. `plot_results.py` is separate: it plots model evaluation results and can also reuse exported dataset-analysis CSVs. `OUTPUT_LAYOUT.md` documents the intended structure of generated result folders.

### What each script does

#### `analyze_dataset.py`

This is the main quantitative statistics pipeline for one benchmark root.

Inputs:
- A benchmark root containing split folders such as `train/`, `dev/`, and `test/`
- Either `documents.jsonl` or `queries.jsonl` plus `candidates.jsonl`
- `ground_truth.jsonl` for positive-pair and retrieval-structure analysis

Outputs under `statistics/`:
- `csv/languages_overall.csv`, `csv/languages_by_doc_type.csv`, `csv/languages_by_split_doc_type.csv`
  - Document counts and percentages by language.
- `csv/genres_overall.csv`, `csv/primary_genres_overall.csv`, `csv/genres_by_language.csv`, `csv/primary_genres_by_language.csv`
  - Fine-grained and primary-genre coverage tables.
- `csv/phases_overall.csv`, `csv/phases_by_language.csv`, `csv/phases_by_split_doc_type.csv`
  - Phase composition tables for datasets that carry `phase` metadata.
- `csv/token_lengths_by_language*.csv`, `csv/token_lengths_by_primary_genre*.csv`
  - Token-length summary statistics by language, genre, and doc type.
- `csv/token_length_bucket_distribution_by_language.csv`
  - Length bucket counts using the current code thresholds:
    - `short <= 10`
    - `medium = 11..100`
    - `long = 101..500`
    - `extra_long > 500`
- `csv/sources_overall.csv`
  - Source distribution table.
- `csv/authors_by_language.csv`
  - Unique author counts by language.
- `csv/docs_per_author_by_language.csv`, `csv/docs_per_author_overall.csv`
  - Per-language and overall document-per-author distribution summaries.
- `csv/author_split_membership.csv`, `csv/author_split_membership_distribution.csv`, `csv/author_split_overlap_matrix.csv`
  - Split-overlap diagnostics for authors.
- `csv/positive_pairs_summary.csv`, `csv/positive_pairs_by_language.csv`
  - Positive-pair alignment checks, including same-language and same-primary-genre rates.
- `csv/retrieval_structure_*.csv`, `csv/candidate_docs_per_author.csv`
  - Query-level positive-candidate structure and candidate inventory by author.

Figures under `statistics/figures/`:
- `language_distribution.png`
  - Countplot of language mix by document type.
- `primary_genre_heatmap.png`
  - Within-language primary-genre mixture heatmap.
- `token_length_boxplot.png`
  - Token-length distribution by language and document type.
- `primary_genre_distribution.png`
  - Top primary genres by document count.
- `phase_distribution_by_language.png`
  - Phase composition by language when phase metadata exists.

#### `qualitative_analysis.py`

This is the diagnostic and leakage-oriented analysis pipeline for one benchmark root.

Outputs under `qualitative/`:

- `csv/basic/summary_overview.json`
  - High-level benchmark summary: total docs, docs with author IDs, unique authors, and document-per-author distribution summary.
- `csv/basic/docs_per_author_raw.csv`
  - Raw author-level document counts for custom inspection.
- `csv/basic/docs_by_language.csv`, `docs_by_primary_genre.csv`, `docs_by_source.csv`, `docs_by_split_and_type.csv`
  - Basic breakdown tables used to contextualize later qualitative checks.

- `csv/entropy/author_genre_entropy.csv`
  - Per-author genre entropy and dominant-genre share.
- `csv/entropy/author_genre_entropy_by_language.csv`
  - Language-level summary of genre entropy.
- `csv/entropy/author_genre_entropy_summary.json`
  - Aggregate multi-genre summary.
- `figures/author_genre_entropy_histogram.png`
  - Distribution of normalized per-author genre entropy.

- `csv/cross_genre/cross_genre_author_pairs.csv`
  - Author overlap counts across primary-genre pairs.
- `csv/cross_genre/cross_genre_authors_by_language.csv`
  - Share of multi-genre authors by dominant language.
- `csv/cross_genre/cross_genre_author_summary.json`
  - Top cross-genre author summary.
- `figures/cross_genre_author_overlap_heatmap.png`
  - Cross-genre overlap heatmap for top genres.

- `csv/split_leakage/author_split_membership.csv`
  - Author membership across `train/dev/test`.
- `csv/split_leakage/author_split_membership_distribution.csv`
  - How many authors appear in one split, two splits, or all splits.
- `csv/split_leakage/author_split_overlap_matrix.csv`
  - Split-pair overlap matrix.
- `csv/split_leakage/author_phase_overlap_matrix.csv`
  - Phase-pair overlap matrix when phase labels exist.
- `csv/split_leakage/author_split_leakage_by_language.csv`
  - Multi-split author rates by dominant language.
- `csv/split_leakage/split_leakage_summary.json`
  - Aggregate split leakage summary.
- `figures/author_split_overlap_heatmap.png`
  - Split-overlap heatmap.
- `figures/author_phase_overlap_heatmap.png`
  - Phase-overlap heatmap.

- `csv/duplicates/exact_duplicate_groups.csv`
  - Exact duplicate groups after case and whitespace normalization.
- `csv/duplicates/exact_duplicate_groups_top200.csv`
  - Largest duplicate groups.
- `csv/duplicates/exact_duplicate_flag_summary.csv`
  - Counts of duplicate groups crossing authors, splits, or phases.
- `csv/duplicates/exact_duplicate_summary.json`
  - Duplicate summary metrics.
- `csv/duplicates/exact_duplicate_preview.md`
  - Human-readable preview of top duplicate groups.
- `figures/exact_duplicate_group_flags.png`
  - Duplicate-group count plot by flag type.

- `csv/topic_pairs/topic_group_candidates.csv`
  - Eligible `(language, primary_genre)` strata for topic-controlled analysis.
- `csv/topic_pairs/topic_cluster_summary.csv`
  - Topic cluster sizes and mined pair counts.
- `csv/topic_pairs/topic_controlled_same_genre_pairs.csv`
  - Nearest-neighbor pairs from the same language, same primary genre, and same topic cluster.
- `csv/topic_pairs/topic_controlled_pairs_summary.csv`
  - Pair counts and similarity summaries by language, genre, and pair type.
- `csv/topic_pairs/topic_controlled_pairs_stats.json`
  - Aggregate topic-pair summary.
- `csv/topic_pairs/topic_controlled_same_genre_pairs_preview.md`
  - Human-readable preview of same-topic same-genre pairs.

- `csv/embedding/embedding_projection_points.csv`
  - Sampled documents with 2D embedding coordinates.
- `csv/embedding/embedding_projection_summary.json`
  - Embedding and projection metadata.
- `figures/embedding_projection_by_language.png`
  - 2D projection colored by language.
- `figures/embedding_projection_by_primary_genre.png`
  - 2D projection colored by primary genre.

- `qualitative_report.md`
  - A short report that summarizes the major outputs above.

#### `analyze_benchmarks.py`

This runs the shared statistics and qualitative pipelines on two benchmark roots and then writes direct side-by-side comparisons.

Current default comparison:
- Dataset A: `processing/outputs/pipeline_phase1_official`
- Dataset B: `processing/second_phase_web_crawling/outputs/pipeline_phase2_official`
- Output: `post_analysis/outputs/phase1_vs_phase2`

Outputs:
- `per_benchmark/<benchmark_name>/statistics/`
  - Full `analyze_dataset.py` outputs for each benchmark.
- `per_benchmark/<benchmark_name>/qualitative/`
  - Full `qualitative_analysis.py` outputs for each benchmark.
- `comparison/tables/benchmark_summary.csv`
  - Benchmark-level headline metrics.
- `comparison/tables/language_distribution.csv`
  - Language counts and normalized shares per benchmark.
- `comparison/tables/primary_genre_distribution.csv`
  - Primary-genre counts and normalized shares per benchmark.
- `comparison/tables/token_length_bucket_distribution.csv`
  - Length-bucket counts and normalized shares per benchmark.
- `comparison/tables/source_distribution.csv`
  - Source counts and normalized shares per benchmark.
- `comparison/tables/split_doc_type_distribution.csv`
  - Split and document-type composition per benchmark.
- `comparison/tables/summary_metrics_long.csv`
  - Melted long-form summary table used for plotting.
- `comparison/figures/*.png`
  - Grouped bar charts for language, primary genre, length buckets, sources, split/type composition, and summary metrics.
- `comparison/benchmark_comparison_report.md`
  - Human-readable comparison report.

#### `authorship_benchmark_analysis.py`

This supplementary script adds the benchmark-specific composition, balance, stage-monitoring, and leakage-risk outputs requested for the authorship benchmark.

Outputs under `benchmark_profile/`:

- `tables/summary_overview.csv` and `summary_overview.json`
  - Total documents, unique authors, average documents per author, language count, genre count, and other headline metrics.
- `tables/unique_authors_by_*.csv`
  - Unique-author counts by language, genre, primary genre, length bucket, and `(language, primary_genre, token_length_bucket)` cells.
- `tables/docs_per_author_distribution_*.csv`
  - Histogram-ready document-per-author distributions overall and by selected metadata groupings.
- `tables/author_balance_*.csv`
  - Group-level author balance and concentration metrics, including top-author share, top-5-author share, Gini, and HHI.
- `tables/language_distribution.csv`, `source_distribution.csv`, `document_length_distribution.csv`, `primary_genre_distribution.csv`, `genre_distribution.csv`
  - Composition tables used for the new charts.
- `figures/primary_genre_subgenre_nested_donut.png`
  - Double-layer donut chart for primary genres and subgenres.
- `figures/language_distribution_pie.png`
  - Language distribution pie chart.
- `figures/document_length_distribution_pie.png`
  - Short / medium / long / extra-long distribution pie chart.
- `figures/source_distribution_pie.png`
  - Source distribution pie chart.
- `figures/author_balance_histogram.png`
  - Histogram with x-axis = documents per author and y-axis = number of authors.
- `tables/stage_document_counts.csv`, `stage_document_flow_edges.csv`, `stage_author_counts.csv`
  - Stage-level monitoring tables for document and author counts.
- `figures/stage_document_sankey.png`
  - Combined phase1 + phase2 stage attribution map for document flow.
- `figures/stage_author_counts.png`
  - Available author-count checkpoints across major stages.
- `tables/quality_filter_examples.csv`, `language_audit_examples.csv`, `sampling_deficit_examples.csv`, `merge_dedup_summary.csv`, `cross_phase_exact_overlap_examples.csv`
  - Stage-level qualitative examples and summaries.
- `reports/stage_examples_report.md`
  - Human-readable stage-by-stage examples report.
- `tables/positive_pair_metadata_alignment*.csv`
  - Query/candidate metadata alignment diagnostics for language, genre, source, and length bucket.
- `tables/candidate_author_metadata_span.csv`, `metadata_cell_risk.csv`
  - Candidate-pool leakage-risk diagnostics.
- `reports/topic_leakage_report.md`
  - Benchmark-specific leakage report plus recommended evaluation protocol.

#### `leakage_audit.py`

This is the reviewer-facing shortcut audit for one benchmark root. It is meant to answer challenges such as "is the benchmark solvable from language or topic metadata alone?" by producing explicit metadata-only baselines and topic-cluster pool diagnostics.

Outputs under `leakage_audit/`:

- `tables/positive_pair_alignment.csv`, `positive_pair_alignment_summary.json`
  - Same-language, same-genre, same-source, and same-length rates for labeled query-positive pairs.
- `tables/candidate_author_metadata_span.csv`, `candidate_author_metadata_span_summary.json`
  - How specialized candidate authors are by language, genre, source, and length.
- `tables/shortcut_pool_sizes_by_query.csv`
  - Per-query author-pool sizes after conditioning on metadata-only schemes such as `language_only`, `language_fine_genre`, and `full_metadata`.
- `tables/shortcut_pool_summary.csv`, `shortcut_pool_summary_by_language.csv`
  - Summary metrics for metadata-only shortcut strength, including expected and deterministic `success@1/@5`.
- `tables/topic_cluster_assignments.csv`, `topic_cluster_strata.csv`
  - Topic-cluster assignments and per-stratum clustering diagnostics for `(language, primary_genre)` groups.
- `tables/topic_shortcut_pool_sizes_by_query.csv`, `topic_alignment_by_language.csv`, `topic_shortcut_summary.json`
  - Topic-cluster shortcut diagnostics, including same-topic positive-pair rates and matched-pool sizes under `(language, primary_genre, topic_cluster)`.
- `reports/leakage_audit_report.md`
  - Short human-readable interpretation of the main leakage metrics.
- `summary.json`
  - Compact machine-readable summary for downstream reporting.

#### `plot_results.py`

This script is for evaluation-result visualization, not benchmark-construction analysis. It reads evaluation outputs from `eval/results/` and optionally merges them with analysis CSVs.

It can generate:
- Overall or macro-average model performance bar charts
- Performance by language
- Performance by genre
- Performance by length bucket
- Dataset-composition visuals reused from analysis CSVs:
  - language distribution bar chart
  - genre pie-grid by language
  - token-length boxplot by language

### Assessment of the proposed new analyses

Your proposed additions are reasonable. Most of them are useful, and several are already partially covered by the current code.

| Proposed item | Status | Recommendation |
| --- | --- | --- |
| Total document count, unique author count, avg docs per author, language count, genre count | Already covered | Keep as headline metrics in every report. |
| Unique authors for each language, genre, and document length | Partially covered | Language is already covered. Add author counts by primary genre and by length bucket. |
| Document distribution per author | Partially covered | Existing raw and summary tables are useful, but add a histogram plus concentration metrics such as Gini, HHI, top-1%, and top-10-author share. |
| Double-layer pie chart for primary genre and subgenre | Not implemented | Good idea. Use a nested donut chart, keep only top subgenres inside each primary genre, and merge the rest into `other`. |
| Language distribution pie chart | Not implemented | Feasible, but for 10 languages a sorted bar chart is often clearer than a pie chart. Keep the bar chart and add the pie chart only for presentation. |
| Document length distribution pie chart | Not implemented | Reasonable if the bucket definition is frozen and explicitly documented. |
| Source distribution pie chart | Not implemented | Reasonable. Also keep the table because source labels are often too long for a pie chart alone. |
| Stage-drop attribution map / Sankey | Not implemented | Strong addition. Plot both document counts and unique-author counts, because author loss and document loss can move differently. |
| Author balance histogram | Not implemented directly | Recommended. Use both raw counts and log-scaled x-axis. |
| 1-3 examples per processing stage | Not fully possible from current post-analysis outputs alone | Best done by instrumenting the upstream processing pipeline to save sampled before/after examples at each stage. |
| Topic leakage analysis for candidate sampling | Partially covered | The current topic-controlled pair mining is a good start. Extend it with stricter matching and explicit leakage-risk metrics. |

### Recommended refinements and additions

If you expand the benchmark analysis, the next additions I would prioritize are:

- Author counts by `primary_genre`, `genre`, `token_length_bucket`, and `(language, primary_genre)` cells.
- Concentration metrics for authors within each language, genre, and length bucket.
- Minimum-author thresholds per `(language, primary_genre, token_length_bucket)` cell so you can flag cells where metadata nearly identifies the author.
- Query/candidate alignment tables for:
  - same language
  - same primary genre
  - same fine-grained genre
  - same source
  - same length bucket
- Hard-negative diagnostics:
  - how often a different-author candidate shares language, genre, source, and length bucket with the query
- Metadata-only leakage baselines:
  - predict author or positive-pair membership from metadata only
  - predict source or phase from text only
- Per-language duplicate and near-duplicate rates.
- Temporal or source split diagnostics if timestamp metadata is available in future pipeline versions.

### Standard practice for topic, genre, and language leakage analysis

For this benchmark, a good leakage-analysis protocol is:

1. Enforce strict metadata matching for positives when possible.
   - Positive candidates should at least match query language and primary genre.
   - If possible, also match fine-grained genre, source, and length bucket.

2. Report same-metadata rates explicitly.
   - `same_language`
   - `same_primary_genre`
   - `same_fine_grained_genre`
   - `same_source`
   - `same_length_bucket`

3. Measure author diversity inside each metadata cell.
   - For each `(language, primary_genre, token_length_bucket)` cell, report:
     - docs
     - unique authors
     - docs per author
     - top-author share

4. Use topic-controlled same-genre sampling.
   - Within one `(language, primary_genre)` stratum, cluster documents by lexical similarity.
   - Compare nearest-neighbor pairs for same-author vs different-author cases.
   - If different-author same-topic pairs remain much easier than they should be, topical leakage is likely.

5. Add a metadata-only baseline.
   - If a model using only language, genre, source, phase, and length predicts author or pair labels unusually well, the benchmark still leaks metadata shortcuts.

6. Add a text-only topic baseline.
   - A simple TF-IDF nearest-neighbor baseline inside matched language/genre buckets is useful for quantifying how much topical similarity alone can recover positives.

### Running the analysis

Install dependencies:

```bash
python3 -m pip install -r post_analysis/requirements.txt
```

Run statistics plus qualitative analysis for the combined benchmark:

```bash
./post_analysis/run_combined_phase1_phase2_analysis.sh
```

Equivalent explicit commands:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python3 -m post_analysis.analyze_dataset \
  --dataset-dir processing/outputs/combined_phase1_phase2 \
  --output-dir post_analysis/outputs/combined_phase1_phase2/statistics \
  --splits all

python3 -m post_analysis.qualitative_analysis \
  --dataset-dir processing/outputs/combined_phase1_phase2 \
  --output-dir post_analysis/outputs/combined_phase1_phase2/qualitative \
  --splits all

python3 -m post_analysis.authorship_benchmark_analysis \
  --dataset-dir processing/outputs/combined_phase1_phase2 \
  --output-dir post_analysis/outputs/combined_phase1_phase2/benchmark_profile \
  --splits all

python3 -m post_analysis.leakage_audit \
  --dataset-dir processing/outputs/combined_phase1_phase2 \
  --output-dir post_analysis/outputs/combined_phase1_phase2/leakage_audit \
  --splits test
```

Run the AuthBench-specific bundle, including the leakage audit:

```bash
./post_analysis/run_authbench_analysis.sh
```

Equivalent explicit command:

```bash
python3 -m post_analysis.leakage_audit \
  --dataset-dir processing/outputs/authbench \
  --output-dir post_analysis/outputs/authbench/leakage_audit \
  --splits test
```

Run Phase 1 vs Phase 2 comparison:

```bash
./post_analysis/run_analyze_benchmarks.sh
```

Equivalent explicit command:

```bash
python3 -m post_analysis.analyze_benchmarks \
  --dataset-a-dir processing/outputs/pipeline_phase1_official \
  --dataset-b-dir processing/second_phase_web_crawling/outputs/pipeline_phase2_official \
  --dataset-a-name pipeline_phase1_official \
  --dataset-b-name pipeline_phase2_official \
  --output-dir post_analysis/outputs/phase1_vs_phase2
```

Run evaluation plotting:

```bash
python3 -m post_analysis.plot_results \
  --results-dir eval/results \
  --csv-dir post_analysis/outputs/combined_phase1_phase2/statistics/csv
```

### Notes on current output folders

- `post_analysis/outputs/combined_phase1_phase2/` is the recommended location for future combined-benchmark analysis runs.
- `post_analysis/outputs/authbench/` is the recommended location for AuthBench-specific runs, including `statistics/`, `qualitative/`, `benchmark_profile/`, and `leakage_audit/`.
- `post_analysis/outputs/phase1_vs_phase2/` is the recommended location for cross-benchmark comparisons.
- `post_analysis/outputs/phase1_official_plus_phase2_all4_all_docs/` is an older combined-benchmark output already present in the repository. It is still useful, but it should be treated as a legacy run directory rather than the canonical name going forward.
