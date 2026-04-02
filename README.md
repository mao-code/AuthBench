# AuthBench: A Large-Scale Multilingual Benchmark for Authorship Representation across Genres and Lengths

AuthBench is a multilingual benchmark for authorship representation built to evaluate whether text representations preserve author-specific signals across languages, genres, and document lengths. The benchmark supports two tasks:

- authorship attribution as open-world same-author retrieval
- authorship verification as same-author binary decision

The current paper release described in [`paper/colm_latex.tex`](/Users/maoxunhuang/Desktop/AuthBench/paper/colm_latex.tex) contains 428,150 documents by 153,825 authors across 10 languages, 9 primary genres, 66 fine-grained genres, and 4 document-length buckets.

## Benchmark profile

- Languages: `en`, `zh`, `hi`, `es`, `fr`, `ar`, `ru`, `de`, `ja`, `ko`
- Primary genres: `social_media`, `literature`, `news`, `blog`, `media_reviews`, `poetry`, `ecommerce_reviews`, `qna`, `research_paper`
- Length buckets:
  - short: 1-10 tokens
  - medium: 11-100 tokens
  - long: 101-500 tokens
  - extra-long: >500 tokens

The paper reports that the benchmark is dominated by `social_media` (40.9%), `literature` (30.0%), and `news` (17.2%), with the remaining six primary genres accounting for the final 11.9%.

## Current split materialization

The paper reports the current split export as:

- Queries: 198,345 total
- Candidates: 229,805 total
- Train: 156,335 queries / 186,184 candidates
- Dev: 21,008 queries / 21,813 candidates
- Test: 21,002 queries / 21,808 candidates

Each final split contains:

- `candidates.jsonl`
- `queries.jsonl`
- `ground_truth.jsonl`

## What is in this repo

- [`processing/`](/Users/maoxunhuang/Desktop/AuthBench/processing): benchmark construction pipeline
- [`eval/`](/Users/maoxunhuang/Desktop/AuthBench/eval): zero-shot evaluation, baselines, fine-tuning, and analysis code
- [`post_analysis/`](/Users/maoxunhuang/Desktop/AuthBench/post_analysis): benchmark analysis scripts
- [`raw_analysis/`](/Users/maoxunhuang/Desktop/AuthBench/raw_analysis): source inspection and intermediate analysis utilities
- [`DATASET.md`](/Users/maoxunhuang/Desktop/AuthBench/DATASET.md): source inventory and release notes
- [`paper/colm_latex.tex`](/Users/maoxunhuang/Desktop/AuthBench/paper/colm_latex.tex): manuscript describing the benchmark and results

## Construction overview

AuthBench combines two kinds of inputs:

- curated public datasets
- public-web crawling pipelines

All sources are mapped into a shared schema and processed with five stages implemented in the current codebase:

1. Build & Normalization
2. Quality Filtering
3. Redundancy Reduction
4. Language Audit
5. Bucket Balanced Sampling

These stages are documented in [`processing/README.md`](/Users/maoxunhuang/Desktop/AuthBench/processing/README.md) and [`processing/PROCESSING.md`](/Users/maoxunhuang/Desktop/AuthBench/processing/PROCESSING.md).

## Source coverage

The paper’s current combined release uses 17 public input sources across both phases of construction. These include Exorde, Babel Briefings, MARC, Blog Authorship, arXiv metadata, Xiaohongshu / Weibo, Douban, Hindi Discourse, several public-domain book corpora, and four crawl-built sources: Stack Exchange, Project Gutenberg, Wikisource, and YouTube comments.

See [`DATASET.md`](/Users/maoxunhuang/Desktop/AuthBench/DATASET.md) for the full source list and realized source composition.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main entrypoints

Build the benchmark:

```bash
python -m processing.construct_benchmark --help
python -m processing.second_phase_web_crawling.run_pipeline --help
python -m processing.combine_phase_benchmarks --help
```

Run evaluation:

```bash
python -m AuthBench.eval.runner --help
python -m AuthBench.eval.train --help
python -m AuthBench.eval.baseline_runner --help
```

## Evaluation summary from the paper

The manuscript reports a unified zero-shot benchmark over 47 neural models and 3 non-neural baselines. In the current paper draft:

- best retrieval model: `multilingual-e5-large` with 0.258 Success@5
- best verification model: `llama3.1-8b-instruct` with 0.076 EER and 0.968 ROC-AUC

These numbers come from the paper draft and should be read as release-paper results, not as a promise that every result artifact is already checked into this repository.
