# AuthBench: A Large-Scale Multilingual Benchmark for Authorship Representation across Genres and Lengths
AuthBench is a multilingual, multi-genre benchmark for authorship representation and attribution. It spans a range of token lengths and targets realistic, cross-domain evaluation.

## What is in this repo
- `processing/`: dataset construction pipeline and configs
- `eval/`: evaluation outputs and metrics
- `post_analysis/`: analysis scripts and notebooks
- `raw_analysis/`: source analysis and intermediate artifacts
- `DATASET.md`: curated dataset sources and references

## Key contributions
1. Multiple languages (en, zh, hi, es, fr, ar, ru, de, ja, ko)
2. Diverse token lengths (short: 1-10, medium: 11-100, long: 101-500, extra-long: >500)
3. Multiple genres (social media, science, news, reviews, blogs, articles)

## Dataset search logic
1. Start with languages (10 major languages listed above)
2. For each language, include diverse genres where data is available
3. Ensure token-length coverage across short to extra-long
4. English has broader genre coverage due to dataset availability; other languages may be less balanced

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data sources
See `DATASET.md` for the full list of upstream datasets and references.

## Processing pipeline
The current pipeline has five stages: Build & Normalization, Quality Filtering, Redundancy Reduction, Language Audit, and Bucket Balanced Sampling.
See `processing/README.md` and `processing/PROCESSING.md` for the current end-to-end data build instructions.
