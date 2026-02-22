# Second-Phase Web Crawling

This folder adds new web-sourced corpora and keeps the downstream benchmark logic exactly aligned with `processing/`:
- filtering/chunking/author caps/sampling/splits: `processing.build_benchmark`
- monitoring dynamics: `processing.monitor_pipeline`
- final post-cleaning/rebalancing outputs: `processing.postprocess`

## Sources Included

1. Stack Exchange Data Dump (Q&A + comments)
   - Official update timeline and access change: data dumps moved off Archive.org and into Stack Exchange infrastructure (monthly cadence noted for 2025):  
     https://meta.stackexchange.com/questions/401324/stack-exchange-data-dump-is-moving-away-from-archive-org
   - Release/timeline clarification post:  
     https://meta.stackexchange.com/questions/396597/data-dumps-releases-timeline-updates-and-clarification
   - Legacy dump schema (`Posts.xml`, `Comments.xml`, etc.):  
     https://archive.org/download/stackexchange/readme.txt
   - Crawler: `crawl_stackexchange.py`

2. Project Gutenberg (books / essays / speeches)
   - Offline catalogs and machine-readable metadata recommendation:  
     https://www.gutenberg.org/ebooks/offline_catalogs.html
   - Feeds index (`pg_catalog.csv`, `rdf-files.tar.*`, text tarball, etc.):  
     https://www.gutenberg.org/cache/epub/feeds/
   - Crawler: `crawl_gutenberg.py`

3. Wikisource (Wikimedia dumps)
   - Dump index pattern (example for English Wikisource):  
     https://dumps.wikimedia.org/enwikisource/latest/
   - Crawler: `crawl_wikisource.py`

4. YouTube comments (YouTube Data API v3)
   - API overview:  
     https://developers.google.com/youtube/v3/docs
   - Crawler: `crawl_ytcomments.py` (expects `YOUTUBE_API_KEY`)

## Files

- `crawl_stackexchange.py`: parses Stack Exchange `Posts.7z` (+ optional `Comments.7z`) into JSONL rows.
- `crawl_gutenberg.py`: uses `pg_catalog.csv` + text URLs to build book/essay/speech rows.
- `crawl_wikisource.py`: parses Wikisource XML dumps into plain-text rows with author heuristics.
- `crawl_ytcomments.py`: collects multilingual YouTube top-level comments into JSONL rows.
- `build_manifest.py`: builds a `processing`-compatible manifest for crawled JSONL files.
- `run_pipeline.py`: end-to-end runner (crawl -> monitor -> build -> postprocess).

## Quick Start

### 1) Crawl sources

```bash
python -m processing.second_phase_web_crawling.run_pipeline \
  --stages crawl \
  --stackexchange-download-mode api \
  --stackexchange-sites stackoverflow.com,es.stackoverflow.com,ru.stackoverflow.com \
  --gutenberg-max-docs 20000 \
  --wikisource-wikis enwikisource,frwikisource \
  --wikisource-max-docs-per-wiki 10000 \
  --ytcomments-max-docs 20000
```

Notes:
- For Stack Exchange, current official dumps may require manual account-based download; place `.7z` (or extracted `.xml`) files under `processing/second_phase_web_crawling/downloads/stackexchange/`.
- Stack Exchange comments are included by default; pass `--stackexchange-skip-comments` to disable.
- `--stackexchange-download-mode api` fetches from the official Stack Exchange API (recommended for small smoke runs).
- `--stackexchange-download-mode archive` exists for legacy/mirror workflows.
- `crawl_ytcomments.py` uses YouTube Data API v3 key auth (`YOUTUBE_API_KEY` environment variable).

### 2) Run monitor/build/postprocess with the same pipeline logic

```bash
python -m processing.second_phase_web_crawling.run_pipeline \
  --stages monitor build postprocess \
  --total-docs 100000 \
  --allow-other-languages \
  --max-documents-per-dataset 10000000 \
  --shuffle-buffer-size 10000 \
  --chunk-probability 0.7 \
  --truncate-to-tokens 2000
```

Outputs:
- Monitor report: `processing/second_phase_web_crawling/outputs/monitoring/`
- Stage-1 benchmark outputs: `processing/second_phase_web_crawling/outputs/stage1/`
- Final postprocessed outputs: `processing/second_phase_web_crawling/outputs/stage2/`

### 3) One-shot script for 4 datasets at 60K target

```bash
bash processing/second_phase_web_crawling/scripts/run_webcrawl_60k_all4.sh
```
