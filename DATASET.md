# AuthBench Source Datasets

This document summarizes the raw input sources used to build the current AuthBench release described in [`paper/colm_latex.tex`](/Users/maoxunhuang/Desktop/AuthBench/paper/colm_latex.tex). AuthBench is constructed from 17 public sources across two phases:

- Phase 1 curated datasets listed in [`processing/datasets_manifest.json`](/Users/maoxunhuang/Desktop/AuthBench/processing/datasets_manifest.json)
- Phase 2 public-web crawls listed in [`processing/second_phase_web_crawling/datasets_manifest.json`](/Users/maoxunhuang/Desktop/AuthBench/processing/second_phase_web_crawling/datasets_manifest.json)

Each source is standardized into a shared schema with `source`, `author_id`, `lang`, `genre`, `content`, and `token_length`, then passed through the five-stage pipeline documented in [`processing/README.md`](/Users/maoxunhuang/Desktop/AuthBench/processing/README.md).

## Current benchmark summary

- Documents: 428,150
- Authors: 153,825
- Languages: 10 (`en`, `zh`, `hi`, `es`, `fr`, `ar`, `ru`, `de`, `ja`, `ko`)
- Primary genres: 9
- Fine-grained genres: 66
- Length buckets:
  - short: 1-10 tokens
  - medium: 11-100 tokens
  - long: 101-500 tokens
  - extra-long: >500 tokens

## Raw sources

| Source | Languages | Primary genre(s) | Upstream scale | Author labels | Notes |
| --- | --- | --- | --- | --- | --- |
| Exorde | Multilingual | Social media, news, forums | 65M+ items per week | Yes | Social/news/forum content with explicit author hash |
| Babel Briefings | 30+ | News headlines | 4.7M | Partial | Publisher or organization level authorship only |
| Amazon Reviews Multi (MARC) | 6 | E-commerce reviews | 200k+ per language | Yes | Multilingual product reviews |
| Blog Authorship Corpus | English | Blogs | 681k posts | Yes | Classic Blogger corpus |
| arXiv Abstracts (metadata snapshot) | English | Research papers | 1.7M+ records | Yes | Uses first author from metadata |
| Xiaohongshu / Weibo | Chinese | Social media | 11k+ | Yes | Chinese social posts/comments |
| Douban Reviews | Chinese | Media, book, music reviews | 13.5M | Yes | Review platform data |
| Hindi Discourse Stories | Hindi | Literature | 53 stories | Yes | Short stories with author labels |
| Spanish PD Books | Spanish | Literature | 300k+ texts | Yes | Public-domain books |
| French PD Books | French | Literature | 289k+ books | Yes | Public-domain books |
| Arabic Classical Poetry | Arabic | Poetry | 70k poems | Yes | Poet-labeled classical poetry |
| Russian PD Corpus | Russian | Literature, periodicals | 8.5k titles | Yes | Public-domain corpus |
| German PD Corpus | German | Literature, newspapers | 260k+ texts | Yes | Public-domain corpus |
| Stack Exchange crawl | Multilingual | Q&A | Crawl-dependent | Yes | Public Q&A posts/comments from the Phase 2 crawl |
| Project Gutenberg crawl | Multilingual | Literature, essays, speeches | Crawl-dependent | Yes | Public-domain or public-access texts from the Phase 2 crawl |
| Wikisource crawl | Multilingual | Literature, drama, speeches | Crawl-dependent | Yes | Public-source texts from the Phase 2 crawl |
| YouTube comments crawl | Multilingual | Social media comments | Crawl-dependent | Yes | Public comments collected through the Phase 2 crawl |

## Realized contribution in the current combined release

These are the retained document counts reported in the paper after filtering, redundancy reduction, cross-phase merge cleanup, and final selection.

| Source | Docs | Share |
| --- | ---: | ---: |
| Exorde | 94,231 | 22.0% |
| Wikisource | 78,984 | 18.4% |
| Babel Briefings | 73,676 | 17.2% |
| YouTube comments | 71,808 | 16.8% |
| Blog Authorship Corpus | 22,494 | 5.3% |
| Project Gutenberg | 18,739 | 4.4% |
| Russian PD Corpus | 12,728 | 3.0% |
| Douban Reviews | 10,424 | 2.4% |
| Xiaohongshu / Weibo | 8,869 | 2.1% |
| French PD Books | 8,761 | 2.0% |
| German PD Corpus | 8,400 | 2.0% |
| Spanish PD Books | 4,961 | 1.2% |
| Amazon Reviews Multi (MARC) | 4,924 | 1.2% |
| Stack Exchange crawl | 4,651 | 1.1% |
| Arabic Classical Poetry | 2,503 | 0.6% |
| arXiv Abstracts | 1,784 | 0.4% |
| Hindi Discourse Stories | 213 | 0.0% |

No configured source is fully absent from the current combined release, but the source distribution is highly skewed: Exorde, Wikisource, Babel Briefings, and YouTube comments together contribute 74.4% of all documents.

## Release and licensing note

The paper distinguishes between:

- Tier A: sources whose normalized text can be redistributed more directly
- Tier B: sources that are safer to support through manifest-only reconstruction

Examples of conservative Tier B handling in the paper include MARC, Blog Authorship, arXiv metadata-derived text, Xiaohongshu, Douban, Hindi Discourse, Project Gutenberg, Wikisource, and YouTube comments. For source-specific licensing and redistribution notes, see the appendix tables in [`paper/colm_latex.tex`](/Users/maoxunhuang/Desktop/AuthBench/paper/colm_latex.tex).

## Notes on terminology

- `Babel Briefings` should not be treated as clean per-user authorship data; its labels are only partial.
- `arXiv` in this repository refers to metadata snapshots and uses the first listed author during preprocessing.
- `Xiaohongshu / Weibo` is the naming used in the paper appendix for the Chinese social-media source currently referenced in the manifest as `xiaohongshu`.
- The current paper describes the full combined benchmark, which includes both the phase-1 curated sources and the phase-2 crawled sources.
