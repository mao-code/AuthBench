from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end runner for second-phase web crawling + unified AuthBench "
            "construction (multi-stage processing + monitoring report)."
        )
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["crawl", "construct"],
        choices=["crawl", "construct"],
        help="Which stages to execute (default: crawl + construct).",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/datasets_manifest.json"),
    )

    parser.add_argument(
        "--stackexchange-output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/stackexchange/stackexchange.jsonl"),
    )
    parser.add_argument(
        "--gutenberg-output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/gutenberg/gutenberg.jsonl"),
    )
    parser.add_argument(
        "--wikisource-output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/wikisource/wikisource.jsonl"),
    )
    parser.add_argument(
        "--ytcomments-output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/ytcomments/ytcomments.jsonl"),
    )
    parser.add_argument("--skip-stackexchange", action="store_true")
    parser.add_argument("--skip-gutenberg", action="store_true")
    parser.add_argument("--skip-wikisource", action="store_true")
    parser.add_argument("--skip-ytcomments", action="store_true")

    # Stack Exchange crawler options
    parser.add_argument(
        "--stackexchange-sites",
        default=(
            "stackoverflow.com,es.stackoverflow.com,ru.stackoverflow.com,ja.stackoverflow.com,"
            "spanish.stackexchange.com,french.stackexchange.com,german.stackexchange.com,"
            "arabic.stackexchange.com,chinese.stackexchange.com,korean.stackexchange.com,"
            "hindi.stackexchange.com"
        ),
        help="Comma-separated sites to crawl.",
    )
    parser.add_argument(
        "--stackexchange-archives-dir",
        type=Path,
        default=Path("processing/second_phase_web_crawling/downloads/stackexchange"),
    )
    parser.add_argument("--stackexchange-skip-comments", action="store_true")
    parser.add_argument("--stackexchange-max-posts-per-site", type=int, default=200000)
    parser.add_argument("--stackexchange-max-comments-per-site", type=int, default=100000)
    parser.add_argument(
        "--stackexchange-download-mode",
        choices=["none", "archive", "api"],
        default="none",
        help="StackExchange ingestion mode (local/archive/api).",
    )
    parser.add_argument("--stackexchange-archive-identifier", default="stackexchange")

    # Gutenberg crawler options
    parser.add_argument("--gutenberg-max-docs", type=int, default=200000)
    parser.add_argument(
        "--gutenberg-languages",
        default="en,es,fr,de,ru,ar,zh,ja,ko,hi",
    )
    parser.add_argument("--gutenberg-min-chars", type=int, default=500)
    parser.add_argument("--gutenberg-refresh-catalog", action="store_true")

    # Wikisource crawler options
    parser.add_argument(
        "--wikisource-wikis",
        default="enwikisource,zhwikisource,hiwikisource,eswikisource,frwikisource,arwikisource,ruwikisource,dewikisource,jawikisource,kowikisource",
    )
    parser.add_argument("--wikisource-max-docs-per-wiki", type=int, default=20000)
    parser.add_argument("--wikisource-max-total-docs", type=int, default=200000)
    parser.add_argument("--wikisource-min-chars", type=int, default=500)

    # YouTube comments crawler options
    parser.add_argument("--ytcomments-max-docs", type=int, default=200000)
    parser.add_argument(
        "--ytcomments-languages",
        default="en,zh,hi,es,fr,ar,ru,de,ja,ko",
    )
    parser.add_argument(
        "--ytcomments-region-map",
        default="en:US,zh:TW,hi:IN,es:ES,fr:FR,ar:EG,ru:RU,de:DE,ja:JP,ko:KR",
    )
    parser.add_argument("--ytcomments-max-video-pages-per-lang", type=int, default=50)
    parser.add_argument("--ytcomments-max-comments-per-video", type=int, default=200)
    parser.add_argument("--ytcomments-max-comment-pages-per-video", type=int, default=8)
    parser.add_argument("--ytcomments-max-empty-pages-per-video", type=int, default=2)
    parser.add_argument("--ytcomments-min-chars", type=int, default=20)
    parser.add_argument("--ytcomments-timeout", type=int, default=60)
    parser.add_argument("--ytcomments-retries", type=int, default=3)
    parser.add_argument("--ytcomments-retry-backoff-sec", type=float, default=1.5)
    parser.add_argument("--ytcomments-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--ytcomments-resume", action="store_true")
    parser.add_argument("--ytcomments-skip-langdetect", action="store_true")

    # Core processing options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processing/second_phase_web_crawling/outputs/pipeline"),
        help="Final unified pipeline output directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional persistent working directory for intermediate artifacts.",
    )
    parser.add_argument(
        "--monitor-report-path",
        type=Path,
        default=None,
        help="Monitoring report path (default: <output-dir>/pipeline_dynamics.json).",
    )
    parser.add_argument("--monitor-overwrite", action="store_true")

    parser.add_argument("--total-docs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--allow-other-languages", action="store_true")
    parser.add_argument("--max-documents-per-dataset", type=int, default=None)
    parser.add_argument("--dataset-max-docs", nargs="*", default=[])
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--no-shuffle-datasets", nargs="*", default=[])
    parser.add_argument("--chunk-probability", type=float, default=0.7)
    parser.add_argument("--truncate-to-tokens", type=int, default=2000)

    parser.add_argument("--post-target-total", type=int, default=None)
    parser.add_argument("--post-skip-langdetect", action="store_true")
    parser.add_argument("--disable-dedup", action="store_true")
    parser.add_argument("--dedup-near-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--dedup-author-similarity-threshold", type=float, default=0.94)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_crawl_stage(args: argparse.Namespace) -> None:
    py = sys.executable

    if not args.skip_stackexchange:
        sites = [s.strip() for s in args.stackexchange_sites.split(",") if s.strip()]
        cmd = [
            py,
            "-m",
            "processing.second_phase_web_crawling.crawl_stackexchange",
            "--input-archives-dir",
            str(args.stackexchange_archives_dir),
            "--output-path",
            str(args.stackexchange_output_path),
            "--download-mode",
            args.stackexchange_download_mode,
            "--archive-identifier",
            args.stackexchange_archive_identifier,
            "--max-posts-per-site",
            str(args.stackexchange_max_posts_per_site),
            "--max-comments-per-site",
            str(args.stackexchange_max_comments_per_site),
            "--sites",
            *sites,
            "--log-level",
            args.log_level,
        ]
        if args.stackexchange_skip_comments:
            cmd.append("--skip-comments")
        _run(cmd)

    if not args.skip_gutenberg:
        cmd = [
            py,
            "-m",
            "processing.second_phase_web_crawling.crawl_gutenberg",
            "--output-path",
            str(args.gutenberg_output_path),
            "--languages",
            args.gutenberg_languages,
            "--max-docs",
            str(args.gutenberg_max_docs),
            "--min-chars",
            str(args.gutenberg_min_chars),
            "--log-level",
            args.log_level,
        ]
        if args.gutenberg_refresh_catalog:
            cmd.append("--refresh-catalog")
        _run(cmd)

    if not args.skip_wikisource:
        cmd = [
            py,
            "-m",
            "processing.second_phase_web_crawling.crawl_wikisource",
            "--output-path",
            str(args.wikisource_output_path),
            "--wikis",
            args.wikisource_wikis,
            "--max-docs-per-wiki",
            str(args.wikisource_max_docs_per_wiki),
            "--max-total-docs",
            str(args.wikisource_max_total_docs),
            "--min-chars",
            str(args.wikisource_min_chars),
            "--log-level",
            args.log_level,
        ]
        _run(cmd)

    if not args.skip_ytcomments:
        cmd = [
            py,
            "-m",
            "processing.second_phase_web_crawling.crawl_ytcomments",
            "--output-path",
            str(args.ytcomments_output_path),
            "--languages",
            args.ytcomments_languages,
            "--max-docs",
            str(args.ytcomments_max_docs),
            "--region-map",
            args.ytcomments_region_map,
            "--max-video-pages-per-lang",
            str(args.ytcomments_max_video_pages_per_lang),
            "--max-comments-per-video",
            str(args.ytcomments_max_comments_per_video),
            "--max-comment-pages-per-video",
            str(args.ytcomments_max_comment_pages_per_video),
            "--max-empty-pages-per-video",
            str(args.ytcomments_max_empty_pages_per_video),
            "--min-chars",
            str(args.ytcomments_min_chars),
            "--timeout",
            str(args.ytcomments_timeout),
            "--retries",
            str(args.ytcomments_retries),
            "--retry-backoff-sec",
            str(args.ytcomments_retry_backoff_sec),
            "--sleep-seconds",
            str(args.ytcomments_sleep_seconds),
            "--log-level",
            args.log_level,
        ]
        if args.ytcomments_resume:
            cmd.append("--resume")
        if args.ytcomments_skip_langdetect:
            cmd.append("--skip-langdetect")
        _run(cmd)

    _run(
        [
            py,
            "-m",
            "processing.second_phase_web_crawling.build_manifest",
            "--manifest-path",
            str(args.manifest_path),
            "--stackexchange-path",
            str(args.stackexchange_output_path),
            "--gutenberg-path",
            str(args.gutenberg_output_path),
            "--wikisource-path",
            str(args.wikisource_output_path),
            "--ytcomments-path",
            str(args.ytcomments_output_path),
            "--log-level",
            args.log_level,
        ]
    )


def run_construct_stage(args: argparse.Namespace) -> None:
    py = sys.executable
    report_path = args.monitor_report_path or (args.output_dir / "pipeline_dynamics.json")
    cmd = [
        py,
        "-m",
        "processing.construct_benchmark",
        "--manifest",
        str(args.manifest_path),
        "--report-path",
        str(report_path),
        "--output-dir",
        str(args.output_dir),
        "--total-docs",
        str(args.total_docs),
        "--seed",
        str(args.seed),
        "--train-ratio",
        str(args.train_ratio),
        "--dev-ratio",
        str(args.dev_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--shuffle-buffer-size",
        str(args.shuffle_buffer_size),
        "--chunk-probability",
        str(args.chunk_probability),
        "--truncate-to-tokens",
        str(args.truncate_to_tokens),
        "--log-level",
        args.log_level,
    ]
    if args.allow_other_languages:
        cmd.append("--allow-other-languages")
    if args.monitor_overwrite:
        cmd.append("--overwrite-report")
    if args.work_dir is not None:
        cmd.extend(["--work-dir", str(args.work_dir)])
    if args.max_documents_per_dataset is not None:
        cmd.extend(["--max-documents-per-dataset", str(args.max_documents_per_dataset)])
    if args.dataset_max_docs:
        cmd.extend(["--dataset-max-docs", *args.dataset_max_docs])
    if args.no_shuffle_datasets:
        cmd.extend(["--no-shuffle-datasets", *args.no_shuffle_datasets])
    if args.post_target_total is not None:
        cmd.extend(["--post-target-total", str(args.post_target_total)])
    if args.post_skip_langdetect:
        cmd.append("--post-skip-langdetect")
    if args.disable_dedup:
        cmd.append("--disable-dedup")
    if args.dedup_near_similarity_threshold is not None:
        cmd.extend(
            ["--dedup-near-similarity-threshold", str(args.dedup_near_similarity_threshold)]
        )
    if args.dedup_author_similarity_threshold is not None:
        cmd.extend(
            ["--dedup-author-similarity-threshold", str(args.dedup_author_similarity_threshold)]
        )
    _run(cmd)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stages = set(args.stages)
    if "crawl" not in stages and not args.manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {args.manifest_path}. Run stage 'crawl' first or provide a manifest."
        )

    if "crawl" in stages:
        run_crawl_stage(args)

    if "construct" in stages:
        run_construct_stage(args)


if __name__ == "__main__":
    main()
