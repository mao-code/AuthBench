from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def dataset_entry(
    *,
    name: str,
    source: str,
    path: Path,
    repo_root: Path,
) -> dict:
    return {
        "name": name,
        "source": source,
        "loader": "jsonl",
        "path": to_repo_relative(path, repo_root),
        "text_field": "text",
        "author_field": "author",
        "lang_field": "lang",
        "genre_field": "genre",
        "raw_id_field": "raw_id",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a processing-compatible manifest for second-phase crawled corpora "
            "(Stack Exchange, Gutenberg, Wikisource)."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/datasets_manifest.json"),
    )
    parser.add_argument(
        "--stackexchange-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/stackexchange/stackexchange.jsonl"),
    )
    parser.add_argument(
        "--gutenberg-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/gutenberg/gutenberg.jsonl"),
    )
    parser.add_argument(
        "--wikisource-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/wikisource/wikisource.jsonl"),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]
    datasets: list[dict] = []

    if args.stackexchange_path.exists():
        datasets.append(
            dataset_entry(
                name="stackexchange_web_crawl",
                source="stackexchange",
                path=args.stackexchange_path,
                repo_root=repo_root,
            )
        )
    else:
        logger.warning("Skipping missing Stack Exchange file: %s", args.stackexchange_path)

    if args.gutenberg_path.exists():
        datasets.append(
            dataset_entry(
                name="project_gutenberg_web_crawl",
                source="project_gutenberg",
                path=args.gutenberg_path,
                repo_root=repo_root,
            )
        )
    else:
        logger.warning("Skipping missing Gutenberg file: %s", args.gutenberg_path)

    if args.wikisource_path.exists():
        datasets.append(
            dataset_entry(
                name="wikisource_web_crawl",
                source="wikisource",
                path=args.wikisource_path,
                repo_root=repo_root,
            )
        )
    else:
        logger.warning("Skipping missing Wikisource file: %s", args.wikisource_path)

    if not datasets:
        raise RuntimeError("No crawled corpus files were found; cannot write an empty manifest.")

    payload = {"datasets": datasets}
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote manifest with %d datasets to %s", len(datasets), args.manifest_path)


if __name__ == "__main__":
    main()

