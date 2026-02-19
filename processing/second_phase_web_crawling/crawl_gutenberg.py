from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from collections import Counter
from pathlib import Path

from .common import JSONLWriter, download_file, http_get_text, normalize_whitespace, write_json

logger = logging.getLogger(__name__)


def parse_languages(raw: str) -> list[str]:
    langs = [part.strip().lower() for part in re.split(r"[;,]", raw or "") if part.strip()]
    return langs


def first_author(raw: str) -> str | None:
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    if not parts:
        return None
    return parts[0]


def classify_genre(title: str, subjects: str, bookshelves: str) -> str:
    blob = " ".join([title, subjects, bookshelves]).lower()
    if any(token in blob for token in ("speech", "speeches", "address", "oration", "essay", "essays", "letters")):
        return "literature/speech_essay"
    if "poetry" in blob:
        return "poetry"
    if "drama" in blob or "play" in blob:
        return "literature/drama"
    return "literature"


def gutenberg_text_urls(ebook_id: int) -> list[str]:
    eid = str(ebook_id)
    return [
        f"https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt",
        f"https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt.utf-8",
        f"https://www.gutenberg.org/files/{eid}/{eid}-0.txt",
        f"https://www.gutenberg.org/files/{eid}/{eid}.txt",
        f"https://www.gutenberg.org/files/{eid}/{eid}-8.txt",
    ]


def fetch_ebook_text(ebook_id: int, timeout: int, retries: int) -> tuple[str | None, str | None]:
    for url in gutenberg_text_urls(ebook_id):
        try:
            text = http_get_text(url, timeout=timeout, retries=retries)
            if text and text.strip():
                return text, url
        except Exception:
            continue
    return None, None


def strip_gutenberg_boilerplate(text: str) -> str:
    body = text.replace("\r\n", "\n").replace("\r", "\n")
    start_re = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    end_re = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    start_match = start_re.search(body)
    end_match = end_re.search(body)
    if start_match and end_match and end_match.start() > start_match.end():
        body = body[start_match.end() : end_match.start()]
    elif start_match:
        body = body[start_match.end() :]
    elif end_match:
        body = body[: end_match.start()]
    return body.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Project Gutenberg corpus (books/essays/speeches) "
            "into AuthBench's JSONL schema."
        )
    )
    parser.add_argument(
        "--catalog-url",
        default="https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
        help="Project Gutenberg metadata catalog URL.",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/downloads/gutenberg/pg_catalog.csv"),
        help="Local cache for the catalog file.",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Force re-download of catalog metadata.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/gutenberg/gutenberg.jsonl"),
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--languages",
        default="en,es,fr,de,ru,ar,zh,ja,ko,hi",
        help="Comma-separated language codes to keep.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50000,
        help="Maximum kept documents.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Drop books shorter than this many characters after cleanup.",
    )
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between ebook requests.",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    keep_langs = {x.strip().lower() for x in args.languages.split(",") if x.strip()}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_path.parent.mkdir(parents=True, exist_ok=True)

    if args.refresh_catalog or not args.catalog_path.exists():
        logger.info("Downloading catalog metadata from %s", args.catalog_url)
        download_file(args.catalog_url, args.catalog_path, overwrite=True)

    writer = JSONLWriter(args.output_path)
    stats = Counter()
    by_lang = Counter()

    with args.catalog_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if writer.count >= args.max_docs:
                break
            stats["catalog_rows_seen"] += 1

            if (row.get("Type") or "").strip() != "Text":
                stats["non_text_type"] += 1
                continue

            text_id = (row.get("Text#") or "").strip()
            if not text_id.isdigit():
                stats["missing_text_id"] += 1
                continue
            ebook_id = int(text_id)

            langs = parse_languages(row.get("Language", ""))
            if not langs:
                stats["missing_language"] += 1
                continue
            lang = next((lg for lg in langs if lg in keep_langs), None)
            if not lang:
                stats["language_filtered"] += 1
                continue

            author = first_author(row.get("Authors", ""))
            if not author:
                stats["missing_author"] += 1
                continue

            raw_text, chosen_url = fetch_ebook_text(
                ebook_id,
                timeout=args.request_timeout,
                retries=args.retries,
            )
            time.sleep(args.sleep_seconds)
            if not raw_text:
                stats["download_failed"] += 1
                continue

            cleaned = strip_gutenberg_boilerplate(raw_text)
            cleaned = normalize_whitespace(cleaned)
            if len(cleaned) < args.min_chars:
                stats["too_short"] += 1
                continue

            title = normalize_whitespace(row.get("Title", ""))
            subjects = normalize_whitespace(row.get("Subjects", ""))
            bookshelves = normalize_whitespace(row.get("Bookshelves", ""))
            genre = classify_genre(title, subjects, bookshelves)

            writer.write(
                {
                    "raw_id": f"gutenberg:{ebook_id}",
                    "author": author,
                    "text": cleaned,
                    "lang": lang,
                    "genre": genre,
                    "title": title,
                    "issued": row.get("Issued"),
                    "subjects": subjects,
                    "bookshelves": bookshelves,
                    "source_url": chosen_url,
                }
            )
            by_lang[lang] += 1
            stats["rows_kept"] += 1

    writer.close()
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        "output_path": str(args.output_path),
        "catalog_path": str(args.catalog_path),
        "rows_written": writer.count,
        "languages_filter": sorted(keep_langs),
        "stats": dict(stats),
        "rows_by_lang": dict(by_lang),
    }
    write_json(summary_path, summary)
    logger.info("Wrote %d rows to %s", writer.count, args.output_path)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()

