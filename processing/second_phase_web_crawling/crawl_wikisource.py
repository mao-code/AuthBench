from __future__ import annotations

import argparse
import bz2
import html
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterator

from .common import JSONLWriter, download_file, normalize_whitespace, write_json

logger = logging.getLogger(__name__)


WIKI_LANG_OVERRIDES = {
    "enwikisource": "en",
    "zhwikisource": "zh",
    "eswikisource": "es",
    "frwikisource": "fr",
    "arwikisource": "ar",
    "ruwikisource": "ru",
    "dewikisource": "de",
    "jawikisource": "ja",
    "kowikisource": "ko",
    "hiwikisource": "hi",
}

AUTHOR_PATTERNS = [
    re.compile(r"\[\[(?:Author|Auteur|Autor|Автор|作者|مؤلف|लेखक|著者|저자)\s*:\s*([^|\]]+)", re.IGNORECASE),
    re.compile(
        r"\|\s*(?:author|auteur|autor|автор|作者|مؤلف|लेखक|著者|저자)\s*=\s*([^\n|}]+)",
        re.IGNORECASE,
    ),
]


def _tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_child_text(elem: ET.Element, target: str) -> str | None:
    for child in elem:
        if _tag(child.tag) == target:
            return child.text
    return None


def wiki_to_lang(wiki: str) -> str:
    if wiki in WIKI_LANG_OVERRIDES:
        return WIKI_LANG_OVERRIDES[wiki]
    if wiki.endswith("wikisource"):
        return wiki.replace("wikisource", "")
    return "en"


def download_dump(wiki: str, dumps_dir: Path, timeout: int, retries: int) -> Path:
    filename = f"{wiki}-latest-pages-articles-multistream.xml.bz2"
    destination = dumps_dir / wiki / filename
    if destination.exists():
        return destination
    url = f"https://dumps.wikimedia.org/{wiki}/latest/{filename}"
    logger.info("Downloading %s", url)
    return download_file(url, destination, timeout=timeout, retries=retries)


def clean_wikitext(text: str) -> str:
    body = text or ""
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    body = re.sub(r"<ref[^>]*>.*?</ref>", " ", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", body)

    # Remove templates iteratively to handle shallow nesting.
    for _ in range(8):
        new_body = re.sub(r"\{\{[^{}]*\}\}", " ", body)
        if new_body == body:
            break
        body = new_body

    body = re.sub(r"\[\[(?:Category|Kategorie|Категория|分类|تصنيف|분류):[^\]]+\]\]", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"\[\[(?:File|Image|Datei|Файл|文件):[^\]]+\]\]", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", body)
    body = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", body)
    body = re.sub(r"\[\[([^\]]+)\]\]", r"\1", body)
    body = body.replace("'''", "").replace("''", "")
    body = body.replace("{|", " ").replace("|}", " ")
    body = re.sub(r"^\|.*$", " ", body, flags=re.MULTILINE)
    body = html.unescape(body)
    return normalize_whitespace(body)


def clean_author(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\{\{[^{}]*\}\}", " ", value)
    value = value.replace("Author:", "").replace("Auteur:", "").replace("Autor:", "")
    value = value.replace("作者:", "").replace("Автор:", "").replace("مؤلف:", "")
    return normalize_whitespace(value)


def extract_author(title: str, wikitext: str) -> str | None:
    for pattern in AUTHOR_PATTERNS:
        match = pattern.search(wikitext)
        if not match:
            continue
        candidate = clean_author(match.group(1))
        if candidate:
            return candidate

    # Heuristic fallback: "Some title (Author Name)"
    paren_match = re.search(r"\(([^()]{3,120})\)\s*$", title or "")
    if paren_match:
        candidate = clean_author(paren_match.group(1))
        if candidate:
            return candidate
    return None


def classify_genre(title: str, text: str) -> str:
    blob = f"{title} {text}".lower()
    if any(token in blob for token in ("speech", "address", "oration", "discours", "discurso", "речь", "خطاب", "भाषण", "演讲", "演說")):
        return "literature/speech_essay"
    if any(token in blob for token in ("essay", "essai", "ensayo", "очерк", "مقال", "निबंध", "随笔")):
        return "literature/speech_essay"
    if any(token in blob for token in ("poem", "poetry", "poésie", "poesía", "стих", "قصيدة", "कविता", "诗")):
        return "poetry"
    return "literature"


def iter_pages(dump_path: Path) -> Iterator[dict]:
    with bz2.open(dump_path, "rb") as fh:
        context = ET.iterparse(fh, events=("end",))
        for _, elem in context:
            if _tag(elem.tag) != "page":
                continue

            title = _find_child_text(elem, "title") or ""
            ns = _find_child_text(elem, "ns") or "0"
            redirect = any(_tag(child.tag) == "redirect" for child in elem)
            page_id: str | None = None
            rev_id: str | None = None
            text: str | None = None
            for child in elem:
                ctag = _tag(child.tag)
                if ctag == "id" and page_id is None:
                    page_id = child.text
                elif ctag == "revision":
                    if rev_id is None:
                        rev_id = _find_child_text(child, "id")
                    text = _find_child_text(child, "text")

            payload = {
                "title": title,
                "ns": ns,
                "redirect": redirect,
                "page_id": page_id,
                "revision_id": rev_id,
                "text": text or "",
            }
            yield payload
            elem.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Wikisource corpus from Wikimedia dumps into AuthBench's JSONL schema."
    )
    parser.add_argument(
        "--wikis",
        default="enwikisource,frwikisource,eswikisource,ruwikisource",
        help="Comma-separated Wikimedia dump project names (e.g., enwikisource).",
    )
    parser.add_argument(
        "--dumps-dir",
        type=Path,
        default=Path("processing/second_phase_web_crawling/downloads/wikisource"),
        help="Directory for .bz2 dump files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/wikisource/wikisource.jsonl"),
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--max-docs-per-wiki",
        type=int,
        default=20000,
        help="Maximum kept docs per wiki.",
    )
    parser.add_argument(
        "--max-total-docs",
        type=int,
        default=100000,
        help="Maximum kept docs overall.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Drop documents shorter than this many chars after text cleanup.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.dumps_dir.mkdir(parents=True, exist_ok=True)

    wikis = [w.strip() for w in args.wikis.split(",") if w.strip()]
    writer = JSONLWriter(args.output_path)

    global_stats = Counter()
    per_wiki_stats: dict[str, dict[str, int]] = {}
    per_lang = Counter()

    for wiki in wikis:
        if writer.count >= args.max_total_docs:
            break
        lang = wiki_to_lang(wiki)
        dump_path = download_dump(wiki, args.dumps_dir, args.timeout, args.retries)
        logger.info("Parsing wiki=%s lang=%s dump=%s", wiki, lang, dump_path)

        kept_for_wiki = 0
        stats = Counter()
        for page in iter_pages(dump_path):
            stats["pages_seen"] += 1
            if kept_for_wiki >= args.max_docs_per_wiki:
                break
            if writer.count >= args.max_total_docs:
                break

            if page["ns"] != "0":
                stats["non_main_namespace"] += 1
                continue
            if page["redirect"]:
                stats["redirect"] += 1
                continue
            if not page["text"]:
                stats["empty_text"] += 1
                continue

            author = extract_author(page["title"], page["text"])
            if not author:
                stats["missing_author"] += 1
                continue

            cleaned = clean_wikitext(page["text"])
            if len(cleaned) < args.min_chars:
                stats["too_short"] += 1
                continue

            genre = classify_genre(page["title"], cleaned[:1200])
            raw_id = f"{wiki}:{page.get('page_id') or 'unknown'}:{page.get('revision_id') or 'unknown'}"
            writer.write(
                {
                    "raw_id": raw_id,
                    "author": author,
                    "text": cleaned,
                    "lang": lang,
                    "genre": genre,
                    "title": page["title"],
                    "wiki": wiki,
                    "page_id": page.get("page_id"),
                    "revision_id": page.get("revision_id"),
                }
            )
            kept_for_wiki += 1
            stats["rows_kept"] += 1
            per_lang[lang] += 1

        per_wiki_stats[wiki] = dict(stats)
        global_stats.update(stats)

    writer.close()
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        "output_path": str(args.output_path),
        "wikis": wikis,
        "rows_written": writer.count,
        "max_docs_per_wiki": args.max_docs_per_wiki,
        "max_total_docs": args.max_total_docs,
        "stats_total": dict(global_stats),
        "stats_per_wiki": per_wiki_stats,
        "rows_by_lang": dict(per_lang),
    }
    write_json(summary_path, summary)
    logger.info("Wrote %d rows to %s", writer.count, args.output_path)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()

