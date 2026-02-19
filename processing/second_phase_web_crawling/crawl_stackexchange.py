from __future__ import annotations

import argparse
import html
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, urlencode

from .common import JSONLWriter, download_file, http_get_text, normalize_whitespace, write_json

logger = logging.getLogger(__name__)

try:
    import py7zr  # type: ignore

    _HAS_PY7ZR = True
except Exception:  # pragma: no cover - optional dependency
    py7zr = None
    _HAS_PY7ZR = False


SITE_PREFIX_LANG_MAP = {
    "ru.": "ru",
    "rus.": "ru",
    "es.": "es",
    "pt.": "pt",
    "ja.": "ja",
    "japanese.": "ja",
    "ko.": "ko",
    "korean.": "ko",
    "zh.": "zh",
    "chinese.": "zh",
    "de.": "de",
    "german.": "de",
    "fr.": "fr",
    "french.": "fr",
    "ar.": "ar",
    "arabic.": "ar",
    "hi.": "hi",
    "hindi.": "hi",
}
SE_API_BASE = "https://api.stackexchange.com/2.3"


def detect_site_language(site: str) -> str:
    host = site.strip().lower()
    if host in {"stackoverflow.com", "mathoverflow.net", "serverfault.com", "superuser.com"}:
        return "en"
    for prefix, lang in SITE_PREFIX_LANG_MAP.items():
        if host.startswith(prefix):
            return lang
    if host.startswith("es.stackoverflow"):
        return "es"
    if host.startswith("ru.stackoverflow"):
        return "ru"
    if host.startswith("pt.stackoverflow"):
        return "pt"
    return "en"


def _strip_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_xml_rows(xml_path: Path) -> Iterator[dict]:
    context = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in context:
        if _strip_tag(elem.tag).lower() == "row":
            yield dict(elem.attrib)
        elem.clear()


def clean_post_html(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = re.sub(r"<pre><code>.*?</code></pre>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return normalize_whitespace(cleaned)


def clean_comment_text(text: str) -> str:
    return normalize_whitespace(html.unescape(text or ""))


def api_site_name(site: str) -> str:
    host = site.strip().lower()
    if host.endswith(".stackexchange.com"):
        return host[: -len(".stackexchange.com")]
    if host.endswith(".com"):
        return host[: -len(".com")]
    if host.endswith(".net"):
        return host[: -len(".net")]
    return host


def api_get(
    endpoint: str,
    *,
    params: dict[str, object],
    timeout: int,
    retries: int,
) -> dict:
    query = urlencode(params, doseq=True)
    url = f"{SE_API_BASE}/{endpoint}?{query}"
    payload = json.loads(http_get_text(url, timeout=timeout, retries=retries))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected API payload type for {url}")
    return payload


def api_owner_to_author(owner: dict | None) -> str | None:
    if not owner or not isinstance(owner, dict):
        return None
    if owner.get("user_id") is not None:
        return str(owner["user_id"])
    display = owner.get("display_name")
    if display:
        return str(display)
    return None


def crawl_site_via_api(
    *,
    site: str,
    lang: str,
    writer: JSONLWriter,
    include_comments: bool,
    max_posts: int | None,
    max_comments: int | None,
    timeout: int,
    retries: int,
) -> Counter:
    stats = Counter()
    target_posts = max_posts if max_posts is not None else 1000
    target_comments = max_comments if max_comments is not None else 1000
    api_site = api_site_name(site)

    def _sleep_backoff(payload: dict) -> None:
        backoff = payload.get("backoff")
        if isinstance(backoff, (int, float)) and backoff > 0:
            logger.info("API backoff=%ss (site=%s)", int(backoff), site)
            time.sleep(float(backoff))

    page = 1
    while stats["posts_kept"] < target_posts:
        page_size = min(100, target_posts - stats["posts_kept"])
        payload = api_get(
            "questions",
            params={
                "order": "desc",
                "sort": "creation",
                "site": api_site,
                "filter": "withbody",
                "pagesize": page_size,
                "page": page,
            },
            timeout=timeout,
            retries=retries,
        )
        _sleep_backoff(payload)
        items = payload.get("items") or []
        if not isinstance(items, list) or not items:
            break
        for item in items:
            stats["question_items_seen"] += 1
            if stats["posts_kept"] >= target_posts:
                break
            if not isinstance(item, dict):
                continue
            author = api_owner_to_author(item.get("owner"))
            if not author:
                stats["question_missing_author"] += 1
                continue
            qid = item.get("question_id")
            if qid is None:
                stats["question_missing_id"] += 1
                continue
            title = normalize_whitespace(item.get("title", ""))
            body = clean_post_html(item.get("body", ""))
            text = normalize_whitespace(f"{title} {body}")
            if not text:
                stats["question_empty_text"] += 1
                continue
            writer.write(
                {
                    "raw_id": f"{site}:api_question:{qid}",
                    "author": author,
                    "text": text,
                    "lang": lang,
                    "genre": "qna/question",
                    "source_site": site,
                    "score": item.get("score"),
                    "creation_date": item.get("creation_date"),
                    "post_type_id": "1",
                }
            )
            stats["posts_kept"] += 1
            stats["rows_kept"] += 1
        if not payload.get("has_more"):
            break
        page += 1

    page = 1
    while stats["posts_kept"] < target_posts:
        page_size = min(100, target_posts - stats["posts_kept"])
        payload = api_get(
            "answers",
            params={
                "order": "desc",
                "sort": "creation",
                "site": api_site,
                "filter": "withbody",
                "pagesize": page_size,
                "page": page,
            },
            timeout=timeout,
            retries=retries,
        )
        _sleep_backoff(payload)
        items = payload.get("items") or []
        if not isinstance(items, list) or not items:
            break
        for item in items:
            stats["answer_items_seen"] += 1
            if stats["posts_kept"] >= target_posts:
                break
            if not isinstance(item, dict):
                continue
            author = api_owner_to_author(item.get("owner"))
            if not author:
                stats["answer_missing_author"] += 1
                continue
            aid = item.get("answer_id")
            if aid is None:
                stats["answer_missing_id"] += 1
                continue
            body = clean_post_html(item.get("body", ""))
            if not body:
                stats["answer_empty_text"] += 1
                continue
            writer.write(
                {
                    "raw_id": f"{site}:api_answer:{aid}",
                    "author": author,
                    "text": body,
                    "lang": lang,
                    "genre": "qna/answer",
                    "source_site": site,
                    "score": item.get("score"),
                    "creation_date": item.get("creation_date"),
                    "post_type_id": "2",
                    "parent_id": item.get("question_id"),
                }
            )
            stats["posts_kept"] += 1
            stats["rows_kept"] += 1
        if not payload.get("has_more"):
            break
        page += 1

    if include_comments and target_comments > 0:
        page = 1
        while stats["comments_kept"] < target_comments:
            page_size = min(100, target_comments - stats["comments_kept"])
            payload = api_get(
                "comments",
                params={
                    "order": "desc",
                    "sort": "creation",
                    "site": api_site,
                    "filter": "withbody",
                    "pagesize": page_size,
                    "page": page,
                },
                timeout=timeout,
                retries=retries,
            )
            _sleep_backoff(payload)
            items = payload.get("items") or []
            if not isinstance(items, list) or not items:
                break
            for item in items:
                stats["comment_items_seen"] += 1
                if stats["comments_kept"] >= target_comments:
                    break
                if not isinstance(item, dict):
                    continue
                author = api_owner_to_author(item.get("owner"))
                if not author:
                    stats["comment_missing_author"] += 1
                    continue
                cid = item.get("comment_id")
                if cid is None:
                    stats["comment_missing_id"] += 1
                    continue
                text = clean_comment_text(item.get("body", ""))
                if not text:
                    stats["comment_empty_text"] += 1
                    continue
                writer.write(
                    {
                        "raw_id": f"{site}:api_comment:{cid}",
                        "author": author,
                        "text": text,
                        "lang": lang,
                        "genre": "qna/comment",
                        "source_site": site,
                        "post_id": item.get("post_id"),
                        "score": item.get("score"),
                        "creation_date": item.get("creation_date"),
                    }
                )
                stats["comments_kept"] += 1
                stats["rows_kept"] += 1
            if not payload.get("has_more"):
                break
            page += 1

    return stats


def _extract_archive_member(archive_path: Path, tmp_dir: Path) -> Path:
    if _HAS_PY7ZR:
        with py7zr.SevenZipFile(archive_path, mode="r") as zf:  # type: ignore[arg-type]
            members = [name for name in zf.getnames() if name.lower().endswith(".xml")]
            if not members:
                raise RuntimeError(f"No XML members found in {archive_path}")
            zf.extract(path=tmp_dir, targets=[members[0]])
        extracted = tmp_dir / members[0]
        if extracted.exists():
            return extracted
        candidates = sorted(tmp_dir.rglob("*.xml"))
        if candidates:
            return candidates[0]
        raise RuntimeError(f"Failed to extract XML from {archive_path}")

    for binary in ("7z", "7zz"):
        if shutil.which(binary):
            proc = subprocess.run(
                [binary, "e", "-y", f"-o{tmp_dir}", str(archive_path), "*.xml"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{binary} failed for {archive_path}:\n{proc.stderr.strip() or proc.stdout.strip()}"
                )
            candidates = sorted(tmp_dir.rglob("*.xml"))
            if not candidates:
                raise RuntimeError(f"{binary} extracted no XML for {archive_path}")
            return candidates[0]

    raise RuntimeError(
        "Cannot read .7z archives. Install `py7zr` (`pip install py7zr`) or install `7z`/`7zz`."
    )


def ensure_archive(
    *,
    archives_dir: Path,
    site: str,
    dump_type: str,
    download_mode: str,
    archive_identifier: str,
    archive_base_url: str,
    timeout: int,
    retries: int,
) -> Path | None:
    local_7z = archives_dir / f"{site}-{dump_type}.7z"
    if local_7z.exists():
        return local_7z
    local_xml = archives_dir / f"{site}-{dump_type}.xml"
    if local_xml.exists():
        return local_xml
    if download_mode != "archive":
        return None

    filename = f"{site}-{dump_type}.7z"
    local_path = archives_dir / filename
    url = f"{archive_base_url.rstrip('/')}/{archive_identifier}/{quote(filename)}"
    logger.info("Downloading %s", url)
    try:
        return download_file(
            url,
            local_path,
            timeout=timeout,
            retries=retries,
        )
    except Exception as exc:
        logger.warning("Failed to download %s (%s).", url, exc)
        return None


def parse_posts_archive(
    archive_path: Path,
    *,
    site: str,
    lang: str,
    writer: JSONLWriter,
    max_rows: int | None,
) -> Counter:
    stats = Counter()
    if archive_path.suffix.lower() == ".xml":
        row_iter = iter_xml_rows(archive_path)
        tmp_context = None
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="authbench_se_posts_")
        xml_path = _extract_archive_member(archive_path, Path(tmp_context.name))
        row_iter = iter_xml_rows(xml_path)

    try:
        for row in row_iter:
            stats["rows_seen"] += 1
            if max_rows is not None and stats["rows_kept"] >= max_rows:
                break

            raw_author = row.get("OwnerUserId") or row.get("OwnerDisplayName")
            if not raw_author:
                stats["missing_author"] += 1
                continue

            post_id = row.get("Id")
            if not post_id:
                stats["missing_post_id"] += 1
                continue

            body = clean_post_html(row.get("Body", ""))
            title = normalize_whitespace(html.unescape(row.get("Title", "")))
            post_type = str(row.get("PostTypeId", "")).strip()
            if post_type == "1":
                text = normalize_whitespace(f"{title} {body}")
                genre = "qna/question"
            elif post_type == "2":
                text = body
                genre = "qna/answer"
            else:
                text = normalize_whitespace(f"{title} {body}")
                genre = "qna/post"

            if not text:
                stats["empty_text"] += 1
                continue

            writer.write(
                {
                    "raw_id": f"{site}:post:{post_id}",
                    "author": str(raw_author),
                    "text": text,
                    "lang": lang,
                    "genre": genre,
                    "source_site": site,
                    "post_type_id": post_type,
                    "score": row.get("Score"),
                    "creation_date": row.get("CreationDate"),
                }
            )
            stats["rows_kept"] += 1
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()
    return stats


def parse_comments_archive(
    archive_path: Path,
    *,
    site: str,
    lang: str,
    writer: JSONLWriter,
    max_rows: int | None,
) -> Counter:
    stats = Counter()
    if archive_path.suffix.lower() == ".xml":
        row_iter = iter_xml_rows(archive_path)
        tmp_context = None
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="authbench_se_comments_")
        xml_path = _extract_archive_member(archive_path, Path(tmp_context.name))
        row_iter = iter_xml_rows(xml_path)

    try:
        for row in row_iter:
            stats["rows_seen"] += 1
            if max_rows is not None and stats["rows_kept"] >= max_rows:
                break

            raw_author = row.get("UserId") or row.get("UserDisplayName")
            if not raw_author:
                stats["missing_author"] += 1
                continue
            comment_id = row.get("Id")
            if not comment_id:
                stats["missing_comment_id"] += 1
                continue

            text = clean_comment_text(row.get("Text", ""))
            if not text:
                stats["empty_text"] += 1
                continue

            writer.write(
                {
                    "raw_id": f"{site}:comment:{comment_id}",
                    "author": str(raw_author),
                    "text": text,
                    "lang": lang,
                    "genre": "qna/comment",
                    "source_site": site,
                    "post_id": row.get("PostId"),
                    "score": row.get("Score"),
                    "creation_date": row.get("CreationDate"),
                }
            )
            stats["rows_kept"] += 1
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Stack Exchange corpus (Posts + optional Comments) into AuthBench's JSONL schema."
        )
    )
    parser.add_argument(
        "--input-archives-dir",
        type=Path,
        default=Path("processing/second_phase_web_crawling/downloads/stackexchange"),
        help="Directory containing <site>-Posts.7z / <site>-Comments.7z archives.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/stackexchange/stackexchange.jsonl"),
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=["stackoverflow.com"],
        help="Stack Exchange site dump prefixes (e.g., stackoverflow.com math.stackexchange.com).",
    )
    parser.add_argument(
        "--include-comments",
        dest="include_comments",
        action="store_true",
        default=True,
        help="Include Comments dumps when available (default: enabled).",
    )
    parser.add_argument(
        "--skip-comments",
        dest="include_comments",
        action="store_false",
        help="Disable Comments ingestion and only keep posts.",
    )
    parser.add_argument(
        "--max-posts-per-site",
        type=int,
        default=None,
        help="Optional cap for kept post rows per site.",
    )
    parser.add_argument(
        "--max-comments-per-site",
        type=int,
        default=None,
        help="Optional cap for kept comment rows per site.",
    )
    parser.add_argument(
        "--download-mode",
        choices=["none", "archive", "api"],
        default="none",
        help="StackExchange ingestion mode: local-only, archive.org mirror, or Stack Exchange API.",
    )
    parser.add_argument(
        "--archive-identifier",
        default="stackexchange",
        help="Archive.org identifier used when --download-mode=archive.",
    )
    parser.add_argument(
        "--archive-base-url",
        default="https://archive.org/download",
        help="Base download URL for archive mode.",
    )
    parser.add_argument("--timeout", type=int, default=120)
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
    args.input_archives_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = JSONLWriter(args.output_path)
    per_site_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)

    for site in args.sites:
        site = site.strip()
        if not site:
            continue
        lang = detect_site_language(site)
        logger.info("Processing site=%s (lang=%s)", site, lang)

        if args.download_mode == "api":
            api_stats = crawl_site_via_api(
                site=site,
                lang=lang,
                writer=writer,
                include_comments=args.include_comments,
                max_posts=args.max_posts_per_site,
                max_comments=args.max_comments_per_site,
                timeout=args.timeout,
                retries=args.retries,
            )
            per_site_stats[site]["api"] = dict(api_stats)
            continue

        posts_archive = ensure_archive(
            archives_dir=args.input_archives_dir,
            site=site,
            dump_type="Posts",
            download_mode=args.download_mode,
            archive_identifier=args.archive_identifier,
            archive_base_url=args.archive_base_url,
            timeout=args.timeout,
            retries=args.retries,
        )
        if posts_archive and posts_archive.exists():
            post_stats = parse_posts_archive(
                posts_archive,
                site=site,
                lang=lang,
                writer=writer,
                max_rows=args.max_posts_per_site,
            )
            per_site_stats[site]["posts"] = dict(post_stats)
        else:
            logger.warning("Posts archive missing for %s", site)
            per_site_stats[site]["posts"] = {"missing_archive": 1}

        if args.include_comments:
            comments_archive = ensure_archive(
                archives_dir=args.input_archives_dir,
                site=site,
                dump_type="Comments",
                download_mode=args.download_mode,
                archive_identifier=args.archive_identifier,
                archive_base_url=args.archive_base_url,
                timeout=args.timeout,
                retries=args.retries,
            )
            if comments_archive and comments_archive.exists():
                comment_stats = parse_comments_archive(
                    comments_archive,
                    site=site,
                    lang=lang,
                    writer=writer,
                    max_rows=args.max_comments_per_site,
                )
                per_site_stats[site]["comments"] = dict(comment_stats)
            else:
                logger.warning("Comments archive missing for %s", site)
                per_site_stats[site]["comments"] = {"missing_archive": 1}

    writer.close()
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        "output_path": str(args.output_path),
        "rows_written": writer.count,
        "sites": args.sites,
        "include_comments": args.include_comments,
        "download_mode": args.download_mode,
        "per_site_stats": per_site_stats,
    }
    write_json(summary_path, summary)
    logger.info("Wrote %d rows to %s", writer.count, args.output_path)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
