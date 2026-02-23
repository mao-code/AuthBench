from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .common import DEFAULT_USER_AGENT, JSONLWriter, normalize_whitespace, write_json

logger = logging.getLogger(__name__)

API_BASE = "https://www.googleapis.com/youtube/v3"
TARGET_LANGUAGES = ["en", "zh", "hi", "es", "fr", "ar", "ru", "de", "ja", "ko"]
DEFAULT_REGION_BY_LANG = {
    "en": "US",
    "zh": "TW",
    "hi": "IN",
    "es": "ES",
    "fr": "FR",
    "ar": "EG",
    "ru": "RU",
    "de": "DE",
    "ja": "JP",
    "ko": "KR",
}
EXPECTED_SCRIPTS = {
    "en": {"latin"},
    "es": {"latin"},
    "fr": {"latin"},
    "de": {"latin"},
    "ar": {"arabic"},
    "ru": {"cyrillic"},
    "zh": {"cjk"},
    "ja": {"cjk", "hiragana", "katakana"},
    "ko": {"hangul", "cjk"},
    "hi": {"devanagari"},
}
NON_LATIN_LANGS = {"zh", "ja", "ko", "hi", "ar", "ru"}
RETRYABLE_403_REASONS = {
    "backendError",
    "internalError",
    "quotaExceeded",
    "rateLimitExceeded",
    "userRateLimitExceeded",
}

try:
    from langdetect import LangDetectException, detect

    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover - optional dependency
    LangDetectException = Exception  # type: ignore[assignment]
    detect = None  # type: ignore[assignment]
    _HAS_LANGDETECT = False


def parse_languages(raw: str) -> list[str]:
    langs = [part.strip().lower() for part in re.split(r"[;,]", raw or "") if part.strip()]
    if not langs:
        return TARGET_LANGUAGES.copy()
    return langs


def parse_region_map(raw: str) -> dict[str, str]:
    region_map = dict(DEFAULT_REGION_BY_LANG)
    if not raw.strip():
        return region_map

    for item in re.split(r"[;,]", raw):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        lang, region = item.split(":", 1)
        lang = lang.strip().lower()
        region = region.strip().upper()
        if lang and region:
            region_map[lang] = region
    return region_map


def _script_of_char(ch: str) -> str:
    import unicodedata

    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "ARABIC" in name:
        return "arabic"
    if "HIRAGANA" in name:
        return "hiragana"
    if "KATAKANA" in name:
        return "katakana"
    if "HANGUL" in name:
        return "hangul"
    if "CJK" in name or "IDEOGRAPH" in name:
        return "cjk"
    if "DEVANAGARI" in name:
        return "devanagari"
    return ""


def script_match_ratio(text: str, lang: str) -> float | None:
    expected = EXPECTED_SCRIPTS.get(lang)
    if not expected:
        return None
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    matches = sum(1 for ch in letters if _script_of_char(ch) in expected)
    return matches / len(letters)


def looks_like_language(
    text: str,
    *,
    lang: str,
    use_langdetect: bool,
    langdetect_min_chars: int,
) -> bool:
    text = normalize_whitespace(text)
    if not text:
        return False

    min_script_ratio = {
        "zh": 0.15,
        "ja": 0.2,
        "ko": 0.2,
        "hi": 0.25,
        "ar": 0.25,
        "ru": 0.25,
    }.get(lang, 0.3)

    script_ratio = script_match_ratio(text, lang)
    if lang in NON_LATIN_LANGS:
        if script_ratio is None or script_ratio < min_script_ratio:
            return False

    if (
        use_langdetect
        and _HAS_LANGDETECT
        and detect is not None
        and len(text) >= langdetect_min_chars
    ):
        try:
            detected = detect(text)
        except LangDetectException:
            return lang in NON_LATIN_LANGS
        if not (detected == lang or detected.startswith(lang) or lang.startswith(detected)):
            return False

    return True


def youtube_api_get(
    endpoint: str,
    *,
    params: dict[str, object],
    timeout: int,
    retries: int,
    retry_backoff_sec: float,
    api_key: str,
) -> dict:
    query = urlencode({**params, "key": api_key}, doseq=True)
    url = f"{API_BASE}/{endpoint}?{query}"
    req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Encoding": "identity"})
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected YouTube API payload type for {url}")
            return payload
        except HTTPError as exc:
            last_exc = exc
            body = exc.read().decode("utf-8", errors="replace")
            reason = ""
            message = ""
            try:
                error_payload = json.loads(body)
                if isinstance(error_payload, dict):
                    error_obj = error_payload.get("error")
                    if isinstance(error_obj, dict):
                        message = str(error_obj.get("message") or "")
                        errors = error_obj.get("errors")
                        if isinstance(errors, list) and errors:
                            first = errors[0]
                            if isinstance(first, dict):
                                reason = str(first.get("reason") or "")
            except json.JSONDecodeError:
                message = body.strip()

            is_retryable_403 = exc.code == 403 and (not reason or reason in RETRYABLE_403_REASONS)
            if attempt < retries and (is_retryable_403 or exc.code in {408, 429, 500, 502, 503, 504}):
                sleep_for = retry_backoff_sec * attempt
                logger.warning(
                    "YouTube API GET failed (%s reason=%s message=%s), retrying in %.1fs [attempt %d/%d]",
                    exc.code,
                    reason or "unknown",
                    message or exc.reason,
                    sleep_for,
                    attempt,
                    retries,
                )
                time.sleep(sleep_for)
                continue

            detail = message or body.strip() or str(exc)
            if reason:
                detail = f"{detail} (reason={reason})"
            raise RuntimeError(f"Failed to fetch {url}: HTTP {exc.code} {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            sleep_for = retry_backoff_sec * attempt
            logger.warning(
                "YouTube API GET failed (%s), retrying in %.1fs [attempt %d/%d]",
                exc,
                sleep_for,
                attempt,
                retries,
            )
            time.sleep(sleep_for)

    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def language_targets(langs: list[str], total_docs: int) -> dict[str, int]:
    if not langs:
        return {}
    base = total_docs // len(langs)
    remainder = total_docs % len(langs)
    targets = {lang: base for lang in langs}
    for lang in langs[:remainder]:
        targets[lang] += 1
    return targets


def extract_author(top_snippet: dict) -> str | None:
    author_channel = top_snippet.get("authorChannelId")
    if isinstance(author_channel, dict):
        value = author_channel.get("value")
        if value:
            return str(value)

    author_name = top_snippet.get("authorDisplayName")
    if author_name:
        return str(author_name)
    return None


def crawl_comments_for_video(
    *,
    video_id: str,
    lang: str,
    keep_limit: int,
    writer: JSONLWriter,
    api_key: str,
    timeout: int,
    retries: int,
    retry_backoff_sec: float,
    max_comments_per_video: int,
    max_comment_pages_per_video: int,
    min_chars: int,
    use_langdetect: bool,
    langdetect_min_chars: int,
    seen_comment_ids: set[str],
) -> Counter:
    stats = Counter()
    next_page_token: str | None = None
    pages_scanned = 0

    while (
        stats["rows_kept"] < keep_limit
        and stats["rows_kept"] < max_comments_per_video
        and pages_scanned < max_comment_pages_per_video
    ):
        page_size = min(100, keep_limit - stats["rows_kept"], max_comments_per_video - stats["rows_kept"])
        if page_size <= 0:
            break

        params: dict[str, object] = {
            "part": "snippet",
            "videoId": video_id,
            "order": "time",
            "textFormat": "plainText",
            "maxResults": page_size,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            payload = youtube_api_get(
                "commentThreads",
                params=params,
                timeout=timeout,
                retries=retries,
                retry_backoff_sec=retry_backoff_sec,
                api_key=api_key,
            )
        except Exception as exc:
            logger.warning("commentThreads failed for video=%s: %s", video_id, exc)
            stats["comment_api_errors"] += 1
            break

        stats["comment_api_calls"] += 1
        pages_scanned += 1

        if "error" in payload:
            error = payload.get("error") or {}
            message = ""
            if isinstance(error, dict):
                message = str(error.get("message") or "")
            logger.debug("commentThreads API error for video=%s: %s", video_id, message)
            stats["comment_api_error_payload"] += 1
            break

        items = payload.get("items") or []
        if not isinstance(items, list) or not items:
            break

        for item in items:
            stats["comments_seen"] += 1
            if stats["rows_kept"] >= keep_limit or stats["rows_kept"] >= max_comments_per_video:
                break
            if not isinstance(item, dict):
                continue

            comment_id = item.get("id")
            if not comment_id:
                stats["missing_comment_id"] += 1
                continue
            comment_id = str(comment_id)
            if comment_id in seen_comment_ids:
                stats["duplicate_comment_id"] += 1
                continue

            thread_snippet = item.get("snippet")
            if not isinstance(thread_snippet, dict):
                stats["missing_thread_snippet"] += 1
                continue

            top_comment = thread_snippet.get("topLevelComment")
            if not isinstance(top_comment, dict):
                stats["missing_top_comment"] += 1
                continue

            top_snippet = top_comment.get("snippet")
            if not isinstance(top_snippet, dict):
                stats["missing_top_snippet"] += 1
                continue

            text = top_snippet.get("textOriginal") or top_snippet.get("textDisplay") or ""
            text = normalize_whitespace(str(text))
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            if not looks_like_language(
                text,
                lang=lang,
                use_langdetect=use_langdetect,
                langdetect_min_chars=langdetect_min_chars,
            ):
                stats["language_filtered"] += 1
                continue

            author = extract_author(top_snippet)
            if not author:
                stats["missing_author"] += 1
                continue

            seen_comment_ids.add(comment_id)
            writer.write(
                {
                    "raw_id": f"youtube:{video_id}:{comment_id}",
                    "author": author,
                    "text": text,
                    "lang": lang,
                    "genre": "social_media/youtube_comment",
                    "video_id": video_id,
                    "comment_id": comment_id,
                    "channel_id": top_snippet.get("channelId"),
                    "published_at": top_snippet.get("publishedAt"),
                    "updated_at": top_snippet.get("updatedAt"),
                    "like_count": top_snippet.get("likeCount"),
                    "source_url": f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
                }
            )
            stats["rows_kept"] += 1

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multilingual YouTube comments corpus into AuthBench's JSONL schema."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("processing/second_phase_web_crawling/corpora/ytcomments/ytcomments.jsonl"),
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(TARGET_LANGUAGES),
        help="Comma-separated language codes to collect.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200000,
        help="Total maximum kept comments across all languages.",
    )
    parser.add_argument(
        "--region-map",
        default=",".join(f"{k}:{v}" for k, v in DEFAULT_REGION_BY_LANG.items()),
        help="Language-to-region mapping as lang:REGION pairs, comma-separated.",
    )
    parser.add_argument(
        "--max-video-pages-per-lang",
        type=int,
        default=50,
        help="Max pages of mostPopular videos to scan per language region (50 videos per page).",
    )
    parser.add_argument(
        "--max-comments-per-video",
        type=int,
        default=200,
        help="Max kept comments per video.",
    )
    parser.add_argument(
        "--max-comment-pages-per-video",
        type=int,
        default=8,
        help="Max commentThreads pages to scan per video.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Drop comments shorter than this many characters.",
    )
    parser.add_argument(
        "--skip-langdetect",
        action="store_true",
        help="Disable optional langdetect checks (script checks still apply for non-Latin languages).",
    )
    parser.add_argument(
        "--langdetect-min-chars",
        type=int,
        default=24,
        help="Run langdetect only for comments with at least this many characters.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("YOUTUBE_API_KEY"),
        help="YouTube Data API key (defaults to YOUTUBE_API_KEY env var).",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=1.5,
        help="Linear backoff seconds multiplier between retries.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between API calls to reduce burst rate.",
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

    if not args.api_key:
        raise RuntimeError(
            "Missing YouTube API key. Set YOUTUBE_API_KEY or pass --api-key explicitly."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    langs = parse_languages(args.languages)
    region_map = parse_region_map(args.region_map)

    lang_targets = language_targets(langs, args.max_docs)
    per_lang_stats: dict[str, Counter] = {lang: Counter() for lang in langs}
    per_lang_written = Counter()
    global_stats = Counter()

    writer = JSONLWriter(args.output_path)
    seen_comment_ids: set[str] = set()

    for lang in langs:
        lang_target = lang_targets.get(lang, 0)
        if lang_target <= 0:
            continue

        region = region_map.get(lang, "US")
        logger.info(
            "Collecting YouTube comments for lang=%s region=%s target=%d",
            lang,
            region,
            lang_target,
        )

        next_video_page_token: str | None = None
        seen_video_ids: set[str] = set()
        video_pages_scanned = 0

        while per_lang_written[lang] < lang_target and video_pages_scanned < args.max_video_pages_per_lang:
            params: dict[str, object] = {
                "part": "snippet",
                "chart": "mostPopular",
                "regionCode": region,
                "maxResults": 50,
                "hl": lang,
            }
            if next_video_page_token:
                params["pageToken"] = next_video_page_token

            try:
                video_payload = youtube_api_get(
                    "videos",
                    params=params,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_backoff_sec=args.retry_backoff_sec,
                    api_key=args.api_key,
                )
            except Exception as exc:
                logger.warning("videos.list failed for lang=%s region=%s: %s", lang, region, exc)
                per_lang_stats[lang]["video_api_errors"] += 1
                break

            per_lang_stats[lang]["video_api_calls"] += 1
            video_pages_scanned += 1

            if "error" in video_payload:
                error = video_payload.get("error") or {}
                message = ""
                if isinstance(error, dict):
                    message = str(error.get("message") or "")
                logger.warning("videos.list API error for lang=%s region=%s: %s", lang, region, message)
                per_lang_stats[lang]["video_api_error_payload"] += 1
                break

            items = video_payload.get("items") or []
            if not isinstance(items, list) or not items:
                break

            for item in items:
                if per_lang_written[lang] >= lang_target:
                    break
                if not isinstance(item, dict):
                    continue

                video_id = item.get("id")
                if not video_id:
                    per_lang_stats[lang]["missing_video_id"] += 1
                    continue

                video_id = str(video_id)
                if video_id in seen_video_ids:
                    per_lang_stats[lang]["duplicate_video_id"] += 1
                    continue
                seen_video_ids.add(video_id)
                per_lang_stats[lang]["videos_seen"] += 1

                keep_limit = lang_target - per_lang_written[lang]
                comment_stats = crawl_comments_for_video(
                    video_id=video_id,
                    lang=lang,
                    keep_limit=keep_limit,
                    writer=writer,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_backoff_sec=args.retry_backoff_sec,
                    max_comments_per_video=args.max_comments_per_video,
                    max_comment_pages_per_video=args.max_comment_pages_per_video,
                    min_chars=args.min_chars,
                    use_langdetect=not args.skip_langdetect,
                    langdetect_min_chars=args.langdetect_min_chars,
                    seen_comment_ids=seen_comment_ids,
                )
                per_lang_stats[lang].update(comment_stats)
                per_lang_written[lang] += int(comment_stats.get("rows_kept", 0))

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            next_video_page_token = video_payload.get("nextPageToken")
            if not next_video_page_token:
                break

        if per_lang_written[lang] < lang_target:
            logger.warning(
                "Language shortfall for %s: requested=%d, collected=%d",
                lang,
                lang_target,
                per_lang_written[lang],
            )

    writer.close()

    for lang in langs:
        global_stats.update(per_lang_stats[lang])

    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        "output_path": str(args.output_path),
        "rows_written": writer.count,
        "target_max_docs": args.max_docs,
        "languages": langs,
        "targets_by_lang": lang_targets,
        "rows_by_lang": dict(per_lang_written),
        "used_langdetect": bool((not args.skip_langdetect) and _HAS_LANGDETECT),
        "stats_total": dict(global_stats),
        "stats_by_lang": {lang: dict(stats) for lang, stats in per_lang_stats.items()},
    }
    write_json(summary_path, summary)

    logger.info("Wrote %d rows to %s", writer.count, args.output_path)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
