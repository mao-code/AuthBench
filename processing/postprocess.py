from __future__ import annotations

import argparse
import json
import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from langdetect import LangDetectException, detect
_LANGDETECT_AVAILABLE = True

from .config import (
    DIRTY_DEFAULTS,
    GENRE_PERCENTS,
    LANGUAGE_PERCENTS,
    LENGTH_BUCKET_PERCENTS,
    make_split_ratios,
)
from .dirty import dirty_reason
from .sampling import build_retrieval_sets, sample_language_docs, split_by_language
from .types import ProcessedDocument
from .utils import count_tokens, length_bucket, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

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


def _script_of_char(ch: str) -> str:
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


def _collapse_spaced_letters(text: str, *, min_run: int = 2) -> str:
    tokens = text.split()
    collapsed: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if len(token) == 1 and token.isalpha():
            run: list[str] = []
            while idx < len(tokens) and len(tokens[idx]) == 1 and tokens[idx].isalpha():
                run.append(tokens[idx])
                idx += 1
            if len(run) >= min_run:
                collapsed.append("".join(run))
            else:
                collapsed.extend(run)
            continue
        collapsed.append(token)
        idx += 1

    cleaned = " ".join(collapsed)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    return cleaned.strip()


def _spacing_stats(text: str):
    tokens = [t for t in text.split() if t]
    single_letter_tokens = [t for t in tokens if len(t) == 1 and t.isalpha()]
    single_ratio = (len(single_letter_tokens) / len(tokens)) if tokens else 0.0
    max_run = 0
    cur = 0
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return single_ratio, max_run


def normalize_spaced_text(
    text: str,
    *,
    collapse_ratio_threshold: float,
    min_run: int,
) -> tuple[str, bool, float, int]:
    single_ratio, max_run = _spacing_stats(text)
    should_collapse = single_ratio >= collapse_ratio_threshold or max_run >= min_run
    if not should_collapse:
        return text, False, single_ratio, max_run
    cleaned = _collapse_spaced_letters(text, min_run=min_run)
    new_ratio, new_max = _spacing_stats(cleaned)
    return cleaned, cleaned != text, new_ratio, new_max


def _script_match_ratio(text: str, lang: str) -> float | None:
    expected = EXPECTED_SCRIPTS.get(lang)
    if not expected:
        return None
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    matches = sum(1 for ch in letters if _script_of_char(ch) in expected)
    return matches / len(letters)


def looks_untranslatable(
    text: str,
    lang: str,
    *,
    min_alpha_ratio: float,
    min_alpha_token_ratio: float,
    max_single_letter_ratio: float,
    max_single_letter_run: int,
    use_langdetect: bool,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    tokens = [t for t in stripped.split() if t]
    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not alpha_tokens:
        return True
    letters = [ch for ch in stripped if ch.isalpha()]

    no_space_chars = [c for c in stripped if not c.isspace()]
    if no_space_chars:
        alpha_ratio = sum(1 for ch in no_space_chars if ch.isalpha()) / len(no_space_chars)
        if alpha_ratio < min_alpha_ratio:
            return True

    alpha_token_ratio = len(alpha_tokens) / len(tokens)
    if alpha_token_ratio < min_alpha_token_ratio:
        return True

    single_ratio, max_run = _spacing_stats(stripped)
    if single_ratio > max_single_letter_ratio or max_run >= max_single_letter_run:
        return True

    script_ratio = _script_match_ratio(stripped, lang)
    min_script_ratio_by_lang = {
        "zh": 0.15,
        "ja": 0.2,
        "ko": 0.2,
        "hi": 0.25,
        "ar": 0.25,
        "ru": 0.25,
    }
    min_script_ratio = min_script_ratio_by_lang.get(lang, 0.3)
    if script_ratio is not None and len(letters) >= 8 and script_ratio < min_script_ratio:
        return True

    if (
        use_langdetect
        and _LANGDETECT_AVAILABLE
        and (script_ratio is None or script_ratio < min_script_ratio + 0.1)
    ):
        try:
            detected = detect(stripped)
            if detected and not (
                detected == lang or detected.startswith(lang) or lang.startswith(detected)
            ):
                return True
        except LangDetectException:
            return True

    return False


def _resolve_doc_id(row: dict) -> str | None:
    for key in ("doc_id", "candidate_id", "query_id"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _row_to_processed_doc(
    row: dict,
    *,
    split: str,
    fallback_author_id: str = "",
) -> ProcessedDocument | None:
    doc_id = _resolve_doc_id(row)
    lang = str(row.get("lang", "") or "").strip().lower()
    if not lang:
        logger.debug("Skipping row with missing lang in %s: %s", split, doc_id)
        return None

    author_id = str(row.get("author_id", "") or fallback_author_id or "").strip()
    if not author_id:
        logger.debug("Skipping row with missing author_id in %s: %s", split, doc_id)
        return None

    text = row.get("content")
    if text is None:
        text = row.get("text", "")
    text = str(text or "")
    token_len = int(row.get("token_length") or count_tokens(text))
    genre = str(row.get("genre", "unknown") or "unknown")
    source = str(row.get("source", "") or "")
    raw_id = str(row.get("raw_id") or doc_id or "")
    return ProcessedDocument(
        raw_id=raw_id,
        doc_id=doc_id,
        author_id=author_id,
        text=text,
        lang=lang,
        source=source,
        genre=genre,
        token_length=token_len,
        length_bucket=length_bucket(token_len),
    )


def _read_query_author_map(ground_truth_path: Path) -> dict[str, str]:
    if not ground_truth_path.exists():
        return {}
    query_to_author: dict[str, str] = {}
    for row in read_jsonl(ground_truth_path):
        query_id = row.get("query_id")
        author_id = row.get("author_id")
        if query_id and author_id:
            query_to_author[str(query_id)] = str(author_id)
    return query_to_author


def _dedup_key(doc: ProcessedDocument, split: str) -> str:
    if doc.doc_id:
        return doc.doc_id
    return f"{split}:{doc.lang}:{doc.source}:{doc.author_id}:{doc.raw_id}"


def _read_stage_documents(base_dir: Path) -> list[ProcessedDocument]:
    docs: list[ProcessedDocument] = []
    seen_keys: set[str] = set()
    documents_files_loaded = 0

    for split in ("train", "dev", "test"):
        split_dir = base_dir / split
        documents_path = split_dir / "documents.jsonl"
        if documents_path.exists():
            documents_files_loaded += 1
            for row in read_jsonl(documents_path):
                doc = _row_to_processed_doc(row, split=split)
                if doc is None:
                    continue
                key = _dedup_key(doc, split)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                docs.append(doc)
            continue

        candidates_path = split_dir / "candidates.jsonl"
        queries_path = split_dir / "queries.jsonl"
        loaded_any = False

        if candidates_path.exists():
            loaded_any = True
            for row in read_jsonl(candidates_path):
                doc = _row_to_processed_doc(row, split=split)
                if doc is None:
                    continue
                key = _dedup_key(doc, split)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                docs.append(doc)

        if queries_path.exists():
            loaded_any = True
            query_to_author = _read_query_author_map(split_dir / "ground_truth.jsonl")
            skipped_missing_author = 0
            for row in read_jsonl(queries_path):
                query_id = row.get("query_id")
                author_id = query_to_author.get(str(query_id)) if query_id else ""
                doc = _row_to_processed_doc(
                    row,
                    split=split,
                    fallback_author_id=author_id,
                )
                if doc is None:
                    skipped_missing_author += 1
                    continue
                key = _dedup_key(doc, split)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                docs.append(doc)
            if skipped_missing_author:
                logger.warning(
                    "Skipped %d query rows without author mapping in %s/%s.",
                    skipped_missing_author,
                    base_dir,
                    split,
                )

        if not loaded_any:
            logger.warning(
                "Missing retrieval files for split %s under %s (expected candidates.jsonl and/or queries.jsonl).",
                split,
                base_dir,
            )

    if documents_files_loaded:
        logger.info(
            "Loaded stage documents from documents.jsonl in %d split(s).",
            documents_files_loaded,
        )
    return docs


def _read_candidates(base_dir: Path) -> list[ProcessedDocument]:
    """
    Backward-compatible alias used by downstream modules.
    """
    return _read_stage_documents(base_dir)


def _summarize_docs(docs: Iterable[ProcessedDocument]) -> dict:
    lang_counter = Counter(doc.lang for doc in docs)
    genre_counter = Counter(doc.genre for doc in docs)
    length_counter = Counter(doc.length_bucket for doc in docs)
    return {
        "total": sum(lang_counter.values()),
        "by_lang": dict(lang_counter),
        "by_genre": dict(genre_counter),
        "by_length_bucket": dict(length_counter),
    }


def filter_documents(
    docs: Iterable[ProcessedDocument], args
) -> tuple[list[ProcessedDocument], Counter, Counter, list[dict]]:
    cleaned: list[ProcessedDocument] = []
    drop_reasons: Counter = Counter()
    drop_by_lang: Counter = Counter()
    drop_records: list[dict] = []
    spacing_collapsed = 0
    total_seen = 0

    for doc in docs:
        total_seen += 1
        text, collapsed, single_ratio, max_run = normalize_spaced_text(
            doc.text,
            collapse_ratio_threshold=args.spacing_collapse_ratio,
            min_run=args.min_spacing_run,
        )
        if collapsed:
            spacing_collapsed += 1
        doc.text = text
        doc.token_length = count_tokens(doc.text)
        doc.length_bucket = length_bucket(doc.token_length)

        if single_ratio > args.max_single_letter_ratio or max_run > args.max_single_letter_run:
            drop_reasons["excessive_spaced_letters"] += 1
            drop_by_lang[f"{doc.lang}:excessive_spaced_letters"] += 1
            drop_records.append(
                {
                    "raw_id": doc.raw_id,
                    "doc_id": doc.doc_id,
                    "lang": doc.lang,
                    "source": doc.source,
                    "reason": "excessive_spaced_letters",
                    "snippet": doc.text[:100],
                }
            )
            continue

        dirty = dirty_reason(
            doc.text,
            doc.token_length,
            unique_token_ratio=DIRTY_DEFAULTS["unique_token_ratio"],
            symbol_ratio=DIRTY_DEFAULTS["symbol_ratio"],
            max_consecutive_symbols=DIRTY_DEFAULTS["max_consecutive_symbols"],
            max_repeated_char_run=DIRTY_DEFAULTS["max_repeated_char_run"],
            source=doc.source,
        )
        if dirty:
            drop_reasons[f"dirty:{dirty}"] += 1
            drop_by_lang[f"{doc.lang}:dirty:{dirty}"] += 1
            drop_records.append(
                {
                    "raw_id": doc.raw_id,
                    "doc_id": doc.doc_id,
                    "lang": doc.lang,
                    "source": doc.source,
                    "reason": f"dirty:{dirty}",
                    "snippet": doc.text[:100],
                }
            )
            continue

        if looks_untranslatable(
            doc.text,
            doc.lang,
            min_alpha_ratio=args.min_alpha_ratio,
            min_alpha_token_ratio=args.min_alpha_token_ratio,
            max_single_letter_ratio=args.max_single_letter_ratio,
            max_single_letter_run=args.max_single_letter_run,
            use_langdetect=not args.skip_langdetect,
        ):
            drop_reasons["untranslatable"] += 1
            drop_by_lang[f"{doc.lang}:untranslatable"] += 1
            drop_records.append(
                {
                    "raw_id": doc.raw_id,
                    "doc_id": doc.doc_id,
                    "lang": doc.lang,
                    "source": doc.source,
                    "reason": "untranslatable",
                    "snippet": doc.text[:100],
                }
            )
            continue

        cleaned.append(doc)

    logger.info(
        "Spacing normalization applied to %d of %d docs (ratio %.3f).",
        spacing_collapsed,
        total_seen,
        (spacing_collapsed / total_seen) if total_seen else 0,
    )

    return cleaned, drop_reasons, drop_by_lang, drop_records


def _normalized_language_percents(available_langs: set[str]) -> dict[str, float]:
    percents = {lang: pct for lang, pct in LANGUAGE_PERCENTS.items() if lang in available_langs and pct > 0}
    if not percents and available_langs:
        pct = 1 / len(available_langs)
        return {lang: pct for lang in available_langs}
    total = sum(percents.values()) or 1.0
    return {lang: pct / total for lang, pct in percents.items()}


def compute_language_targets(
    docs_by_lang: dict[str, list[ProcessedDocument]],
    total_target: int,
) -> tuple[dict[str, int], dict]:
    available_total = sum(len(v) for v in docs_by_lang.values())
    total_target = min(total_target, available_total)
    percents = _normalized_language_percents(set(docs_by_lang))

    targets = {lang: int(round(total_target * pct)) for lang, pct in percents.items()}
    diff = total_target - sum(targets.values())
    if diff:
        ordered = sorted(percents.items(), key=lambda kv: kv[1], reverse=True)
        idx = 0
        while diff != 0 and ordered:
            lang = ordered[idx % len(ordered)][0]
            targets[lang] = targets.get(lang, 0) + (1 if diff > 0 else -1)
            diff += -1 if diff > 0 else 1
            idx += 1

    for lang, lang_docs in docs_by_lang.items():
        cap = len(lang_docs)
        if targets.get(lang, 0) > cap:
            targets[lang] = cap

    remaining = total_target - sum(targets.values())
    if remaining > 0:
        ordered = sorted(percents.items(), key=lambda kv: kv[1], reverse=True)
        for lang, _ in ordered:
            if remaining <= 0:
                break
            cap = len(docs_by_lang[lang]) - targets.get(lang, 0)
            if cap <= 0:
                continue
            add = min(cap, remaining)
            targets[lang] += add
            remaining -= add

    log = {
        "available_total": available_total,
        "requested_total": total_target,
        "final_total": sum(targets.values()),
        "per_language": {
            lang: {
                "target": targets.get(lang, 0),
                "available": len(docs),
                "percent_weight": percents.get(lang, 0.0),
            }
            for lang, docs in docs_by_lang.items()
        },
    }
    return targets, log


def sample_documents(
    docs: Iterable[ProcessedDocument],
    targets: dict[str, int],
    rng: random.Random,
) -> tuple[list[ProcessedDocument], list[dict]]:
    docs_by_lang: dict[str, list[ProcessedDocument]] = defaultdict(list)
    for doc in docs:
        docs_by_lang[doc.lang].append(doc)

    selected: list[ProcessedDocument] = []
    sampling_log: list[dict] = []
    for lang, target in targets.items():
        lang_docs = docs_by_lang.get(lang, [])
        lang_docs.sort(key=lambda d: (d.source, d.author_id, d.raw_id))
        sampled, deficits = sample_language_docs(
            lang,
            lang_docs,
            target,
            GENRE_PERCENTS.get(lang),
            LENGTH_BUCKET_PERCENTS,
            rng,
        )
        selected.extend(sampled)
        sampling_log.extend(deficits)

    return selected, sampling_log


def write_outputs(
    splits: dict[str, list[ProcessedDocument]],
    output_dir: Path,
    rng: random.Random,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    for split_name, docs in splits.items():
        candidates, queries, ground_truth = build_retrieval_sets(docs, rng)
        split_path = output_dir / split_name
        write_jsonl(split_path / "candidates.jsonl", candidates)
        write_jsonl(split_path / "queries.jsonl", queries)
        write_jsonl(split_path / "ground_truth.jsonl", ground_truth)
        split_summary[split_name] = {
            "documents": len(docs),
            "candidates": len(candidates),
            "queries": len(queries),
            "ground_truth": len(ground_truth),
            "documents_by_lang": Counter(doc.lang for doc in docs),
            "candidates_by_lang": Counter(doc["lang"] for doc in candidates),
            "queries_by_lang": Counter(doc["lang"] for doc in queries),
        }
    return split_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process AuthBench outputs to clean and rebalance.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Existing output directory produced by build_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for cleaned outputs (defaults to <input_dir>_postprocessed).",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional cap on total documents after filtering; defaults to all cleaned docs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--spacing-collapse-ratio",
        type=float,
        default=0.25,
        help="Collapse letter-by-letter text when single-letter tokens exceed this ratio.",
    )
    parser.add_argument(
        "--min-spacing-run",
        type=int,
        default=2,
        help="Minimum consecutive single-letter tokens to trigger collapsing.",
    )
    parser.add_argument(
        "--max-single-letter-ratio",
        type=float,
        default=0.45,
        help="Drop entries if >this fraction of tokens are single letters after normalization.",
    )
    parser.add_argument(
        "--max-single-letter-run",
        type=int,
        default=10,
        help="Drop entries if any run of single-letter tokens meets/exceeds this value after normalization.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.25,
        help="Minimum fraction of alphabetic characters (non-space) required to keep a row.",
    )
    parser.add_argument(
        "--min-alpha-token-ratio",
        type=float,
        default=0.5,
        help="Minimum fraction of tokens containing alphabetic characters required to keep a row.",
    )
    parser.add_argument(
        "--skip-langdetect",
        action="store_true",
        help="Disable optional langdetect consistency check.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for the final dataset.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Dev split ratio for the final dataset.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio for the final dataset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (INFO, DEBUG).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rng = random.Random(args.seed)

    output_dir = args.output_dir or Path(f"{args.input_dir}_postprocessed")

    logger.info("Loading stage documents from %s", args.input_dir)
    raw_docs = _read_stage_documents(args.input_dir)
    logger.info("Loaded %d documents across splits.", len(raw_docs))
    logger.info("Input by language: %s", dict(_summarize_docs(raw_docs)["by_lang"]))

    filtered_docs, drop_reasons, drop_by_lang, drop_records = filter_documents(raw_docs, args)
    if not filtered_docs:
        raise RuntimeError("No documents remain after filtering.")
    logger.info(
        "Kept %d documents after filtering (dropped %d).",
        len(filtered_docs),
        sum(drop_reasons.values()),
    )
    if drop_reasons:
        logger.info(
            "Top drop reasons: %s",
            ", ".join(f"{k}={v}" for k, v in drop_reasons.most_common(10)),
        )
        logger.info(
            "Top drop reasons by language: %s",
            ", ".join(f"{k}={v}" for k, v in drop_by_lang.most_common(10)),
        )
    logger.info("After filter by language: %s", dict(_summarize_docs(filtered_docs)["by_lang"]))

    total_target = args.target_total or len(filtered_docs)
    docs_by_lang: dict[str, list[ProcessedDocument]] = defaultdict(list)
    for doc in filtered_docs:
        docs_by_lang[doc.lang].append(doc)

    lang_targets, lang_log = compute_language_targets(docs_by_lang, total_target)
    logger.info(
        "Language targets (requested %d): %s",
        total_target,
        ", ".join(f"{k}={v}" for k, v in sorted(lang_targets.items())),
    )
    selected_docs, sampling_log = sample_documents(filtered_docs, lang_targets, rng)
    if sampling_log:
        logger.info("Sampling shortfalls: %d entries (see summary file).", len(sampling_log))
    logger.info("After sampling by language: %s", dict(_summarize_docs(selected_docs)["by_lang"]))

    split_ratios = make_split_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)
    splits = split_by_language(selected_docs, split_ratios, rng)
    split_summary = write_outputs(splits, output_dir, rng)
    for split_name, stats in split_summary.items():
        logger.info(
            "[%s] docs=%d candidates=%d queries=%d gt=%d docs_by_lang=%s cand_by_lang=%s queries_by_lang=%s",
            split_name,
            stats["documents"],
            stats["candidates"],
            stats["queries"],
            stats["ground_truth"],
            dict(stats["documents_by_lang"]),
            dict(stats["candidates_by_lang"]),
            dict(stats["queries_by_lang"]),
        )

    dirty_log_path = output_dir / "postprocess_dirty.log"
    if drop_records:
        write_jsonl(dirty_log_path, drop_records)
        logger.info("Wrote dirty log with %d entries to %s", len(drop_records), dirty_log_path)
    else:
        logger.info("No dirty log written; no documents were dropped during postprocessing.")

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "before_filter": _summarize_docs(raw_docs),
        "after_filter": _summarize_docs(filtered_docs),
        "after_sampling": _summarize_docs(selected_docs),
        "drop_reasons": dict(drop_reasons),
        "drop_reasons_by_lang": dict(drop_by_lang),
        "dirty_log_path": str(dirty_log_path) if drop_records else None,
        "dirty_log_count": len(drop_records),
        "language_targets": lang_log,
        "sampling_deficits": sampling_log,
        "splits": split_summary,
    }
    (output_dir / "postprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote cleaned outputs to %s", output_dir.resolve())


if __name__ == "__main__":
    main()

    """
    python -m processing.postprocess \
        --input-dir processing/outputs/official_ttl300k_cap10M_sf10k \
        --target-total 10000  # sanity check
    """
