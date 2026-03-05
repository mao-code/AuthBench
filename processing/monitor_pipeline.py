from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .build_benchmark import AuthorAccumulator, _parse_dataset_caps, buffered_shuffle
from .chunker import chunk_document, truncate_raw_document
from .config import (
    AUTHOR_DOC_LIMITS,
    CHUNKING_DEFAULTS,
    DIRTY_DEFAULTS,
    GENRE_PERCENTS,
    LANGUAGE_PERCENTS,
    LENGTH_BUCKET_PERCENTS,
    TARGET_TOTAL_DOCS,
    build_sampling_targets,
    default_manifest_path,
    make_split_ratios,
)
from .datasets import iter_dataset, load_manifest
from .dirty import dirty_reason
from .sampling import (
    assign_document_ids,
    build_retrieval_sets,
    sample_language_docs,
    sample_to_targets,
    split_by_language,
)
from .types import ProcessedDocument
from .utils import count_tokens, hash_author, length_bucket

logger = logging.getLogger(__name__)

try:
    from langdetect import LangDetectException, detect

    _LANGDETECT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LangDetectException = Exception  # type: ignore[assignment]
    detect = None  # type: ignore[assignment]
    _LANGDETECT_AVAILABLE = False


def _coarse_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    if ":" in reason:
        return reason.split(":", 1)[0]
    return reason


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {k: counter[k] for k in sorted(counter)}


def _summarize_docs(docs: Iterable[ProcessedDocument]) -> dict:
    lang = Counter()
    genre = Counter()
    bucket = Counter()
    total = 0
    for doc in docs:
        total += 1
        lang[doc.lang] += 1
        genre[doc.genre] += 1
        bucket[doc.length_bucket] += 1
    return {
        "total": total,
        "by_lang": _sorted_counter(lang),
        "by_genre": _sorted_counter(genre),
        "by_length_bucket": _sorted_counter(bucket),
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


def _spacing_stats(text: str) -> tuple[float, int]:
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
        and detect is not None
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


def _normalized_language_percents(available_langs: set[str]) -> dict[str, float]:
    percents = {
        lang: pct
        for lang, pct in LANGUAGE_PERCENTS.items()
        if lang in available_langs and pct > 0
    }
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


def monitor_build_stage(
    *,
    manifest_path: Path,
    total_docs: int,
    split_ratios,
    seed: int,
    sanity_check: bool,
    sanity_limit: int | None,
    max_documents_per_dataset: int | None,
    shuffle_buffer_size: int,
    no_shuffle_datasets: set[str],
    dataset_max_docs: dict[str, int],
    allow_other_languages: bool,
    chunking_params: dict,
    truncate_to_tokens: int | None,
) -> tuple[dict, list[ProcessedDocument]]:
    stage_start = time.perf_counter()
    rng = random.Random(seed)

    configs = load_manifest(manifest_path)
    author_accumulator = AuthorAccumulator(
        min_docs=AUTHOR_DOC_LIMITS["min_docs"],
        max_docs=AUTHOR_DOC_LIMITS["max_docs"],
        rng=rng,
    )

    global_ingest = Counter()
    global_dirty = Counter()
    dataset_stats: dict[str, dict] = {}
    clean_doc_count = 0

    for cfg in configs:
        logger.info("Monitoring dataset '%s' (source=%s)", cfg.name, cfg.source)
        per_dataset = {
            "source": cfg.source,
            "raw_docs_seen": 0,
            "loader_errors": 0,
            "processing_errors": 0,
            "empty_text_dropped": 0,
            "source_docs_over_max_tokens": 0,
            "source_docs_chunked": 0,
            "source_docs_kept_over_limit": 0,
            "chunks_created": 0,
            "chunks_truncated": 0,
            "dirty_dropped": 0,
            "dirty_by_reason": Counter(),
            "clean_chunks_kept": 0,
            "capped_out": False,
        }
        dataset_stats[cfg.name] = per_dataset

        dataset_iter = iter(iter_dataset(cfg, sanity_limit if sanity_check else None))
        dataset_key = {cfg.name.lower(), cfg.source.lower()}
        if not (dataset_key & no_shuffle_datasets):
            dataset_iter = buffered_shuffle(dataset_iter, shuffle_buffer_size, rng)

        cap_override = next((dataset_max_docs[k] for k in dataset_key if k in dataset_max_docs), None)
        effective_cap = cap_override if cap_override is not None else max_documents_per_dataset
        per_dataset_processed = 0

        while True:
            try:
                raw_doc = next(dataset_iter)
            except StopIteration:
                break
            except Exception:
                per_dataset["loader_errors"] += 1
                global_ingest["loader_errors"] += 1
                continue

            per_dataset_processed += 1
            if effective_cap is not None and per_dataset_processed > effective_cap:
                per_dataset["capped_out"] = True
                break

            per_dataset["raw_docs_seen"] += 1
            global_ingest["raw_docs_seen"] += 1

            if not raw_doc.text or not str(raw_doc.text).strip():
                per_dataset["empty_text_dropped"] += 1
                global_ingest["empty_text_dropped"] += 1
                continue

            try:
                source_token_len = count_tokens(raw_doc.text)
                if source_token_len > chunking_params["max_tokens"]:
                    per_dataset["source_docs_over_max_tokens"] += 1
                    global_ingest["source_docs_over_max_tokens"] += 1

                chunks = chunk_document(
                    raw_doc,
                    max_tokens=chunking_params["max_tokens"],
                    target_chunk_tokens=chunking_params["target_chunk_tokens"],
                    min_chunk_tokens=chunking_params["min_chunk_tokens"],
                    chunk_probability=chunking_params.get("chunk_probability", 1.0),
                    rng=rng,
                )
                if len(chunks) > 1:
                    per_dataset["source_docs_chunked"] += 1
                    global_ingest["source_docs_chunked"] += 1
                elif source_token_len > chunking_params["max_tokens"]:
                    per_dataset["source_docs_kept_over_limit"] += 1
                    global_ingest["source_docs_kept_over_limit"] += 1

                per_dataset["chunks_created"] += len(chunks)
                global_ingest["chunks_created"] += len(chunks)

                for chunk in chunks:
                    if truncate_to_tokens and count_tokens(chunk.text) > truncate_to_tokens:
                        chunk = truncate_raw_document(chunk, truncate_to_tokens)
                        per_dataset["chunks_truncated"] += 1
                        global_ingest["chunks_truncated"] += 1

                    token_len = count_tokens(chunk.text)
                    reason = dirty_reason(
                        chunk.text,
                        token_len,
                        unique_token_ratio=DIRTY_DEFAULTS["unique_token_ratio"],
                        symbol_ratio=DIRTY_DEFAULTS["symbol_ratio"],
                        max_consecutive_symbols=DIRTY_DEFAULTS["max_consecutive_symbols"],
                        max_repeated_char_run=DIRTY_DEFAULTS["max_repeated_char_run"],
                        source=chunk.source,
                    )
                    if reason:
                        coarse = _coarse_reason(reason) or "unknown"
                        per_dataset["dirty_dropped"] += 1
                        per_dataset["dirty_by_reason"][coarse] += 1
                        global_ingest["dirty_dropped"] += 1
                        global_dirty[coarse] += 1
                        continue

                    author_accumulator.add(
                        ProcessedDocument(
                            raw_id=chunk.raw_id,
                            author_id=hash_author(chunk.source, chunk.author),
                            text=chunk.text,
                            lang=chunk.lang,
                            source=chunk.source,
                            genre=chunk.genre,
                            token_length=token_len,
                            length_bucket=length_bucket(token_len),
                            metadata=chunk.metadata,
                        )
                    )
                    per_dataset["clean_chunks_kept"] += 1
                    global_ingest["clean_chunks_kept"] += 1
                    clean_doc_count += 1
            except Exception:
                per_dataset["processing_errors"] += 1
                global_ingest["processing_errors"] += 1
                continue

    selected_docs, underfull_docs, dropped_authors = author_accumulator.finalize()
    authors_seen = len(author_accumulator._buckets)

    selected_docs = list(selected_docs)
    fallback_authors_added: list[str] = []
    fallback_docs_added = 0
    if AUTHOR_DOC_LIMITS["fallback_min_docs"] < AUTHOR_DOC_LIMITS["min_docs"]:
        fallback_group = defaultdict(list)
        for doc in underfull_docs:
            fallback_group[doc.author_id].append(doc)
        for author_id, docs in fallback_group.items():
            if len(docs) >= AUTHOR_DOC_LIMITS["fallback_min_docs"]:
                docs = sorted(docs, key=lambda d: (d.lang, d.source, d.raw_id))
                docs = docs[: AUTHOR_DOC_LIMITS["max_docs"]]
                selected_docs.extend(docs)
                fallback_authors_added.append(author_id)
                fallback_docs_added += len(docs)

    before_sampling_count = len(selected_docs)
    targets = build_sampling_targets(total_docs=total_docs)
    sampled_docs, sampling_log = sample_to_targets(
        selected_docs, targets, rng, allow_other_languages=allow_other_languages
    )

    final_docs = assign_document_ids(sampled_docs)
    splits = split_by_language(final_docs, split_ratios, rng)

    split_summary: dict[str, dict] = {}
    stage1_docs_for_stage2: list[ProcessedDocument] = []
    for split_name, docs in splits.items():
        candidates, queries, ground_truth = build_retrieval_sets(docs, rng)
        split_summary[split_name] = {
            "documents": len(docs),
            "candidates": len(candidates),
            "queries": len(queries),
            "ground_truth": len(ground_truth),
            "documents_by_lang": _sorted_counter(Counter(doc.lang for doc in docs)),
            "candidates_by_lang": _sorted_counter(Counter(row["lang"] for row in candidates)),
            "queries_by_lang": _sorted_counter(Counter(row["lang"] for row in queries)),
        }
        # Stage-2 now consumes full stage-1 split docs rather than candidate-only rows.
        stage1_docs_for_stage2.extend(replace(doc) for doc in docs)

    for cfg_name in dataset_stats:
        dataset_stats[cfg_name]["dirty_by_reason"] = _sorted_counter(
            dataset_stats[cfg_name]["dirty_by_reason"]
        )

    build_summary = {
        "runtime_seconds": round(time.perf_counter() - stage_start, 3),
        "inputs": {
            "seed": seed,
            "manifest": str(manifest_path),
            "dataset_count": len(configs),
            "total_docs_target": total_docs,
            "allow_other_languages": allow_other_languages,
            "sanity_check": sanity_check,
            "sanity_limit": sanity_limit,
            "max_documents_per_dataset": max_documents_per_dataset,
            "dataset_max_docs": dict(sorted(dataset_max_docs.items())),
            "shuffle_buffer_size": shuffle_buffer_size,
            "no_shuffle_datasets": sorted(no_shuffle_datasets),
            "chunking": {
                "max_tokens": chunking_params["max_tokens"],
                "target_chunk_tokens": chunking_params["target_chunk_tokens"],
                "min_chunk_tokens": chunking_params["min_chunk_tokens"],
                "chunk_probability": chunking_params.get("chunk_probability", 1.0),
            },
            "truncate_to_tokens": truncate_to_tokens,
        },
        "per_dataset": dataset_stats,
        "ingestion_and_filtering": {
            "raw_docs_seen": global_ingest["raw_docs_seen"],
            "loader_errors": global_ingest["loader_errors"],
            "processing_errors": global_ingest["processing_errors"],
            "empty_text_dropped": global_ingest["empty_text_dropped"],
            "source_docs_over_max_tokens": global_ingest["source_docs_over_max_tokens"],
            "source_docs_chunked": global_ingest["source_docs_chunked"],
            "source_docs_kept_over_limit": global_ingest["source_docs_kept_over_limit"],
            "chunks_created": global_ingest["chunks_created"],
            "chunks_truncated": global_ingest["chunks_truncated"],
            "dirty_dropped": global_ingest["dirty_dropped"],
            "dirty_dropped_by_reason": _sorted_counter(global_dirty),
            "clean_docs_before_author_filter": clean_doc_count,
        },
        "author_filtering": {
            "authors_seen": authors_seen,
            "authors_dropped_min_docs": len(dropped_authors),
            "underfull_docs_pool": len(underfull_docs),
            "fallback_authors_added": len(fallback_authors_added),
            "fallback_docs_added": fallback_docs_added,
            "docs_after_author_filter": len(selected_docs),
            "docs_removed_by_author_filter": clean_doc_count - len(selected_docs),
        },
        "sampling": {
            "docs_before_sampling": before_sampling_count,
            "requested_total": targets.total_docs,
            "sampled_total": len(final_docs),
            "docs_removed_by_sampling": before_sampling_count - len(final_docs),
            "requested_by_lang": dict(sorted(targets.language_targets.items())),
            "available_by_lang": _sorted_counter(Counter(doc.lang for doc in selected_docs)),
            "sampled_by_lang": _sorted_counter(Counter(doc.lang for doc in final_docs)),
            "sampling_log_entries": len(sampling_log),
            "sampling_log": sampling_log,
        },
        "split_and_retrieval": split_summary,
        "stage1_documents_for_stage2": _summarize_docs(stage1_docs_for_stage2),
        "stage1_candidates_for_postprocess": _summarize_docs(stage1_docs_for_stage2),
    }
    return build_summary, stage1_docs_for_stage2


def monitor_postprocess_stage(
    *,
    stage1_docs: list[ProcessedDocument],
    seed: int,
    split_ratios,
    target_total: int | None,
    spacing_collapse_ratio: float,
    min_spacing_run: int,
    max_single_letter_ratio: float,
    max_single_letter_run: int,
    min_alpha_ratio: float,
    min_alpha_token_ratio: float,
    skip_langdetect: bool,
) -> dict:
    stage_start = time.perf_counter()
    rng = random.Random(seed)

    input_docs = [replace(doc) for doc in stage1_docs]
    filtered_docs: list[ProcessedDocument] = []
    drop_reasons: Counter = Counter()
    drop_reasons_by_lang: Counter = Counter()
    spacing_collapsed = 0

    for doc in input_docs:
        text, collapsed, single_ratio, max_run = normalize_spaced_text(
            doc.text,
            collapse_ratio_threshold=spacing_collapse_ratio,
            min_run=min_spacing_run,
        )
        if collapsed:
            spacing_collapsed += 1
        doc.text = text
        doc.token_length = count_tokens(doc.text)
        doc.length_bucket = length_bucket(doc.token_length)

        if single_ratio > max_single_letter_ratio or max_run > max_single_letter_run:
            reason = "excessive_spaced_letters"
            drop_reasons[reason] += 1
            drop_reasons_by_lang[f"{doc.lang}:{reason}"] += 1
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
            reason = f"dirty:{_coarse_reason(dirty)}"
            drop_reasons[reason] += 1
            drop_reasons_by_lang[f"{doc.lang}:{reason}"] += 1
            continue

        if looks_untranslatable(
            doc.text,
            doc.lang,
            min_alpha_ratio=min_alpha_ratio,
            min_alpha_token_ratio=min_alpha_token_ratio,
            max_single_letter_ratio=max_single_letter_ratio,
            max_single_letter_run=max_single_letter_run,
            use_langdetect=not skip_langdetect,
        ):
            reason = "untranslatable"
            drop_reasons[reason] += 1
            drop_reasons_by_lang[f"{doc.lang}:{reason}"] += 1
            continue

        filtered_docs.append(doc)

    if not filtered_docs:
        return {
            "runtime_seconds": round(time.perf_counter() - stage_start, 3),
            "inputs": {
                "seed": seed,
                "input_stage1_docs": len(input_docs),
                "input_candidates": len(input_docs),
                "target_total": target_total,
                "skip_langdetect": skip_langdetect,
                "spacing_collapse_ratio": spacing_collapse_ratio,
                "min_spacing_run": min_spacing_run,
                "max_single_letter_ratio": max_single_letter_ratio,
                "max_single_letter_run": max_single_letter_run,
                "min_alpha_ratio": min_alpha_ratio,
                "min_alpha_token_ratio": min_alpha_token_ratio,
            },
            "before_filter": _summarize_docs(input_docs),
            "filtering": {
                "spacing_collapsed_docs": spacing_collapsed,
                "dropped_total": sum(drop_reasons.values()),
                "drop_reasons": _sorted_counter(drop_reasons),
                "drop_reasons_by_lang": _sorted_counter(drop_reasons_by_lang),
                "kept_after_filter": 0,
            },
            "after_filter": _summarize_docs([]),
            "language_targets": {
                "available_total": 0,
                "requested_total": 0,
                "final_total": 0,
                "per_language": {},
            },
            "sampling": {
                "kept_before_sampling": 0,
                "selected_total": 0,
                "dropped_by_sampling": 0,
                "selected_by_lang": {},
                "sampling_log_entries": 0,
                "sampling_log": [],
            },
            "after_sampling": _summarize_docs([]),
            "split_and_retrieval": {},
            "warning": "No documents remain after postprocess filtering.",
        }

    total_target = target_total or len(filtered_docs)
    docs_by_lang: dict[str, list[ProcessedDocument]] = defaultdict(list)
    for doc in filtered_docs:
        docs_by_lang[doc.lang].append(doc)
    lang_targets, lang_log = compute_language_targets(docs_by_lang, total_target)

    selected_docs, sampling_log = sample_documents(filtered_docs, lang_targets, rng)
    splits = split_by_language(selected_docs, split_ratios, rng)

    split_summary: dict[str, dict] = {}
    for split_name, docs in splits.items():
        candidates, queries, ground_truth = build_retrieval_sets(docs, rng)
        split_summary[split_name] = {
            "documents": len(docs),
            "candidates": len(candidates),
            "queries": len(queries),
            "ground_truth": len(ground_truth),
            "documents_by_lang": _sorted_counter(Counter(doc.lang for doc in docs)),
            "candidates_by_lang": _sorted_counter(Counter(row["lang"] for row in candidates)),
            "queries_by_lang": _sorted_counter(Counter(row["lang"] for row in queries)),
        }

    return {
        "runtime_seconds": round(time.perf_counter() - stage_start, 3),
        "inputs": {
            "seed": seed,
            "input_stage1_docs": len(input_docs),
            "input_candidates": len(input_docs),
            "target_total": target_total,
            "skip_langdetect": skip_langdetect,
            "spacing_collapse_ratio": spacing_collapse_ratio,
            "min_spacing_run": min_spacing_run,
            "max_single_letter_ratio": max_single_letter_ratio,
            "max_single_letter_run": max_single_letter_run,
            "min_alpha_ratio": min_alpha_ratio,
            "min_alpha_token_ratio": min_alpha_token_ratio,
        },
        "before_filter": _summarize_docs(input_docs),
        "filtering": {
            "spacing_collapsed_docs": spacing_collapsed,
            "dropped_total": sum(drop_reasons.values()),
            "drop_reasons": _sorted_counter(drop_reasons),
            "drop_reasons_by_lang": _sorted_counter(drop_reasons_by_lang),
            "kept_after_filter": len(filtered_docs),
        },
        "after_filter": _summarize_docs(filtered_docs),
        "language_targets": lang_log,
        "sampling": {
            "kept_before_sampling": len(filtered_docs),
            "selected_total": len(selected_docs),
            "dropped_by_sampling": len(filtered_docs) - len(selected_docs),
            "selected_by_lang": _sorted_counter(Counter(doc.lang for doc in selected_docs)),
            "sampling_log_entries": len(sampling_log),
            "sampling_log": sampling_log,
        },
        "after_sampling": _summarize_docs(selected_docs),
        "split_and_retrieval": split_summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the AuthBench processing pipeline with the same seed and emit "
            "stage-by-stage monitoring statistics without rewriting final benchmark files."
        )
    )

    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-docs", type=int, default=TARGET_TOTAL_DOCS)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--post-train-ratio", type=float, default=None)
    parser.add_argument("--post-dev-ratio", type=float, default=None)
    parser.add_argument("--post-test-ratio", type=float, default=None)

    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--sanity-limit", type=int, default=2000)
    parser.add_argument("--max-documents-per-dataset", type=int, default=None)
    parser.add_argument("--dataset-max-docs", nargs="*", default=[])
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--no-shuffle-datasets", nargs="*", default=[])
    parser.add_argument("--allow-other-languages", action="store_true")

    parser.add_argument(
        "--max-chunk-tokens", type=int, default=CHUNKING_DEFAULTS["max_tokens"]
    )
    parser.add_argument(
        "--target-chunk-tokens", type=int, default=CHUNKING_DEFAULTS["target_chunk_tokens"]
    )
    parser.add_argument(
        "--min-chunk-tokens", type=int, default=CHUNKING_DEFAULTS["min_chunk_tokens"]
    )
    parser.add_argument(
        "--chunk-probability",
        type=float,
        default=CHUNKING_DEFAULTS.get("chunk_probability", 1.0),
    )
    parser.add_argument("--truncate-to-tokens", type=int, default=None)

    parser.add_argument("--post-target-total", type=int, default=None)
    parser.add_argument("--post-spacing-collapse-ratio", type=float, default=0.25)
    parser.add_argument("--post-min-spacing-run", type=int, default=2)
    parser.add_argument("--post-max-single-letter-ratio", type=float, default=0.45)
    parser.add_argument("--post-max-single-letter-run", type=int, default=10)
    parser.add_argument("--post-min-alpha-ratio", type=float, default=0.25)
    parser.add_argument("--post-min-alpha-token-ratio", type=float, default=0.5)
    parser.add_argument("--post-skip-langdetect", action="store_true")

    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "Where to write the monitoring JSON (default: "
            "processing/outputs/monitoring/pipeline_dynamics_seed<seed>.json)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite report file if it already exists.",
    )
    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    build_split_ratios = make_split_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)
    post_split_ratios = make_split_ratios(
        args.post_train_ratio if args.post_train_ratio is not None else args.train_ratio,
        args.post_dev_ratio if args.post_dev_ratio is not None else args.dev_ratio,
        args.post_test_ratio if args.post_test_ratio is not None else args.test_ratio,
    )

    report_path = args.report_path or (
        Path(__file__).parent
        / "outputs"
        / "monitoring"
        / f"pipeline_dynamics_seed{args.seed}.json"
    )
    report_path = report_path.resolve()
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Report already exists: {report_path}. Pass --overwrite or choose --report-path."
        )

    dataset_max_docs = _parse_dataset_caps(args.dataset_max_docs)
    no_shuffle_datasets = set(map(str.lower, args.no_shuffle_datasets or []))
    chunking_params = {
        "max_tokens": args.max_chunk_tokens,
        "target_chunk_tokens": args.target_chunk_tokens,
        "min_chunk_tokens": args.min_chunk_tokens,
        "chunk_probability": args.chunk_probability,
    }

    overall_start = time.perf_counter()
    build_summary, stage1_docs = monitor_build_stage(
        manifest_path=args.manifest,
        total_docs=args.total_docs,
        split_ratios=build_split_ratios,
        seed=args.seed,
        sanity_check=args.sanity_check,
        sanity_limit=args.sanity_limit,
        max_documents_per_dataset=args.max_documents_per_dataset,
        shuffle_buffer_size=args.shuffle_buffer_size,
        no_shuffle_datasets=no_shuffle_datasets,
        dataset_max_docs=dataset_max_docs,
        allow_other_languages=args.allow_other_languages,
        chunking_params=chunking_params,
        truncate_to_tokens=args.truncate_to_tokens,
    )

    post_summary = monitor_postprocess_stage(
        stage1_docs=stage1_docs,
        seed=args.seed,
        split_ratios=post_split_ratios,
        target_total=args.post_target_total,
        spacing_collapse_ratio=args.post_spacing_collapse_ratio,
        min_spacing_run=args.post_min_spacing_run,
        max_single_letter_ratio=args.post_max_single_letter_ratio,
        max_single_letter_run=args.post_max_single_letter_run,
        min_alpha_ratio=args.post_min_alpha_ratio,
        min_alpha_token_ratio=args.post_min_alpha_token_ratio,
        skip_langdetect=args.post_skip_langdetect,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "report_runtime_seconds": round(time.perf_counter() - overall_start, 3),
        "build_stage": build_summary,
        "postprocess_stage": post_summary,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote monitoring report to %s", report_path)


if __name__ == "__main__":
    main()
