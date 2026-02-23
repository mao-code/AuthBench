from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .types import ProcessedDocument

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class DedupConfig:
    exact_text: bool = True
    near_text: bool = True
    near_similarity_threshold: float = 0.92
    near_lsh_bands: int = 4
    min_tokens_for_near: int = 20
    near_same_language_only: bool = True
    author_similarity: bool = True
    author_similarity_threshold: float = 0.94
    author_cross_source_only: bool = True
    author_same_language_only: bool = True
    author_profile_docs: int = 3
    max_bucket_size: int = 512


def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _simhash64(text: str) -> int:
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return 0

    # Token unigrams and bigrams provide a lightweight lexical+context signal.
    features = list(tokens)
    features.extend(f"{tokens[i]}::{tokens[i + 1]}" for i in range(len(tokens) - 1))
    weighted = Counter(features)

    vector = [0] * 64
    for feature, weight in weighted.items():
        h = _hash64(feature)
        for bit in range(64):
            if h & (1 << bit):
                vector[bit] += weight
            else:
                vector[bit] -= weight

    out = 0
    for bit, val in enumerate(vector):
        if val >= 0:
            out |= 1 << bit
    return out


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _lsh_keys(simhash: int, bands: int) -> list[tuple[int, int]]:
    if bands <= 0 or 64 % bands != 0:
        raise ValueError(f"near_lsh_bands must be a positive divisor of 64; got {bands}.")
    width = 64 // bands
    mask = (1 << width) - 1
    return [(band, (simhash >> (band * width)) & mask) for band in range(bands)]


def _doc_sort_key(doc: ProcessedDocument) -> tuple[int, str, str, str, str]:
    return (
        -int(doc.token_length),
        doc.lang,
        doc.source,
        doc.author_id,
        doc.raw_id,
    )


def _author_signature_source(author_docs: list[ProcessedDocument]) -> str:
    by_source = Counter(doc.source for doc in author_docs)
    if not by_source:
        return ""
    return sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def deduplicate_documents(
    docs: Iterable[ProcessedDocument],
    *,
    config: DedupConfig | None = None,
) -> tuple[list[ProcessedDocument], dict]:
    cfg = config or DedupConfig()
    if not 0 < cfg.near_similarity_threshold <= 1:
        raise ValueError(
            f"near_similarity_threshold must be in (0, 1], got {cfg.near_similarity_threshold}."
        )
    if not 0 < cfg.author_similarity_threshold <= 1:
        raise ValueError(
            f"author_similarity_threshold must be in (0, 1], got {cfg.author_similarity_threshold}."
        )

    ordered = sorted(list(docs), key=_doc_sort_key)
    total_input = len(ordered)
    drop_reasons = Counter()

    exact_seen: dict[str, int] = {}
    near_index: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    near_fingerprints: list[int] = []
    near_langs: list[str] = []
    kept_after_near: list[ProcessedDocument] = []

    near_hamming_max = int(math.floor((1.0 - cfg.near_similarity_threshold) * 64.0))
    near_candidates_checked = 0
    near_pairs_matched = 0

    for doc in ordered:
        normalized = _normalize_text(doc.text)
        if not normalized:
            drop_reasons["empty_after_normalization"] += 1
            continue

        if cfg.exact_text:
            exact_key = hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()
            if exact_key in exact_seen:
                drop_reasons["exact_text_duplicate"] += 1
                continue
            exact_seen[exact_key] = 1

        simhash = 0
        is_near_dup = False
        if cfg.near_text and doc.token_length >= cfg.min_tokens_for_near:
            simhash = _simhash64(normalized)
            candidate_indices: set[int] = set()
            lang_key = doc.lang if cfg.near_same_language_only else "*"
            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                bucket = near_index.get((lsh_key[0], lsh_key[1], lang_key))
                if bucket:
                    candidate_indices.update(bucket)

            for idx in sorted(candidate_indices):
                near_candidates_checked += 1
                if cfg.near_same_language_only and near_langs[idx] != doc.lang:
                    continue
                dist = _hamming_distance(simhash, near_fingerprints[idx])
                if dist <= near_hamming_max:
                    is_near_dup = True
                    near_pairs_matched += 1
                    break

        if is_near_dup:
            drop_reasons["near_text_duplicate"] += 1
            continue

        kept_idx = len(kept_after_near)
        kept_after_near.append(doc)
        near_fingerprints.append(simhash)
        near_langs.append(doc.lang)

        if cfg.near_text and doc.token_length >= cfg.min_tokens_for_near:
            lang_key = doc.lang if cfg.near_same_language_only else "*"
            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                key = (lsh_key[0], lsh_key[1], lang_key)
                bucket = near_index[key]
                if len(bucket) < cfg.max_bucket_size:
                    bucket.append(kept_idx)

    kept_after_author = list(kept_after_near)
    author_pairs_matched = 0
    dropped_authors: set[str] = set()

    if cfg.author_similarity:
        docs_by_author: dict[str, list[ProcessedDocument]] = defaultdict(list)
        for doc in kept_after_near:
            docs_by_author[doc.author_id].append(doc)

        profile_items: list[tuple[str, str, str, int, int, int]] = []
        for author_id, author_docs in docs_by_author.items():
            ordered_docs = sorted(author_docs, key=_doc_sort_key)
            profile_docs = ordered_docs[: max(1, cfg.author_profile_docs)]
            normalized_parts = [_normalize_text(d.text) for d in profile_docs]
            normalized_parts = [p for p in normalized_parts if p]
            if not normalized_parts:
                continue
            profile_text = " ".join(normalized_parts)
            profile_hash = _simhash64(profile_text)
            lang = profile_docs[0].lang
            source = _author_signature_source(author_docs)
            total_tokens = sum(d.token_length for d in author_docs)
            profile_items.append(
                (
                    author_id,
                    lang,
                    source,
                    len(author_docs),
                    total_tokens,
                    profile_hash,
                )
            )

        profile_items.sort(key=lambda x: (-x[3], -x[4], x[1], x[2], x[0]))
        author_hamming_max = int(math.floor((1.0 - cfg.author_similarity_threshold) * 64.0))
        author_index: dict[tuple[int, int, str], list[int]] = defaultdict(list)

        for idx, (author_id, lang, source, doc_count, total_tokens, simhash) in enumerate(profile_items):
            if author_id in dropped_authors:
                continue

            lang_key = lang if cfg.author_same_language_only else "*"
            candidate_indices: set[int] = set()
            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                bucket = author_index.get((lsh_key[0], lsh_key[1], lang_key))
                if bucket:
                    candidate_indices.update(bucket)

            drop_current = False
            for cand_idx in sorted(candidate_indices):
                (
                    cand_author_id,
                    cand_lang,
                    cand_source,
                    cand_doc_count,
                    cand_total_tokens,
                    cand_simhash,
                ) = profile_items[cand_idx]
                if cand_author_id in dropped_authors:
                    continue
                if cfg.author_same_language_only and cand_lang != lang:
                    continue
                if cfg.author_cross_source_only and cand_source == source:
                    continue

                dist = _hamming_distance(simhash, cand_simhash)
                if dist > author_hamming_max:
                    continue

                author_pairs_matched += 1
                # Keep the stronger author profile to reduce false positives.
                keep_current = (
                    (doc_count > cand_doc_count)
                    or (doc_count == cand_doc_count and total_tokens > cand_total_tokens)
                    or (
                        doc_count == cand_doc_count
                        and total_tokens == cand_total_tokens
                        and author_id < cand_author_id
                    )
                )
                if keep_current:
                    dropped_authors.add(cand_author_id)
                else:
                    dropped_authors.add(author_id)
                    drop_current = True
                    break

            if drop_current:
                continue

            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                key = (lsh_key[0], lsh_key[1], lang_key)
                bucket = author_index[key]
                if len(bucket) < cfg.max_bucket_size:
                    bucket.append(idx)

        if dropped_authors:
            kept_after_author = [doc for doc in kept_after_near if doc.author_id not in dropped_authors]
            drop_reasons["near_author_duplicate"] += len(kept_after_near) - len(kept_after_author)

    summary = {
        "input_docs": total_input,
        "after_exact_and_near_doc_dedup": len(kept_after_near),
        "after_author_dedup": len(kept_after_author),
        "dropped_total": total_input - len(kept_after_author),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "authors_before": len({doc.author_id for doc in kept_after_near}),
        "authors_after": len({doc.author_id for doc in kept_after_author}),
        "authors_dropped": len(dropped_authors),
        "near_similarity_threshold": cfg.near_similarity_threshold,
        "near_hamming_max": near_hamming_max,
        "near_candidates_checked": near_candidates_checked,
        "near_pairs_matched": near_pairs_matched,
        "author_similarity_threshold": cfg.author_similarity_threshold,
        "author_pairs_matched": author_pairs_matched,
        "config": {
            "exact_text": cfg.exact_text,
            "near_text": cfg.near_text,
            "near_lsh_bands": cfg.near_lsh_bands,
            "min_tokens_for_near": cfg.min_tokens_for_near,
            "near_same_language_only": cfg.near_same_language_only,
            "author_similarity": cfg.author_similarity,
            "author_cross_source_only": cfg.author_cross_source_only,
            "author_same_language_only": cfg.author_same_language_only,
            "author_profile_docs": cfg.author_profile_docs,
            "max_bucket_size": cfg.max_bucket_size,
        },
    }
    return kept_after_author, summary
