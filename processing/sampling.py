from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Iterable

from .types import ProcessedDocument, SamplingTargets, SplitRatios
from .utils import deterministic_shuffle

logger = logging.getLogger(__name__)


def enforce_author_limits(
    docs: Iterable[ProcessedDocument],
    *,
    min_docs: int,
    max_docs: int,
    rng,
):
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.author_id].append(doc)

    selected: list[ProcessedDocument] = []
    discarded_authors: list[str] = []
    underfull_docs: list[ProcessedDocument] = []

    for author_id, author_docs in grouped.items():
        ordered = sorted(author_docs, key=lambda d: (d.lang, d.source, d.raw_id))
        if len(ordered) < min_docs:
            underfull_docs.extend(ordered)
            discarded_authors.append(author_id)
            continue
        if len(ordered) > max_docs:
            ordered = deterministic_shuffle(ordered, rng)[:max_docs]
        selected.extend(ordered)

    return selected, underfull_docs, discarded_authors


def _bucket_targets(total: int, bucket_percents: dict[str, float]) -> dict[str, int]:
    targets = {bucket: int(round(total * pct)) for bucket, pct in bucket_percents.items()}
    diff = total - sum(targets.values())
    if diff and targets:
        first_key = next(iter(targets))
        targets[first_key] += diff
    return targets


def _genre_key(genre: str, available_targets: dict[str, int]) -> str:
    for target in available_targets:
        if genre == target or genre.startswith(f"{target}/"):
            return target
    return genre


def _sample_bucketed(
    docs: list[ProcessedDocument],
    target: int,
    length_bucket_percents: dict[str, float],
    rng,
):
    buckets = defaultdict(list)
    for doc in docs:
        buckets[doc.length_bucket].append(doc)

    for bucket_docs in buckets.values():
        bucket_docs.sort(key=lambda d: (d.source, d.author_id, d.raw_id))

    bucket_targets = _bucket_targets(target, length_bucket_percents)
    selected: list[ProcessedDocument] = []
    leftovers: list[ProcessedDocument] = []
    bucket_deficits: dict[str, int] = {}

    for bucket, docs_in_bucket in buckets.items():
        ordered = deterministic_shuffle(docs_in_bucket, rng)
        desired = bucket_targets.get(bucket, 0)
        take = min(len(ordered), desired)
        selected.extend(ordered[:take])
        leftovers.extend(ordered[take:])
        if take < desired:
            bucket_deficits[bucket] = desired - take

    remaining = target - len(selected)
    if remaining > 0:
        fill_pool = deterministic_shuffle(leftovers, rng)
        selected.extend(fill_pool[:remaining])

    return selected[:target], bucket_deficits


def sample_language_docs(
    lang: str,
    docs: list[ProcessedDocument],
    lang_target: int,
    genre_percent_map: dict[str, float] | None,
    length_bucket_percents: dict[str, float],
    rng,
):
    genre_percent_map = genre_percent_map or {}
    genre_targets = {
        genre: int(round(lang_target * pct)) for genre, pct in genre_percent_map.items()
    }
    if genre_targets:
        diff = lang_target - sum(genre_targets.values())
        if diff:
            first_key = next(iter(genre_targets))
            genre_targets[first_key] += diff
    else:
        genre_targets = {"all": lang_target}

    grouped = defaultdict(list)
    untargeted: list[ProcessedDocument] = []
    for doc in docs:
        key = _genre_key(doc.genre, genre_targets) if genre_targets else "all"
        if key in genre_targets:
            grouped[key].append(doc)
        else:
            untargeted.append(doc)

    selected: list[ProcessedDocument] = []
    deficits: list[dict] = []
    leftovers: list[ProcessedDocument] = []

    for genre, target in genre_targets.items():
        genre_docs = grouped.get(genre, [])
        sampled, bucket_deficits = _sample_bucketed(genre_docs, target, length_bucket_percents, rng)
        selected.extend(sampled)
        # preserve any unused docs for redistribution
        used_ids = {doc.raw_id for doc in sampled}
        leftovers.extend([doc for doc in genre_docs if doc.raw_id not in used_ids])

        if len(sampled) < target:
            deficits.append(
                {
                    "lang": lang,
                    "genre": genre,
                    "needed": target,
                    "selected": len(sampled),
                    "bucket_deficits": bucket_deficits,
                }
            )

    remaining = lang_target - len(selected)
    if remaining > 0:
        spill_pool = leftovers + untargeted
        spill_pool = deterministic_shuffle(spill_pool, rng)
        selected.extend(spill_pool[:remaining])
        if len(spill_pool) < remaining:
            deficits.append(
                {
                    "lang": lang,
                    "genre": "spill",
                    "needed": remaining,
                    "selected": len(spill_pool),
                    "bucket_deficits": {},
                }
            )

    return selected[:lang_target], deficits


def sample_to_targets(
    docs: Iterable[ProcessedDocument],
    targets: SamplingTargets,
    rng,
    *,
    allow_other_languages: bool = True,
):
    docs_by_lang = defaultdict(list)
    for doc in docs:
        docs_by_lang[doc.lang].append(doc)

    selected: list[ProcessedDocument] = []
    sampling_log: list[dict] = []

    targeted_langs = set(targets.language_targets.keys())
    for lang in targeted_langs:
        lang_docs = docs_by_lang.get(lang, [])
        requested_target = targets.language_targets[lang]
        lang_target = min(requested_target, len(lang_docs))
        if lang_target < requested_target:
            sampling_log.append(
                {
                    "lang": lang,
                    "type": "language_shortfall",
                    "requested": requested_target,
                    "available": len(lang_docs),
                }
            )
        lang_docs.sort(key=lambda d: (d.source, d.author_id, d.raw_id))
        sampled, deficits = sample_language_docs(
            lang,
            lang_docs,
            lang_target,
            targets.genre_percents.get(lang),
            targets.length_bucket_percents,
            rng,
        )
        selected.extend(sampled)
        sampling_log.extend(deficits)

    selected_count = len(selected)
    remaining_budget = max(targets.total_docs - selected_count, 0)

    if allow_other_languages and remaining_budget > 0:
        other_docs: list[ProcessedDocument] = []
        for lang, lang_docs in docs_by_lang.items():
            if lang in targeted_langs:
                continue
            lang_docs.sort(key=lambda d: (d.source, d.author_id, d.raw_id))
            other_docs.extend(lang_docs)
        other_docs = deterministic_shuffle(other_docs, rng)
        selected.extend(other_docs[:remaining_budget])

    return selected, sampling_log


def assign_document_ids(
    docs: Iterable[ProcessedDocument], prefix: str = "doc"
) -> list[ProcessedDocument]:
    ordered = sorted(docs, key=lambda d: (d.lang, d.source, d.author_id, d.raw_id))
    for idx, doc in enumerate(ordered):
        doc.doc_id = f"{prefix}_{idx:06d}"
    return ordered


def _split_counts(total: int, split_order: list[tuple[str, float]]) -> dict[str, int]:
    """
    Allocate integer counts across splits using fractional remainder tie-breaking.
    """

    counts: dict[str, int] = {}
    remainders: list[tuple[str, float, int]] = []

    for idx, (name, ratio) in enumerate(split_order):
        raw = total * ratio
        base = int(raw)
        counts[name] = base
        remainders.append((name, raw - base, idx))

    assigned = sum(counts.values())
    leftover = total - assigned
    if leftover > 0:
        remainders.sort(key=lambda x: (-x[1], x[2]))
        for i in range(leftover):
            name, _, _ = remainders[i % len(remainders)]
            counts[name] += 1

    return counts


def split_by_language(
    docs: Iterable[ProcessedDocument],
    split_ratios: SplitRatios,
    rng: random.Random | None = None,
):
    """
    Stratified split by language while preserving genre and length-bucket mix.

    Each language is partitioned independently. Within a language, documents are
    bucketed by (genre, length_bucket), shuffled deterministically, and divided
    according to the requested split ratios so that dev/test are not biased
    toward late-sorting sources.
    """

    rng = rng or random.Random(13)
    split_order = split_ratios.as_list()
    splits = {name: [] for name, _ in split_order}

    docs_by_lang = defaultdict(list)
    for doc in docs:
        docs_by_lang[doc.lang].append(doc)

    for lang, lang_docs in sorted(docs_by_lang.items()):
        buckets: dict[tuple[str, str], list[ProcessedDocument]] = defaultdict(list)
        for doc in lang_docs:
            buckets[(doc.genre, doc.length_bucket)].append(doc)

        for bucket_key in sorted(buckets):
            bucket_docs = buckets[bucket_key]
            shuffled = deterministic_shuffle(bucket_docs, rng)
            counts = _split_counts(len(shuffled), split_order)
            cursor = 0
            for split_name, _ in split_order:
                take = counts.get(split_name, 0)
                if take > 0:
                    splits[split_name].extend(shuffled[cursor : cursor + take])
                    cursor += take

    return splits


def split_by_author_language(
    docs: Iterable[ProcessedDocument],
    split_ratios: SplitRatios,
    rng: random.Random | None = None,
):
    """
    Keep each author in a single split while approximately preserving
    per-language document ratios across train/dev/test.
    """

    rng = rng or random.Random(13)
    split_order = split_ratios.as_list()
    split_names = [name for name, _ in split_order]
    split_index = {name: idx for idx, (name, _) in enumerate(split_order)}
    splits = {name: [] for name in split_names}

    docs_by_author = defaultdict(list)
    for doc in docs:
        docs_by_author[doc.author_id].append(doc)

    authors_by_lang: dict[str, list[tuple[str, list[ProcessedDocument]]]] = defaultdict(list)
    for author_id, author_docs in docs_by_author.items():
        lang_counts = Counter(doc.lang for doc in author_docs)
        dominant_lang = sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        ordered_docs = sorted(author_docs, key=lambda d: (d.lang, d.source, d.raw_id))
        authors_by_lang[dominant_lang].append((author_id, ordered_docs))

    for lang, author_groups in sorted(authors_by_lang.items()):
        author_groups = deterministic_shuffle(author_groups, rng)
        author_groups.sort(key=lambda item: (-len(item[1]), item[0]))

        total_docs = sum(len(group_docs) for _, group_docs in author_groups)
        target_counts = _split_counts(total_docs, split_order)
        current_counts = {name: 0 for name in split_names}

        for author_id, author_docs in author_groups:
            del author_id
            chosen_split = max(
                split_names,
                key=lambda name: (
                    target_counts[name] - current_counts[name],
                    -current_counts[name],
                    -split_index[name],
                ),
            )
            splits[chosen_split].extend(author_docs)
            current_counts[chosen_split] += len(author_docs)

        logger.info(
            "Author-aware split for lang=%s total_docs=%d targets=%s realized=%s",
            lang,
            total_docs,
            target_counts,
            current_counts,
        )

    return splits


def build_retrieval_sets(split_docs: list[ProcessedDocument], rng):
    """
    Build retrieval sets where each author contributes 1–2 candidate docs and 1–2 query docs.
    Authors with fewer than 2 docs are skipped because they cannot form a positive pair.
    """
    candidates = []
    queries = []
    ground_truth = []

    docs_by_author = defaultdict(list)
    for doc in split_docs:
        docs_by_author[doc.author_id].append(doc)

    for author_id, author_docs in docs_by_author.items():
        if len(author_docs) < 2:
            continue
        ordered = sorted(author_docs, key=lambda d: (d.lang, d.source, d.raw_id))
        ordered = deterministic_shuffle(ordered, rng)

        candidate_pool = ordered[: min(2, len(ordered))]
        remaining = ordered[len(candidate_pool) :]
        query_pool = remaining[: min(2, len(remaining))]

        if not query_pool:
            if len(candidate_pool) >= 2:
                query_pool = [candidate_pool.pop()]
            else:
                continue

        if not candidate_pool:
            continue

        candidate_ids: list[str] = []
        for doc in candidate_pool:
            candidates.append(
                {
                    "candidate_id": doc.doc_id,
                    "author_id": doc.author_id,
                    "lang": doc.lang,
                    "genre": doc.genre,
                    "content": doc.text,
                    "source": doc.source,
                    "token_length": doc.token_length,
                }
            )
            candidate_ids.append(doc.doc_id)

        for query_doc in query_pool:
            queries.append(
                {
                    "query_id": query_doc.doc_id,
                    "lang": query_doc.lang,
                    "genre": query_doc.genre,
                    "content": query_doc.text,
                    "source": query_doc.source,
                    "token_length": query_doc.token_length,
                }
            )
            ground_truth.append(
                {
                    "query_id": query_doc.doc_id,
                    "positive_ids": candidate_ids,
                    "author_id": author_id,
                }
            )

    return candidates, queries, ground_truth
