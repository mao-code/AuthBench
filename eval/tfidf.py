from __future__ import annotations

from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from collections import defaultdict
import random

import numpy as np
import torch

from AuthBench.eval.data import AuthBenchSplit
from AuthBench.eval.metrics import (
    aggregate_ranking_metrics,
    compute_eer,
    compute_roc_auc,
    ranking_metrics_for_query,
)
from AuthBench.eval.pools import build_topic_candidate_index, build_topic_pool


@dataclass
class TfidfIndex:
    vectorizer: Any
    candidate_matrix: Any
    query_matrix: Any


def _clean_label(value: Optional[str], default: str = "unknown") -> str:
    if not value:
        return default
    return str(value)


def _length_bucket_from_record(record: dict) -> str:
    if record is None:
        return "unknown"
    bucket = record.get("length_bucket")
    if bucket:
        return str(bucket)
    token_len = record.get("token_length")
    try:
        token_len_int = int(token_len)
    except Exception:
        return "unknown"
    if token_len_int <= 10:
        return "short"
    if token_len_int <= 100:
        return "medium"
    if token_len_int <= 500:
        return "long"
    return "extra_long"


def _candidate_pool_stats(counts: List[int]) -> Dict[str, float]:
    if not counts:
        return {}
    avg = sum(counts) / len(counts)
    return {
        "num_candidates": avg,
        "min_num_candidates": min(counts),
        "max_num_candidates": max(counts),
    }


def build_tfidf_index(
    split: AuthBenchSplit,
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: Optional[int] = None,
    min_df: int = 1,
    lowercase: bool = True,
    include_queries_in_fit: bool = False,
) -> Tuple[AuthBenchSplit, TfidfIndex]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    working = split.limited(max_queries=max_queries, max_candidates=max_candidates)
    candidate_texts = [c["content"] for c in working.candidates]
    query_texts = [q["content"] for q in working.queries]

    corpus = candidate_texts + query_texts if include_queries_in_fit else candidate_texts
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=lowercase,
        norm="l2",
    )
    vectorizer.fit(corpus)
    candidate_matrix = vectorizer.transform(candidate_texts)
    query_matrix = vectorizer.transform(query_texts)
    return working, TfidfIndex(vectorizer, candidate_matrix, query_matrix)


def evaluate_tfidf_representation(
    split: AuthBenchSplit,
    ks: Sequence[int] = (1, 3, 5, 10),
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: Optional[int] = None,
    min_df: int = 1,
    lowercase: bool = True,
    include_queries_in_fit: bool = False,
    tfidf_index: Optional[TfidfIndex] = None,
    working_split: Optional[AuthBenchSplit] = None,
) -> Dict[str, object]:
    if candidate_pool not in ("all", "topic"):
        raise ValueError(f"Unknown candidate_pool: {candidate_pool}")

    if tfidf_index is None or working_split is None:
        working_split, tfidf_index = build_tfidf_index(
            split,
            max_queries=max_queries,
            max_candidates=max_candidates,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            lowercase=lowercase,
            include_queries_in_fit=include_queries_in_fit,
        )
    working = working_split

    candidate_ids = [c["candidate_id"] for c in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    candidate_matrix = tfidf_index.candidate_matrix
    query_matrix = tfidf_index.query_matrix

    metrics_per_query = []
    per_lang: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_genre: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_length: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    candidate_counts: List[int] = []
    per_lang_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_genre_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_length_counts: DefaultDict[str, List[int]] = defaultdict(list)

    topic_candidates = (
        build_topic_candidate_index(working.candidates) if candidate_pool == "topic" else None
    )

    if candidate_pool == "topic":
        for idx, query_record in enumerate(working.queries):
            query_id = query_record["query_id"]
            positives = working.positives_by_query.get(query_id, [])
            pos_indices = [candidate_index[cid] for cid in positives if cid in candidate_index]
            if not pos_indices:
                continue

            pool_indices = build_topic_pool(
                query_record=query_record,
                query_id=query_id,
                candidate_ids=candidate_ids,
                candidate_indices_by_topic=topic_candidates or {},
                positive_indices=pos_indices,
                max_candidates=max_topic_candidates,
                seed=topic_seed,
            )
            if not pool_indices:
                continue
            pool_index = {idx: pos for pos, idx in enumerate(pool_indices)}
            pool_pos_indices = [pool_index[idx] for idx in pos_indices if idx in pool_index]
            if not pool_pos_indices:
                continue
            scores = query_matrix[idx].dot(candidate_matrix[pool_indices].T)
            scores = np.asarray(scores.toarray()).ravel()
            metrics = ranking_metrics_for_query(torch.from_numpy(scores).float(), pool_pos_indices, ks)
            pool_size = len(pool_indices)

            metrics_per_query.append(metrics)
            lang = _clean_label(query_record.get("lang") or query_record.get("language"))
            genre = _clean_label(query_record.get("genre"))
            length_bucket = _length_bucket_from_record(query_record)
            per_lang[lang].append(metrics)
            per_genre[genre].append(metrics)
            per_length[length_bucket].append(metrics)
            candidate_counts.append(pool_size)
            per_lang_counts[lang].append(pool_size)
            per_genre_counts[genre].append(pool_size)
            per_length_counts[length_bucket].append(pool_size)
    else:
        batch_size = 256
        for start in range(0, len(working.queries), batch_size):
            end = min(start + batch_size, len(working.queries))
            score_matrix = query_matrix[start:end].dot(candidate_matrix.T)
            score_matrix = np.asarray(score_matrix.toarray(), dtype=np.float32)
            for row, idx in enumerate(range(start, end)):
                query_record = working.queries[idx]
                query_id = query_record["query_id"]
                positives = working.positives_by_query.get(query_id, [])
                pos_indices = [candidate_index[cid] for cid in positives if cid in candidate_index]
                if not pos_indices:
                    continue
                metrics = ranking_metrics_for_query(
                    torch.from_numpy(score_matrix[row]).float(),
                    pos_indices,
                    ks,
                )
                metrics_per_query.append(metrics)
                lang = _clean_label(query_record.get("lang") or query_record.get("language"))
                genre = _clean_label(query_record.get("genre"))
                length_bucket = _length_bucket_from_record(query_record)
                per_lang[lang].append(metrics)
                per_genre[genre].append(metrics)
                per_length[length_bucket].append(metrics)

    aggregated = aggregate_ranking_metrics(metrics_per_query)
    aggregated["num_queries"] = len(metrics_per_query)
    if candidate_pool == "topic":
        aggregated.update(_candidate_pool_stats(candidate_counts))
        aggregated["by_language"] = _aggregate_grouped_ranking(
            per_lang, per_lang_counts, candidate_ids
        )
        aggregated["by_genre"] = _aggregate_grouped_ranking(
            per_genre, per_genre_counts, candidate_ids
        )
        aggregated["by_length_bucket"] = _aggregate_grouped_ranking(
            per_length, per_length_counts, candidate_ids
        )
    else:
        aggregated["num_candidates"] = len(candidate_ids)
        aggregated["by_language"] = _aggregate_grouped_ranking(per_lang, None, candidate_ids)
        aggregated["by_genre"] = _aggregate_grouped_ranking(per_genre, None, candidate_ids)
        aggregated["by_length_bucket"] = _aggregate_grouped_ranking(
            per_length, None, candidate_ids
        )
    return aggregated


def _aggregate_grouped_ranking(
    grouped: DefaultDict[str, List[Dict[str, float]]],
    candidate_counts: Optional[DefaultDict[str, List[int]]],
    candidate_ids: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    num_candidates = len(candidate_ids)
    for key, per_query in grouped.items():
        if not per_query:
            continue
        agg = aggregate_ranking_metrics(per_query)
        agg["num_queries"] = len(per_query)
        if candidate_counts is not None and candidate_counts.get(key):
            agg.update(_candidate_pool_stats(candidate_counts[key]))
        else:
            agg["num_candidates"] = num_candidates
        aggregated[key] = agg
    return aggregated


def evaluate_tfidf_attribution(
    split: AuthBenchSplit,
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    negatives_per_query: int = 10,
    negative_strategy: str = "all",
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: Optional[int] = None,
    min_df: int = 1,
    lowercase: bool = True,
    include_queries_in_fit: bool = False,
    tfidf_index: Optional[TfidfIndex] = None,
    working_split: Optional[AuthBenchSplit] = None,
    seed: int = 13,
) -> Dict[str, object]:
    if candidate_pool not in ("all", "topic"):
        raise ValueError(f"Unknown candidate_pool: {candidate_pool}")

    if tfidf_index is None or working_split is None:
        working_split, tfidf_index = build_tfidf_index(
            split,
            max_queries=max_queries,
            max_candidates=max_candidates,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            lowercase=lowercase,
            include_queries_in_fit=include_queries_in_fit,
        )
    working = working_split

    candidate_ids = [c["candidate_id"] for c in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    candidate_matrix = tfidf_index.candidate_matrix
    query_matrix = tfidf_index.query_matrix
    rng = random.Random(seed)

    positive_scores: List[float] = []
    negative_scores: List[float] = []
    query_counter = 0
    positive_pairs = 0
    negative_pairs = 0
    candidate_counts: List[int] = []
    pos_by_lang: DefaultDict[str, List[float]] = defaultdict(list)
    neg_by_lang: DefaultDict[str, List[float]] = defaultdict(list)
    pos_by_genre: DefaultDict[str, List[float]] = defaultdict(list)
    neg_by_genre: DefaultDict[str, List[float]] = defaultdict(list)
    pos_by_length: DefaultDict[str, List[float]] = defaultdict(list)
    neg_by_length: DefaultDict[str, List[float]] = defaultdict(list)
    query_count_by_lang: DefaultDict[str, int] = defaultdict(int)
    query_count_by_genre: DefaultDict[str, int] = defaultdict(int)
    query_count_by_length: DefaultDict[str, int] = defaultdict(int)
    pos_pairs_by_lang: DefaultDict[str, int] = defaultdict(int)
    neg_pairs_by_lang: DefaultDict[str, int] = defaultdict(int)
    pos_pairs_by_genre: DefaultDict[str, int] = defaultdict(int)
    neg_pairs_by_genre: DefaultDict[str, int] = defaultdict(int)
    pos_pairs_by_length: DefaultDict[str, int] = defaultdict(int)
    neg_pairs_by_length: DefaultDict[str, int] = defaultdict(int)
    per_lang_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_genre_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_length_counts: DefaultDict[str, List[int]] = defaultdict(list)

    topic_candidates = (
        build_topic_candidate_index(working.candidates) if candidate_pool == "topic" else None
    )

    if candidate_pool == "topic":
        for idx, query_record in enumerate(working.queries):
            query_id = query_record["query_id"]
            positives = working.positives_by_query.get(query_id, [])
            pos_indices = [candidate_index[cid] for cid in positives if cid in candidate_index]
            if not pos_indices:
                continue

            pool_indices = build_topic_pool(
                query_record=query_record,
                query_id=query_id,
                candidate_ids=candidate_ids,
                candidate_indices_by_topic=topic_candidates or {},
                positive_indices=pos_indices,
                max_candidates=max_topic_candidates,
                seed=topic_seed,
            )
            if not pool_indices:
                continue
            pool_index = {idx: pos for pos, idx in enumerate(pool_indices)}
            pool_pos_indices = [pool_index[idx] for idx in pos_indices if idx in pool_index]
            if not pool_pos_indices:
                continue
            scores = query_matrix[idx].dot(candidate_matrix[pool_indices].T)
            scores = np.asarray(scores.toarray()).ravel()
            neg_pool = [i for i in range(len(pool_indices)) if i not in pool_pos_indices]
            pool_size = len(pool_indices)

            query_counter += 1
            pos_vals = [scores[i] for i in pool_pos_indices]
            positive_scores.extend(pos_vals)
            positive_pairs += len(pos_vals)

            lang = _clean_label(query_record.get("lang") or query_record.get("language"))
            genre = _clean_label(query_record.get("genre"))
            length_bucket = _length_bucket_from_record(query_record)
            pos_by_lang[lang].extend(pos_vals)
            pos_by_genre[genre].extend(pos_vals)
            pos_by_length[length_bucket].extend(pos_vals)
            query_count_by_lang[lang] += 1
            query_count_by_genre[genre] += 1
            query_count_by_length[length_bucket] += 1
            pos_pairs_by_lang[lang] += len(pos_vals)
            pos_pairs_by_genre[genre] += len(pos_vals)
            pos_pairs_by_length[length_bucket] += len(pos_vals)

            if negative_strategy == "all":
                chosen = neg_pool
            else:
                if negatives_per_query is None or negatives_per_query >= len(neg_pool):
                    chosen = neg_pool
                else:
                    chosen = rng.sample(neg_pool, negatives_per_query)
            if chosen:
                neg_vals = [scores[i] for i in chosen]
                negative_scores.extend(neg_vals)
                negative_pairs += len(neg_vals)
                neg_by_lang[lang].extend(neg_vals)
                neg_by_genre[genre].extend(neg_vals)
                neg_by_length[length_bucket].extend(neg_vals)
                neg_pairs_by_lang[lang] += len(neg_vals)
                neg_pairs_by_genre[genre] += len(neg_vals)
                neg_pairs_by_length[length_bucket] += len(neg_vals)

            candidate_counts.append(pool_size)
            per_lang_counts[lang].append(pool_size)
            per_genre_counts[genre].append(pool_size)
            per_length_counts[length_bucket].append(pool_size)
    else:
        all_indices = list(range(len(candidate_ids)))
        for idx, query_record in enumerate(working.queries):
            query_id = query_record["query_id"]
            positives = working.positives_by_query.get(query_id, [])
            pos_indices = [candidate_index[cid] for cid in positives if cid in candidate_index]
            if not pos_indices:
                continue

            positive_set = set(pos_indices)
            neg_pool = [i for i in all_indices if i not in positive_set]
            if negative_strategy == "all":
                chosen = neg_pool
            else:
                if negatives_per_query is None or negatives_per_query >= len(neg_pool):
                    chosen = neg_pool
                else:
                    chosen = rng.sample(neg_pool, negatives_per_query)
            selected = pos_indices + chosen
            scores = query_matrix[idx].dot(candidate_matrix[selected].T)
            scores = np.asarray(scores.toarray()).ravel()

            query_counter += 1
            pos_vals = scores[: len(pos_indices)].tolist()
            positive_scores.extend(pos_vals)
            positive_pairs += len(pos_vals)

            lang = _clean_label(query_record.get("lang") or query_record.get("language"))
            genre = _clean_label(query_record.get("genre"))
            length_bucket = _length_bucket_from_record(query_record)
            pos_by_lang[lang].extend(pos_vals)
            pos_by_genre[genre].extend(pos_vals)
            pos_by_length[length_bucket].extend(pos_vals)
            query_count_by_lang[lang] += 1
            query_count_by_genre[genre] += 1
            query_count_by_length[length_bucket] += 1
            pos_pairs_by_lang[lang] += len(pos_vals)
            pos_pairs_by_genre[genre] += len(pos_vals)
            pos_pairs_by_length[length_bucket] += len(pos_vals)

            neg_vals = scores[len(pos_indices) :].tolist()
            if neg_vals:
                negative_scores.extend(neg_vals)
                negative_pairs += len(neg_vals)
                neg_by_lang[lang].extend(neg_vals)
                neg_by_genre[genre].extend(neg_vals)
                neg_by_length[length_bucket].extend(neg_vals)
                neg_pairs_by_lang[lang] += len(neg_vals)
                neg_pairs_by_genre[genre] += len(neg_vals)
                neg_pairs_by_length[length_bucket] += len(neg_vals)

    if not positive_scores or not negative_scores:
        raise RuntimeError("EER requires at least one positive and one negative score.")

    eer = compute_eer(positive_scores, negative_scores)
    roc_auc = compute_roc_auc(positive_scores, negative_scores)
    result = {
        "eer": eer,
        "roc_auc": roc_auc,
        "num_queries": query_counter,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "negatives_per_query": negatives_per_query,
        "negative_strategy": negative_strategy,
    }
    if candidate_pool == "topic":
        result.update(_candidate_pool_stats(candidate_counts))
        result["by_language"] = _aggregate_grouped_eer(
            pos_by_lang,
            neg_by_lang,
            query_count_by_lang,
            pos_pairs_by_lang,
            neg_pairs_by_lang,
            per_lang_counts,
        )
        result["by_genre"] = _aggregate_grouped_eer(
            pos_by_genre,
            neg_by_genre,
            query_count_by_genre,
            pos_pairs_by_genre,
            neg_pairs_by_genre,
            per_genre_counts,
        )
        result["by_length_bucket"] = _aggregate_grouped_eer(
            pos_by_length,
            neg_by_length,
            query_count_by_length,
            pos_pairs_by_length,
            neg_pairs_by_length,
            per_length_counts,
        )
    else:
        result["num_candidates"] = len(candidate_ids)
        result["by_language"] = _aggregate_grouped_eer(
            pos_by_lang, neg_by_lang, query_count_by_lang, pos_pairs_by_lang, neg_pairs_by_lang
        )
        result["by_genre"] = _aggregate_grouped_eer(
            pos_by_genre, neg_by_genre, query_count_by_genre, pos_pairs_by_genre, neg_pairs_by_genre
        )
        result["by_length_bucket"] = _aggregate_grouped_eer(
            pos_by_length, neg_by_length, query_count_by_length, pos_pairs_by_length, neg_pairs_by_length
        )
    return result


def _aggregate_grouped_eer(
    positive_scores: DefaultDict[str, List[float]],
    negative_scores: DefaultDict[str, List[float]],
    query_counts: DefaultDict[str, int],
    positive_pairs: DefaultDict[str, int],
    negative_pairs: DefaultDict[str, int],
    candidate_counts: Optional[DefaultDict[str, List[int]]] = None,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for key, pos_scores in positive_scores.items():
        neg_scores = negative_scores.get(key, [])
        if not pos_scores or not neg_scores:
            continue
        grouped[key] = {
            "eer": compute_eer(pos_scores, neg_scores),
            "roc_auc": compute_roc_auc(pos_scores, neg_scores),
            "num_queries": query_counts.get(key, 0),
            "positive_pairs": positive_pairs.get(key, len(pos_scores)),
            "negative_pairs": negative_pairs.get(key, len(neg_scores)),
        }
        if candidate_counts is not None and candidate_counts.get(key):
            grouped[key].update(_candidate_pool_stats(candidate_counts[key]))
    return grouped
