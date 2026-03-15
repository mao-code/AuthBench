from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import DefaultDict, Dict, List, Optional, Sequence

from AuthBench.eval.metrics import aggregate_ranking_metrics, compute_eer, compute_roc_auc


@dataclass(frozen=True)
class PairExample:
    query_idx: int
    candidate_idx: int
    label: int


def clean_label(value: Optional[str], default: str = "unknown") -> str:
    if not value:
        return default
    return str(value)


def length_bucket_from_record(record: dict | None) -> str:
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


def candidate_pool_stats(counts: List[int]) -> Dict[str, float]:
    if not counts:
        return {}
    avg = sum(counts) / len(counts)
    return {
        "num_candidates": avg,
        "min_num_candidates": min(counts),
        "max_num_candidates": max(counts),
    }


def aggregate_grouped_ranking(
    grouped: DefaultDict[str, List[Dict[str, float]]],
    num_candidates: int,
    candidate_counts: Optional[DefaultDict[str, List[int]]] = None,
) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    for key, per_query in grouped.items():
        if not per_query:
            continue
        agg = aggregate_ranking_metrics(per_query)
        agg["num_queries"] = len(per_query)
        if candidate_counts is not None and candidate_counts.get(key):
            agg.update(candidate_pool_stats(candidate_counts[key]))
        else:
            agg["num_candidates"] = num_candidates
        aggregated[key] = agg
    return aggregated


def aggregate_grouped_eer(
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
            grouped[key].update(candidate_pool_stats(candidate_counts[key]))
    return grouped


def sample_pair_examples(
    split,
    positives_per_query: int = 1,
    negatives_per_query: int = 1,
    max_examples: int | None = None,
    seed: int = 13,
) -> List[PairExample]:
    rng = random.Random(seed)
    candidate_ids = [record["candidate_id"] for record in split.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    query_index = {record["query_id"]: idx for idx, record in enumerate(split.queries)}

    ground_truth = list(split.ground_truth)
    rng.shuffle(ground_truth)

    examples: List[PairExample] = []
    num_candidates = len(candidate_ids)
    for entry in ground_truth:
        qid = entry["query_id"]
        qidx = query_index.get(qid)
        if qidx is None:
            continue

        positive_indices = [
            candidate_index[cid]
            for cid in split.positives_by_query.get(qid, [])
            if cid in candidate_index
        ]
        if not positive_indices:
            continue

        if positives_per_query > 0 and len(positive_indices) > positives_per_query:
            positive_indices = rng.sample(positive_indices, positives_per_query)

        for cidx in positive_indices:
            examples.append(PairExample(query_idx=qidx, candidate_idx=cidx, label=1))
            if max_examples is not None and len(examples) >= max_examples:
                return examples

        if negatives_per_query <= 0:
            continue

        positive_set = set(positive_indices)
        negatives: List[int] = []
        limit = min(negatives_per_query * max(1, len(positive_indices)), num_candidates - len(positive_set))
        while len(negatives) < limit:
            sampled = rng.randrange(num_candidates)
            if sampled in positive_set:
                continue
            negatives.append(sampled)
            positive_set.add(sampled)

        for cidx in negatives:
            examples.append(PairExample(query_idx=qidx, candidate_idx=cidx, label=0))
            if max_examples is not None and len(examples) >= max_examples:
                return examples

    return examples
