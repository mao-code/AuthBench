from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


def _ideal_dcg(num_positives: int, k: int) -> float:
    top = min(num_positives, k)
    if top == 0:
        return 0.0
    return sum(1.0 / math.log2(idx + 2) for idx in range(top))


def ranking_metrics_for_query(
    scores: torch.Tensor, positive_indices: Sequence[int], ks: Sequence[int]
) -> Dict[str, float]:
    """
    Compute Recall@K, Success@K, nDCG@K, and MRR for a single query.

    Args:
        scores: Similarity scores for every candidate (1D tensor).
        positive_indices: Candidate indices that are positives for the query.
        ks: List of K values to evaluate at.
    """

    if scores.ndim != 1:
        raise ValueError("Scores must be a 1D tensor.")
    if not positive_indices:
        return {f"recall@{k}": 0.0 for k in ks} | {f"success@{k}": 0.0 for k in ks} | {
            f"ndcg@{k}": 0.0 for k in ks
        } | {"mrr": 0.0}

    positive_set = set(int(i) for i in positive_indices)
    num_pos = len(positive_set)
    max_k = min(max(ks), scores.numel())

    top_scores, top_indices = torch.topk(scores, k=max_k)
    hits = [1 if int(idx) in positive_set else 0 for idx in top_indices]

    metrics: Dict[str, float] = {}
    for k in ks:
        k = min(k, max_k)
        k_hits = hits[:k]
        hit_count = sum(k_hits)
        metrics[f"recall@{k}"] = hit_count / num_pos
        metrics[f"success@{k}"] = 1.0 if hit_count > 0 else 0.0
        idcg = _ideal_dcg(num_pos, k)
        dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(k_hits))
        metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

    positive_scores = scores[list(positive_set)]
    best_positive = positive_scores.max()
    better = (scores > best_positive).sum().item()
    metrics["mrr"] = 1.0 / (better + 1)
    return metrics


def aggregate_ranking_metrics(
    per_query_metrics: List[Mapping[str, float]]
) -> Dict[str, float]:
    """Average per-query metrics."""

    if not per_query_metrics:
        return {}
    accumulator: Dict[str, float] = {}
    for metrics in per_query_metrics:
        for key, value in metrics.items():
            accumulator[key] = accumulator.get(key, 0.0) + float(value)
    total = len(per_query_metrics)
    return {key: value / total for key, value in accumulator.items()}


def compute_eer(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Compute the Equal Error Rate (EER) given positive and negative scores."""

    pos = np.asarray(list(positive_scores), dtype=np.float64)
    neg = np.asarray(list(negative_scores), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Positive and negative score arrays must be non-empty.")

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    P = float(pos.size)
    N = float(neg.size)

    tp = 0.0
    fp = 0.0
    fprs = [0.0]
    fnrs = [1.0]
    start = 0
    while start < scores_sorted.size:
        end = start + 1
        while end < scores_sorted.size and scores_sorted[end] == scores_sorted[start]:
            end += 1
        block = labels_sorted[start:end]
        tp += float(block.sum())
        fp += float(block.size - block.sum())
        fpr = fp / N
        fnr = (P - tp) / P
        fprs.append(fpr)
        fnrs.append(fnr)
        start = end

    fprs = np.asarray(fprs)
    fnrs = np.asarray(fnrs)
    diff = fprs - fnrs
    crossing = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if len(crossing) == 0:
        return float(fprs[np.argmin(np.abs(diff))])

    idx = crossing[0]
    x0, x1 = fprs[idx], fprs[idx + 1]
    y0, y1 = fnrs[idx], fnrs[idx + 1]

    denom = (y0 - x0) + (x1 - y1)
    if denom == 0:
        return float((x0 + y0) / 2)
    t = (y0 - x0) / denom
    t = min(max(t, 0.0), 1.0)
    eer = x0 + t * (x1 - x0)
    return float(eer)


def compute_roc_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Compute ROC-AUC from positive and negative scores.

    The implementation uses average ranks for ties, which makes it equivalent to the
    Mann-Whitney U statistic normalized by the number of positive-negative pairs.
    """

    pos = np.asarray(list(positive_scores), dtype=np.float64)
    neg = np.asarray(list(negative_scores), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Positive and negative score arrays must be non-empty.")

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    num_pos = float(pos.size)
    num_neg = float(neg.size)
    pos_rank_sum = ranks[labels == 1].sum()
    auc = (pos_rank_sum - (num_pos * (num_pos + 1.0) / 2.0)) / (num_pos * num_neg)
    return float(auc)
