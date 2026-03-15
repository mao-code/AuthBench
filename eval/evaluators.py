from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence

import torch

from AuthBench.eval.data import AuthBenchSplit
from AuthBench.eval.embedder import HuggingFaceEmbedder
from AuthBench.eval.metrics import (
    aggregate_ranking_metrics,
    compute_eer,
    compute_roc_auc,
    ranking_metrics_for_query,
)
from AuthBench.eval.pools import build_topic_candidate_index, build_topic_pool


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


def _aggregate_grouped_ranking_metrics(
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
            agg.update(_candidate_pool_stats(candidate_counts[key]))
        else:
            agg["num_candidates"] = num_candidates
        aggregated[key] = agg
    return aggregated


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


def _maxsim_scores(
    query_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    device: torch.device,
    candidate_batch_size: int = 64,
    query_mask: Optional[torch.Tensor] = None,
    candidate_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute late-interaction scores (average of per-token max similarity) for one query
    against every candidate, batching candidates to stay within memory.
    """

    scores = []
    query_tokens = query_tokens.to(device)
    query_mask = query_mask.to(device).float() if query_mask is not None else None
    for start in range(0, candidate_tokens.size(0), candidate_batch_size):
        cand_chunk = candidate_tokens[start : start + candidate_batch_size].to(device)
        cand_mask_chunk = (
            candidate_masks[start : start + candidate_batch_size].to(device) if candidate_masks is not None else None
        )
        # query_tokens: (q_len, d); cand_chunk: (b, c_len, d)
        sim = torch.einsum("qd,bkd->bqk", query_tokens, cand_chunk)
        if cand_mask_chunk is not None:
            sim = sim.masked_fill(cand_mask_chunk.unsqueeze(1) == 0, float("-inf"))
        max_sim = sim.max(dim=2).values  # (b, q_len)
        if query_mask is not None:
            norm = query_mask.sum().clamp_min(1e-6)
            max_sim = (max_sim * query_mask).sum(dim=1) / norm
        else:
            max_sim = max_sim.mean(dim=1)
        scores.append(max_sim.cpu())
    return torch.cat(scores, dim=0)


def evaluate_authorship_representation(
    split: AuthBenchSplit,
    embedder: HuggingFaceEmbedder,
    batch_size: int = 32,
    ks: Sequence[int] = (1, 3, 5, 10),
    query_prefix: str = "",
    doc_prefix: str = "",
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    late_interaction: bool = False,
    candidate_chunk_size: int = 128,
    score_device: Optional[str] = None,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
) -> Dict[str, object]:
    """Evaluate retrieval-style authorship representation metrics."""

    working = split.limited(max_queries=max_queries, max_candidates=max_candidates)
    candidate_ids = [c["candidate_id"] for c in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}

    candidate_embeddings = embedder.encode_texts(
        [c["content"] for c in working.candidates],
        batch_size=batch_size,
        prefix=doc_prefix,
        return_tokens=late_interaction,
        show_progress=True,
    )
    query_embeddings = embedder.encode_texts(
        [q["content"] for q in working.queries],
        batch_size=batch_size,
        prefix=query_prefix,
        return_tokens=late_interaction,
        show_progress=True,
    )

    if candidate_pool not in ("all", "topic"):
        raise ValueError(f"Unknown candidate_pool: {candidate_pool}")

    metrics_per_query = []
    per_lang: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_genre: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_length: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    candidate_counts: List[int] = []
    per_lang_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_genre_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_length_counts: DefaultDict[str, List[int]] = defaultdict(list)

    device = torch.device(score_device or embedder.device)
    candidate_vectors = candidate_embeddings.vectors.to(device)
    query_vectors = query_embeddings.vectors
    candidate_masks = candidate_embeddings.attention_masks

    topic_candidates = (
        build_topic_candidate_index(working.candidates) if candidate_pool == "topic" else None
    )

    if candidate_pool == "all":
        for start in range(0, query_vectors.size(0), batch_size):
            q_batch = query_vectors[start : start + batch_size].to(device)
            if late_interaction:
                if (
                    candidate_embeddings.token_embeddings is None
                    or query_embeddings.token_embeddings is None
                    or candidate_masks is None
                    or query_embeddings.attention_masks is None
                ):
                    raise ValueError("Token embeddings are required for late interaction scoring.")
                query_tokens_batch = query_embeddings.token_embeddings[
                    start : start + q_batch.size(0)
                ]
                query_mask_batch = query_embeddings.attention_masks[start : start + q_batch.size(0)]

            sims = []
            if late_interaction:
                for idx_in_batch, query_tokens in enumerate(query_tokens_batch):
                    scores = _maxsim_scores(
                        query_tokens=query_tokens,
                        candidate_tokens=candidate_embeddings.token_embeddings,
                        device=device,
                        candidate_batch_size=candidate_chunk_size,
                        query_mask=query_mask_batch[idx_in_batch],
                        candidate_masks=candidate_masks,
                    )
                    sims.append(scores)
                sim_matrix = torch.stack(sims, dim=0)
            else:
                sim_matrix = torch.matmul(q_batch, candidate_vectors.T)

            batch_query_indices = list(range(start, min(start + batch_size, len(working.queries))))
            query_ids = [working.queries[i]["query_id"] for i in batch_query_indices]
            for row, query_id in enumerate(query_ids):
                positives = working.positives_by_query.get(query_id, [])
                positive_indices = [
                    candidate_index[cid] for cid in positives if cid in candidate_index
                ]
                if not positive_indices:
                    continue
                metrics = ranking_metrics_for_query(sim_matrix[row].cpu(), positive_indices, ks)
                metrics_per_query.append(metrics)
                query_record = working.queries[batch_query_indices[row]]
                lang = _clean_label(query_record.get("lang") or query_record.get("language"))
                genre = _clean_label(query_record.get("genre"))
                length_bucket = _length_bucket_from_record(query_record)
                per_lang[lang].append(metrics)
                per_genre[genre].append(metrics)
                per_length[length_bucket].append(metrics)
    else:
        if (
            late_interaction
            and (
                candidate_embeddings.token_embeddings is None
                or query_embeddings.token_embeddings is None
                or query_embeddings.attention_masks is None
            )
        ):
            raise ValueError("Token embeddings and attention masks are required for late interaction.")

        for idx, query_record in enumerate(working.queries):
            query_id = query_record["query_id"]
            positives = working.positives_by_query.get(query_id, [])
            positive_indices = [
                candidate_index[cid] for cid in positives if cid in candidate_index
            ]
            if not positive_indices:
                continue
            pool_indices = build_topic_pool(
                query_record=query_record,
                query_id=query_id,
                candidate_ids=candidate_ids,
                candidate_indices_by_topic=topic_candidates or {},
                positive_indices=positive_indices,
                max_candidates=max_topic_candidates,
                seed=topic_seed,
            )
            if not pool_indices:
                continue
            pool_index = {idx: pos for pos, idx in enumerate(pool_indices)}
            pool_positive_indices = [
                pool_index[idx] for idx in positive_indices if idx in pool_index
            ]
            if not pool_positive_indices:
                continue

            if late_interaction:
                query_tokens = query_embeddings.token_embeddings[idx]
                query_mask = (
                    query_embeddings.attention_masks[idx]
                    if query_embeddings.attention_masks is not None
                    else None
                )
                cand_tokens = candidate_embeddings.token_embeddings[pool_indices]
                cand_masks = candidate_masks[pool_indices] if candidate_masks is not None else None
                scores = _maxsim_scores(
                    query_tokens=query_tokens,
                    candidate_tokens=cand_tokens,
                    device=device,
                    candidate_batch_size=candidate_chunk_size,
                    query_mask=query_mask,
                    candidate_masks=cand_masks,
                )
            else:
                query_vec = query_vectors[idx].to(device)
                scores = torch.matmul(query_vec, candidate_vectors[pool_indices].T).cpu()

            metrics = ranking_metrics_for_query(scores.cpu(), pool_positive_indices, ks)
            metrics_per_query.append(metrics)

            pool_size = len(pool_indices)
            candidate_counts.append(pool_size)
            lang = _clean_label(query_record.get("lang") or query_record.get("language"))
            genre = _clean_label(query_record.get("genre"))
            length_bucket = _length_bucket_from_record(query_record)
            per_lang[lang].append(metrics)
            per_genre[genre].append(metrics)
            per_length[length_bucket].append(metrics)
            per_lang_counts[lang].append(pool_size)
            per_genre_counts[genre].append(pool_size)
            per_length_counts[length_bucket].append(pool_size)

    aggregated = aggregate_ranking_metrics(metrics_per_query)
    aggregated["num_queries"] = len(metrics_per_query)
    if candidate_pool == "all":
        aggregated["num_candidates"] = len(candidate_ids)
        aggregated["by_language"] = _aggregate_grouped_ranking_metrics(per_lang, len(candidate_ids))
        aggregated["by_genre"] = _aggregate_grouped_ranking_metrics(per_genre, len(candidate_ids))
        aggregated["by_length_bucket"] = _aggregate_grouped_ranking_metrics(
            per_length, len(candidate_ids)
        )
    else:
        aggregated.update(_candidate_pool_stats(candidate_counts))
        aggregated["by_language"] = _aggregate_grouped_ranking_metrics(
            per_lang, len(candidate_ids), per_lang_counts
        )
        aggregated["by_genre"] = _aggregate_grouped_ranking_metrics(
            per_genre, len(candidate_ids), per_genre_counts
        )
        aggregated["by_length_bucket"] = _aggregate_grouped_ranking_metrics(
            per_length, len(candidate_ids), per_length_counts
        )
    return aggregated


def evaluate_authorship_attribution(
    split: AuthBenchSplit,
    embedder: HuggingFaceEmbedder,
    batch_size: int = 32,
    query_prefix: str = "",
    doc_prefix: str = "",
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    negatives_per_query: int = 50,
    negative_strategy: str = "all",
    late_interaction: bool = False,
    candidate_chunk_size: int = 128,
    score_device: Optional[str] = None,
    seed: int = 13,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
) -> Dict[str, object]:
    """Evaluate EER and ROC-AUC for authorship attribution / verification."""

    working = split.limited(max_queries=max_queries, max_candidates=max_candidates)
    candidate_ids = [c["candidate_id"] for c in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    rng = random.Random(seed)

    candidate_embeddings = embedder.encode_texts(
        [c["content"] for c in working.candidates],
        batch_size=batch_size,
        prefix=doc_prefix,
        return_tokens=late_interaction,
        show_progress=True,
    )
    query_embeddings = embedder.encode_texts(
        [q["content"] for q in working.queries],
        batch_size=batch_size,
        prefix=query_prefix,
        return_tokens=late_interaction,
        show_progress=True,
    )

    if candidate_pool not in ("all", "topic"):
        raise ValueError(f"Unknown candidate_pool: {candidate_pool}")

    device = torch.device(score_device or embedder.device)
    candidate_vectors = candidate_embeddings.vectors.to(device)
    candidate_masks = candidate_embeddings.attention_masks
    positive_scores = []
    negative_scores = []
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

    query_vectors = query_embeddings.vectors
    if late_interaction:
        if (
            candidate_embeddings.token_embeddings is None
            or candidate_masks is None
            or query_embeddings.token_embeddings is None
            or query_embeddings.attention_masks is None
        ):
            raise ValueError("Token embeddings and attention masks are required for late interaction.")

    topic_candidates = (
        build_topic_candidate_index(working.candidates) if candidate_pool == "topic" else None
    )

    if candidate_pool == "all":
        for start in range(0, query_vectors.size(0), batch_size):
            q_batch = query_vectors[start : start + batch_size].to(device)
            if late_interaction:
                query_tokens_batch = query_embeddings.token_embeddings[
                    start : start + q_batch.size(0)
                ]
                query_mask_batch = (
                    query_embeddings.attention_masks[start : start + q_batch.size(0)]
                    if query_embeddings.attention_masks is not None
                    else None
                )

            if late_interaction:
                batch_scores = []
                for idx_in_batch, query_tokens in enumerate(query_tokens_batch):
                    scores = _maxsim_scores(
                        query_tokens=query_tokens,
                        candidate_tokens=candidate_embeddings.token_embeddings,
                        device=device,
                        candidate_batch_size=candidate_chunk_size,
                        query_mask=query_mask_batch[idx_in_batch] if query_mask_batch is not None else None,
                        candidate_masks=candidate_masks,
                    )
                    batch_scores.append(scores)
                sim_matrix = torch.stack(batch_scores, dim=0)
            else:
                sim_matrix = torch.matmul(q_batch, candidate_vectors.T)

            batch_query_indices = list(range(start, min(start + batch_size, len(working.queries))))
            query_ids = [working.queries[i]["query_id"] for i in batch_query_indices]
            for row, query_id in enumerate(query_ids):
                positives = working.positives_by_query.get(query_id, [])
                pos_indices = [
                    candidate_index[cid] for cid in positives if cid in candidate_index
                ]
                if not pos_indices:
                    continue
                query_counter += 1
                scores = sim_matrix[row].cpu()
                pos_vals = scores[pos_indices].tolist()
                positive_scores.extend(pos_vals)
                positive_pairs += len(pos_vals)
                query_record = working.queries[batch_query_indices[row]]
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

                neg_pool = [idx for idx in range(len(candidate_ids)) if idx not in pos_indices]
                if negative_strategy == "all":
                    chosen = neg_pool
                else:
                    if negatives_per_query is None or negatives_per_query >= len(neg_pool):
                        chosen = neg_pool
                    else:
                        chosen = rng.sample(neg_pool, negatives_per_query)
                if chosen:
                    neg_vals = scores[chosen].tolist()
                    negative_scores.extend(neg_vals)
                    negative_pairs += len(neg_vals)
                    neg_by_lang[lang].extend(neg_vals)
                    neg_by_genre[genre].extend(neg_vals)
                    neg_by_length[length_bucket].extend(neg_vals)
                    neg_pairs_by_lang[lang] += len(neg_vals)
                    neg_pairs_by_genre[genre] += len(neg_vals)
                    neg_pairs_by_length[length_bucket] += len(neg_vals)
    else:
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

            if late_interaction:
                query_tokens = query_embeddings.token_embeddings[idx]
                query_mask = (
                    query_embeddings.attention_masks[idx]
                    if query_embeddings.attention_masks is not None
                    else None
                )
                cand_tokens = candidate_embeddings.token_embeddings[pool_indices]
                cand_masks = candidate_masks[pool_indices] if candidate_masks is not None else None
                scores = _maxsim_scores(
                    query_tokens=query_tokens,
                    candidate_tokens=cand_tokens,
                    device=device,
                    candidate_batch_size=candidate_chunk_size,
                    query_mask=query_mask,
                    candidate_masks=cand_masks,
                )
            else:
                query_vec = query_vectors[idx].to(device)
                scores = torch.matmul(query_vec, candidate_vectors[pool_indices].T).cpu()

            query_counter += 1
            pos_vals = scores[pool_pos_indices].tolist()
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

            neg_pool = [i for i in range(len(pool_indices)) if i not in pool_pos_indices]
            if negative_strategy == "all":
                chosen = neg_pool
            else:
                if negatives_per_query is None or negatives_per_query >= len(neg_pool):
                    chosen = neg_pool
                else:
                    chosen = rng.sample(neg_pool, negatives_per_query)
            if chosen:
                neg_vals = scores[chosen].tolist()
                negative_scores.extend(neg_vals)
                negative_pairs += len(neg_vals)
                neg_by_lang[lang].extend(neg_vals)
                neg_by_genre[genre].extend(neg_vals)
                neg_by_length[length_bucket].extend(neg_vals)
                neg_pairs_by_lang[lang] += len(neg_vals)
                neg_pairs_by_genre[genre] += len(neg_vals)
                neg_pairs_by_length[length_bucket] += len(neg_vals)

            pool_size = len(pool_indices)
            candidate_counts.append(pool_size)
            per_lang_counts[lang].append(pool_size)
            per_genre_counts[genre].append(pool_size)
            per_length_counts[length_bucket].append(pool_size)

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

    if candidate_pool == "all":
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
    else:
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
    return result
