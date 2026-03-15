from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import DefaultDict, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from AuthBench.eval.baseline_utils import (
    PairExample,
    aggregate_grouped_eer,
    aggregate_grouped_ranking,
    candidate_pool_stats,
    clean_label,
    length_bucket_from_record,
    sample_pair_examples,
)
from AuthBench.eval.data import AuthBenchSplit
from AuthBench.eval.metrics import (
    aggregate_ranking_metrics,
    compute_eer,
    compute_roc_auc,
    ranking_metrics_for_query,
)
from AuthBench.eval.pools import build_topic_candidate_index, build_topic_pool


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _as_float_array(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray()).ravel().astype(np.float32, copy=False)
    if hasattr(matrix, "A1"):
        return np.asarray(matrix.A1).astype(np.float32, copy=False)
    return np.asarray(matrix).ravel().astype(np.float32, copy=False)


def _safe_ratio_similarity(query_value: float, candidate_values: np.ndarray) -> np.ndarray:
    candidate_values = candidate_values.astype(np.float32, copy=False)
    numer = np.minimum(query_value, candidate_values)
    denom = np.maximum(np.maximum(query_value, candidate_values), 1.0)
    return numer / denom


def _extract_surface_features(texts: Sequence[str]) -> np.ndarray:
    features = np.zeros((len(texts), 7), dtype=np.float32)
    for idx, text in enumerate(texts):
        raw = text or ""
        chars = max(len(raw), 1)
        tokens = raw.split()
        token_count = max(len(tokens), 1)
        punct = sum(1 for ch in raw if not ch.isalnum() and not ch.isspace())
        digits = sum(1 for ch in raw if ch.isdigit())
        upper = sum(1 for ch in raw if ch.isupper())
        spaces = sum(1 for ch in raw if ch.isspace())
        mean_token_len = float(sum(len(tok) for tok in tokens)) / token_count if tokens else 0.0
        features[idx] = np.asarray(
            [
                float(len(raw)),
                float(len(tokens)),
                punct / chars,
                digits / chars,
                upper / chars,
                spaces / chars,
                mean_token_len,
            ],
            dtype=np.float32,
        )
    return features


def _surface_query_features(query_features: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
    sims = np.empty((candidate_features.shape[0], 7), dtype=np.float32)
    sims[:, 0] = _safe_ratio_similarity(float(query_features[0]), candidate_features[:, 0])
    sims[:, 1] = _safe_ratio_similarity(float(query_features[1]), candidate_features[:, 1])
    sims[:, 2:] = 1.0 - np.abs(candidate_features[:, 2:] - query_features[2:])
    np.clip(sims[:, 2:], 0.0, 1.0, out=sims[:, 2:])
    return sims


def _surface_paired_features(query_features: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
    sims = np.empty((candidate_features.shape[0], 7), dtype=np.float32)
    sims[:, 0] = np.minimum(query_features[:, 0], candidate_features[:, 0]) / np.maximum(
        np.maximum(query_features[:, 0], candidate_features[:, 0]),
        1.0,
    )
    sims[:, 1] = np.minimum(query_features[:, 1], candidate_features[:, 1]) / np.maximum(
        np.maximum(query_features[:, 1], candidate_features[:, 1]),
        1.0,
    )
    sims[:, 2:] = 1.0 - np.abs(candidate_features[:, 2:] - query_features[:, 2:])
    np.clip(sims[:, 2:], 0.0, 1.0, out=sims[:, 2:])
    return sims


@dataclass
class LinearPairCalibrator:
    feature_names: List[str]
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Sequence[str],
    ) -> "LinearPairCalibrator":
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=13,
        )
        model.fit(features_scaled, labels)
        scale = scaler.scale_.astype(np.float32, copy=False)
        scale[scale == 0] = 1.0
        return cls(
            feature_names=list(feature_names),
            mean_=scaler.mean_.astype(np.float32, copy=False),
            scale_=scale,
            coef_=model.coef_[0].astype(np.float32, copy=False),
            intercept_=float(model.intercept_[0]),
        )

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        normalized = (features - self.mean_) / self.scale_
        return normalized @ self.coef_ + self.intercept_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(features))

    def metadata(self) -> Dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "coef": {name: float(value) for name, value in zip(self.feature_names, self.coef_)},
            "intercept": float(self.intercept_),
        }

    def linear_parameters(self) -> tuple[np.ndarray, float]:
        weights = self.coef_ / self.scale_
        bias = float(self.intercept_ - np.dot(self.mean_ / self.scale_, self.coef_))
        return weights.astype(np.float32, copy=False), bias


def _resolve_candidate_pool(
    working: AuthBenchSplit,
    candidate_ids: Sequence[str],
    candidate_index: Dict[str, int],
    topic_candidates,
    candidate_pool: str,
    max_topic_candidates: Optional[int],
    topic_seed: int,
    query_record: dict,
) -> tuple[list[int], list[int]]:
    query_id = query_record["query_id"]
    positive_indices = [
        candidate_index[cid]
        for cid in working.positives_by_query.get(query_id, [])
        if cid in candidate_index
    ]
    if not positive_indices:
        return [], []

    if candidate_pool == "topic":
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
            return [], []
        pool_lookup = {idx: pos for pos, idx in enumerate(pool_indices)}
        pool_positive_indices = [pool_lookup[idx] for idx in positive_indices if idx in pool_lookup]
        return list(pool_indices), pool_positive_indices

    return list(range(len(candidate_ids))), positive_indices


def _evaluate_representation_from_scorer(
    working: AuthBenchSplit,
    score_query_fn,
    ks: Sequence[int] = (1, 3, 5, 10),
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
) -> Dict[str, object]:
    candidate_ids = [record["candidate_id"] for record in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    topic_candidates = (
        build_topic_candidate_index(working.candidates) if candidate_pool == "topic" else None
    )

    metrics_per_query = []
    per_lang: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_genre: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_length: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    candidate_counts: List[int] = []
    per_lang_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_genre_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_length_counts: DefaultDict[str, List[int]] = defaultdict(list)

    for query_idx, query_record in enumerate(working.queries):
        pool_indices, pool_positive_indices = _resolve_candidate_pool(
            working=working,
            candidate_ids=candidate_ids,
            candidate_index=candidate_index,
            topic_candidates=topic_candidates,
            candidate_pool=candidate_pool,
            max_topic_candidates=max_topic_candidates,
            topic_seed=topic_seed,
            query_record=query_record,
        )
        if not pool_indices or not pool_positive_indices:
            continue

        scores = score_query_fn(query_idx, pool_indices)
        metrics = ranking_metrics_for_query(
            torch.from_numpy(scores).float(),
            pool_positive_indices,
            ks,
        )
        metrics_per_query.append(metrics)

        lang = clean_label(query_record.get("lang") or query_record.get("language"))
        genre = clean_label(query_record.get("genre"))
        length_bucket = length_bucket_from_record(query_record)
        per_lang[lang].append(metrics)
        per_genre[genre].append(metrics)
        per_length[length_bucket].append(metrics)

        if candidate_pool == "topic":
            pool_size = len(pool_indices)
            candidate_counts.append(pool_size)
            per_lang_counts[lang].append(pool_size)
            per_genre_counts[genre].append(pool_size)
            per_length_counts[length_bucket].append(pool_size)

    aggregated = aggregate_ranking_metrics(metrics_per_query)
    aggregated["num_queries"] = len(metrics_per_query)
    if candidate_pool == "topic":
        aggregated.update(candidate_pool_stats(candidate_counts))
        aggregated["by_language"] = aggregate_grouped_ranking(per_lang, len(candidate_ids), per_lang_counts)
        aggregated["by_genre"] = aggregate_grouped_ranking(per_genre, len(candidate_ids), per_genre_counts)
        aggregated["by_length_bucket"] = aggregate_grouped_ranking(
            per_length, len(candidate_ids), per_length_counts
        )
    else:
        aggregated["num_candidates"] = len(candidate_ids)
        aggregated["by_language"] = aggregate_grouped_ranking(per_lang, len(candidate_ids))
        aggregated["by_genre"] = aggregate_grouped_ranking(per_genre, len(candidate_ids))
        aggregated["by_length_bucket"] = aggregate_grouped_ranking(per_length, len(candidate_ids))
    return aggregated


def _evaluate_representation_all_pool_batches(
    working: AuthBenchSplit,
    batch_score_fn,
    batch_size: int = 32,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, object]:
    candidate_ids = [record["candidate_id"] for record in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}

    metrics_per_query = []
    per_lang: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_genre: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_length: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)

    for start in range(0, len(working.queries), batch_size):
        end = min(start + batch_size, len(working.queries))
        score_matrix = batch_score_fn(list(range(start, end)))
        for row, query_idx in enumerate(range(start, end)):
            query_record = working.queries[query_idx]
            positive_indices = [
                candidate_index[cid]
                for cid in working.positives_by_query.get(query_record["query_id"], [])
                if cid in candidate_index
            ]
            if not positive_indices:
                continue
            metrics = ranking_metrics_for_query(
                torch.from_numpy(score_matrix[row]).float(),
                positive_indices,
                ks,
            )
            metrics_per_query.append(metrics)
            lang = clean_label(query_record.get("lang") or query_record.get("language"))
            genre = clean_label(query_record.get("genre"))
            length_bucket = length_bucket_from_record(query_record)
            per_lang[lang].append(metrics)
            per_genre[genre].append(metrics)
            per_length[length_bucket].append(metrics)

    aggregated = aggregate_ranking_metrics(metrics_per_query)
    aggregated["num_queries"] = len(metrics_per_query)
    aggregated["num_candidates"] = len(candidate_ids)
    aggregated["by_language"] = aggregate_grouped_ranking(per_lang, len(candidate_ids))
    aggregated["by_genre"] = aggregate_grouped_ranking(per_genre, len(candidate_ids))
    aggregated["by_length_bucket"] = aggregate_grouped_ranking(per_length, len(candidate_ids))
    return aggregated


def _evaluate_attribution_from_scorer(
    working: AuthBenchSplit,
    score_query_fn,
    negatives_per_query: int = 50,
    negative_strategy: str = "all",
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    seed: int = 13,
) -> Dict[str, object]:
    candidate_ids = [record["candidate_id"] for record in working.candidates]
    candidate_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    topic_candidates = (
        build_topic_candidate_index(working.candidates) if candidate_pool == "topic" else None
    )

    rng = np.random.default_rng(seed)
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

    for query_idx, query_record in enumerate(working.queries):
        pool_indices, pool_positive_indices = _resolve_candidate_pool(
            working=working,
            candidate_ids=candidate_ids,
            candidate_index=candidate_index,
            topic_candidates=topic_candidates,
            candidate_pool=candidate_pool,
            max_topic_candidates=max_topic_candidates,
            topic_seed=topic_seed,
            query_record=query_record,
        )
        if not pool_indices or not pool_positive_indices:
            continue

        neg_pool = np.asarray(
            [idx for idx in range(len(pool_indices)) if idx not in set(pool_positive_indices)],
            dtype=np.int32,
        )
        if negative_strategy == "all" or negatives_per_query is None or negatives_per_query >= len(neg_pool):
            chosen_neg = neg_pool
        else:
            chosen_neg = rng.choice(neg_pool, size=negatives_per_query, replace=False)

        score_positions = list(pool_positive_indices) + chosen_neg.tolist()
        selected_candidate_indices = [pool_indices[pos] for pos in score_positions]
        selected_scores = score_query_fn(query_idx, selected_candidate_indices)

        pos_vals = selected_scores[: len(pool_positive_indices)]
        neg_vals = selected_scores[len(pool_positive_indices) :]
        if pos_vals.size == 0 or neg_vals.size == 0:
            continue

        query_counter += 1
        positive_scores.extend(pos_vals.tolist())
        negative_scores.extend(neg_vals.tolist())
        positive_pairs += int(pos_vals.size)
        negative_pairs += int(neg_vals.size)

        lang = clean_label(query_record.get("lang") or query_record.get("language"))
        genre = clean_label(query_record.get("genre"))
        length_bucket = length_bucket_from_record(query_record)
        pos_by_lang[lang].extend(pos_vals.tolist())
        neg_by_lang[lang].extend(neg_vals.tolist())
        pos_by_genre[genre].extend(pos_vals.tolist())
        neg_by_genre[genre].extend(neg_vals.tolist())
        pos_by_length[length_bucket].extend(pos_vals.tolist())
        neg_by_length[length_bucket].extend(neg_vals.tolist())
        query_count_by_lang[lang] += 1
        query_count_by_genre[genre] += 1
        query_count_by_length[length_bucket] += 1
        pos_pairs_by_lang[lang] += int(pos_vals.size)
        neg_pairs_by_lang[lang] += int(neg_vals.size)
        pos_pairs_by_genre[genre] += int(pos_vals.size)
        neg_pairs_by_genre[genre] += int(neg_vals.size)
        pos_pairs_by_length[length_bucket] += int(pos_vals.size)
        neg_pairs_by_length[length_bucket] += int(neg_vals.size)

        if candidate_pool == "topic":
            pool_size = len(pool_indices)
            candidate_counts.append(pool_size)
            per_lang_counts[lang].append(pool_size)
            per_genre_counts[genre].append(pool_size)
            per_length_counts[length_bucket].append(pool_size)

    if not positive_scores or not negative_scores:
        raise RuntimeError("EER requires at least one positive and one negative score.")

    result = {
        "eer": compute_eer(positive_scores, negative_scores),
        "roc_auc": compute_roc_auc(positive_scores, negative_scores),
        "num_queries": query_counter,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "negatives_per_query": negatives_per_query,
        "negative_strategy": negative_strategy,
    }
    if candidate_pool == "topic":
        result.update(candidate_pool_stats(candidate_counts))
        result["by_language"] = aggregate_grouped_eer(
            pos_by_lang,
            neg_by_lang,
            query_count_by_lang,
            pos_pairs_by_lang,
            neg_pairs_by_lang,
            per_lang_counts,
        )
        result["by_genre"] = aggregate_grouped_eer(
            pos_by_genre,
            neg_by_genre,
            query_count_by_genre,
            pos_pairs_by_genre,
            neg_pairs_by_genre,
            per_genre_counts,
        )
        result["by_length_bucket"] = aggregate_grouped_eer(
            pos_by_length,
            neg_by_length,
            query_count_by_length,
            pos_pairs_by_length,
            neg_pairs_by_length,
            per_length_counts,
        )
    else:
        result["num_candidates"] = len(candidate_ids)
        result["by_language"] = aggregate_grouped_eer(
            pos_by_lang,
            neg_by_lang,
            query_count_by_lang,
            pos_pairs_by_lang,
            neg_pairs_by_lang,
        )
        result["by_genre"] = aggregate_grouped_eer(
            pos_by_genre,
            neg_by_genre,
            query_count_by_genre,
            pos_pairs_by_genre,
            neg_pairs_by_genre,
        )
        result["by_length_bucket"] = aggregate_grouped_eer(
            pos_by_length,
            neg_by_length,
            query_count_by_length,
            pos_pairs_by_length,
            neg_pairs_by_length,
        )
    return result


@dataclass
class NgramSplitData:
    query_char: sparse.csr_matrix
    candidate_char: sparse.csr_matrix
    query_word: sparse.csr_matrix
    candidate_word: sparse.csr_matrix
    query_surface: np.ndarray
    candidate_surface: np.ndarray


@dataclass
class NgramStylometricBaseline:
    char_vectorizer: HashingVectorizer
    word_vectorizer: HashingVectorizer
    calibrator: LinearPairCalibrator
    train_pair_count: int

    @classmethod
    def fit(
        cls,
        train_split: AuthBenchSplit,
        max_train_examples: int = 120000,
        seed: int = 13,
        char_features: int = 2**16,
        word_features: int = 2**15,
    ) -> "NgramStylometricBaseline":
        char_vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=char_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        word_vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            n_features=word_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )

        train_queries = [record["content"] for record in train_split.queries]
        train_candidates = [record["content"] for record in train_split.candidates]
        query_char = char_vectorizer.transform(train_queries).tocsr()
        candidate_char = char_vectorizer.transform(train_candidates).tocsr()
        query_word = word_vectorizer.transform(train_queries).tocsr()
        candidate_word = word_vectorizer.transform(train_candidates).tocsr()
        query_surface = _extract_surface_features(train_queries)
        candidate_surface = _extract_surface_features(train_candidates)

        examples = sample_pair_examples(
            train_split,
            positives_per_query=1,
            negatives_per_query=1,
            max_examples=max_train_examples,
            seed=seed,
        )
        query_indices = np.asarray([example.query_idx for example in examples], dtype=np.int32)
        candidate_indices = np.asarray([example.candidate_idx for example in examples], dtype=np.int32)
        labels = np.asarray([example.label for example in examples], dtype=np.int32)

        char_sims = _as_float_array(
            query_char[query_indices].multiply(candidate_char[candidate_indices]).sum(axis=1)
        )
        word_sims = _as_float_array(
            query_word[query_indices].multiply(candidate_word[candidate_indices]).sum(axis=1)
        )
        pair_surface = _surface_paired_features(
            query_surface[query_indices],
            candidate_surface[candidate_indices],
        )
        features = np.column_stack([char_sims, word_sims, pair_surface]).astype(np.float32, copy=False)
        calibrator = LinearPairCalibrator.fit(
            features,
            labels,
            feature_names=[
                "char_cosine",
                "word_cosine",
                "char_len_ratio",
                "token_len_ratio",
                "punctuation_similarity",
                "digit_similarity",
                "uppercase_similarity",
                "whitespace_similarity",
                "mean_token_length_similarity",
            ],
        )
        return cls(
            char_vectorizer=char_vectorizer,
            word_vectorizer=word_vectorizer,
            calibrator=calibrator,
            train_pair_count=len(examples),
        )

    def transform_split(
        self,
        split: AuthBenchSplit,
        max_queries: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> tuple[AuthBenchSplit, NgramSplitData]:
        working = split.limited(max_queries=max_queries, max_candidates=max_candidates)
        query_texts = [record["content"] for record in working.queries]
        candidate_texts = [record["content"] for record in working.candidates]
        return working, NgramSplitData(
            query_char=self.char_vectorizer.transform(query_texts).tocsr(),
            candidate_char=self.char_vectorizer.transform(candidate_texts).tocsr(),
            query_word=self.word_vectorizer.transform(query_texts).tocsr(),
            candidate_word=self.word_vectorizer.transform(candidate_texts).tocsr(),
            query_surface=_extract_surface_features(query_texts),
            candidate_surface=_extract_surface_features(candidate_texts),
        )

    def score_query(self, data: NgramSplitData, query_idx: int, candidate_indices: Sequence[int]) -> np.ndarray:
        indices = np.asarray(candidate_indices, dtype=np.int32)
        char_scores = _as_float_array(data.query_char[query_idx].dot(data.candidate_char[indices].T))
        word_scores = _as_float_array(data.query_word[query_idx].dot(data.candidate_word[indices].T))
        surface = _surface_query_features(data.query_surface[query_idx], data.candidate_surface[indices])
        features = np.column_stack([char_scores, word_scores, surface]).astype(np.float32, copy=False)
        return self.calibrator.decision_function(features).astype(np.float32, copy=False)

    def metadata(self) -> Dict[str, object]:
        return {
            "char_ngrams": [3, 5],
            "word_ngrams": [1, 2],
            "char_features": int(self.char_vectorizer.n_features),
            "word_features": int(self.word_vectorizer.n_features),
            "train_pair_count": int(self.train_pair_count),
            "pair_calibrator": self.calibrator.metadata(),
        }


@dataclass
class PPMPreparedSplit:
    counts: sparse.csr_matrix
    totals: np.ndarray
    corrections: sparse.csr_matrix
    base_log_probs: np.ndarray


@dataclass
class PPMBaseline:
    vectorizer: HashingVectorizer
    n_features: int
    alpha: float
    calibrator: LinearPairCalibrator
    train_pair_count: int
    order: int

    @classmethod
    def fit(
        cls,
        train_split: AuthBenchSplit,
        max_train_examples: int = 80000,
        order: int = 4,
        n_features: int = 8192,
        alpha: float = 0.5,
        seed: int = 13,
    ) -> "PPMBaseline":
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(order + 1, order + 1),
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            lowercase=False,
        )
        train_queries = [record["content"] for record in train_split.queries]
        train_candidates = [record["content"] for record in train_split.candidates]
        query_prepared = cls._prepare_counts(vectorizer, train_queries, n_features, alpha)
        candidate_prepared = cls._prepare_counts(vectorizer, train_candidates, n_features, alpha)

        examples = sample_pair_examples(
            train_split,
            positives_per_query=1,
            negatives_per_query=1,
            max_examples=max_train_examples,
            seed=seed,
        )
        features = np.empty((len(examples), 2), dtype=np.float32)
        labels = np.asarray([example.label for example in examples], dtype=np.int32)
        for idx, example in enumerate(examples):
            q_idx = example.query_idx
            c_idx = example.candidate_idx
            query_under_candidate = cls._one_way_entropy(
                source_counts=query_prepared.counts[q_idx],
                source_total=float(query_prepared.totals[q_idx]),
                model_correction=candidate_prepared.corrections[c_idx],
                model_base_log_prob=float(candidate_prepared.base_log_probs[c_idx]),
            )
            candidate_under_query = cls._one_way_entropy(
                source_counts=candidate_prepared.counts[c_idx],
                source_total=float(candidate_prepared.totals[c_idx]),
                model_correction=query_prepared.corrections[q_idx],
                model_base_log_prob=float(query_prepared.base_log_probs[q_idx]),
            )
            features[idx, 0] = 0.5 * (query_under_candidate + candidate_under_query)
            features[idx, 1] = abs(query_under_candidate - candidate_under_query)

        calibrator = LinearPairCalibrator.fit(
            features,
            labels,
            feature_names=["mean_cross_entropy", "cross_entropy_gap"],
        )
        return cls(
            vectorizer=vectorizer,
            n_features=n_features,
            alpha=alpha,
            calibrator=calibrator,
            train_pair_count=len(examples),
            order=order,
        )

    @staticmethod
    def _prepare_counts(
        vectorizer: HashingVectorizer,
        texts: Sequence[str],
        n_features: int,
        alpha: float,
    ) -> PPMPreparedSplit:
        counts = vectorizer.transform(texts).tocsr().astype(np.float32)
        totals = _as_float_array(counts.sum(axis=1))
        safe_totals = np.maximum(totals, 1.0)
        corrections = counts.copy().tocsr()
        if corrections.nnz:
            corrections.data = np.log(corrections.data + alpha).astype(np.float32, copy=False) - math.log(alpha)
        base_log_probs = (
            np.float32(math.log(alpha))
            - np.log(safe_totals + np.float32(alpha * n_features)).astype(np.float32, copy=False)
        )
        return PPMPreparedSplit(
            counts=counts,
            totals=safe_totals.astype(np.float32, copy=False),
            corrections=corrections,
            base_log_probs=base_log_probs.astype(np.float32, copy=False),
        )

    @staticmethod
    def _one_way_entropy(
        source_counts: sparse.csr_matrix,
        source_total: float,
        model_correction: sparse.csr_matrix,
        model_base_log_prob: float,
    ) -> float:
        if source_total <= 0:
            return float(-model_base_log_prob)
        overlap = float(source_counts.multiply(model_correction).sum())
        avg_log_prob = model_base_log_prob + overlap / source_total
        return float(-avg_log_prob)

    def prepare_split(
        self,
        split: AuthBenchSplit,
        max_queries: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> tuple[AuthBenchSplit, PPMPreparedSplit, PPMPreparedSplit]:
        working = split.limited(max_queries=max_queries, max_candidates=max_candidates)
        query_texts = [record["content"] for record in working.queries]
        candidate_texts = [record["content"] for record in working.candidates]
        query_prepared = self._prepare_counts(self.vectorizer, query_texts, self.n_features, self.alpha)
        candidate_prepared = self._prepare_counts(
            self.vectorizer,
            candidate_texts,
            self.n_features,
            self.alpha,
        )
        return working, query_prepared, candidate_prepared

    def score_query(
        self,
        query_prepared: PPMPreparedSplit,
        candidate_prepared: PPMPreparedSplit,
        query_idx: int,
        candidate_indices: Sequence[int],
    ) -> np.ndarray:
        indices = np.asarray(candidate_indices, dtype=np.int32)
        query_counts = query_prepared.counts[query_idx]
        query_total = float(query_prepared.totals[query_idx])
        candidate_corrections = candidate_prepared.corrections[indices]
        candidate_counts = candidate_prepared.counts[indices]
        candidate_totals = candidate_prepared.totals[indices]
        candidate_base = candidate_prepared.base_log_probs[indices]

        forward = -(
            candidate_base
            + _as_float_array(query_counts.dot(candidate_corrections.T)) / max(query_total, 1.0)
        )

        query_base = float(query_prepared.base_log_probs[query_idx])
        query_correction = query_prepared.corrections[query_idx]
        query_dense = np.zeros(self.n_features, dtype=np.float32)
        if query_correction.nnz:
            query_dense[query_correction.indices] = query_correction.data
        backward = -(
            query_base
            + _as_float_array(candidate_counts.dot(query_dense)) / np.maximum(candidate_totals, 1.0)
        )

        features = np.column_stack(
            [
                0.5 * (forward + backward),
                np.abs(forward - backward),
            ]
        ).astype(np.float32, copy=False)
        return self.calibrator.decision_function(features).astype(np.float32, copy=False)

    def metadata(self) -> Dict[str, object]:
        return {
            "order": int(self.order),
            "hash_features": int(self.n_features),
            "alpha": float(self.alpha),
            "train_pair_count": int(self.train_pair_count),
            "pair_calibrator": self.calibrator.metadata(),
            "implementation_note": "Fixed-order hashed character language-model approximation of PPM-style scoring.",
        }


def evaluate_ngram_representation(
    split: AuthBenchSplit,
    model: NgramStylometricBaseline,
    ks: Sequence[int] = (1, 3, 5, 10),
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
) -> Dict[str, object]:
    working, data = model.transform_split(split, max_queries=max_queries, max_candidates=max_candidates)
    if candidate_pool == "all":
        weights, bias = model.calibrator.linear_parameters()

        def batch_score_fn(query_indices: Sequence[int]) -> np.ndarray:
            char_scores = data.query_char[query_indices].dot(data.candidate_char.T).toarray().astype(np.float32)
            word_scores = data.query_word[query_indices].dot(data.candidate_word.T).toarray().astype(np.float32)
            scores = np.empty_like(char_scores, dtype=np.float32)
            candidate_surface = data.candidate_surface
            for row, query_idx in enumerate(query_indices):
                surface = _surface_query_features(data.query_surface[query_idx], candidate_surface)
                scores[row] = (
                    bias
                    + weights[0] * char_scores[row]
                    + weights[1] * word_scores[row]
                    + surface @ weights[2:]
                )
            return scores

        result = _evaluate_representation_all_pool_batches(
            working,
            batch_score_fn=batch_score_fn,
            batch_size=32,
            ks=ks,
        )
    else:
        result = _evaluate_representation_from_scorer(
            working,
            lambda query_idx, candidate_indices: model.score_query(data, query_idx, candidate_indices),
            ks=ks,
            candidate_pool=candidate_pool,
            max_topic_candidates=max_topic_candidates,
            topic_seed=topic_seed,
        )
    result["baseline"] = model.metadata()
    return result


def evaluate_ngram_attribution(
    split: AuthBenchSplit,
    model: NgramStylometricBaseline,
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    negatives_per_query: int = 50,
    negative_strategy: str = "all",
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    seed: int = 13,
) -> Dict[str, object]:
    working, data = model.transform_split(split, max_queries=max_queries, max_candidates=max_candidates)
    result = _evaluate_attribution_from_scorer(
        working,
        lambda query_idx, candidate_indices: model.score_query(data, query_idx, candidate_indices),
        negatives_per_query=negatives_per_query,
        negative_strategy=negative_strategy,
        candidate_pool=candidate_pool,
        max_topic_candidates=max_topic_candidates,
        topic_seed=topic_seed,
        seed=seed,
    )
    result["baseline"] = model.metadata()
    return result


def evaluate_ppm_representation(
    split: AuthBenchSplit,
    model: PPMBaseline,
    ks: Sequence[int] = (1, 3, 5, 10),
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
) -> Dict[str, object]:
    working, query_prepared, candidate_prepared = model.prepare_split(
        split,
        max_queries=max_queries,
        max_candidates=max_candidates,
    )
    if candidate_pool == "all":
        weights, bias = model.calibrator.linear_parameters()
        candidate_base = candidate_prepared.base_log_probs
        candidate_counts = candidate_prepared.counts
        candidate_corrections = candidate_prepared.corrections
        candidate_totals = np.maximum(candidate_prepared.totals, 1.0)

        def batch_score_fn(query_indices: Sequence[int]) -> np.ndarray:
            query_counts = query_prepared.counts[query_indices]
            query_totals = np.maximum(query_prepared.totals[query_indices], 1.0)
            forward = -(
                candidate_base[None, :]
                + query_counts.dot(candidate_corrections.T).toarray().astype(np.float32)
                / query_totals[:, None]
            )

            backward = np.empty_like(forward, dtype=np.float32)
            for row, query_idx in enumerate(query_indices):
                query_base = float(query_prepared.base_log_probs[query_idx])
                query_correction = query_prepared.corrections[query_idx]
                query_dense = np.zeros(model.n_features, dtype=np.float32)
                if query_correction.nnz:
                    query_dense[query_correction.indices] = query_correction.data
                backward[row] = -(
                    query_base
                    + _as_float_array(candidate_counts.dot(query_dense)) / candidate_totals
                )

            mean_entropy = 0.5 * (forward + backward)
            gap_entropy = np.abs(forward - backward)
            return bias + weights[0] * mean_entropy + weights[1] * gap_entropy

        result = _evaluate_representation_all_pool_batches(
            working,
            batch_score_fn=batch_score_fn,
            batch_size=16,
            ks=ks,
        )
    else:
        result = _evaluate_representation_from_scorer(
            working,
            lambda query_idx, candidate_indices: model.score_query(
                query_prepared,
                candidate_prepared,
                query_idx,
                candidate_indices,
            ),
            ks=ks,
            candidate_pool=candidate_pool,
            max_topic_candidates=max_topic_candidates,
            topic_seed=topic_seed,
        )
    result["baseline"] = model.metadata()
    return result


def evaluate_ppm_attribution(
    split: AuthBenchSplit,
    model: PPMBaseline,
    max_queries: Optional[int] = None,
    max_candidates: Optional[int] = None,
    negatives_per_query: int = 50,
    negative_strategy: str = "all",
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    seed: int = 13,
) -> Dict[str, object]:
    working, query_prepared, candidate_prepared = model.prepare_split(
        split,
        max_queries=max_queries,
        max_candidates=max_candidates,
    )
    result = _evaluate_attribution_from_scorer(
        working,
        lambda query_idx, candidate_indices: model.score_query(
            query_prepared,
            candidate_prepared,
            query_idx,
            candidate_indices,
        ),
        negatives_per_query=negatives_per_query,
        negative_strategy=negative_strategy,
        candidate_pool=candidate_pool,
        max_topic_candidates=max_topic_candidates,
        topic_seed=topic_seed,
        seed=seed,
    )
    result["baseline"] = model.metadata()
    return result
