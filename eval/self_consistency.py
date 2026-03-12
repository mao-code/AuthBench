from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from AuthBench.eval.data import AuthBenchSplit
from AuthBench.eval.embedder import EmbeddingResult
from AuthBench.eval.evaluators import (
    _aggregate_grouped_eer,
    _aggregate_grouped_ranking_metrics,
    _candidate_pool_stats,
    _clean_label,
    _length_bucket_from_record,
)
from AuthBench.eval.hf_utils import load_causal_lm_model, load_tokenizer
from AuthBench.eval.metrics import (
    aggregate_ranking_metrics,
    compute_eer,
    ranking_metrics_for_query,
)
from AuthBench.eval.pools import build_topic_candidate_index, build_topic_pool


DEFAULT_STYLE_PROMPT_TEMPLATE = """You are extracting authorship style signals for authorship verification.
Read the document and write a short style profile that focuses on writing style rather than topic.
Describe lexical choice, syntax, tone, discourse habits, punctuation, formatting, and recurring rhetorical patterns when present.
Do not summarize the topic and do not copy long spans from the document.

Document:
{text}

Style profile:"""

DEFAULT_SELF_CONSISTENCY_RETRIEVAL_KS: Tuple[int, ...] = (1, 3, 5, 10)
# Backward-compatible alias for older imports.
DEFAULT_VOTED_RETRIEVAL_KS: Tuple[int, ...] = DEFAULT_SELF_CONSISTENCY_RETRIEVAL_KS


def _chunk_iterable(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _score_device(device: Optional[str]) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _move_to_score_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(device=device, dtype=torch.float32)
    return tensor.to(device)


def _validate_requested_ks(ks: Sequence[int]) -> Tuple[int, ...]:
    cleaned = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not cleaned:
        raise ValueError("At least one positive K value is required for self-consistency evaluation.")
    return cleaned


def _validate_sampled_embeddings(
    query_embeddings: "SampledEmbeddingResult",
    candidate_embeddings: "SampledEmbeddingResult",
    split: AuthBenchSplit,
) -> None:
    if query_embeddings.sample_vectors.ndim != 3 or candidate_embeddings.sample_vectors.ndim != 3:
        raise ValueError("Self-consistency sampled embeddings must be rank-3 tensors.")
    if query_embeddings.sample_vectors.size(0) != len(split.queries):
        raise ValueError(
            "Query embedding count does not match the split. "
            f"Expected {len(split.queries)}, got {query_embeddings.sample_vectors.size(0)}."
        )
    if candidate_embeddings.sample_vectors.size(0) != len(split.candidates):
        raise ValueError(
            "Candidate embedding count does not match the split. "
            f"Expected {len(split.candidates)}, got {candidate_embeddings.sample_vectors.size(0)}."
        )
    if query_embeddings.num_votes != candidate_embeddings.num_votes:
        raise ValueError(
            "Query and candidate sampled embeddings must expose the same number of samples. "
            f"Got {query_embeddings.num_votes} and {candidate_embeddings.num_votes}."
        )


def _aggregate_batch_scores(
    query_sample_vectors: torch.Tensor,
    candidate_sample_vectors: torch.Tensor,
    *,
    score_device: Optional[str] = None,
) -> torch.Tensor:
    device = _score_device(score_device)
    num_votes = int(query_sample_vectors.size(1))
    aggregated_scores = None

    for sample_idx in range(num_votes):
        query_vectors = _move_to_score_device(query_sample_vectors[:, sample_idx, :], device)
        candidate_vectors = _move_to_score_device(candidate_sample_vectors[:, sample_idx, :], device)
        sample_scores = torch.matmul(query_vectors, candidate_vectors.T)
        aggregated_scores = sample_scores if aggregated_scores is None else aggregated_scores + sample_scores

    if aggregated_scores is None:
        return torch.empty(
            (int(query_sample_vectors.size(0)), int(candidate_sample_vectors.size(0))),
            dtype=torch.float32,
        )
    return aggregated_scores.cpu()


def _aggregate_query_scores(
    query_sample_vectors: torch.Tensor,
    candidate_sample_vectors: torch.Tensor,
    *,
    score_device: Optional[str] = None,
) -> torch.Tensor:
    device = _score_device(score_device)
    num_votes = int(query_sample_vectors.size(0))
    aggregated_scores = None

    for sample_idx in range(num_votes):
        query_vector = _move_to_score_device(query_sample_vectors[sample_idx], device)
        candidate_vectors = _move_to_score_device(candidate_sample_vectors[:, sample_idx, :], device)
        sample_scores = torch.matmul(candidate_vectors, query_vector)
        aggregated_scores = sample_scores if aggregated_scores is None else aggregated_scores + sample_scores

    if aggregated_scores is None:
        return torch.empty(int(candidate_sample_vectors.size(0)), dtype=torch.float32)
    return aggregated_scores.cpu()


@dataclass(frozen=True)
class SelfConsistencyConfig:
    num_samples: int = 4
    top_k: int = 50
    temperature: float = 0.8
    max_new_tokens: int = 96
    prompt_template: str = DEFAULT_STYLE_PROMPT_TEMPLATE
    include_original: bool = False

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("self-consistency num_samples must be positive.")
        if self.top_k <= 0:
            raise ValueError("self-consistency top_k must be positive.")
        if self.temperature <= 0:
            raise ValueError("self-consistency temperature must be positive.")
        if self.max_new_tokens <= 0:
            raise ValueError("self-consistency max_new_tokens must be positive.")

    @property
    def total_votes(self) -> int:
        return self.num_samples + (1 if self.include_original else 0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampledEmbeddingResult:
    sample_vectors: torch.Tensor
    ids: Optional[List[str]] = None

    @property
    def num_votes(self) -> int:
        if self.sample_vectors.ndim < 2:
            return 0
        return int(self.sample_vectors.size(1))


class SelfConsistencyCausalLMEmbedder:
    """Sample multiple style embeddings per text and sum retrieval scores across samples."""

    def __init__(
        self,
        model_name_or_path: str,
        config: SelfConsistencyConfig,
        device: Optional[str] = None,
        max_length: int = 512,
        no_truncation: bool = False,
        pooling: str = "mean",
        normalize: bool = True,
        torch_dtype: Optional[str] = None,
        truncation_side: str = "right",
        trust_remote_code: bool = False,
        allow_remote_code_fallback: bool = True,
    ):
        if "{text}" not in config.prompt_template:
            raise ValueError("self-consistency prompt template must include a '{text}' placeholder.")

        self.model_name_or_path = model_name_or_path
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.no_truncation = no_truncation
        self.pooling = pooling
        self.normalize = normalize
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        self.tokenizer = load_tokenizer(
            model_name_or_path,
            truncation_side=truncation_side,
            trust_remote_code=trust_remote_code,
            allow_remote_code_fallback=allow_remote_code_fallback,
        )
        pad_added = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_added = True
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model = load_causal_lm_model(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
            allow_remote_code_fallback=allow_remote_code_fallback,
        )
        if pad_added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    @property
    def dimension(self) -> Optional[int]:
        return getattr(self.model.config, "hidden_size", None) or getattr(
            self.model.config, "d_model", None
        )

    def _apply_prefix(self, texts: Sequence[str], prefix: str) -> List[str]:
        if prefix:
            return [prefix + text for text in texts]
        return list(texts)

    def _resolve_model_max_length(self) -> Optional[int]:
        candidates: List[int] = []
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_max and tokenizer_max < 1_000_000:
            candidates.append(int(tokenizer_max))
        for attr in ("max_position_embeddings", "max_seq_len", "n_positions"):
            value = getattr(self.model.config, attr, None)
            if value:
                candidates.append(int(value))
        return min(candidates) if candidates else None

    def _pool_outputs(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            pooled = hidden_state[:, 0]
        elif self.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden_state[torch.arange(hidden_state.size(0), device=hidden_state.device), lengths]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        return pooled

    def _last_hidden_state(self, model_outputs) -> torch.Tensor:
        hidden_state = getattr(model_outputs, "last_hidden_state", None)
        if hidden_state is not None:
            return hidden_state
        hidden_states = getattr(model_outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise ValueError("Causal LM outputs did not expose hidden states for pooling.")

    def _tokenize(
        self,
        texts: Sequence[str],
        *,
        padding,
        max_length: Optional[int],
        truncation: bool,
    ) -> dict:
        tokens = self.tokenizer(
            list(texts),
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in tokens.items()}

    def _encode_direct(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress: bool,
        progress_desc: str,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        iterator = _chunk_iterable(list(texts), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=progress_desc,
                total=(len(texts) + batch_size - 1) // batch_size,
            )

        with torch.inference_mode():
            for batch in iterator:
                if self.no_truncation:
                    padding_strategy = True
                    max_length = self._resolve_model_max_length()
                    truncation = max_length is not None
                else:
                    padding_strategy = True
                    truncation = True
                    max_length = self.max_length
                tokens = self._tokenize(
                    batch,
                    padding=padding_strategy,
                    max_length=max_length,
                    truncation=truncation,
                )
                model_outputs = self.model(**tokens, output_hidden_states=True, return_dict=True)
                hidden_state = self._last_hidden_state(model_outputs)
                pooled = self._pool_outputs(hidden_state, tokens["attention_mask"])
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu())

        if not outputs:
            dim = self.dimension or 0
            return torch.empty((0, dim), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def _generate_style_summaries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        prefix: str,
        show_progress: bool,
    ) -> List[str]:
        summaries: List[str] = []
        source_texts = self._apply_prefix(texts, prefix)
        iterator = _chunk_iterable(source_texts, batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Style sampling",
                total=(len(source_texts) + batch_size - 1) // batch_size,
            )

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            with torch.inference_mode():
                for batch in iterator:
                    prompts = [self.config.prompt_template.format(text=text) for text in batch]
                    if self.no_truncation:
                        max_length = self._resolve_model_max_length()
                        truncation = max_length is not None
                    else:
                        max_length = self.max_length
                        truncation = True
                    tokens = self._tokenize(
                        prompts,
                        padding=True,
                        max_length=max_length,
                        truncation=truncation,
                    )
                    input_length = tokens["input_ids"].size(1)
                    generated = self.model.generate(
                        **tokens,
                        do_sample=True,
                        top_k=self.config.top_k,
                        temperature=self.config.temperature,
                        max_new_tokens=self.config.max_new_tokens,
                        num_return_sequences=self.config.num_samples,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    generated_tokens = generated[:, input_length:]
                    decoded = self.tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True,
                    )
                    for text in decoded:
                        cleaned = text.strip()
                        summaries.append(cleaned or "style profile unavailable")
        finally:
            self.tokenizer.padding_side = original_padding_side

        expected = len(texts) * self.config.num_samples
        if len(summaries) != expected:
            raise RuntimeError(
                f"Expected {expected} generated style summaries, but received {len(summaries)}."
            )
        return summaries

    def encode_texts_samples(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        prefix: str = "",
        show_progress: bool = False,
    ) -> SampledEmbeddingResult:
        texts_list = list(texts)
        was_training = self.model.training
        self.model.eval()
        try:
            if not texts_list:
                dim = self.dimension or 0
                return SampledEmbeddingResult(
                    sample_vectors=torch.empty(
                        (0, self.config.total_votes, dim),
                        dtype=torch.float32,
                    )
                )

            sampled_summaries = self._generate_style_summaries(
                texts_list,
                batch_size=batch_size,
                prefix=prefix,
                show_progress=show_progress,
            )
            sampled_vectors = self._encode_direct(
                sampled_summaries,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_desc="Style embedding",
            ).view(len(texts_list), self.config.num_samples, -1)

            if self.config.include_original:
                original_vectors = self._encode_direct(
                    self._apply_prefix(texts_list, prefix),
                    batch_size=batch_size,
                    show_progress=False,
                    progress_desc="Embedding",
                ).unsqueeze(1)
                sampled_vectors = torch.cat([sampled_vectors, original_vectors], dim=1)

            return SampledEmbeddingResult(sample_vectors=sampled_vectors.cpu())
        finally:
            if was_training:
                self.model.train()

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        prefix: str = "",
        return_tokens: bool = False,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        # Kept for compatibility; the evaluation path uses `encode_texts_samples()`
        # and sums per-sample query-candidate similarities before reranking.
        if return_tokens:
            raise ValueError(
                "Self-consistency embeddings only expose pooled vectors. Disable late interaction."
            )

        sampled = self.encode_texts_samples(
            texts,
            batch_size=batch_size,
            prefix=prefix,
            show_progress=show_progress,
        )
        vectors = sampled.sample_vectors.mean(dim=1)
        if self.normalize:
            vectors = F.normalize(vectors, p=2, dim=1)
        return EmbeddingResult(vectors=vectors.cpu())


def encode_self_consistency_split_embeddings(
    split: AuthBenchSplit,
    embedder: SelfConsistencyCausalLMEmbedder,
    *,
    batch_size: int = 32,
    query_prefix: str = "",
    doc_prefix: str = "",
) -> Tuple[SampledEmbeddingResult, SampledEmbeddingResult]:
    candidate_embeddings = embedder.encode_texts_samples(
        [candidate["content"] for candidate in split.candidates],
        batch_size=batch_size,
        prefix=doc_prefix,
        show_progress=True,
    )
    query_embeddings = embedder.encode_texts_samples(
        [query["content"] for query in split.queries],
        batch_size=batch_size,
        prefix=query_prefix,
        show_progress=True,
    )
    return query_embeddings, candidate_embeddings


def evaluate_self_consistency_representation(
    split: AuthBenchSplit,
    query_embeddings: SampledEmbeddingResult,
    candidate_embeddings: SampledEmbeddingResult,
    *,
    batch_size: int = 32,
    ks: Sequence[int] = DEFAULT_SELF_CONSISTENCY_RETRIEVAL_KS,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    score_device: Optional[str] = None,
) -> Dict[str, object]:
    _validate_sampled_embeddings(query_embeddings, candidate_embeddings, split)
    ks = _validate_requested_ks(ks)

    candidate_ids = [candidate["candidate_id"] for candidate in split.candidates]
    candidate_index = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}

    metrics_per_query: List[Dict[str, float]] = []
    per_lang: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_genre: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    per_length: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    candidate_counts: List[int] = []
    per_lang_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_genre_counts: DefaultDict[str, List[int]] = defaultdict(list)
    per_length_counts: DefaultDict[str, List[int]] = defaultdict(list)

    if candidate_pool not in ("all", "topic"):
        raise ValueError(f"Unknown candidate_pool: {candidate_pool}")

    if candidate_pool == "all":
        for start in range(0, len(split.queries), batch_size):
            end = min(start + batch_size, len(split.queries))
            aggregated_scores = _aggregate_batch_scores(
                query_embeddings.sample_vectors[start:end],
                candidate_embeddings.sample_vectors,
                score_device=score_device,
            )
            for row, query_idx in enumerate(range(start, end)):
                query_record = split.queries[query_idx]
                query_id = query_record["query_id"]
                positive_indices = [
                    candidate_index[candidate_id]
                    for candidate_id in split.positives_by_query.get(query_id, [])
                    if candidate_id in candidate_index
                ]
                if not positive_indices:
                    continue

                metrics = ranking_metrics_for_query(
                    aggregated_scores[row],
                    positive_indices,
                    ks=ks,
                )
                metrics_per_query.append(metrics)
                lang = _clean_label(query_record.get("lang") or query_record.get("language"))
                genre = _clean_label(query_record.get("genre"))
                length_bucket = _length_bucket_from_record(query_record)
                per_lang[lang].append(metrics)
                per_genre[genre].append(metrics)
                per_length[length_bucket].append(metrics)
    else:
        topic_candidates = build_topic_candidate_index(split.candidates)
        for query_idx, query_record in enumerate(split.queries):
            query_id = query_record["query_id"]
            positive_indices = [
                candidate_index[candidate_id]
                for candidate_id in split.positives_by_query.get(query_id, [])
                if candidate_id in candidate_index
            ]
            if not positive_indices:
                continue

            pool_indices = build_topic_pool(
                query_record=query_record,
                query_id=query_id,
                candidate_ids=candidate_ids,
                candidate_indices_by_topic=topic_candidates,
                positive_indices=positive_indices,
                max_candidates=max_topic_candidates,
                seed=topic_seed,
            )
            if not pool_indices:
                continue

            pool_index = {candidate_idx: local_idx for local_idx, candidate_idx in enumerate(pool_indices)}
            pool_positive_indices = [
                pool_index[candidate_idx]
                for candidate_idx in positive_indices
                if candidate_idx in pool_index
            ]
            if not pool_positive_indices:
                continue

            aggregated_scores = _aggregate_query_scores(
                query_embeddings.sample_vectors[query_idx],
                candidate_embeddings.sample_vectors[pool_indices],
                score_device=score_device,
            )
            metrics = ranking_metrics_for_query(
                aggregated_scores,
                pool_positive_indices,
                ks=ks,
            )
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
    aggregated["ranking_aggregation"] = "sum_sample_scores_rerank"
    aggregated["num_votes"] = query_embeddings.num_votes
    if candidate_pool == "all":
        aggregated["num_candidates"] = len(candidate_ids)
        aggregated["by_language"] = _aggregate_grouped_ranking_metrics(per_lang, len(candidate_ids))
        aggregated["by_genre"] = _aggregate_grouped_ranking_metrics(per_genre, len(candidate_ids))
        aggregated["by_length_bucket"] = _aggregate_grouped_ranking_metrics(
            per_length,
            len(candidate_ids),
        )
    else:
        aggregated.update(_candidate_pool_stats(candidate_counts))
        aggregated["by_language"] = _aggregate_grouped_ranking_metrics(
            per_lang,
            len(candidate_ids),
            per_lang_counts,
        )
        aggregated["by_genre"] = _aggregate_grouped_ranking_metrics(
            per_genre,
            len(candidate_ids),
            per_genre_counts,
        )
        aggregated["by_length_bucket"] = _aggregate_grouped_ranking_metrics(
            per_length,
            len(candidate_ids),
            per_length_counts,
        )
    return aggregated


def evaluate_self_consistency_attribution(
    split: AuthBenchSplit,
    query_embeddings: SampledEmbeddingResult,
    candidate_embeddings: SampledEmbeddingResult,
    *,
    batch_size: int = 32,
    negatives_per_query: int = 50,
    negative_strategy: str = "sample",
    seed: int = 13,
    candidate_pool: str = "all",
    max_topic_candidates: Optional[int] = None,
    topic_seed: int = 13,
    score_device: Optional[str] = None,
) -> Dict[str, object]:
    _validate_sampled_embeddings(query_embeddings, candidate_embeddings, split)
    rng = random.Random(seed)

    candidate_ids = [candidate["candidate_id"] for candidate in split.candidates]
    candidate_index = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}

    if candidate_pool not in ("all", "topic"):
        raise ValueError(f"Unknown candidate_pool: {candidate_pool}")

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

    if candidate_pool == "all":
        for start in range(0, len(split.queries), batch_size):
            end = min(start + batch_size, len(split.queries))
            aggregated_scores = _aggregate_batch_scores(
                query_embeddings.sample_vectors[start:end],
                candidate_embeddings.sample_vectors,
                score_device=score_device,
            )
            for row, query_idx in enumerate(range(start, end)):
                query_record = split.queries[query_idx]
                query_id = query_record["query_id"]
                positive_indices = [
                    candidate_index[candidate_id]
                    for candidate_id in split.positives_by_query.get(query_id, [])
                    if candidate_id in candidate_index
                ]
                if not positive_indices:
                    continue

                scores = aggregated_scores[row]
                query_counter += 1
                pos_vals = scores[positive_indices].tolist()
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

                negative_pool = [idx for idx in range(len(candidate_ids)) if idx not in positive_indices]
                if negative_strategy == "all":
                    chosen = negative_pool
                else:
                    if negatives_per_query is None or negatives_per_query >= len(negative_pool):
                        chosen = negative_pool
                    else:
                        chosen = rng.sample(negative_pool, negatives_per_query)
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
        topic_candidates = build_topic_candidate_index(split.candidates)
        for query_idx, query_record in enumerate(split.queries):
            query_id = query_record["query_id"]
            positive_indices = [
                candidate_index[candidate_id]
                for candidate_id in split.positives_by_query.get(query_id, [])
                if candidate_id in candidate_index
            ]
            if not positive_indices:
                continue

            pool_indices = build_topic_pool(
                query_record=query_record,
                query_id=query_id,
                candidate_ids=candidate_ids,
                candidate_indices_by_topic=topic_candidates,
                positive_indices=positive_indices,
                max_candidates=max_topic_candidates,
                seed=topic_seed,
            )
            if not pool_indices:
                continue

            pool_index = {candidate_idx: local_idx for local_idx, candidate_idx in enumerate(pool_indices)}
            pool_positive_indices = [
                pool_index[candidate_idx]
                for candidate_idx in positive_indices
                if candidate_idx in pool_index
            ]
            if not pool_positive_indices:
                continue

            scores = _aggregate_query_scores(
                query_embeddings.sample_vectors[query_idx],
                candidate_embeddings.sample_vectors[pool_indices],
                score_device=score_device,
            )

            query_counter += 1
            pos_vals = scores[pool_positive_indices].tolist()
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

            negative_pool = [idx for idx in range(len(pool_indices)) if idx not in pool_positive_indices]
            if negative_strategy == "all":
                chosen = negative_pool
            else:
                if negatives_per_query is None or negatives_per_query >= len(negative_pool):
                    chosen = negative_pool
                else:
                    chosen = rng.sample(negative_pool, negatives_per_query)
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

    result = {
        "eer": compute_eer(positive_scores, negative_scores),
        "num_queries": query_counter,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "negatives_per_query": negatives_per_query,
        "negative_strategy": negative_strategy,
        "score_aggregation": "sum_sample_scores",
        "num_votes": query_embeddings.num_votes,
    }

    if candidate_pool == "all":
        result["num_candidates"] = len(candidate_ids)
        result["by_language"] = _aggregate_grouped_eer(
            pos_by_lang,
            neg_by_lang,
            query_count_by_lang,
            pos_pairs_by_lang,
            neg_pairs_by_lang,
        )
        result["by_genre"] = _aggregate_grouped_eer(
            pos_by_genre,
            neg_by_genre,
            query_count_by_genre,
            pos_pairs_by_genre,
            neg_pairs_by_genre,
        )
        result["by_length_bucket"] = _aggregate_grouped_eer(
            pos_by_length,
            neg_by_length,
            query_count_by_length,
            pos_pairs_by_length,
            neg_pairs_by_length,
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
