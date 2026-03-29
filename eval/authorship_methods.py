from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from AuthBench.eval.data import AuthBenchSplit
from AuthBench.eval.embedder import EmbeddingResult


AUTHORSHIP_METHODS = ("standard", "part", "luar", "stel")


def pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        pooled = hidden_states[:, 0]
    elif pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), lengths]
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")
    return F.normalize(pooled, p=2, dim=1)


def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def symmetric_infonce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = torch.matmul(F.normalize(a, dim=1), F.normalize(b, dim=1).T) / max(temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def cosine_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    distance = lambda x, y: 1 - F.cosine_similarity(x, y)
    return F.triplet_margin_with_distance_loss(
        anchor,
        positive,
        negative,
        distance_function=distance,
        margin=margin,
        reduction="mean",
    )


def _safe_text(value: object) -> str:
    return str(value or "")


@dataclass
class AuthorTextExample:
    author_id: str
    doc_id: str
    text: str
    genre: str = ""
    source: str = ""
    lang: str = ""
    token_length: int = 0


@dataclass
class AuthorshipMethodArtifacts:
    method: str
    pooling: str
    part_hidden_size: int
    luar_embedding_size: int
    luar_window_size: int
    luar_max_episode_docs: int
    luar_eval_episode_docs: int
    luar_temperature: float
    stel_margin: float
    stel_control_keys: List[str]


def build_author_text_map(split: AuthBenchSplit) -> Dict[str, List[AuthorTextExample]]:
    author_to_examples: Dict[str, List[AuthorTextExample]] = defaultdict(list)

    for record in split.queries:
        author_id = split.author_by_query.get(record["query_id"])
        if not author_id:
            continue
        author_to_examples[author_id].append(
            AuthorTextExample(
                author_id=author_id,
                doc_id=record["query_id"],
                text=_safe_text(record.get("content")),
                genre=_safe_text(record.get("genre")),
                source=_safe_text(record.get("source")),
                lang=_safe_text(record.get("lang") or record.get("language")),
                token_length=int(record.get("token_length") or 0),
            )
        )

    for record in split.candidates:
        author_id = record.get("author_id")
        if not author_id:
            continue
        author_to_examples[author_id].append(
            AuthorTextExample(
                author_id=str(author_id),
                doc_id=record["candidate_id"],
                text=_safe_text(record.get("content")),
                genre=_safe_text(record.get("genre")),
                source=_safe_text(record.get("source")),
                lang=_safe_text(record.get("lang") or record.get("language")),
                token_length=int(record.get("token_length") or 0),
            )
        )

    return dict(author_to_examples)


class AuthorDataset(Dataset):
    def __init__(
        self,
        author_to_examples: Dict[str, List[AuthorTextExample]],
        *,
        min_docs: int = 2,
        max_authors: Optional[int] = None,
        seed: int = 13,
    ):
        author_ids = [author_id for author_id, docs in author_to_examples.items() if len(docs) >= min_docs]
        if max_authors is not None and max_authors < len(author_ids):
            rng = random.Random(seed)
            author_ids = rng.sample(author_ids, max_authors)
        self.author_ids = author_ids

    def __len__(self) -> int:
        return len(self.author_ids)

    def __getitem__(self, idx: int) -> str:
        return self.author_ids[idx]


def _sample_two_examples(examples: Sequence[AuthorTextExample]) -> Tuple[AuthorTextExample, AuthorTextExample]:
    if len(examples) >= 2:
        return tuple(random.sample(list(examples), 2))  # type: ignore[return-value]
    only = examples[0]
    return only, only


def _tokenize_texts(
    tokenizer,
    texts: Sequence[str],
    *,
    prefix: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        [prefix + text for text in texts],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


class PartBatchCollator:
    def __init__(self, author_to_examples, tokenizer, max_length: int, query_prefix: str, doc_prefix: str):
        self.author_to_examples = author_to_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def __call__(self, author_ids: Sequence[str]):
        anchors: List[str] = []
        positives: List[str] = []
        for author_id in author_ids:
            anchor, positive = _sample_two_examples(self.author_to_examples[author_id])
            anchors.append(anchor.text)
            positives.append(positive.text)
        return (
            _tokenize_texts(
                self.tokenizer,
                anchors,
                prefix=self.query_prefix,
                max_length=self.max_length,
            ),
            _tokenize_texts(
                self.tokenizer,
                positives,
                prefix=self.doc_prefix,
                max_length=self.max_length,
            ),
        )


def _sample_episode_size(max_episode_docs: int) -> int:
    if max_episode_docs <= 1:
        return 1
    # LUAR samples the number of documents with a Beta(3, 1) bias toward larger episodes.
    value = random.betavariate(3.0, 1.0)
    return max(1, min(max_episode_docs, math.ceil(1 + (max_episode_docs - 1) * value)))


def _prepare_window_from_token_ids(tokenizer, token_ids: List[int], window_size: int) -> Tuple[List[int], List[int]]:
    if not token_ids:
        filler_id = tokenizer.unk_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        token_ids = [filler_id]
    prepared = tokenizer.prepare_for_model(
        token_ids,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=window_size,
        return_attention_mask=True,
    )
    return prepared["input_ids"], prepared["attention_mask"]


def _token_window_ids(tokenizer, text: str, window_size: int) -> Tuple[List[int], List[int]]:
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(token_ids) > window_size:
        start = random.randint(0, len(token_ids) - window_size)
        token_ids = token_ids[start : start + window_size]
    return _prepare_window_from_token_ids(tokenizer, token_ids, window_size)


def _sample_examples_with_replacement(
    examples: Sequence[AuthorTextExample],
    count: int,
) -> List[AuthorTextExample]:
    if len(examples) >= count:
        return random.sample(list(examples), count)
    return random.choices(list(examples), k=count)


class LuarBatchCollator:
    def __init__(
        self,
        author_to_examples,
        tokenizer,
        *,
        window_size: int,
        max_episode_docs: int,
    ):
        self.author_to_examples = author_to_examples
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_episode_docs = max_episode_docs

    def __call__(self, author_ids: Sequence[str]):
        episode_size = _sample_episode_size(self.max_episode_docs)
        anchor_ids: List[List[List[int]]] = []
        anchor_masks: List[List[List[int]]] = []
        positive_ids: List[List[List[int]]] = []
        positive_masks: List[List[List[int]]] = []

        for author_id in author_ids:
            docs = self.author_to_examples[author_id]
            anchor_docs = _sample_examples_with_replacement(docs, episode_size)
            positive_docs = _sample_examples_with_replacement(docs, episode_size)

            episode_anchor_ids: List[List[int]] = []
            episode_anchor_masks: List[List[int]] = []
            episode_positive_ids: List[List[int]] = []
            episode_positive_masks: List[List[int]] = []

            for example in anchor_docs:
                ids, mask = _token_window_ids(self.tokenizer, example.text, self.window_size)
                episode_anchor_ids.append(ids)
                episode_anchor_masks.append(mask)
            for example in positive_docs:
                ids, mask = _token_window_ids(self.tokenizer, example.text, self.window_size)
                episode_positive_ids.append(ids)
                episode_positive_masks.append(mask)

            anchor_ids.append(episode_anchor_ids)
            anchor_masks.append(episode_anchor_masks)
            positive_ids.append(episode_positive_ids)
            positive_masks.append(episode_positive_masks)

        anchor_inputs = {
            "input_ids": torch.tensor(anchor_ids, dtype=torch.long),
            "attention_mask": torch.tensor(anchor_masks, dtype=torch.long),
        }
        positive_inputs = {
            "input_ids": torch.tensor(positive_ids, dtype=torch.long),
            "attention_mask": torch.tensor(positive_masks, dtype=torch.long),
        }
        return anchor_inputs, positive_inputs


def _control_value(example: AuthorTextExample, keys: Sequence[str]) -> Tuple[str, ...]:
    return tuple(_safe_text(getattr(example, key, "")) for key in keys)


class StelTripletDataset(Dataset):
    def __init__(
        self,
        author_to_examples: Dict[str, List[AuthorTextExample]],
        *,
        control_keys: Sequence[str],
        max_authors: Optional[int] = None,
        seed: int = 13,
    ):
        self.author_to_examples = {
            author_id: docs for author_id, docs in author_to_examples.items() if len(docs) >= 2
        }
        author_ids = list(self.author_to_examples.keys())
        if max_authors is not None and max_authors < len(author_ids):
            rng = random.Random(seed)
            author_ids = rng.sample(author_ids, max_authors)
            self.author_to_examples = {author_id: self.author_to_examples[author_id] for author_id in author_ids}
        self.author_ids = author_ids
        self.control_keys = list(control_keys)
        self.all_examples = [example for docs in self.author_to_examples.values() for example in docs]

        self.control_priority: List[Tuple[str, ...]] = []
        if len(self.control_keys) >= 2:
            self.control_priority.append(tuple(self.control_keys))
        for key in self.control_keys:
            self.control_priority.append((key,))

        self.control_pools: Dict[Tuple[str, ...], Dict[Tuple[str, ...], List[AuthorTextExample]]] = {}
        for key_group in self.control_priority:
            pool: Dict[Tuple[str, ...], List[AuthorTextExample]] = defaultdict(list)
            for example in self.all_examples:
                value = _control_value(example, key_group)
                if any(value):
                    pool[value].append(example)
            self.control_pools[key_group] = dict(pool)

    def __len__(self) -> int:
        return len(self.author_ids)

    def _sample_negative(self, anchor: AuthorTextExample) -> AuthorTextExample:
        for key_group in self.control_priority:
            value = _control_value(anchor, key_group)
            if not any(value):
                continue
            pool = [
                candidate
                for candidate in self.control_pools.get(key_group, {}).get(value, [])
                if candidate.author_id != anchor.author_id
            ]
            if pool:
                return random.choice(pool)

        while True:
            candidate = random.choice(self.all_examples)
            if candidate.author_id != anchor.author_id:
                return candidate

    def __getitem__(self, idx: int) -> Dict[str, str]:
        author_id = self.author_ids[idx]
        anchor, positive = _sample_two_examples(self.author_to_examples[author_id])
        negative = self._sample_negative(anchor)
        return {
            "anchor_text": anchor.text,
            "positive_text": positive.text,
            "negative_text": negative.text,
        }


class StelTripletCollator:
    def __init__(self, tokenizer, max_length: int, query_prefix: str, doc_prefix: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def __call__(self, batch: Sequence[Dict[str, str]]):
        return (
            _tokenize_texts(
                self.tokenizer,
                [item["anchor_text"] for item in batch],
                prefix=self.query_prefix,
                max_length=self.max_length,
            ),
            _tokenize_texts(
                self.tokenizer,
                [item["positive_text"] for item in batch],
                prefix=self.doc_prefix,
                max_length=self.max_length,
            ),
            _tokenize_texts(
                self.tokenizer,
                [item["negative_text"] for item in batch],
                prefix=self.doc_prefix,
                max_length=self.max_length,
            ),
        )


class DynamicLSTMHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lens = attention_mask.sum(-1).clamp_min(1)
        batch_size, seq_len, _ = hidden_states.shape

        seq_lens_sorted, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        hidden_sorted = hidden_states.index_select(0, idx_sort)
        packed = pack_padded_sequence(hidden_sorted, seq_lens_sorted.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
        unpacked = unpacked.index_select(0, idx_unsort)

        split = unpacked.view(batch_size, seq_len, 2, self.hidden_size)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_indices = (seq_lens - 1).to(hidden_states.device)
        return torch.cat([split[batch_indices, last_indices, 0], split[batch_indices, 0, 1]], dim=-1)


class LuarSelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, x)


class AuthorshipTrainingModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        method: str,
        pooling: str,
        part_hidden_size: int,
        luar_embedding_size: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.method = method
        self.pooling = pooling
        self.hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "d_model")
        if self.hidden_size is None:
            raise ValueError("Could not determine hidden size from model config.")

        self.part_head: Optional[nn.Module] = None
        self.luar_attention: Optional[nn.Module] = None
        self.luar_projection: Optional[nn.Module] = None

        if method == "part":
            hidden = max(1, part_hidden_size)
            self.part_head = DynamicLSTMHead(self.hidden_size, hidden)
            self.output_dim = hidden * 2
        elif method == "luar":
            self.luar_attention = LuarSelfAttention(self.hidden_size)
            self.luar_projection = nn.Linear(self.hidden_size, luar_embedding_size)
            self.output_dim = luar_embedding_size
        else:
            self.output_dim = self.hidden_size

    def encode_documents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.method == "part":
            if self.part_head is None:
                raise RuntimeError("PART head is not initialized.")
            pooled = self.part_head(hidden_states, attention_mask)
            return F.normalize(pooled, p=2, dim=1)
        return pool_hidden_states(hidden_states, attention_mask, self.pooling)

    def encode_episodes(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.method != "luar":
            raise ValueError(f"Episode encoding is only supported for LUAR, not {self.method}.")
        if self.luar_attention is None or self.luar_projection is None:
            raise RuntimeError("LUAR modules are not initialized.")

        batch_size, episode_size, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * episode_size, seq_len)
        flat_mask = attention_mask.reshape(batch_size * episode_size, seq_len)
        outputs = self.base_model(input_ids=flat_ids, attention_mask=flat_mask)
        doc_embeddings = mean_pool_hidden_states(outputs.last_hidden_state, flat_mask)
        doc_embeddings = doc_embeddings.reshape(batch_size, episode_size, -1)
        attended = self.luar_attention(doc_embeddings)
        pooled = attended.max(dim=1).values
        projected = self.luar_projection(pooled)
        return F.normalize(projected, p=2, dim=1)

    def export_artifacts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        payload: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.part_head is not None:
            payload["part_head"] = self.part_head.state_dict()
        if self.luar_attention is not None:
            payload["luar_attention"] = self.luar_attention.state_dict()
        if self.luar_projection is not None:
            payload["luar_projection"] = self.luar_projection.state_dict()
        return payload


class AuthorshipMethodEmbedder:
    def __init__(
        self,
        model: AuthorshipTrainingModel,
        tokenizer,
        *,
        method: str,
        device: torch.device,
        max_length: int,
        query_prefix: str,
        doc_prefix: str,
        luar_window_size: int,
        luar_eval_episode_docs: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.device = device
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.luar_window_size = luar_window_size
        self.luar_eval_episode_docs = max(1, luar_eval_episode_docs)

    def _apply_prefix(self, texts: Sequence[str], prefix: str) -> List[str]:
        if prefix:
            return [prefix + text for text in texts]
        return list(texts)

    def _encode_simple_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        prefix: str,
        show_progress: bool,
    ) -> EmbeddingResult:
        texts = self._apply_prefix(texts, prefix)
        vectors: List[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()

        iterator: Iterable[List[str]] = (
            texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
        )
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc="Embedding",
            )

        with torch.inference_mode():
            for batch in iterator:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                embeddings = self.model.encode_documents(inputs["input_ids"], inputs["attention_mask"])
                vectors.append(embeddings.cpu())

        if was_training:
            self.model.train()

        return EmbeddingResult(vectors=torch.cat(vectors, dim=0) if vectors else torch.empty(0))

    def _build_eval_episode(self, text: str) -> Tuple[List[List[int]], List[List[int]]]:
        token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not token_ids:
            token_ids = [self.tokenizer.unk_token_id or self.tokenizer.pad_token_id or 0]
        chunks = [token_ids[start : start + self.luar_window_size] for start in range(0, len(token_ids), self.luar_window_size)]
        if not chunks:
            chunks = [token_ids]
        if len(chunks) >= self.luar_eval_episode_docs:
            if len(chunks) == self.luar_eval_episode_docs:
                selected = chunks
            else:
                indices = torch.linspace(0, len(chunks) - 1, steps=self.luar_eval_episode_docs)
                selected = [chunks[int(round(index.item()))] for index in indices]
        else:
            selected = list(chunks)
            while len(selected) < self.luar_eval_episode_docs:
                selected.append(chunks[-1])

        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        for chunk in selected:
            ids, mask = _prepare_window_from_token_ids(self.tokenizer, chunk, self.luar_window_size)
            input_ids.append(ids)
            attention_masks.append(mask)
        return input_ids, attention_masks

    def _encode_luar_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        prefix: str,
        show_progress: bool,
    ) -> EmbeddingResult:
        texts = self._apply_prefix(texts, prefix)
        vectors: List[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()

        iterator: Iterable[List[str]] = (
            texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
        )
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc="Embedding",
            )

        with torch.inference_mode():
            for batch in iterator:
                batch_ids: List[List[List[int]]] = []
                batch_masks: List[List[List[int]]] = []
                for text in batch:
                    input_ids, attention_masks = self._build_eval_episode(text)
                    batch_ids.append(input_ids)
                    batch_masks.append(attention_masks)
                inputs = {
                    "input_ids": torch.tensor(batch_ids, dtype=torch.long, device=self.device),
                    "attention_mask": torch.tensor(batch_masks, dtype=torch.long, device=self.device),
                }
                embeddings = self.model.encode_episodes(inputs["input_ids"], inputs["attention_mask"])
                vectors.append(embeddings.cpu())

        if was_training:
            self.model.train()

        return EmbeddingResult(vectors=torch.cat(vectors, dim=0) if vectors else torch.empty(0))

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        prefix: str = "",
        return_tokens: bool = False,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        if return_tokens:
            raise ValueError(f"Token-level outputs are not supported for authorship method '{self.method}'.")
        if self.method == "luar":
            return self._encode_luar_texts(
                texts,
                batch_size=batch_size,
                prefix=prefix,
                show_progress=show_progress,
            )
        return self._encode_simple_texts(
            texts,
            batch_size=batch_size,
            prefix=prefix,
            show_progress=show_progress,
        )


def build_authorship_training_components(
    split: AuthBenchSplit,
    tokenizer,
    args,
):
    author_to_examples = build_author_text_map(split)
    max_train_authors = args.max_train_authors

    if args.authorship_method == "part":
        dataset = AuthorDataset(
            author_to_examples,
            min_docs=2,
            max_authors=max_train_authors,
            seed=args.seed,
        )
        collate = PartBatchCollator(
            author_to_examples,
            tokenizer,
            max_length=args.max_length,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
        )
        return dataset, collate

    if args.authorship_method == "luar":
        dataset = AuthorDataset(
            author_to_examples,
            min_docs=2,
            max_authors=max_train_authors,
            seed=args.seed,
        )
        collate = LuarBatchCollator(
            author_to_examples,
            tokenizer,
            window_size=args.luar_window_size,
            max_episode_docs=args.luar_max_episode_docs,
        )
        return dataset, collate

    if args.authorship_method == "stel":
        dataset = StelTripletDataset(
            author_to_examples,
            control_keys=args.stel_control_keys,
            max_authors=max_train_authors,
            seed=args.seed,
        )
        collate = StelTripletCollator(
            tokenizer,
            max_length=args.max_length,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
        )
        return dataset, collate

    raise ValueError(f"Unsupported authorship method: {args.authorship_method}")


def compute_authorship_method_loss(
    model: AuthorshipTrainingModel,
    batch,
    args,
    device: torch.device,
) -> torch.Tensor:
    if args.authorship_method == "part":
        anchor_inputs, positive_inputs = batch
        anchor_inputs = {key: value.to(device) for key, value in anchor_inputs.items()}
        positive_inputs = {key: value.to(device) for key, value in positive_inputs.items()}
        anchor_embeddings = model.encode_documents(anchor_inputs["input_ids"], anchor_inputs["attention_mask"])
        positive_embeddings = model.encode_documents(positive_inputs["input_ids"], positive_inputs["attention_mask"])
        return symmetric_infonce_loss(anchor_embeddings, positive_embeddings, args.temperature)

    if args.authorship_method == "luar":
        anchor_inputs, positive_inputs = batch
        anchor_inputs = {key: value.to(device) for key, value in anchor_inputs.items()}
        positive_inputs = {key: value.to(device) for key, value in positive_inputs.items()}
        anchor_embeddings = model.encode_episodes(anchor_inputs["input_ids"], anchor_inputs["attention_mask"])
        positive_embeddings = model.encode_episodes(positive_inputs["input_ids"], positive_inputs["attention_mask"])
        return symmetric_infonce_loss(anchor_embeddings, positive_embeddings, args.luar_temperature)

    if args.authorship_method == "stel":
        anchor_inputs, positive_inputs, negative_inputs = batch
        anchor_inputs = {key: value.to(device) for key, value in anchor_inputs.items()}
        positive_inputs = {key: value.to(device) for key, value in positive_inputs.items()}
        negative_inputs = {key: value.to(device) for key, value in negative_inputs.items()}
        anchor_embeddings = model.encode_documents(anchor_inputs["input_ids"], anchor_inputs["attention_mask"])
        positive_embeddings = model.encode_documents(positive_inputs["input_ids"], positive_inputs["attention_mask"])
        negative_embeddings = model.encode_documents(negative_inputs["input_ids"], negative_inputs["attention_mask"])
        return cosine_triplet_margin_loss(
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
            args.stel_margin,
        )

    raise ValueError(f"Unsupported authorship method: {args.authorship_method}")


def build_authorship_eval_embedder(
    model: AuthorshipTrainingModel,
    tokenizer,
    args,
    device: torch.device,
) -> AuthorshipMethodEmbedder:
    return AuthorshipMethodEmbedder(
        model,
        tokenizer,
        method=args.authorship_method,
        device=device,
        max_length=args.max_length,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        luar_window_size=args.luar_window_size,
        luar_eval_episode_docs=args.luar_eval_episode_docs,
    )


def build_method_artifacts(args) -> AuthorshipMethodArtifacts:
    return AuthorshipMethodArtifacts(
        method=args.authorship_method,
        pooling=args.pooling,
        part_hidden_size=args.part_hidden_size,
        luar_embedding_size=args.luar_embedding_size,
        luar_window_size=args.luar_window_size,
        luar_max_episode_docs=args.luar_max_episode_docs,
        luar_eval_episode_docs=args.luar_eval_episode_docs,
        luar_temperature=args.luar_temperature,
        stel_margin=args.stel_margin,
        stel_control_keys=list(args.stel_control_keys),
    )


def save_authorship_artifacts(
    save_dir: Path,
    model: AuthorshipTrainingModel,
    args,
) -> None:
    artifacts = build_method_artifacts(args)
    (save_dir / "authorship_method_config.json").write_text(
        json.dumps(asdict(artifacts), indent=2),
        encoding="utf-8",
    )
    state = model.export_artifacts()
    if state:
        torch.save(state, save_dir / "authorship_method_head.pt")
