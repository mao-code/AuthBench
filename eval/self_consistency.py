from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from AuthBench.eval.embedder import EmbeddingResult
from AuthBench.eval.hf_utils import load_causal_lm_model, load_tokenizer


DEFAULT_STYLE_PROMPT_TEMPLATE = """You are extracting authorship style signals for authorship verification.
Read the document and write a short style profile that focuses on writing style rather than topic.
Describe lexical choice, syntax, tone, discourse habits, punctuation, formatting, and recurring rhetorical patterns when present.
Do not summarize the topic and do not copy long spans from the document.

Document:
{text}

Style profile:"""


def _chunk_iterable(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


@dataclass(frozen=True)
class SelfConsistencyConfig:
    num_samples: int = 4
    top_k: int = 50
    temperature: float = 0.8
    max_new_tokens: int = 96
    prompt_template: str = DEFAULT_STYLE_PROMPT_TEMPLATE
    include_original: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class SelfConsistencyCausalLMEmbedder:
    """Build pooled style embeddings by sampling multiple style descriptions per text."""

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

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        prefix: str = "",
        return_tokens: bool = False,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        if return_tokens:
            raise ValueError(
                "Self-consistency embeddings only expose pooled vectors. Disable late interaction."
            )

        texts_list = list(texts)
        was_training = self.model.training
        self.model.eval()
        try:
            if not texts_list:
                dim = self.dimension or 0
                return EmbeddingResult(vectors=torch.empty((0, dim), dtype=torch.float32))

            sampled_summaries = self._generate_style_summaries(
                texts_list,
                batch_size=batch_size,
                prefix=prefix,
                show_progress=show_progress,
            )
            summary_vectors = self._encode_direct(
                sampled_summaries,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_desc="Style embedding",
            )
            vectors = summary_vectors.view(len(texts_list), self.config.num_samples, -1).mean(dim=1)

            if self.config.include_original:
                original_vectors = self._encode_direct(
                    self._apply_prefix(texts_list, prefix),
                    batch_size=batch_size,
                    show_progress=False,
                    progress_desc="Embedding",
                )
                vectors = 0.5 * (vectors + original_vectors)

            if self.normalize:
                vectors = F.normalize(vectors, p=2, dim=1)
            return EmbeddingResult(vectors=vectors.cpu())
        finally:
            if was_training:
                self.model.train()
