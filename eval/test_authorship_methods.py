from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from AuthBench.eval.authorship_methods import (
    AuthorTextExample,
    AuthorshipMethodEmbedder,
    AuthorshipTrainingModel,
    LuarBatchCollator,
    StelTripletDataset,
    compute_authorship_method_loss,
)


class DummyTokenizer:
    cls_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, text, add_special_tokens: bool = False, truncation: bool = False, **kwargs):
        if isinstance(text, str):
            token_ids = list(range(10, 10 + len(text.split())))
            if add_special_tokens:
                return self.prepare_for_model(
                    token_ids,
                    add_special_tokens=True,
                    truncation=truncation,
                    max_length=kwargs.get("max_length"),
                    padding=kwargs.get("padding"),
                    return_attention_mask=True,
                )
            return {"input_ids": token_ids}
        raise TypeError("DummyTokenizer only supports single-string tokenization in these tests.")

    def prepare_for_model(
        self,
        token_ids,
        *,
        add_special_tokens: bool,
        padding,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
    ):
        ids = list(token_ids)
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pad_length = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_length
            mask = mask + [0] * pad_length
        return {"input_ids": ids, "attention_mask": mask}


class DummyBaseModel(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        base = torch.arange(hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = base.view(1, 1, hidden_size).repeat(batch_size, seq_len, 1)
        hidden = hidden + input_ids.unsqueeze(-1).float() * 0.01
        return SimpleNamespace(last_hidden_state=self.proj(hidden))


class DummyLuarModel:
    def encode_episodes(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del input_ids, attention_mask
        return torch.tensor(
            [
                [[1.0, 0.0], [0.9, 0.1]],
                [[0.0, 1.0], [0.1, 0.9]],
            ]
        )


class DummyEvalModel:
    def __init__(self):
        self.training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class AuthorshipMethodTests(unittest.TestCase):
    def test_part_freezes_base_encoder_only(self):
        model = AuthorshipTrainingModel(
            DummyBaseModel(),
            method="part",
            pooling="mean",
            part_hidden_size=4,
            part_temperature_init=0.07,
            luar_embedding_size=4,
        )
        model.set_base_encoder_trainable(False)

        self.assertTrue(all(not p.requires_grad for p in model.base_model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.part_head.parameters()))
        self.assertTrue(model.part_logit_scale.requires_grad)

    def test_luar_collator_produces_two_views_per_author(self):
        tokenizer = DummyTokenizer()
        author_to_examples = {
            "a1": [
                AuthorTextExample(author_id="a1", doc_id="d1", text="one two three four five"),
                AuthorTextExample(author_id="a1", doc_id="d2", text="six seven eight nine ten"),
            ],
            "a2": [
                AuthorTextExample(author_id="a2", doc_id="d3", text="alpha beta gamma delta epsilon"),
                AuthorTextExample(author_id="a2", doc_id="d4", text="zeta eta theta iota kappa"),
            ],
        }
        collator = LuarBatchCollator(
            author_to_examples,
            tokenizer,
            window_size=32,
            episode_length=16,
            samples_per_author=2,
        )

        batch = collator(["a1", "a2"])
        self.assertEqual(batch["input_ids"].shape[:3], (2, 2, batch["input_ids"].shape[2]))
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)

    def test_luar_eval_uses_full_document_windows_by_default(self):
        embedder = AuthorshipMethodEmbedder(
            DummyEvalModel(),
            DummyTokenizer(),
            method="luar",
            device=torch.device("cpu"),
            max_length=512,
            query_prefix="",
            doc_prefix="",
            luar_window_size=32,
            luar_max_eval_windows=None,
        )
        long_text = " ".join(f"tok{i}" for i in range(75))
        input_ids, _ = embedder._build_eval_episode(long_text)
        self.assertEqual(len(input_ids), 3)

    def test_stel_negative_sampling_prioritizes_source_then_genre(self):
        author_to_examples = {
            "anchor": [
                AuthorTextExample(author_id="anchor", doc_id="a1", text="anchor text", source="src", genre="gen"),
                AuthorTextExample(author_id="anchor", doc_id="a2", text="anchor pos", source="src", genre="gen"),
            ],
            "same_both": [
                AuthorTextExample(author_id="same_both", doc_id="b1", text="same both", source="src", genre="gen"),
                AuthorTextExample(author_id="same_both", doc_id="b2", text="same both 2", source="src", genre="gen"),
            ],
            "same_source": [
                AuthorTextExample(author_id="same_source", doc_id="c1", text="same source", source="src", genre="other"),
                AuthorTextExample(author_id="same_source", doc_id="c2", text="same source 2", source="src", genre="other"),
            ],
            "same_genre": [
                AuthorTextExample(author_id="same_genre", doc_id="d1", text="same genre", source="other", genre="gen"),
                AuthorTextExample(author_id="same_genre", doc_id="d2", text="same genre 2", source="other", genre="gen"),
            ],
        }
        dataset = StelTripletDataset(author_to_examples, control_keys=["source", "genre"])
        anchor = author_to_examples["anchor"][0]
        negative = dataset._sample_negative(anchor)
        self.assertEqual(negative.author_id, "same_both")

        reduced = {k: v for k, v in author_to_examples.items() if k != "same_both"}
        dataset = StelTripletDataset(reduced, control_keys=["source", "genre"])
        anchor = reduced["anchor"][0]
        negative = dataset._sample_negative(anchor)
        self.assertEqual(negative.author_id, "same_source")

    def test_luar_loss_uses_supervised_contrastive_views(self):
        args = SimpleNamespace(authorship_method="luar", luar_temperature=0.01)
        batch = {
            "input_ids": torch.zeros(2, 2, 3, 32, dtype=torch.long),
            "attention_mask": torch.ones(2, 2, 3, 32, dtype=torch.long),
        }
        loss = compute_authorship_method_loss(DummyLuarModel(), batch, args, torch.device("cpu"))
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
