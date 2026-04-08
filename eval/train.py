from __future__ import annotations

import argparse
from functools import partial
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Ensure the repo parent is on sys.path so `AuthBench` imports resolve when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from AuthBench.eval.authorship_methods import (
    AUTHORSHIP_METHODS,
    AuthorshipTrainingModel,
    build_authorship_eval_embedder,
    build_authorship_training_components,
    build_method_artifacts,
    compute_authorship_method_loss,
    save_authorship_artifacts,
)
from AuthBench.eval.data import PairDataset, build_positive_pairs, load_split
from AuthBench.eval.embedder import HuggingFaceEmbedder
from AuthBench.eval.evaluators import (
    evaluate_authorship_attribution,
    evaluate_authorship_representation,
)
from AuthBench.eval.hf_utils import load_model, load_tokenizer
from AuthBench.utilities import model_registry


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "processing"
    / "outputs"
    / "official_ttl300k_cap10M_sf10k_postprocessed"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pool_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
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


def collate_pairs(
    batch,
    tokenizer,
    max_length: int,
    query_prefix: str,
    doc_prefix: str,
):
    query_texts = [query_prefix + item["query_text"] for item in batch]
    cand_texts = [doc_prefix + item["candidate_text"] for item in batch]
    query_inputs = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    cand_inputs = tokenizer(
        cand_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return query_inputs, cand_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive training on AuthBench.")
    parser.set_defaults(part_freeze_encoder=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model", default="e5-large-v2", help="Model key from model_registry or HF repo id.")
    parser.add_argument(
        "--authorship-method",
        choices=AUTHORSHIP_METHODS,
        default="standard",
        help="Training recipe: standard query-candidate InfoNCE, PART, LUAR, or STEL.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Do not save model/tokenizer weights; only write the training summary JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, help="Optional max training steps (overrides epochs).")
    parser.add_argument("--max-train-pairs", type=int, help="Cap number of training pairs.")
    parser.add_argument(
        "--max-train-authors",
        type=int,
        help="Optional cap on the number of training authors for PART/LUAR/STEL.",
    )
    parser.add_argument("--max-eval-queries", type=int, help="Cap queries during evaluation.")
    parser.add_argument("--max-eval-candidates", type=int, help="Cap candidates during evaluation.")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps.")
    parser.add_argument(
        "--eval-fraction-epoch",
        type=float,
        help="Evaluate every fraction of an epoch (e.g., 0.5 => mid-epoch). Overrides --eval-every.",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action="store_true",
        help="Force an evaluation at the end of every epoch (in addition to any step-based intervals).",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=("mean", "cls", "last"), default="mean")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--late-interaction", action="store_true", help="Enable late interaction during eval.")
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=128,
        help="Candidate token batch size for late interaction scoring.",
    )
    parser.add_argument("--negatives-per-query", type=int, default=50)
    parser.add_argument("--negative-strategy", choices=("sample", "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank. Set >0 to enable LoRA adapters (e.g., 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha. Defaults to 2x rank when --lora-rank > 0.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument(
        "--lora-bias",
        choices=("none", "all", "lora_only"),
        default="none",
        help="Bias training strategy for LoRA adapters.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["all-linear"],
        help="LoRA target modules (space or comma separated). Use all-linear for broad coverage.",
    )
    parser.add_argument(
        "--eval-ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Ranking metric cutoffs (e.g., --eval-ks 5 or --eval-ks 1 3 5).",
    )
    parser.add_argument(
        "--part-hidden-size",
        type=int,
        default=512,
        help="Hidden size per LSTM direction for the PART BiLSTM head.",
    )
    parser.add_argument(
        "--part-freeze-encoder",
        dest="part_freeze_encoder",
        action="store_true",
        help="Freeze the base encoder and train only the PART head (default).",
    )
    parser.add_argument(
        "--part-train-encoder",
        dest="part_freeze_encoder",
        action="store_false",
        help="Allow PART to tune the base encoder (for example via LoRA).",
    )
    parser.add_argument(
        "--part-temperature-init",
        type=float,
        default=0.07,
        help="Initial learnable PART temperature/logit-scale parameter.",
    )
    parser.add_argument(
        "--luar-window-size",
        type=int,
        default=32,
        help="Token window size for each LUAR excerpt.",
    )
    parser.add_argument(
        "--luar-episode-length",
        type=int,
        default=16,
        help="Maximum number of windows/doc-units sampled in one LUAR training episode.",
    )
    parser.add_argument(
        "--luar-samples-per-author",
        type=int,
        default=2,
        help="Number of LUAR episodes sampled per author for supervised contrastive training.",
    )
    parser.add_argument(
        "--luar-max-eval-windows",
        type=int,
        default=None,
        help="Optional cap on LUAR eval-time windows per document. Defaults to using every 32-token window.",
    )
    parser.add_argument(
        "--luar-max-episode-docs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--luar-eval-episode-docs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--luar-embedding-size",
        type=int,
        default=512,
        help="Output dimensionality of the LUAR episode projection layer.",
    )
    parser.add_argument(
        "--luar-temperature",
        type=float,
        default=0.01,
        help="Temperature used by LUAR's contrastive loss.",
    )
    parser.add_argument(
        "--stel-margin",
        type=float,
        default=0.5,
        help="Triplet margin for STEL's CAV objective.",
    )
    parser.add_argument(
        "--stel-control-keys",
        nargs="+",
        default=["source", "genre"],
        help="Metadata keys used to content-control STEL negatives, in fallback order.",
    )
    parser.add_argument("--log-file", type=Path, help="Optional JSONL log of evaluation metrics.")
    parser.add_argument("--wandb-project", help="If set, log metrics to this Weights & Biases project.")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name.")
    parser.add_argument("--wandb-entity", help="Optional W&B entity/org.")
    parser.add_argument("--wandb-tags", nargs="*", help="Optional list of W&B tags.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders. Even without this flag, "
        "a retry with trust_remote_code=True is attempted when transformers requests it.",
    )
    parser.add_argument(
        "--no-auto-trust-remote-code",
        action="store_true",
        help="Disable the automatic retry with trust_remote_code=True when a model repo "
        "contains custom code.",
    )
    args = parser.parse_args()
    if args.luar_max_episode_docs is not None:
        args.luar_episode_length = args.luar_max_episode_docs
    if args.luar_eval_episode_docs is not None:
        if args.luar_max_eval_windows is not None and args.luar_max_eval_windows != args.luar_eval_episode_docs:
            raise ValueError("--luar-max-eval-windows conflicts with legacy --luar-eval-episode-docs.")
        args.luar_max_eval_windows = args.luar_eval_episode_docs
    if args.luar_episode_length < 1:
        raise ValueError("--luar-episode-length must be >= 1.")
    if args.luar_samples_per_author < 2:
        raise ValueError("--luar-samples-per-author must be >= 2.")
    if args.luar_max_eval_windows is not None and args.luar_max_eval_windows < 1:
        raise ValueError("--luar-max-eval-windows must be >= 1 when set.")
    return args


def encode(model, inputs, pooling: str) -> torch.Tensor:
    outputs = model(**inputs)
    return pool_embeddings(outputs.last_hidden_state, inputs["attention_mask"], pooling)


def maybe_log(log_path: Optional[Path], record: Dict[str, object]) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _flatten(prefix: str, metrics: Dict[str, object]) -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}"
        if isinstance(value, dict):
            flat.update(_flatten(full_key, value))
        else:
            flat[full_key] = value
    return flat


def _init_wandb(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Add it to your environment to enable logging.") from exc
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        tags=args.wandb_tags,
        config={
            "model": args.model,
            "authorship_method": args.authorship_method,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "grad_accum": args.grad_accum,
            "max_train_pairs": args.max_train_pairs,
            "max_train_authors": args.max_train_authors,
            "max_eval_queries": args.max_eval_queries,
            "max_eval_candidates": args.max_eval_candidates,
            "negatives_per_query": args.negatives_per_query,
            "negative_strategy": args.negative_strategy,
            "late_interaction": args.late_interaction,
            "eval_ks": args.eval_ks,
            "part_hidden_size": args.part_hidden_size,
            "part_freeze_encoder": args.part_freeze_encoder,
            "part_temperature_init": args.part_temperature_init,
            "luar_window_size": args.luar_window_size,
            "luar_episode_length": args.luar_episode_length,
            "luar_samples_per_author": args.luar_samples_per_author,
            "luar_max_eval_windows": args.luar_max_eval_windows,
            "luar_embedding_size": args.luar_embedding_size,
            "luar_temperature": args.luar_temperature,
            "stel_margin": args.stel_margin,
            "stel_control_keys": args.stel_control_keys,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_bias": args.lora_bias,
            "lora_target_modules": args.lora_target_modules,
            "dataset_root": str(args.dataset_root),
        },
    )
    return run


def _resolve_lora_target_modules(raw_targets: List[str]) -> Union[str, List[str]]:
    parsed: List[str] = []
    for item in raw_targets:
        for token in re.split(r"[\s,]+", item):
            token = token.strip()
            if token:
                parsed.append(token)
    if not parsed:
        return "all-linear"
    if len(parsed) == 1 and parsed[0] in {"all", "all-linear"}:
        return "all-linear"
    return parsed


def _enable_padding(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _count_parameters(model) -> Tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def _apply_lora_if_enabled(model, args: argparse.Namespace) -> Tuple[torch.nn.Module, Dict[str, object]]:
    lora_details: Dict[str, object] = {
        "enabled": False,
        "rank": 0,
        "alpha": None,
        "dropout": 0.0,
        "bias": "none",
        "target_modules": [],
    }
    if args.lora_rank <= 0:
        return model, lora_details
    if args.authorship_method == "part" and args.part_freeze_encoder:
        print(
            "[WARN] PART freezes the encoder by default; skipping LoRA adapters. "
            "Use --part-train-encoder to enable encoder tuning."
        )
        return model, lora_details

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA requested (--lora-rank > 0) but `peft` is not installed. "
            "Install peft to enable adapter training."
        ) from exc

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2
    target_modules = _resolve_lora_target_modules(args.lora_target_modules)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    print(
        "Applied LoRA adapters "
        f"(rank={args.lora_rank}, alpha={lora_alpha}, dropout={args.lora_dropout}, "
        f"target_modules={target_modules})."
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    lora_details = {
        "enabled": True,
        "rank": args.lora_rank,
        "alpha": lora_alpha,
        "dropout": args.lora_dropout,
        "bias": args.lora_bias,
        "target_modules": target_modules,
    }
    return model, lora_details


def run_evaluations(
    embedder: HuggingFaceEmbedder,
    split_name: str,
    split,
    args: argparse.Namespace,
    step: int,
) -> Dict[str, object]:
    rep = evaluate_authorship_representation(
        split=split,
        embedder=embedder,
        batch_size=max(args.batch_size, 8),
        ks=tuple(args.eval_ks),
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        max_queries=args.max_eval_queries,
        max_candidates=args.max_eval_candidates,
        late_interaction=args.late_interaction,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    attr = evaluate_authorship_attribution(
        split=split,
        embedder=embedder,
        batch_size=max(args.batch_size, 8),
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        max_queries=args.max_eval_queries,
        max_candidates=args.max_eval_candidates,
        negatives_per_query=args.negatives_per_query,
        negative_strategy=args.negative_strategy,
        late_interaction=args.late_interaction,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    payload = {
        "step": step,
        "split": split_name,
        "representation": rep,
        "attribution": attr,
    }
    return payload


def _log_eval_to_wandb(wandb_run, eval_payload: Dict[str, object], prefix: str) -> None:
    if not wandb_run:
        return
    step = eval_payload.get("step")
    log_record: Dict[str, object] = {"event": prefix, "step": step, f"{prefix}/split": eval_payload.get("split")}
    if "representation" in eval_payload:
        log_record.update(_flatten(f"{prefix}/representation", eval_payload["representation"]))
    if "attribution" in eval_payload:
        log_record.update(_flatten(f"{prefix}/attribution", eval_payload["attribution"]))
    wandb_run.log(log_record, step=step)


def train() -> int:
    args = parse_args()
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be >= 0.")
    if args.authorship_method != "standard" and args.late_interaction:
        raise ValueError("--late-interaction is only supported for authorship_method=standard.")
    if args.authorship_method == "standard" and args.max_train_authors is not None:
        print("[WARN] --max-train-authors is ignored for authorship_method=standard.")
    eval_ks = sorted({k for k in args.eval_ks if k > 0})
    if not eval_ks:
        raise ValueError("eval_ks must contain at least one positive integer.")
    args.eval_ks = eval_ks
    set_seed(args.seed)
    wandb_run = _init_wandb(args)
    loss_history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, object]] = []

    def record_eval(event: str, payload: Dict[str, object], wandb_prefix: str) -> None:
        enriched = {"event": event, **payload}
        maybe_log(args.log_file, enriched)
        eval_history.append(enriched)
        _log_eval_to_wandb(wandb_run, payload, prefix=wandb_prefix)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = (
        model_registry.get_hf_repo(args.model) if args.model in model_registry.MODEL_HF_PATHS else args.model
    )

    print(f"Loading data from {args.dataset_root} ...")
    train_split = load_split(args.dataset_root, "train")
    dev_split = load_split(args.dataset_root, "dev")
    test_split = load_split(args.dataset_root, "test")

    train_pairs = None
    if args.authorship_method == "standard":
        train_pairs = build_positive_pairs(train_split, max_pairs=args.max_train_pairs, seed=args.seed)
        if not train_pairs:
            raise RuntimeError("No training pairs found. Check dataset paths or processing output.")

    allow_remote_code_fallback = not args.no_auto_trust_remote_code
    tokenizer = load_tokenizer(
        model_name, trust_remote_code=args.trust_remote_code, allow_remote_code_fallback=allow_remote_code_fallback
    )
    base_model = load_model(
        model_name, trust_remote_code=args.trust_remote_code, allow_remote_code_fallback=allow_remote_code_fallback
    )
    _enable_padding(tokenizer, base_model)
    base_model, lora_details = _apply_lora_if_enabled(base_model, args)

    if args.authorship_method == "standard":
        train_model = base_model
    else:
        train_model = AuthorshipTrainingModel(
            base_model,
            method=args.authorship_method,
            pooling=args.pooling,
            part_hidden_size=args.part_hidden_size,
            part_temperature_init=args.part_temperature_init,
            luar_embedding_size=args.luar_embedding_size,
        )
        if args.authorship_method == "part" and args.part_freeze_encoder:
            train_model.set_base_encoder_trainable(False)

    train_model.to(device)
    trainable_params, total_params = _count_parameters(train_model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    if args.authorship_method == "standard":
        train_dataset = PairDataset(train_pairs)
        collate = partial(
            collate_pairs,
            tokenizer=tokenizer,
            max_length=args.max_length,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
        )
    else:
        train_dataset, collate = build_authorship_training_components(train_split, tokenizer, args)
        if len(train_dataset) == 0:
            raise RuntimeError(
                f"No eligible training examples found for authorship method '{args.authorship_method}'."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    steps_per_epoch = max(1, len(train_loader))
    eval_every_steps = args.eval_every
    if args.eval_fraction_epoch is not None:
        eval_every_steps = max(1, int(round(steps_per_epoch * args.eval_fraction_epoch)))
    if eval_every_steps <= 0:
        eval_every_steps = 1

    total_steps = args.max_steps or math.ceil(len(train_loader) * args.epochs)
    trainable_parameters = [param for param in train_model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found; check LoRA config and model setup.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=args.fp16)
    optimizer.zero_grad(set_to_none=True)

    # Step-0 eval
    if args.authorship_method == "standard":
        eval_embedder = HuggingFaceEmbedder(
            model_name_or_path=model_name,
            model=train_model,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
            pooling=args.pooling,
            trust_remote_code=args.trust_remote_code,
            allow_remote_code_fallback=allow_remote_code_fallback,
        )
    else:
        eval_embedder = build_authorship_eval_embedder(train_model, tokenizer, args, device)
    initial_eval = run_evaluations(eval_embedder, "dev", dev_split, args, step=0)
    print("Step 0 evaluation:", json.dumps(initial_eval, indent=2))
    record_eval("eval/step0", initial_eval, wandb_prefix="eval")

    step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            train_model.train()
            with autocast(enabled=args.fp16):
                if args.authorship_method == "standard":
                    query_inputs, cand_inputs = batch
                    # Move tensors to device in the main process to avoid CUDA init inside DataLoader workers.
                    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
                    cand_inputs = {k: v.to(device) for k, v in cand_inputs.items()}
                    query_emb = encode(train_model, query_inputs, args.pooling)
                    cand_emb = encode(train_model, cand_inputs, args.pooling)
                    logits = torch.matmul(query_emb, cand_emb.T) / args.temperature
                    labels = torch.arange(logits.size(0), device=device)
                    method_loss = F.cross_entropy(logits, labels)
                else:
                    method_loss = compute_authorship_method_loss(train_model, batch, args, device)
                loss = method_loss / args.grad_accum
                loss_value = float(loss.item())

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            step += 1
            loss_history.append({"step": step, "loss": loss_value})
            if step % 50 == 0:
                print(f"step={step} loss={loss_value:.4f}")
                if wandb_run:
                    current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.learning_rate
                    wandb_run.log({"train/loss": loss_value, "train/lr": current_lr, "step": step}, step=step)

            if step % eval_every_steps == 0 or step == total_steps:
                eval_payload = run_evaluations(eval_embedder, "dev", dev_split, args, step=step)
                print(f"Dev evaluation at step {step}:", json.dumps(eval_payload, indent=2))
                record_eval("eval/dev", eval_payload, wandb_prefix="eval")

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        if args.eval_every_epoch:
            eval_payload = run_evaluations(eval_embedder, "dev", dev_split, args, step=step)
            print(f"End-of-epoch evaluation at step {step}:", json.dumps(eval_payload, indent=2))
            record_eval("eval/epoch_end", eval_payload, wandb_prefix="eval")

    print("Final evaluations on dev and test ...")
    final_dev = run_evaluations(eval_embedder, "dev", dev_split, args, step=step)
    final_test = run_evaluations(eval_embedder, "test", test_split, args, step=step)
    print("Dev:", json.dumps(final_dev, indent=2))
    print("Test:", json.dumps(final_test, indent=2))
    record_eval("final/dev", final_dev, wandb_prefix="final/dev")
    record_eval("final/test", final_test, wandb_prefix="final/test")

    save_path = args.output_dir / args.model
    save_path.mkdir(parents=True, exist_ok=True)
    if args.authorship_method != "standard":
        save_authorship_artifacts(save_path, train_model, args)
    summary_path = save_path / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "hf_repo": model_name,
                "authorship_method": args.authorship_method,
                "authorship_method_config": (
                    build_method_artifacts(args).__dict__ if args.authorship_method != "standard" else None
                ),
                "steps": step,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "temperature": args.temperature,
                "pooling": args.pooling,
                "max_length": args.max_length,
                "eval_ks": args.eval_ks,
                "lora": lora_details,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "loss_history": loss_history,
                "eval_history": eval_history,
                "dev_metrics": final_dev,
                "test_metrics": final_test,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved training summary to {summary_path}")
    if not args.skip_checkpoint:
        base_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")
    else:
        print("Skipped checkpoint save (--skip-checkpoint enabled).")
    if wandb_run:
        wandb_run.save(str(summary_path))
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
