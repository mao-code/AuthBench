from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Ensure the repo parent is on sys.path so `AuthBench` imports resolve when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from AuthBench.eval.data import load_split
from AuthBench.eval.embedder import HuggingFaceEmbedder
from AuthBench.eval.evaluators import (
    evaluate_authorship_attribution,
    evaluate_authorship_representation,
)
from AuthBench.eval.self_consistency import (
    DEFAULT_SELF_CONSISTENCY_RETRIEVAL_KS,
    DEFAULT_STYLE_PROMPT_TEMPLATE,
    SelfConsistencyCausalLMEmbedder,
    SelfConsistencyConfig,
    encode_self_consistency_split_embeddings,
    evaluate_self_consistency_attribution,
    evaluate_self_consistency_representation,
)
from AuthBench.utilities import model_registry


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "processing"
    / "outputs"
    / "official_ttl300k_cap10M_sf10k_postprocessed"
)
DEFAULT_RETRIEVAL_KS = DEFAULT_SELF_CONSISTENCY_RETRIEVAL_KS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AuthBench models on processed splits.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--task", choices=("representation", "attribution", "both"), default="both")
    parser.add_argument("--models", nargs="+", default=["e5-large-v2"])
    parser.add_argument("--all-models", action="store_true", help="Evaluate every model in model_registry.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--no-truncation",
        action="store_true",
        help="Disable truncation and pad to the longest sequence in each batch.",
    )
    parser.add_argument("--pooling", default="mean", choices=("mean", "cls", "last"))
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--torch-dtype", default=None, help="Optional torch dtype, e.g., bf16 or float16.")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--max-queries", type=int, help="Optional cap on queries for quick runs.")
    parser.add_argument("--max-candidates", type=int, help="Optional cap on candidates for quick runs.")
    parser.add_argument("--negatives-per-query", type=int, default=50)
    parser.add_argument(
        "--negative-strategy",
        choices=("sample", "all"),
        default="sample",
        help="How to choose negatives for attribution EER.",
    )
    parser.add_argument(
        "--candidate-pool",
        choices=("all", "topic"),
        default="all",
        help="Candidate pool strategy: all candidates or topic-matched candidates.",
    )
    parser.add_argument(
        "--max-topic-candidates",
        type=int,
        default=None,
        help="Optional cap for topic-matched candidate pools.",
    )
    parser.add_argument(
        "--topic-seed",
        type=int,
        default=13,
        help="Seed for deterministic sampling of topic-matched pools.",
    )
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=128,
        help="Chunk size for candidate token batches when using late interaction.",
    )
    parser.add_argument("--late-interaction", action="store_true", help="Use max-sim scoring over tokens.")
    parser.add_argument("--output-json", type=Path, help="Save metrics to JSON.")
    parser.add_argument("--wandb-project", help="If set, log metrics to this Weights & Biases project.")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name.")
    parser.add_argument("--wandb-entity", help="Optional W&B entity/org.")
    parser.add_argument("--wandb-tags", nargs="*", help="Optional list of W&B tags.")
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Sample multiple style descriptions from supported causal LLMs, sum the per-sample query-candidate scores, and rerank candidates.",
    )
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=4,
        help="Number of sampled style descriptions / sampled document embeddings per document when --self-consistency is enabled.",
    )
    parser.add_argument(
        "--self-consistency-top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff for self-consistency generation.",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for self-consistency generation.",
    )
    parser.add_argument(
        "--self-consistency-max-new-tokens",
        type=int,
        default=96,
        help="Maximum number of generated tokens per sampled style description.",
    )
    parser.add_argument(
        "--self-consistency-include-original",
        action="store_true",
        help="Add the direct document embedding as one extra sampled embedding in the score sum.",
    )
    parser.add_argument(
        "--self-consistency-prompt",
        default=DEFAULT_STYLE_PROMPT_TEMPLATE,
        help="Prompt template for style generation; must include a {text} placeholder.",
    )
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
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.all_models:
        if args.self_consistency:
            return model_registry.self_consistency_model_names()
        return sorted(model_registry.MODEL_HF_PATHS.keys())
    return args.models


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
            "split": args.split,
            "task": args.task,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "no_truncation": args.no_truncation,
            "pooling": args.pooling,
            "max_queries": args.max_queries,
            "max_candidates": args.max_candidates,
            "negatives_per_query": args.negatives_per_query,
            "negative_strategy": args.negative_strategy,
            "late_interaction": args.late_interaction,
            "candidate_pool": args.candidate_pool,
            "max_topic_candidates": args.max_topic_candidates,
            "topic_seed": args.topic_seed,
            "dataset_root": str(args.dataset_root),
            "self_consistency": args.self_consistency,
            "self_consistency_samples": args.self_consistency_samples,
            "self_consistency_top_k": args.self_consistency_top_k,
            "self_consistency_temperature": args.self_consistency_temperature,
            "self_consistency_max_new_tokens": args.self_consistency_max_new_tokens,
            "self_consistency_include_original": args.self_consistency_include_original,
            "self_consistency_total_votes": args.self_consistency_samples
            + int(args.self_consistency_include_original),
            "self_consistency_aggregation_strategy": "sum_sample_scores_rerank"
            if args.self_consistency
            else None,
        },
    )
    return run


def main() -> int:
    args = parse_args()
    if args.self_consistency and args.late_interaction:
        raise ValueError("Self-consistency only supports pooled vectors; disable --late-interaction.")
    wandb_run = _init_wandb(args)
    split = load_split(args.dataset_root, args.split)

    model_names = resolve_models(args)
    results: Dict[str, Dict[str, object]] = {}
    log_step = 0
    allow_remote_code_fallback = not args.no_auto_trust_remote_code

    for model_name in model_names:
        repo = model_registry.get_hf_repo(model_name) if model_name in model_registry.MODEL_HF_PATHS else model_name
        print(f"\n=== Evaluating {model_name} ({repo}) on {args.split} ===")
        use_self_consistency = args.self_consistency
        if (
            args.self_consistency
            and model_name in model_registry.MODEL_HF_PATHS
            and not model_registry.supports_self_consistency(model_name)
        ):
            raise ValueError(
                f"Model '{model_name}' does not support generation-based self-consistency. "
                "Use one of the causal LLM checkpoints from model_registry.self_consistency_model_names()."
            )

        if use_self_consistency:
            embedder = SelfConsistencyCausalLMEmbedder(
                repo,
                config=SelfConsistencyConfig(
                    num_samples=args.self_consistency_samples,
                    top_k=args.self_consistency_top_k,
                    temperature=args.self_consistency_temperature,
                    max_new_tokens=args.self_consistency_max_new_tokens,
                    prompt_template=args.self_consistency_prompt,
                    include_original=args.self_consistency_include_original,
                ),
                device=args.device,
                max_length=args.max_length,
                no_truncation=args.no_truncation,
                pooling=args.pooling,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                allow_remote_code_fallback=allow_remote_code_fallback,
            )
        else:
            embedder = HuggingFaceEmbedder(
                repo,
                device=args.device,
                max_length=args.max_length,
                no_truncation=args.no_truncation,
                pooling=args.pooling,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                allow_remote_code_fallback=allow_remote_code_fallback,
            )

        model_result: Dict[str, object] = {"hf_repo": repo}
        working_split = None
        sampled_query_embeddings = None
        sampled_candidate_embeddings = None
        if use_self_consistency:
            working_split = split.limited(
                max_queries=args.max_queries,
                max_candidates=args.max_candidates,
            )
            sampled_query_embeddings, sampled_candidate_embeddings = encode_self_consistency_split_embeddings(
                working_split,
                embedder,
                batch_size=args.batch_size,
                query_prefix=args.query_prefix,
                doc_prefix=args.doc_prefix,
            )
            model_result["self_consistency"] = {
                "enabled": True,
                "num_samples": args.self_consistency_samples,
                "top_k": args.self_consistency_top_k,
                "temperature": args.self_consistency_temperature,
                "max_new_tokens": args.self_consistency_max_new_tokens,
                "include_original": args.self_consistency_include_original,
                "total_votes": args.self_consistency_samples
                + int(args.self_consistency_include_original),
                "prompt_template": args.self_consistency_prompt,
                "aggregation_strategy": "sum_sample_scores_rerank",
                "score_aggregation": "sum_sample_scores",
                "retrieval_ks": list(DEFAULT_RETRIEVAL_KS),
            }
        if args.task in ("representation", "both"):
            if use_self_consistency:
                rep_metrics = evaluate_self_consistency_representation(
                    split=working_split,
                    query_embeddings=sampled_query_embeddings,
                    candidate_embeddings=sampled_candidate_embeddings,
                    batch_size=args.batch_size,
                    ks=DEFAULT_RETRIEVAL_KS,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                    score_device=args.device,
                )
            else:
                rep_metrics = evaluate_authorship_representation(
                    split=split,
                    embedder=embedder,
                    batch_size=args.batch_size,
                    ks=DEFAULT_RETRIEVAL_KS,
                    query_prefix=args.query_prefix,
                    doc_prefix=args.doc_prefix,
                    max_queries=args.max_queries,
                    max_candidates=args.max_candidates,
                    late_interaction=args.late_interaction,
                    candidate_chunk_size=args.candidate_chunk_size,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                )
            model_result["representation"] = rep_metrics
            print("Representation metrics:", json.dumps(rep_metrics, indent=2))

        if args.task in ("attribution", "both"):
            if use_self_consistency:
                attr_metrics = evaluate_self_consistency_attribution(
                    split=working_split,
                    query_embeddings=sampled_query_embeddings,
                    candidate_embeddings=sampled_candidate_embeddings,
                    batch_size=args.batch_size,
                    negatives_per_query=args.negatives_per_query,
                    negative_strategy=args.negative_strategy,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                    score_device=args.device,
                )
            else:
                attr_metrics = evaluate_authorship_attribution(
                    split=split,
                    embedder=embedder,
                    batch_size=args.batch_size,
                    query_prefix=args.query_prefix,
                    doc_prefix=args.doc_prefix,
                    max_queries=args.max_queries,
                    max_candidates=args.max_candidates,
                    negatives_per_query=args.negatives_per_query,
                    negative_strategy=args.negative_strategy,
                    late_interaction=args.late_interaction,
                    candidate_chunk_size=args.candidate_chunk_size,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                )
            model_result["attribution"] = attr_metrics
            print("Attribution metrics:", json.dumps(attr_metrics, indent=2))

        results[model_name] = model_result
        if wandb_run:
            to_log: Dict[str, object] = {
                "model": model_name,
                "hf_repo": repo,
                "split": args.split,
                "task": args.task,
                "step": log_step,
            }
            if "representation" in model_result:
                to_log.update(_flatten(f"{model_name}/representation", model_result["representation"]))
            if "attribution" in model_result:
                to_log.update(_flatten(f"{model_name}/attribution", model_result["attribution"]))
            wandb_run.log(to_log, step=log_step)
            log_step += 1

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {args.output_json}")

    if wandb_run:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
