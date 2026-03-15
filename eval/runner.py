from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
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
from AuthBench.utilities import model_registry


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "processing"
    / "outputs"
    / "official_ttl300k_cap10M_sf10k_postprocessed"
)
DEFAULT_RETRIEVAL_KS = (1, 3, 5, 10)


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
        default="all",
        help="How to choose negatives for attribution metrics.",
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

    return wandb.init(
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
        },
    )


def main() -> int:
    args = parse_args()
    wandb_run = _init_wandb(args)
    split = load_split(args.dataset_root, args.split)

    model_names = resolve_models(args)
    results: Dict[str, Dict[str, object]] = {}
    log_step = 0
    allow_remote_code_fallback = not args.no_auto_trust_remote_code

    for model_name in model_names:
        repo = model_registry.get_hf_repo(model_name) if model_name in model_registry.MODEL_HF_PATHS else model_name
        print(f"\n=== Evaluating {model_name} ({repo}) on {args.split} ===")

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
        if args.task in ("representation", "both"):
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
