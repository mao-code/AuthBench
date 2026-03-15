from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

# Ensure the repo parent is on sys.path so `AuthBench` imports resolve when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from AuthBench.eval.data import load_split
from AuthBench.eval.non_transformer_baselines import (
    NgramStylometricBaseline,
    PPMBaseline,
    evaluate_ngram_attribution,
    evaluate_ngram_representation,
    evaluate_ppm_attribution,
    evaluate_ppm_representation,
)
from AuthBench.eval.tfidf import (
    build_tfidf_index,
    evaluate_tfidf_attribution,
    evaluate_tfidf_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-transformer AuthBench baselines.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--task", choices=("representation", "attribution", "both"), default="both")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["tfidf", "ngram", "ppm"],
        choices=("tfidf", "ngram", "ppm"),
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-pool", choices=("all", "topic"), default="all")
    parser.add_argument("--max-topic-candidates", type=int, default=None)
    parser.add_argument("--topic-seed", type=int, default=13)
    parser.add_argument("--negatives-per-query", type=int, default=50)
    parser.add_argument("--negative-strategy", choices=("sample", "all"), default="all")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-json", type=Path, required=True)

    parser.add_argument("--ngram-max-train-examples", type=int, default=120000)
    parser.add_argument("--ngram-char-features", type=int, default=2**16)
    parser.add_argument("--ngram-word-features", type=int, default=2**15)

    parser.add_argument("--ppm-max-train-examples", type=int, default=80000)
    parser.add_argument("--ppm-order", type=int, default=4)
    parser.add_argument("--ppm-hash-features", type=int, default=8192)
    parser.add_argument("--ppm-alpha", type=float, default=0.5)
    return parser.parse_args()


def _evaluate_tfidf(args: argparse.Namespace, eval_split) -> Dict[str, object]:
    working, tfidf_index = build_tfidf_index(
        eval_split,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    result: Dict[str, object] = {}
    if args.task in ("representation", "both"):
        result["representation"] = evaluate_tfidf_representation(
            eval_split,
            max_queries=args.max_queries,
            max_candidates=args.max_candidates,
            candidate_pool=args.candidate_pool,
            max_topic_candidates=args.max_topic_candidates,
            topic_seed=args.topic_seed,
            analyzer="char_wb",
            ngram_range=(3, 5),
            tfidf_index=tfidf_index,
            working_split=working,
        )
    if args.task in ("attribution", "both"):
        result["attribution"] = evaluate_tfidf_attribution(
            eval_split,
            max_queries=args.max_queries,
            max_candidates=args.max_candidates,
            negatives_per_query=args.negatives_per_query,
            negative_strategy=args.negative_strategy,
            candidate_pool=args.candidate_pool,
            max_topic_candidates=args.max_topic_candidates,
            topic_seed=args.topic_seed,
            analyzer="char_wb",
            ngram_range=(3, 5),
            tfidf_index=tfidf_index,
            working_split=working,
        )
    result["baseline"] = {"implementation": "existing_tfidf_char_wb_3_5"}
    return result


def main() -> None:
    args = parse_args()
    eval_split = load_split(args.dataset_root, args.split)
    train_split = None

    results: Dict[str, Dict[str, object]] = {}
    for baseline_name in args.baselines:
        print(f"\n=== Running baseline: {baseline_name} on split={args.split} ===")
        if baseline_name == "tfidf":
            results["tfidf"] = _evaluate_tfidf(args, eval_split)
            continue

        if train_split is None:
            train_split = load_split(args.dataset_root, args.train_split)

        if baseline_name == "ngram":
            model = NgramStylometricBaseline.fit(
                train_split,
                max_train_examples=args.ngram_max_train_examples,
                seed=args.seed,
                char_features=args.ngram_char_features,
                word_features=args.ngram_word_features,
            )
            result: Dict[str, object] = {}
            if args.task in ("representation", "both"):
                result["representation"] = evaluate_ngram_representation(
                    eval_split,
                    model,
                    max_queries=args.max_queries,
                    max_candidates=args.max_candidates,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                )
            if args.task in ("attribution", "both"):
                result["attribution"] = evaluate_ngram_attribution(
                    eval_split,
                    model,
                    max_queries=args.max_queries,
                    max_candidates=args.max_candidates,
                    negatives_per_query=args.negatives_per_query,
                    negative_strategy=args.negative_strategy,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                    seed=args.seed,
                )
            results["ngram"] = result
            continue

        if baseline_name == "ppm":
            model = PPMBaseline.fit(
                train_split,
                max_train_examples=args.ppm_max_train_examples,
                order=args.ppm_order,
                n_features=args.ppm_hash_features,
                alpha=args.ppm_alpha,
                seed=args.seed,
            )
            result = {}
            if args.task in ("representation", "both"):
                result["representation"] = evaluate_ppm_representation(
                    eval_split,
                    model,
                    max_queries=args.max_queries,
                    max_candidates=args.max_candidates,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                )
            if args.task in ("attribution", "both"):
                result["attribution"] = evaluate_ppm_attribution(
                    eval_split,
                    model,
                    max_queries=args.max_queries,
                    max_candidates=args.max_candidates,
                    negatives_per_query=args.negatives_per_query,
                    negative_strategy=args.negative_strategy,
                    candidate_pool=args.candidate_pool,
                    max_topic_candidates=args.max_topic_candidates,
                    topic_seed=args.topic_seed,
                    seed=args.seed,
                )
            results["ppm"] = result
            continue

        raise ValueError(f"Unknown baseline: {baseline_name}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote baseline metrics to {args.output_json}")


if __name__ == "__main__":
    main()
