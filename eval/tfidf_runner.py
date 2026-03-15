from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo parent is on sys.path so `AuthBench` imports resolve when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from AuthBench.eval.data import load_split
from AuthBench.eval.tfidf import (
    build_tfidf_index,
    evaluate_tfidf_attribution,
    evaluate_tfidf_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF baseline evaluation.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--task", choices=("representation", "attribution", "both"), default="both")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-pool", choices=("all", "topic"), default="all")
    parser.add_argument("--max-topic-candidates", type=int, default=None)
    parser.add_argument("--topic-seed", type=int, default=13)
    parser.add_argument("--negatives-per-query", type=int, default=50)
    parser.add_argument("--negative-strategy", choices=("sample", "all"), default="all")
    parser.add_argument("--analyzer", default="char_wb")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing before vectorizing.",
    )
    parser.add_argument(
        "--include-queries-in-fit",
        action="store_true",
        help="Include queries when fitting TF-IDF (default: candidates only).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-name", default="tfidf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_split(args.dataset_root, args.split)
    ngram_range = (args.ngram_min, args.ngram_max)

    working, tfidf_index = build_tfidf_index(
        split,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        analyzer=args.analyzer,
        ngram_range=ngram_range,
        max_features=args.max_features,
        min_df=args.min_df,
        lowercase=not args.no_lowercase,
        include_queries_in_fit=args.include_queries_in_fit,
    )

    result = {}
    if args.task in ("representation", "both"):
        result["representation"] = evaluate_tfidf_representation(
            split,
            max_queries=args.max_queries,
            max_candidates=args.max_candidates,
            candidate_pool=args.candidate_pool,
            max_topic_candidates=args.max_topic_candidates,
            topic_seed=args.topic_seed,
            analyzer=args.analyzer,
            ngram_range=ngram_range,
            max_features=args.max_features,
            min_df=args.min_df,
            lowercase=not args.no_lowercase,
            include_queries_in_fit=args.include_queries_in_fit,
            tfidf_index=tfidf_index,
            working_split=working,
        )
    if args.task in ("attribution", "both"):
        result["attribution"] = evaluate_tfidf_attribution(
            split,
            max_queries=args.max_queries,
            max_candidates=args.max_candidates,
            negatives_per_query=args.negatives_per_query,
            negative_strategy=args.negative_strategy,
            candidate_pool=args.candidate_pool,
            max_topic_candidates=args.max_topic_candidates,
            topic_seed=args.topic_seed,
            analyzer=args.analyzer,
            ngram_range=ngram_range,
            max_features=args.max_features,
            min_df=args.min_df,
            lowercase=not args.no_lowercase,
            include_queries_in_fit=args.include_queries_in_fit,
            tfidf_index=tfidf_index,
            working_split=working,
        )

    payload = {args.model_name: result}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote TF-IDF metrics to {args.output_json}")


if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m eval.tfidf_runner \
    --dataset-root processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced
    """
