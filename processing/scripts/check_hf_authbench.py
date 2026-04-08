#!/usr/bin/env python3
"""Smoke-test an AuthBench dataset hosted on the Hugging Face Hub.

This script prints:
1. available configs and split sizes from Hub metadata
2. a few example rows per config
3. lightweight consistency checks across queries, candidates, documents, and
   ground_truth for one split

Example:
  python processing/scripts/check_hf_authbench.py
  python processing/scripts/check_hf_authbench.py --repo-id MaoXun/AuthBench --split test
"""

from __future__ import annotations

import argparse
from itertools import islice
from typing import Iterable

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder


DEFAULT_REPO_ID = "MaoXun/AuthBench"
CONFIGS = ("documents", "queries", "candidates", "ground_truth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test AuthBench on the Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "dev", "test"),
        help="Split to sample and validate in more detail.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of example rows to print per config.",
    )
    parser.add_argument(
        "--check-rows",
        type=int,
        default=200,
        help="Number of rows to inspect for the lightweight consistency checks.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def preview_value(value: object, limit: int = 100) -> str:
    if isinstance(value, str):
        compact = value.replace("\n", "\\n")
        return compact if len(compact) <= limit else compact[: limit - 3] + "..."
    return str(value)


def print_split_metadata(repo_id: str, configs: Iterable[str]) -> None:
    print_header("Hub Metadata")
    for config in configs:
        builder = load_dataset_builder(repo_id, config)
        print(f"[{config}]")
        for split_name, info in builder.info.splits.items():
            print(
                f"  {split_name}: num_examples={info.num_examples:,} num_bytes={info.num_bytes:,}"
            )


def print_examples(repo_id: str, config: str, split: str, sample_size: int) -> None:
    ds = load_dataset(repo_id, config, split=split, streaming=True)
    rows = list(islice(ds, sample_size))
    columns = list(rows[0].keys()) if rows else []
    print(f"[{config}] rows={len(rows)} columns={columns}")
    for idx, row in enumerate(rows):
        preview = {key: preview_value(value) for key, value in row.items()}
        print(f"  row {idx}: {preview}")


def load_small_slice(repo_id: str, config: str, split: str, n: int) -> list[dict]:
    ds = load_dataset(repo_id, config, split=split, streaming=True)
    return list(islice(ds, n))


def run_consistency_checks(repo_id: str, split: str, check_rows: int) -> None:
    print_header(f"Consistency Checks ({split})")
    documents = load_small_slice(repo_id, "documents", split, check_rows)
    queries = load_small_slice(repo_id, "queries", split, check_rows)
    candidates = load_small_slice(repo_id, "candidates", split, check_rows)
    ground_truth = load_small_slice(repo_id, "ground_truth", split, check_rows)

    document_ids = {row["doc_id"] for row in documents}
    query_ids = [row["query_id"] for row in queries]
    candidate_ids = {row["candidate_id"] for row in candidates}
    gt_query_ids = [row["query_id"] for row in ground_truth]

    builder_queries = load_dataset_builder(repo_id, "queries").info.splits[split].num_examples
    builder_ground_truth = load_dataset_builder(repo_id, "ground_truth").info.splits[split].num_examples

    print(
        f"metadata check: queries count == ground_truth count -> "
        f"{builder_queries:,} vs {builder_ground_truth:,} "
        f"({'OK' if builder_queries == builder_ground_truth else 'MISMATCH'})"
    )
    print(
        f"sample uniqueness: query_ids={len(set(query_ids))}/{len(query_ids)} "
        f"candidate_ids={len(candidate_ids)}/{len(candidates)} "
        f"doc_ids={len(document_ids)}/{len(documents)}"
    )

    missing_gt_queries = [qid for qid in gt_query_ids if qid not in set(query_ids)]
    print(
        f"ground_truth sample query coverage in sampled queries: "
        f"{len(gt_query_ids) - len(missing_gt_queries)}/{len(gt_query_ids)}"
    )

    gt_positive_total = sum(len(row["positive_ids"]) for row in ground_truth)
    gt_missing_positive_refs = sum(
        1
        for row in ground_truth
        for candidate_id in row["positive_ids"]
        if candidate_id not in candidate_ids
    )
    print(
        f"ground_truth sampled positives: total={gt_positive_total} "
        f"missing_in_sampled_candidates={gt_missing_positive_refs}"
    )

    missing_query_ids = set(query_ids)
    missing_candidate_ids = set(candidate_ids)
    for row in load_dataset(repo_id, "documents", split=split, streaming=True):
        doc_id = row["doc_id"]
        missing_query_ids.discard(doc_id)
        missing_candidate_ids.discard(doc_id)
        if not missing_query_ids and not missing_candidate_ids:
            break
    print(
        f"documents coverage for sampled ids: "
        f"queries_found={len(query_ids) - len(missing_query_ids)}/{len(query_ids)} "
        f"candidates_found={len(candidate_ids) - len(missing_candidate_ids)}/{len(candidate_ids)}"
    )


def main() -> None:
    args = parse_args()
    configs = get_dataset_config_names(args.repo_id)
    print(f"repo_id={args.repo_id}")
    print(f"configs={configs}")

    expected = list(CONFIGS)
    if configs != expected:
        print(f"warning: expected configs {expected}, got {configs}")

    print_split_metadata(args.repo_id, configs)

    print_header(f"Examples ({args.split})")
    for config in configs:
        print_examples(args.repo_id, config, args.split, args.sample_size)

    run_consistency_checks(args.repo_id, args.split, args.check_rows)


if __name__ == "__main__":
    main()
