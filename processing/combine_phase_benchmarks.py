from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import make_split_ratios
from .deduplication import DedupConfig, deduplicate_documents
from .postprocess import _read_candidates, compute_language_targets, sample_documents
from .sampling import assign_document_ids, build_retrieval_sets, split_by_language
from .types import ProcessedDocument
from .utils import write_jsonl

logger = logging.getLogger(__name__)


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {k: counter[k] for k in sorted(counter)}


def _summarize_docs(docs: Iterable[ProcessedDocument]) -> dict:
    by_lang = Counter()
    by_genre = Counter()
    by_source = Counter()
    total = 0
    for doc in docs:
        total += 1
        by_lang[doc.lang] += 1
        by_genre[doc.genre] += 1
        by_source[doc.source] += 1
    return {
        "total": total,
        "by_lang": _sorted_counter(by_lang),
        "by_genre": _sorted_counter(by_genre),
        "by_source": _sorted_counter(by_source),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _text_key(text: str) -> bytes:
    normalized = _normalize_text(text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def _sample_pool(
    *,
    docs: list[ProcessedDocument],
    target_total: int,
    rng: random.Random,
) -> tuple[list[ProcessedDocument], dict, list[dict]]:
    docs_by_lang: dict[str, list[ProcessedDocument]] = {}
    for doc in docs:
        docs_by_lang.setdefault(doc.lang, []).append(doc)
    lang_targets, lang_log = compute_language_targets(docs_by_lang, target_total)
    selected, sampling_log = sample_documents(docs, lang_targets, rng)
    return selected, lang_log, sampling_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine phase-1 and phase-2 AuthBench outputs into one benchmark "
            "with target total size and minimum phase-2 share."
        )
    )
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to write merge monitoring report (defaults to <output_dir>/merge_summary.json).",
    )
    parser.add_argument("--total-docs", type=int, default=1_000_000)
    parser.add_argument("--min-phase2-share", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--allow-lower-phase2-share", action="store_true")

    parser.add_argument("--disable-dedup", action="store_true")
    parser.add_argument("--dedup-near-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--dedup-author-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--dedup-min-tokens-for-near", type=int, default=20)
    parser.add_argument("--dedup-lsh-bands", type=int, default=4)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if not 0 < args.min_phase2_share <= 1:
        raise ValueError(f"--min-phase2-share must be in (0,1], got {args.min_phase2_share}")

    split_total = args.train_ratio + args.dev_ratio + args.test_ratio
    if split_total <= 0:
        raise ValueError("Split ratios must sum to > 0.")

    train_ratio = args.train_ratio / split_total
    dev_ratio = args.dev_ratio / split_total
    test_ratio = args.test_ratio / split_total

    report_path = args.report_path or (args.output_dir / "merge_summary.json")
    rng = random.Random(args.seed)
    start_time = time.perf_counter()

    logger.info("Loading phase-1 candidates from %s", args.phase1_dir)
    phase1_docs = _read_candidates(args.phase1_dir)
    logger.info("Loading phase-2 candidates from %s", args.phase2_dir)
    phase2_docs = _read_candidates(args.phase2_dir)
    phase1_loaded_count = len(phase1_docs)
    phase2_loaded_count = len(phase2_docs)

    for doc in phase1_docs:
        doc.metadata = {**doc.metadata, "phase": "phase1"}
    for doc in phase2_docs:
        doc.metadata = {**doc.metadata, "phase": "phase2"}

    phase1_dedup_summary = {"skipped": True}
    phase2_dedup_summary = {"skipped": True}
    if not args.disable_dedup:
        dedup_cfg = DedupConfig(
            near_similarity_threshold=args.dedup_near_similarity_threshold,
            author_similarity_threshold=args.dedup_author_similarity_threshold,
            min_tokens_for_near=args.dedup_min_tokens_for_near,
            near_lsh_bands=args.dedup_lsh_bands,
        )
        phase1_docs, phase1_dedup_summary = deduplicate_documents(phase1_docs, config=dedup_cfg)
        phase2_docs, phase2_dedup_summary = deduplicate_documents(phase2_docs, config=dedup_cfg)
    phase1_internal_dedup_count = len(phase1_docs)
    phase2_internal_dedup_count = len(phase2_docs)

    # Prefer phase-2 examples on exact text overlap to preserve newly crawled coverage.
    phase2_text_keys = {
        _text_key(doc.text)
        for doc in phase2_docs
        if doc.text and doc.token_length > 0
    }
    phase1_before_overlap = len(phase1_docs)
    phase1_docs = [doc for doc in phase1_docs if _text_key(doc.text) not in phase2_text_keys]
    phase1_overlap_removed = phase1_before_overlap - len(phase1_docs)

    phase2_min = int(math.ceil(args.total_docs * args.min_phase2_share))
    phase1_cap = max(args.total_docs - phase2_min, 0)

    phase1_target = min(len(phase1_docs), phase1_cap)
    phase2_target = args.total_docs - phase1_target

    if len(phase2_docs) < phase2_target:
        if not args.allow_lower_phase2_share:
            raise RuntimeError(
                "Phase-2 pool is too small after deduplication: "
                f"required={phase2_target}, available={len(phase2_docs)}. "
                "Use --allow-lower-phase2-share to proceed with best effort."
            )
        phase2_target = len(phase2_docs)
        phase1_target = min(len(phase1_docs), args.total_docs - phase2_target)

    logger.info(
        "Sampling phase targets: phase1=%d phase2=%d total=%d",
        phase1_target,
        phase2_target,
        phase1_target + phase2_target,
    )

    phase1_selected, phase1_lang_log, phase1_sampling_log = _sample_pool(
        docs=phase1_docs,
        target_total=phase1_target,
        rng=rng,
    )
    phase2_selected, phase2_lang_log, phase2_sampling_log = _sample_pool(
        docs=phase2_docs,
        target_total=phase2_target,
        rng=rng,
    )

    combined_docs = phase1_selected + phase2_selected
    combined_docs = assign_document_ids(combined_docs, prefix="mix")

    split_ratios = make_split_ratios(train_ratio, dev_ratio, test_ratio)
    splits = split_by_language(
        combined_docs,
        split_ratios=split_ratios,
        rng=rng,
    )

    split_summary: dict[str, dict] = {}
    for split_name, docs in splits.items():
        candidates, queries, ground_truth = build_retrieval_sets(docs, rng)
        split_path = args.output_dir / split_name
        write_jsonl(split_path / "candidates.jsonl", candidates)
        write_jsonl(split_path / "queries.jsonl", queries)
        write_jsonl(split_path / "ground_truth.jsonl", ground_truth)
        split_summary[split_name] = {
            "documents": len(docs),
            "candidates": len(candidates),
            "queries": len(queries),
            "ground_truth": len(ground_truth),
            "documents_by_lang": _sorted_counter(Counter(doc.lang for doc in docs)),
        }

    phase2_share = (len(phase2_selected) / len(combined_docs)) if combined_docs else 0.0
    if phase2_share < args.min_phase2_share and not args.allow_lower_phase2_share:
        raise RuntimeError(
            "Final merged share for phase-2 is below requested minimum: "
            f"required>={args.min_phase2_share:.3f}, got={phase2_share:.3f}."
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
        "inputs": {
            "phase1_dir": str(args.phase1_dir),
            "phase2_dir": str(args.phase2_dir),
            "output_dir": str(args.output_dir),
            "total_docs": args.total_docs,
            "min_phase2_share": args.min_phase2_share,
            "seed": args.seed,
            "ratios": {"train": train_ratio, "dev": dev_ratio, "test": test_ratio},
        },
        "stage_counts": {
            "phase1_loaded": phase1_loaded_count,
            "phase2_loaded": phase2_loaded_count,
            "phase1_after_internal_dedup": phase1_internal_dedup_count,
            "phase2_after_internal_dedup": phase2_internal_dedup_count,
            "phase1_removed_by_cross_phase_exact": phase1_overlap_removed,
            "phase1_pool_after_cross_phase_exact": len(phase1_docs),
            "phase1_target": phase1_target,
            "phase2_target": phase2_target,
            "phase1_selected": len(phase1_selected),
            "phase2_selected": len(phase2_selected),
            "combined_selected": len(combined_docs),
            "phase2_share_final": phase2_share,
        },
        "phase1_dedup": phase1_dedup_summary,
        "phase2_dedup": phase2_dedup_summary,
        "phase1_pool_summary": _summarize_docs(phase1_docs),
        "phase2_pool_summary": _summarize_docs(phase2_docs),
        "phase1_selected_summary": _summarize_docs(phase1_selected),
        "phase2_selected_summary": _summarize_docs(phase2_selected),
        "combined_summary": _summarize_docs(combined_docs),
        "phase1_language_targets": phase1_lang_log,
        "phase2_language_targets": phase2_lang_log,
        "phase1_sampling_log": phase1_sampling_log,
        "phase2_sampling_log": phase2_sampling_log,
        "splits": split_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote combined benchmark outputs to %s", args.output_dir.resolve())
    logger.info("Wrote merge summary report to %s", report_path.resolve())
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
