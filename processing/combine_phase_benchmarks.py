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
from .language_audit import LanguageAuditConfig, run_language_audit
from .postprocess import _read_stage_documents, compute_language_targets, sample_documents, write_outputs
from .sampling import assign_document_ids, split_by_author_language
from .types import ProcessedDocument

logger = logging.getLogger(__name__)

DEFAULT_PHASE1_DIR = Path("processing/outputs/pipeline_phase1_official")
DEFAULT_PHASE2_DIR = Path("processing/second_phase_web_crawling/outputs/pipeline_phase2_official")
DEFAULT_COMBINED_OUTPUT_DIR = Path("processing/outputs/combined_phase1_official_phase2_all4_all_docs")


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
    if target_total >= len(docs):
        return (
            list(docs),
            {
                "mode": "all_docs",
                "available_total": len(docs),
                "requested_total": target_total,
                "final_total": len(docs),
            },
            [],
        )

    docs_by_lang: dict[str, list[ProcessedDocument]] = {}
    for doc in docs:
        docs_by_lang.setdefault(doc.lang, []).append(doc)
    lang_targets, lang_log = compute_language_targets(docs_by_lang, target_total)
    selected, sampling_log = sample_documents(docs, lang_targets, rng)
    return selected, lang_log, sampling_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine phase-1 and phase-2 AuthBench outputs into one benchmark. "
            "Defaults are wired to the official phase1 output and the all4 phase2 webcrawl output."
        )
    )
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMBINED_OUTPUT_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to write merge monitoring report (defaults to <output_dir>/merge_summary.json).",
    )
    parser.add_argument(
        "--total-docs",
        type=int,
        default=None,
        help="Final target docs. Default uses all available docs after dedup/overlap removal.",
    )
    parser.add_argument("--min-phase2-share", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--allow-lower-phase2-share", action="store_true")
    parser.add_argument(
        "--take-all-docs",
        action="store_true",
        help=(
            "Skip phase-wise sampling and keep all loaded docs after optional dedup/overlap handling. "
            "Best for full-pool merges."
        ),
    )
    parser.add_argument(
        "--disable-cross-phase-overlap-removal",
        action="store_true",
        help="Do not remove phase-1 docs that exactly overlap phase-2 text.",
    )

    parser.add_argument("--disable-dedup", action="store_true")
    parser.add_argument("--dedup-near-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--dedup-author-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--dedup-min-tokens-for-near", type=int, default=20)
    parser.add_argument("--dedup-lsh-bands", type=int, default=4)

    parser.add_argument(
        "--retag-languages",
        action="store_true",
        help="Run automated language audit and retag high-confidence mismatches before combining.",
    )
    parser.add_argument("--lang-audit-min-detect-chars", type=int, default=80)
    parser.add_argument("--lang-audit-max-text-chars", type=int, default=3000)
    parser.add_argument("--lang-audit-min-confidence", type=float, default=0.85)
    parser.add_argument("--lang-audit-min-script-chars", type=int, default=8)
    parser.add_argument("--lang-audit-max-detect-docs", type=int, default=200000)
    parser.add_argument("--lang-audit-max-suspects", type=int, default=5000)
    parser.add_argument("--lang-audit-drop-detected-mismatches", action="store_true")

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if not 0 <= args.min_phase2_share <= 1:
        raise ValueError(f"--min-phase2-share must be in [0,1], got {args.min_phase2_share}")

    split_total = args.train_ratio + args.dev_ratio + args.test_ratio
    if split_total <= 0:
        raise ValueError("Split ratios must sum to > 0.")

    train_ratio = args.train_ratio / split_total
    dev_ratio = args.dev_ratio / split_total
    test_ratio = args.test_ratio / split_total

    report_path = args.report_path or (args.output_dir / "merge_summary.json")
    rng = random.Random(args.seed)
    start_time = time.perf_counter()

    logger.info("Loading phase-1 stage docs from %s", args.phase1_dir)
    phase1_docs = _read_stage_documents(args.phase1_dir)
    logger.info("Loading phase-2 stage docs from %s", args.phase2_dir)
    phase2_docs = _read_stage_documents(args.phase2_dir)
    phase1_loaded_count = len(phase1_docs)
    phase2_loaded_count = len(phase2_docs)

    for doc in phase1_docs:
        doc.metadata = {**doc.metadata, "phase": "phase1"}
    for doc in phase2_docs:
        doc.metadata = {**doc.metadata, "phase": "phase2"}

    phase1_lang_audit_summary = {"skipped": True}
    phase2_lang_audit_summary = {"skipped": True}
    if args.retag_languages:
        logger.info("Running language retag audit on phase pools.")
        audit_cfg = LanguageAuditConfig(
            enabled=True,
            min_detect_chars=args.lang_audit_min_detect_chars,
            max_text_chars=args.lang_audit_max_text_chars,
            min_confidence=args.lang_audit_min_confidence,
            min_script_chars=args.lang_audit_min_script_chars,
            max_detect_docs=args.lang_audit_max_detect_docs,
            max_suspects=args.lang_audit_max_suspects,
            drop_detected_mismatches=args.lang_audit_drop_detected_mismatches,
            seed=args.seed,
        )
        phase1_docs, phase1_lang_audit_summary, _ = run_language_audit(phase1_docs, config=audit_cfg)
        phase2_docs, phase2_lang_audit_summary, _ = run_language_audit(phase2_docs, config=audit_cfg)

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

    phase1_overlap_removed = 0
    if args.disable_cross_phase_overlap_removal:
        logger.info("Skipping cross-phase exact-overlap removal.")
    else:
        # Prefer phase-2 examples on exact text overlap to preserve newly crawled coverage.
        phase2_text_keys = {
            _text_key(doc.text)
            for doc in phase2_docs
            if doc.text and doc.token_length > 0
        }
        phase1_before_overlap = len(phase1_docs)
        phase1_docs = [doc for doc in phase1_docs if _text_key(doc.text) not in phase2_text_keys]
        phase1_overlap_removed = phase1_before_overlap - len(phase1_docs)

    available_total = len(phase1_docs) + len(phase2_docs)
    if args.take_all_docs:
        if args.total_docs is not None and args.total_docs < available_total:
            raise ValueError(
                "--take-all-docs cannot be combined with --total-docs smaller than available docs."
            )
        target_total = available_total
        if args.total_docs is not None and args.total_docs > available_total:
            logger.warning(
                "Requested --total-docs=%d exceeds available pool (%d). Using %d.",
                args.total_docs,
                available_total,
                target_total,
            )

        phase1_target = len(phase1_docs)
        phase2_target = len(phase2_docs)
        phase1_selected = list(phase1_docs)
        phase2_selected = list(phase2_docs)
        phase1_lang_log = {
            "mode": "take_all_docs",
            "available_total": len(phase1_docs),
            "selected_total": len(phase1_selected),
        }
        phase2_lang_log = {
            "mode": "take_all_docs",
            "available_total": len(phase2_docs),
            "selected_total": len(phase2_selected),
        }
        phase1_sampling_log: list[dict] = []
        phase2_sampling_log: list[dict] = []
        logger.info(
            "Take-all mode enabled: selecting full pools phase1=%d phase2=%d total=%d",
            phase1_target,
            phase2_target,
            phase1_target + phase2_target,
        )
    else:
        if args.total_docs is None:
            target_total = available_total
        else:
            target_total = min(args.total_docs, available_total)
            if args.total_docs > available_total:
                logger.warning(
                    "Requested --total-docs=%d exceeds available pool (%d). Using %d.",
                    args.total_docs,
                    available_total,
                    target_total,
                )

        phase2_min = int(math.ceil(target_total * args.min_phase2_share))
        phase1_cap = max(target_total - phase2_min, 0)

        phase1_target = min(len(phase1_docs), phase1_cap)
        phase2_target = target_total - phase1_target

        if len(phase2_docs) < phase2_target:
            if not args.allow_lower_phase2_share:
                raise RuntimeError(
                    "Phase-2 pool is too small after deduplication: "
                    f"required={phase2_target}, available={len(phase2_docs)}. "
                    "Use --allow-lower-phase2-share to proceed with best effort."
                )
            phase2_target = len(phase2_docs)
            phase1_target = min(len(phase1_docs), target_total - phase2_target)

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
    splits = split_by_author_language(
        combined_docs,
        split_ratios=split_ratios,
        rng=rng,
    )

    split_summary = write_outputs(
        splits,
        args.output_dir,
        rng,
        retain_all_docs=True,
        write_documents_jsonl=True,
        metadata_fields=("phase", "input_split", "input_doc_type"),
    )
    split_summary = {
        split: {
            **details,
            "documents_by_lang": _sorted_counter(details["documents_by_lang"]),
            "candidates_by_lang": _sorted_counter(details["candidates_by_lang"]),
            "queries_by_lang": _sorted_counter(details["queries_by_lang"]),
        }
        for split, details in split_summary.items()
    }

    phase2_share = (len(phase2_selected) / len(combined_docs)) if combined_docs else 0.0
    if (
        phase2_share < args.min_phase2_share
        and not args.allow_lower_phase2_share
        and target_total > 0
    ):
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
            "requested_total_docs": args.total_docs,
            "effective_total_docs": target_total,
            "min_phase2_share": args.min_phase2_share,
            "take_all_docs": args.take_all_docs,
            "disable_cross_phase_overlap_removal": args.disable_cross_phase_overlap_removal,
            "seed": args.seed,
            "ratios": {"train": train_ratio, "dev": dev_ratio, "test": test_ratio},
            "preserve_all_selected_docs_in_output": True,
            "write_documents_jsonl": True,
            "author_aware_split": True,
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
            "combined_exported_documents": sum(details["documents"] for details in split_summary.values()),
            "combined_exported_candidates": sum(details["candidates"] for details in split_summary.values()),
            "combined_exported_queries": sum(details["queries"] for details in split_summary.values()),
        },
        "phase1_dedup": phase1_dedup_summary,
        "phase2_dedup": phase2_dedup_summary,
        "phase1_language_audit": phase1_lang_audit_summary,
        "phase2_language_audit": phase2_lang_audit_summary,
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
