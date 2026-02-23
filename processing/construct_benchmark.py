from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from . import build_benchmark
from .config import CHUNKING_DEFAULTS, TARGET_TOTAL_DOCS, default_manifest_path, make_split_ratios
from .deduplication import DedupConfig, deduplicate_documents
from .postprocess import (
    _read_candidates,
    compute_language_targets,
    filter_documents,
    sample_documents,
    write_outputs,
)
from .sampling import split_by_language
from .types import ProcessedDocument
from .utils import length_bucket, write_jsonl

logger = logging.getLogger(__name__)


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {k: counter[k] for k in sorted(counter)}


def _summarize_docs(docs: Iterable[ProcessedDocument]) -> dict:
    lang_counter = Counter()
    genre_counter = Counter()
    bucket_counter = Counter()
    total = 0
    for doc in docs:
        total += 1
        lang_counter[doc.lang] += 1
        genre_counter[doc.genre] += 1
        bucket_counter[doc.length_bucket or length_bucket(doc.token_length)] += 1
    return {
        "total": total,
        "by_lang": _sorted_counter(lang_counter),
        "by_genre": _sorted_counter(genre_counter),
        "by_length_bucket": _sorted_counter(bucket_counter),
    }


def _parse_dataset_caps(pairs: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for pair in pairs or []:
        if "=" not in pair:
            logger.warning("Ignoring dataset cap '%s' (expected name=value).", pair)
            continue
        key, val = pair.split("=", 1)
        key = key.strip().lower()
        try:
            caps[key] = int(val)
        except ValueError:
            logger.warning("Ignoring dataset cap '%s' (value must be int).", pair)
    return caps


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return default


def _to_plain_jsonable(value):
    if isinstance(value, Counter):
        return {k: value[k] for k in sorted(value)}
    if isinstance(value, dict):
        return {k: _to_plain_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_jsonable(v) for v in value]
    return value


def _postprocess_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        spacing_collapse_ratio=args.post_spacing_collapse_ratio,
        min_spacing_run=args.post_min_spacing_run,
        max_single_letter_ratio=args.post_max_single_letter_ratio,
        max_single_letter_run=args.post_max_single_letter_run,
        min_alpha_ratio=args.post_min_alpha_ratio,
        min_alpha_token_ratio=args.post_min_alpha_token_ratio,
        skip_langdetect=args.post_skip_langdetect,
    )


def run(args: argparse.Namespace) -> dict:
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    split_ratios = make_split_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)
    post_split_ratios = make_split_ratios(
        args.post_train_ratio if args.post_train_ratio is not None else args.train_ratio,
        args.post_dev_ratio if args.post_dev_ratio is not None else args.dev_ratio,
        args.post_test_ratio if args.post_test_ratio is not None else args.test_ratio,
    )

    stage1_dir = args.stage1_output_dir
    final_dir = args.output_dir
    report_path = args.report_path
    if report_path.exists() and not args.overwrite_report:
        raise FileExistsError(
            f"Report already exists: {report_path}. Pass --overwrite-report to replace it."
        )

    dataset_max_docs = _parse_dataset_caps(args.dataset_max_docs)
    no_shuffle_datasets = set(map(str.lower, args.no_shuffle_datasets or []))
    chunking_params = {
        "max_tokens": args.max_chunk_tokens,
        "target_chunk_tokens": args.target_chunk_tokens,
        "min_chunk_tokens": args.min_chunk_tokens,
        "chunk_probability": args.chunk_probability,
    }

    overall_start = time.perf_counter()
    rng = random.Random(args.seed)

    build_start = time.perf_counter()
    logger.info("Stage 1/3: build benchmark candidates -> %s", stage1_dir)
    build_benchmark.run(
        manifest_path=args.manifest,
        output_dir=stage1_dir,
        total_docs=args.total_docs,
        split_ratios=split_ratios,
        seed=args.seed,
        sanity_check=args.sanity_check,
        sanity_limit=args.sanity_limit,
        max_documents_per_dataset=args.max_documents_per_dataset,
        shuffle_buffer_size=args.shuffle_buffer_size,
        no_shuffle_datasets=no_shuffle_datasets,
        dataset_max_docs=dataset_max_docs,
        allow_other_languages=args.allow_other_languages,
        chunking_params=chunking_params,
        truncate_to_tokens=args.truncate_to_tokens,
    )
    build_runtime = round(time.perf_counter() - build_start, 3)

    stage1_summary_path = stage1_dir / "processing_summary.json"
    stage1_sampling_shortfall_path = stage1_dir / "sampling_shortfall.json"
    build_summary = _load_json(stage1_summary_path, default={})
    build_sampling_shortfall = _load_json(stage1_sampling_shortfall_path, default=[])

    stage1_docs = _read_candidates(stage1_dir)
    logger.info("Loaded %d stage-1 candidate docs for stage-2 processing.", len(stage1_docs))

    post_start = time.perf_counter()
    logger.info("Stage 2/3: postprocess filtering + dedup + final sampling -> %s", final_dir)
    post_args = _postprocess_namespace(args)
    filtered_docs, drop_reasons, drop_by_lang, drop_records = filter_documents(stage1_docs, post_args)
    if not filtered_docs:
        raise RuntimeError("No documents remain after postprocess filtering.")

    dedup_summary: dict
    dedup_docs = filtered_docs
    if args.disable_dedup:
        dedup_summary = {
            "skipped": True,
            "reason": "Deduplication disabled by --disable-dedup.",
            "input_docs": len(filtered_docs),
            "after_author_dedup": len(filtered_docs),
            "dropped_total": 0,
        }
    else:
        dedup_cfg = DedupConfig(
            exact_text=not args.dedup_disable_exact_text,
            near_text=not args.dedup_disable_near_text,
            near_similarity_threshold=args.dedup_near_similarity_threshold,
            near_lsh_bands=args.dedup_lsh_bands,
            min_tokens_for_near=args.dedup_min_tokens_for_near,
            near_same_language_only=not args.dedup_allow_cross_language_near,
            author_similarity=not args.dedup_disable_author_similarity,
            author_similarity_threshold=args.dedup_author_similarity_threshold,
            author_cross_source_only=not args.dedup_allow_same_source_author_similarity,
            author_same_language_only=not args.dedup_allow_cross_language_author_similarity,
            author_profile_docs=args.dedup_author_profile_docs,
            max_bucket_size=args.dedup_max_bucket_size,
        )
        dedup_docs, dedup_summary = deduplicate_documents(filtered_docs, config=dedup_cfg)
        if not dedup_docs:
            raise RuntimeError("No documents remain after deduplication.")

    target_total = args.post_target_total or len(dedup_docs)
    docs_by_lang: dict[str, list[ProcessedDocument]] = {}
    for doc in dedup_docs:
        docs_by_lang.setdefault(doc.lang, []).append(doc)
    lang_targets, lang_log = compute_language_targets(docs_by_lang, target_total)
    selected_docs, sampling_log = sample_documents(dedup_docs, lang_targets, rng)
    splits = split_by_language(selected_docs, post_split_ratios, rng)
    split_summary = write_outputs(splits, final_dir, rng)

    dirty_log_path = final_dir / "postprocess_dirty.log"
    if drop_records:
        write_jsonl(dirty_log_path, drop_records)
        logger.info("Wrote postprocess drop log to %s (%d rows)", dirty_log_path, len(drop_records))

    post_summary = {
        "input_dir": str(stage1_dir),
        "output_dir": str(final_dir),
        "before_filter": _summarize_docs(stage1_docs),
        "after_filter": _summarize_docs(filtered_docs),
        "after_dedup": _summarize_docs(dedup_docs),
        "after_sampling": _summarize_docs(selected_docs),
        "drop_reasons": dict(drop_reasons),
        "drop_reasons_by_lang": dict(drop_by_lang),
        "dirty_log_path": str(dirty_log_path) if drop_records else None,
        "dirty_log_count": len(drop_records),
        "language_targets": lang_log,
        "sampling_deficits": sampling_log,
        "deduplication": dedup_summary,
        "splits": _to_plain_jsonable(split_summary),
        "runtime_seconds": round(time.perf_counter() - post_start, 3),
    }
    (final_dir / "postprocessing_summary.json").write_text(
        json.dumps(_to_plain_jsonable(post_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Stage 3/3: writing unified monitoring report -> %s", report_path)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "report_runtime_seconds": round(time.perf_counter() - overall_start, 3),
        "pipeline_inputs": {
            "manifest": str(args.manifest),
            "stage1_output_dir": str(stage1_dir),
            "final_output_dir": str(final_dir),
            "total_docs": args.total_docs,
            "post_target_total": args.post_target_total,
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
            "post_train_ratio": args.post_train_ratio,
            "post_dev_ratio": args.post_dev_ratio,
            "post_test_ratio": args.post_test_ratio,
            "allow_other_languages": args.allow_other_languages,
            "max_documents_per_dataset": args.max_documents_per_dataset,
            "dataset_max_docs": dict(sorted(dataset_max_docs.items())),
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "no_shuffle_datasets": sorted(no_shuffle_datasets),
            "chunking": chunking_params,
            "truncate_to_tokens": args.truncate_to_tokens,
            "post_filtering": {
                "spacing_collapse_ratio": args.post_spacing_collapse_ratio,
                "min_spacing_run": args.post_min_spacing_run,
                "max_single_letter_ratio": args.post_max_single_letter_ratio,
                "max_single_letter_run": args.post_max_single_letter_run,
                "min_alpha_ratio": args.post_min_alpha_ratio,
                "min_alpha_token_ratio": args.post_min_alpha_token_ratio,
                "skip_langdetect": args.post_skip_langdetect,
            },
        },
        "build_stage": {
            "runtime_seconds": build_runtime,
            "summary_path": str(stage1_summary_path),
            "sampling_shortfall_path": str(stage1_sampling_shortfall_path),
            "summary": build_summary,
            "sampling_shortfall": build_sampling_shortfall,
            "stage1_candidates_loaded_for_postprocess": _summarize_docs(stage1_docs),
        },
        "postprocess_dedup_stage": _to_plain_jsonable(post_summary),
        "stage_transitions": {
            "build_after_sampling_total": build_summary.get("after_sampling", {}).get("total"),
            "post_before_filter_total": post_summary["before_filter"]["total"],
            "post_after_filter_total": post_summary["after_filter"]["total"],
            "post_after_dedup_total": post_summary["after_dedup"]["total"],
            "post_after_sampling_total": post_summary["after_sampling"]["total"],
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_to_plain_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Unified pipeline complete. Final outputs at %s", final_dir.resolve())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified AuthBench construction pipeline: build + postprocess + dedup + "
            "monitoring report in one run."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--stage1-output-dir", type=Path, default=Path("processing/outputs/stage1"))
    parser.add_argument("--output-dir", type=Path, default=Path("processing/outputs/stage2"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("processing/outputs/monitoring/pipeline_dynamics.json"),
    )
    parser.add_argument("--overwrite-report", action="store_true")

    parser.add_argument("--total-docs", type=int, default=TARGET_TOTAL_DOCS)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--post-train-ratio", type=float, default=None)
    parser.add_argument("--post-dev-ratio", type=float, default=None)
    parser.add_argument("--post-test-ratio", type=float, default=None)

    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--sanity-limit", type=int, default=2000)
    parser.add_argument("--allow-other-languages", action="store_true")
    parser.add_argument("--max-documents-per-dataset", type=int, default=None)
    parser.add_argument("--dataset-max-docs", nargs="*", default=[])
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--no-shuffle-datasets", nargs="*", default=[])

    parser.add_argument("--max-chunk-tokens", type=int, default=CHUNKING_DEFAULTS["max_tokens"])
    parser.add_argument(
        "--target-chunk-tokens",
        type=int,
        default=CHUNKING_DEFAULTS["target_chunk_tokens"],
    )
    parser.add_argument("--min-chunk-tokens", type=int, default=CHUNKING_DEFAULTS["min_chunk_tokens"])
    parser.add_argument(
        "--chunk-probability",
        type=float,
        default=CHUNKING_DEFAULTS.get("chunk_probability", 1.0),
    )
    parser.add_argument("--truncate-to-tokens", type=int, default=None)

    parser.add_argument("--post-target-total", type=int, default=None)
    parser.add_argument("--post-spacing-collapse-ratio", type=float, default=0.25)
    parser.add_argument("--post-min-spacing-run", type=int, default=2)
    parser.add_argument("--post-max-single-letter-ratio", type=float, default=0.45)
    parser.add_argument("--post-max-single-letter-run", type=int, default=10)
    parser.add_argument("--post-min-alpha-ratio", type=float, default=0.25)
    parser.add_argument("--post-min-alpha-token-ratio", type=float, default=0.5)
    parser.add_argument("--post-skip-langdetect", action="store_true")

    parser.add_argument("--disable-dedup", action="store_true")
    parser.add_argument("--dedup-disable-exact-text", action="store_true")
    parser.add_argument("--dedup-disable-near-text", action="store_true")
    parser.add_argument("--dedup-near-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--dedup-min-tokens-for-near", type=int, default=20)
    parser.add_argument("--dedup-lsh-bands", type=int, default=4)
    parser.add_argument("--dedup-allow-cross-language-near", action="store_true")
    parser.add_argument("--dedup-disable-author-similarity", action="store_true")
    parser.add_argument("--dedup-author-similarity-threshold", type=float, default=0.94)
    parser.add_argument("--dedup-author-profile-docs", type=int, default=3)
    parser.add_argument("--dedup-allow-same-source-author-similarity", action="store_true")
    parser.add_argument("--dedup-allow-cross-language-author-similarity", action="store_true")
    parser.add_argument("--dedup-max-bucket-size", type=int, default=512)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
