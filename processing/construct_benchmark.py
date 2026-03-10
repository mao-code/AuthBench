from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from . import build_benchmark
from .config import CHUNKING_DEFAULTS, TARGET_TOTAL_DOCS, default_manifest_path, make_split_ratios
from .deduplication import DedupConfig, deduplicate_documents
from .language_audit import LanguageAuditConfig, run_language_audit
from .postprocess import (
    _read_stage_documents,
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


def _language_audit_config(args: argparse.Namespace) -> LanguageAuditConfig:
    return LanguageAuditConfig(
        enabled=not args.disable_lang_audit,
        min_detect_chars=args.lang_audit_min_detect_chars,
        max_text_chars=args.lang_audit_max_text_chars,
        min_confidence=args.lang_audit_min_confidence,
        min_script_chars=args.lang_audit_min_script_chars,
        max_detect_docs=args.lang_audit_max_detect_docs,
        max_suspects=args.lang_audit_max_suspects,
        drop_detected_mismatches=args.lang_audit_drop_detected_mismatches,
        seed=args.seed,
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or (output_dir / "pipeline_dynamics.json")
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
    temp_work_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temp_work_dir = tempfile.TemporaryDirectory(
            prefix=f".pipeline_work_{output_dir.name}_",
            dir=str(output_dir.parent),
        )
        work_dir = Path(temp_work_dir.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    build_dir = work_dir / "build_artifacts"
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_start = time.perf_counter()
        logger.info("Stage 1/5: build author-filtered corpus -> %s", build_dir)
        build_benchmark.run(
            manifest_path=args.manifest,
            output_dir=build_dir,
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
            defer_sampling=True,
            write_retrieval_artifacts=False,
        )
        build_runtime = round(time.perf_counter() - build_start, 3)

        build_summary_path = build_dir / "processing_summary.json"
        build_sampling_shortfall_path = build_dir / "sampling_shortfall.json"
        build_summary = _load_json(build_summary_path, default={})
        build_sampling_shortfall = _load_json(build_sampling_shortfall_path, default=[])

        build_docs = _read_stage_documents(build_dir)
        if not build_docs:
            raise RuntimeError(f"No build-stage docs found under {build_dir}.")
        logger.info("Loaded %d docs from build stage.", len(build_docs))

        finalize_start = time.perf_counter()
        logger.info("Stage 2/5: quality filtering")
        post_args = _postprocess_namespace(args)
        filtered_docs, drop_reasons, drop_by_lang, drop_records = filter_documents(build_docs, post_args)
        if not filtered_docs:
            raise RuntimeError("No documents remain after quality filtering.")

        logger.info("Stage 3/5: deduplication")
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

        logger.info("Stage 4/5: language audit")
        audit_start = time.perf_counter()
        audit_cfg = _language_audit_config(args)
        audited_docs, language_audit_summary, language_suspects = run_language_audit(
            dedup_docs,
            config=audit_cfg,
        )
        language_audit_summary = _to_plain_jsonable(language_audit_summary)
        language_audit_summary["runtime_seconds"] = round(time.perf_counter() - audit_start, 3)
        if not audited_docs:
            raise RuntimeError("No documents remain after language audit.")

        language_audit_log = output_dir / "language_audit_suspects.jsonl"
        if language_suspects:
            write_jsonl(language_audit_log, language_suspects)
            logger.info(
                "Wrote language-audit suspects to %s (%d rows)",
                language_audit_log,
                len(language_suspects),
            )

        logger.info("Stage 5/5: final balanced sampling + split writing -> %s", output_dir)
        target_total = args.post_target_total if args.post_target_total is not None else args.total_docs
        docs_by_lang: dict[str, list[ProcessedDocument]] = {}
        for doc in audited_docs:
            docs_by_lang.setdefault(doc.lang, []).append(doc)
        lang_targets, lang_log = compute_language_targets(docs_by_lang, target_total)
        selected_docs, sampling_log = sample_documents(audited_docs, lang_targets, rng)
        splits = split_by_language(selected_docs, post_split_ratios, rng)
        split_summary = write_outputs(splits, output_dir, rng)

        quality_drop_log = output_dir / "quality_filter_drops.log"
        if drop_records:
            write_jsonl(quality_drop_log, drop_records)
            logger.info("Wrote quality-filter drop log to %s (%d rows)", quality_drop_log, len(drop_records))

        finalize_summary = {
            "input_docs": _summarize_docs(build_docs),
            "after_filter": _summarize_docs(filtered_docs),
            "after_dedup": _summarize_docs(dedup_docs),
            "after_language_audit": _summarize_docs(audited_docs),
            "after_sampling": _summarize_docs(selected_docs),
            "drop_reasons": dict(drop_reasons),
            "drop_reasons_by_lang": dict(drop_by_lang),
            "quality_drop_log_path": str(quality_drop_log) if drop_records else None,
            "quality_drop_log_count": len(drop_records),
            "language_audit": language_audit_summary,
            "language_audit_log_path": str(language_audit_log) if language_suspects else None,
            "language_audit_log_count": len(language_suspects),
            "language_targets": lang_log,
            "sampling_deficits": sampling_log,
            "deduplication": dedup_summary,
            "splits": _to_plain_jsonable(split_summary),
            "runtime_seconds": round(time.perf_counter() - finalize_start, 3),
        }

        build_output_total = build_summary.get("build_output", {}).get("total")
        if build_output_total is None:
            build_output_total = build_summary.get("after_sampling", {}).get("total")

        pipeline_summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "build": {
                "runtime_seconds": build_runtime,
                "summary": build_summary,
                "sampling_shortfall": build_sampling_shortfall,
            },
            "finalize": _to_plain_jsonable(finalize_summary),
        }
        (output_dir / "pipeline_summary.json").write_text(
            json.dumps(_to_plain_jsonable(pipeline_summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Writing monitoring dynamics -> %s", report_path)
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "report_runtime_seconds": round(time.perf_counter() - overall_start, 3),
            "pipeline_inputs": {
                "manifest": str(args.manifest),
                "output_dir": str(output_dir),
                "work_dir": str(args.work_dir) if args.work_dir else None,
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
                "quality_filtering": {
                    "spacing_collapse_ratio": args.post_spacing_collapse_ratio,
                    "min_spacing_run": args.post_min_spacing_run,
                    "max_single_letter_ratio": args.post_max_single_letter_ratio,
                    "max_single_letter_run": args.post_max_single_letter_run,
                    "min_alpha_ratio": args.post_min_alpha_ratio,
                    "min_alpha_token_ratio": args.post_min_alpha_token_ratio,
                    "skip_langdetect": args.post_skip_langdetect,
                },
                "language_audit": {
                    "enabled": not args.disable_lang_audit,
                    "min_detect_chars": args.lang_audit_min_detect_chars,
                    "max_text_chars": args.lang_audit_max_text_chars,
                    "min_confidence": args.lang_audit_min_confidence,
                    "min_script_chars": args.lang_audit_min_script_chars,
                    "max_detect_docs": args.lang_audit_max_detect_docs,
                    "max_suspects": args.lang_audit_max_suspects,
                    "drop_detected_mismatches": args.lang_audit_drop_detected_mismatches,
                },
            },
            "stages": {
                "build": {
                    "runtime_seconds": build_runtime,
                    "summary": build_summary,
                    "sampling_shortfall": build_sampling_shortfall,
                    "documents_for_finalize": _summarize_docs(build_docs),
                },
                "quality_dedup_sampling": _to_plain_jsonable(finalize_summary),
            },
            "stage_transitions": {
                "build_after_sampling_total": build_output_total,
                "build_output_total": build_output_total,
                "build_sampling_deferred": bool(
                    build_summary.get("sampling", {}).get("deferred_to_finalize")
                ),
                "quality_input_total": finalize_summary["input_docs"]["total"],
                "quality_after_filter_total": finalize_summary["after_filter"]["total"],
                "after_dedup_total": finalize_summary["after_dedup"]["total"],
                "after_language_audit_total": finalize_summary["after_language_audit"]["total"],
                "final_after_sampling_total": finalize_summary["after_sampling"]["total"],
            },
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_to_plain_jsonable(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Unified pipeline complete. Final outputs at %s", output_dir.resolve())
        return report
    finally:
        if temp_work_dir is not None:
            temp_work_dir.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified AuthBench construction pipeline: build + quality filter + dedup + "
            "language audit + "
            "monitoring report in one run."
        )
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--output-dir", type=Path, default=Path("processing/outputs/pipeline"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Optional persistent working directory for intermediate build artifacts. "
            "If unset, a temporary directory is used and cleaned up automatically."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Monitoring report path (default: <output-dir>/pipeline_dynamics.json).",
    )
    parser.add_argument("--overwrite-report", action="store_true")

    parser.add_argument(
        "--total-docs",
        type=int,
        default=TARGET_TOTAL_DOCS,
        help="Final benchmark target used when --post-target-total is not provided.",
    )
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

    parser.add_argument(
        "--post-target-total",
        type=int,
        default=None,
        help="Optional explicit final benchmark target after filtering, dedup, and language audit.",
    )
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

    parser.add_argument("--disable-lang-audit", action="store_true")
    parser.add_argument("--lang-audit-min-detect-chars", type=int, default=80)
    parser.add_argument("--lang-audit-max-text-chars", type=int, default=3000)
    parser.add_argument("--lang-audit-min-confidence", type=float, default=0.85)
    parser.add_argument("--lang-audit-min-script-chars", type=int, default=8)
    parser.add_argument("--lang-audit-max-detect-docs", type=int, default=50000)
    parser.add_argument("--lang-audit-max-suspects", type=int, default=5000)
    parser.add_argument("--lang-audit-drop-detected-mismatches", action="store_true")

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
