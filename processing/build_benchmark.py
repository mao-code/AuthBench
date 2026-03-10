from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from itertools import islice

from tqdm import tqdm

from .chunker import chunk_document
from .config import (
    AUTHOR_DOC_LIMITS,
    CHUNKING_DEFAULTS,
    DIRTY_DEFAULTS,
    TARGET_TOTAL_DOCS,
    build_sampling_targets,
    default_manifest_path,
    make_split_ratios,
)
from .datasets import iter_dataset, load_manifest
from .dirty import dirty_reason
from .sampling import (
    assign_document_ids,
    build_retrieval_sets,
    sample_to_targets,
    split_by_language,
)
from .types import ProcessedDocument
from .utils import count_tokens, hash_author, length_bucket, write_jsonl
from .chunker import truncate_raw_document

logger = logging.getLogger(__name__)


class StreamingJSONLWriter:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self.count = 0

    def write(self, item: dict):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self):
        if self._fh:
            self._fh.close()


class AuthorAccumulator:
    """
    Maintains a bounded, per-author reservoir so we don't keep every document in memory.
    """

    def __init__(self, *, min_docs: int, max_docs: int, rng: random.Random):
        self.min_docs = min_docs
        self.max_docs = max_docs
        self.rng = rng
        self._buckets: dict[str, dict] = {}

    def add(self, doc: ProcessedDocument):
        bucket = self._buckets.get(doc.author_id)
        if bucket is None:
            bucket = {"docs": [], "count": 0}
            self._buckets[doc.author_id] = bucket

        bucket["count"] += 1
        docs = bucket["docs"]
        if len(docs) < self.max_docs:
            docs.append(doc)
            return

        # Reservoir sample to keep memory bounded while giving each doc a fair chance.
        replace_idx = self.rng.randint(0, bucket["count"] - 1)
        if replace_idx < self.max_docs:
            docs[replace_idx] = doc

    def finalize(self):
        selected: list[ProcessedDocument] = []
        underfull: list[ProcessedDocument] = []
        dropped_authors: list[str] = []

        for author_id, bucket in self._buckets.items():
            docs_sorted = sorted(bucket["docs"], key=lambda d: (d.lang, d.source, d.raw_id))
            if bucket["count"] < self.min_docs:
                underfull.extend(docs_sorted)
                dropped_authors.append(author_id)
                continue
            selected.extend(docs_sorted)

        return selected, underfull, dropped_authors


def _summarize(docs: list[ProcessedDocument]) -> dict:
    by_lang = Counter(doc.lang for doc in docs)
    by_genre = Counter(doc.genre for doc in docs)
    by_source = Counter(doc.source for doc in docs)
    return {
        "total": len(docs),
        "by_lang": dict(by_lang),
        "by_genre": dict(by_genre),
        "by_source": dict(by_source),
    }


def _lang_distribution(docs: list[ProcessedDocument], key_fn):
    grouped = defaultdict(Counter)
    for doc in docs:
        grouped[doc.lang][key_fn(doc)] += 1
    return {lang: dict(counter) for lang, counter in grouped.items()}


def buffered_shuffle(iterator, buffer_size: int, rng: random.Random):
    """
    Shuffle a streaming iterator using a bounded buffer to avoid loading
    entire datasets into memory. This gives a near-random order without
    replacement while keeping memory usage predictable.
    """
    if buffer_size <= 0:
        yield from iterator
        return

    # A rolling bag
    buffer = list(islice(iterator, buffer_size))
    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)
        try:
            # The window move on item per iteration and add on new sample in the buffer
            buffer.append(next(iterator))
        except StopIteration:
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()
            break


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    total_docs: int,
    split_ratios,
    seed: int,
    sanity_check: bool,
    sanity_limit: int | None,
    max_documents_per_dataset: int | None,
    shuffle_buffer_size: int,
    no_shuffle_datasets: set[str],
    dataset_max_docs: dict[str, int],
    allow_other_languages: bool,
    chunking_params: dict,
    truncate_to_tokens: int | None,
    defer_sampling: bool = False,
    write_retrieval_artifacts: bool = True,
):
    rng = random.Random(seed)
    configs = load_manifest(manifest_path)
    if max_documents_per_dataset is not None:
        logger.info(
            "Applying global cap per dataset after shuffling: %d (still respecting manifest and sanity caps).",
            max_documents_per_dataset,
        )
    if dataset_max_docs:
        logger.info(
            "Dataset-specific caps after shuffling: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(dataset_max_docs.items())),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dirty_logger = StreamingJSONLWriter(output_dir / "dirty_docs.log")
    error_logger = StreamingJSONLWriter(output_dir / "errored_docs.log")
    author_accumulator = AuthorAccumulator(
        min_docs=AUTHOR_DOC_LIMITS["min_docs"],
        max_docs=AUTHOR_DOC_LIMITS["max_docs"],
        rng=rng,
    )
    clean_doc_count = 0

    for cfg in configs:
        logger.info("Loading dataset '%s' (source=%s)", cfg.name, cfg.source)
        pbar = tqdm(desc=f"{cfg.name}", unit="doc")
        dataset_iter = iter(iter_dataset(cfg, sanity_limit if sanity_check else None))
        dataset_key = {cfg.name.lower(), cfg.source.lower()}
        if dataset_key & no_shuffle_datasets:
            dataset_iter = dataset_iter
        else:
            dataset_iter = buffered_shuffle(dataset_iter, shuffle_buffer_size, rng)
        cap_override = next((dataset_max_docs[k] for k in dataset_key if k in dataset_max_docs), None)
        effective_cap = cap_override if cap_override is not None else max_documents_per_dataset
        per_dataset_processed = 0
        while True:
            try:
                raw_doc = next(dataset_iter)
            except StopIteration:
                break
            except Exception as exc:  # Skip individual files/rows that fail to load
                logger.warning("Skipping data entry from '%s' due to read error: %s", cfg.name, exc)
                error_logger.write({"dataset": cfg.name, "source": cfg.source, "error": str(exc)})
                continue

            per_dataset_processed += 1
            if effective_cap is not None and per_dataset_processed > effective_cap:
                break

            pbar.update(1)

            if not raw_doc.text or not str(raw_doc.text).strip():
                continue
            try:
                chunks = chunk_document(
                    raw_doc,
                    max_tokens=chunking_params["max_tokens"],
                    target_chunk_tokens=chunking_params["target_chunk_tokens"],
                    min_chunk_tokens=chunking_params["min_chunk_tokens"],
                    chunk_probability=chunking_params.get("chunk_probability", 1.0),
                    rng=rng,
                )
                for chunk in chunks:
                    if truncate_to_tokens and count_tokens(chunk.text) > truncate_to_tokens:
                        chunk = truncate_raw_document(chunk, truncate_to_tokens)
                    token_len = count_tokens(chunk.text)
                    reason = dirty_reason(
                        chunk.text,
                        token_len,
                        unique_token_ratio=DIRTY_DEFAULTS["unique_token_ratio"],
                        symbol_ratio=DIRTY_DEFAULTS["symbol_ratio"],
                        max_consecutive_symbols=DIRTY_DEFAULTS["max_consecutive_symbols"],
                        max_repeated_char_run=DIRTY_DEFAULTS["max_repeated_char_run"],
                        source=chunk.source,
                    )
                    if reason:
                        dirty_logger.write(
                            {
                                "source": chunk.source,
                                "raw_id": chunk.raw_id,
                                "reason": reason,
                                "chunk_text": chunk.text[:100],
                            }
                        )
                        continue
                    author_accumulator.add(
                        ProcessedDocument(
                            raw_id=chunk.raw_id,
                            author_id=hash_author(chunk.source, chunk.author),
                            text=chunk.text,
                            lang=chunk.lang,
                            source=chunk.source,
                            genre=chunk.genre,
                            token_length=token_len,
                            length_bucket=length_bucket(token_len),
                            metadata=chunk.metadata,
                        )
                    )
                    clean_doc_count += 1
            except Exception as exc:  # Skip documents that fail during processing
                logger.warning(
            "Skipping raw document %s from '%s' due to processing error: %s",
                    getattr(raw_doc, "raw_id", "<unknown>"),
                    cfg.name,
                    exc,
                )
                error_logger.write(
                    {
                        "dataset": cfg.name,
                        "source": cfg.source,
                        "raw_id": getattr(raw_doc, "raw_id", None),
                        "error": str(exc),
                    }
                )
                continue
        pbar.close()

    dirty_logger.close()
    error_logger.close()

    selected_docs, underfull_docs, dropped_authors = author_accumulator.finalize()

    if not selected_docs and not underfull_docs:
        raise RuntimeError("No clean documents found. Check manifest paths and thresholds.")

    authors_seen = len(author_accumulator._buckets)
    logger.info(
        "Processed %d clean documents before author filtering (tracked %d authors).",
        clean_doc_count,
        authors_seen,
    )

    fallback_authors_added: list[str] = []
    if AUTHOR_DOC_LIMITS["fallback_min_docs"] < AUTHOR_DOC_LIMITS["min_docs"]:
        fallback_group = defaultdict(list)
        for doc in underfull_docs:
            fallback_group[doc.author_id].append(doc)
        for author_id, docs in fallback_group.items():
            if len(docs) >= AUTHOR_DOC_LIMITS["fallback_min_docs"]:
                docs = sorted(docs, key=lambda d: (d.lang, d.source, d.raw_id))
                docs = docs[: AUTHOR_DOC_LIMITS["max_docs"]]
                selected_docs.extend(docs)
                fallback_authors_added.append(author_id)

    logger.info(
        "Kept %d documents after author filtering (%d authors dropped, %d fallback authors re-added).",
        len(selected_docs),
        len(dropped_authors),
        len(fallback_authors_added),
    )

    sampling_log: list[dict] = []
    if defer_sampling:
        logger.info(
            "Deferring balanced bucket sampling to the final stage; carrying forward %d author-filtered docs.",
            len(selected_docs),
        )
        carried_docs = selected_docs
    else:
        targets = build_sampling_targets(total_docs=total_docs)
        carried_docs, sampling_log = sample_to_targets(
            selected_docs, targets, rng, allow_other_languages=allow_other_languages
        )
        logger.info("Sampled %d documents toward target %d.", len(carried_docs), targets.total_docs)

    final_docs = assign_document_ids(carried_docs)
    total_authors = len({doc.author_id for doc in final_docs})
    avg_docs_per_author = (len(final_docs) / total_authors) if total_authors else 0
    genre_by_language = _lang_distribution(final_docs, lambda d: d.genre)
    length_bucket_by_language = _lang_distribution(final_docs, lambda d: d.length_bucket)
    splits = split_by_language(final_docs, split_ratios, rng)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, docs in splits.items():
        split_path = output_dir / split_name
        document_rows = [
            {
                "doc_id": doc.doc_id,
                "raw_id": doc.raw_id,
                "author_id": doc.author_id,
                "lang": doc.lang,
                "genre": doc.genre,
                "content": doc.text,
                "source": doc.source,
                "token_length": doc.token_length,
                "length_bucket": doc.length_bucket,
            }
            for doc in docs
        ]
        write_jsonl(split_path / "documents.jsonl", document_rows)
        if write_retrieval_artifacts:
            candidates, queries, ground_truth = build_retrieval_sets(docs, rng)
            write_jsonl(split_path / "candidates.jsonl", candidates)
            write_jsonl(split_path / "queries.jsonl", queries)
            write_jsonl(split_path / "ground_truth.jsonl", ground_truth)
            logger.info(
                "[%s] docs=%d candidates=%d queries=%d",
                split_name,
                len(docs),
                len(candidates),
                len(queries),
            )
        else:
            logger.info("[%s] docs=%d retrieval_artifacts=skipped", split_name, len(docs))

    summary = {
        "inputs": {
            "clean_documents": clean_doc_count,
            "dirty_documents": dirty_logger.count,
            "dropped_authors": len(dropped_authors),
            "errored_documents": error_logger.count,
        },
        "after_author_filter": _summarize(selected_docs),
        "after_sampling": _summarize(final_docs),
        "build_output": _summarize(final_docs),
        "splits": {name: _summarize(docs) for name, docs in splits.items()},
        # "fallback_authors": fallback_authors_added,
        "sampling": {
            "deferred_to_finalize": defer_sampling,
            "requested_total_docs": total_docs,
            "write_retrieval_artifacts": write_retrieval_artifacts,
        },
        "final_statistics": {
            "unique_authors": total_authors,
            "avg_docs_per_author": avg_docs_per_author,
            "genre_by_language": genre_by_language,
            "length_bucket_by_language": length_bucket_by_language,
        },
    }
    (output_dir / "processing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "sampling_shortfall.json").write_text(
        json.dumps(sampling_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Wrote outputs to %s", output_dir.resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the AuthBench benchmark from raw datasets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_path(),
        help="Path to dataset manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Where to write processed splits.",
    )
    parser.add_argument(
        "--total-docs",
        type=int,
        default=TARGET_TOTAL_DOCS,
        help="Total target documents across all languages.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Dev/validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Enable fast sanity mode (caps documents per dataset).",
    )
    parser.add_argument(
        "--sanity-limit",
        type=int,
        default=2000,
        help="Max documents per dataset in sanity mode. Ignored otherwise.",
    )
    parser.add_argument(
        "--max-documents-per-dataset",
        type=int,
        default=None,
        help="Global cap per dataset applied after shuffling (manifest/sanity caps still apply).",
    )
    parser.add_argument(
        "--dataset-max-docs",
        nargs="*",
        default=[],
        help="Per-dataset caps applied after shuffling, e.g. german_pd=1000 french_pd_books=800 (case-insensitive names/sources).",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=0,
        help="If >0, randomize dataset order with a bounded shuffle buffer to reduce monotony in streams.",
    )
    parser.add_argument(
        "--no-shuffle-datasets",
        nargs="*",
        default=[],
        help="Dataset names or sources to skip shuffling (case-insensitive).",
    )
    parser.add_argument(
        "--allow-other-languages",
        action="store_true",
        help="Allow filling leftover budget with languages not in the target table.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=CHUNKING_DEFAULTS["max_tokens"],
        help="Upper bound on tokens per output document after chunking.",
    )
    parser.add_argument(
        "--target-chunk-tokens",
        type=int,
        default=CHUNKING_DEFAULTS["target_chunk_tokens"],
        help="Target size when grouping sentences/paragraphs into chunks.",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=CHUNKING_DEFAULTS["min_chunk_tokens"],
        help="Minimum chunk size unless it is the final remainder.",
    )
    parser.add_argument(
        "--chunk-probability",
        type=float,
        default=CHUNKING_DEFAULTS.get("chunk_probability", 1.0),
        help="Probability to chunk an over-length document; set <1.0 to occasionally keep long docs intact.",
    )
    parser.add_argument(
        "--truncate-to-tokens",
        type=int,
        default=None,
        help="If set, truncate each (possibly chunked) document to this token cap using punctuation-aware boundaries.",
    )
    parser.add_argument(
        "--defer-sampling",
        action="store_true",
        help="Carry all author-filtered documents forward and let a later pipeline stage apply the final benchmark cap.",
    )
    parser.add_argument(
        "--skip-retrieval-artifacts",
        action="store_true",
        help="Write only documents.jsonl for each split and skip candidates/queries/ground_truth generation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (INFO, DEBUG).",
    )
    return parser.parse_args()


def _parse_dataset_caps(pairs: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for pair in pairs or []:
        if "=" not in pair:
            logger.warning("Ignoring dataset cap '%s' (expected format name=value).", pair)
            continue
        key, val = pair.split("=", 1)
        key = key.strip().lower()
        try:
            caps[key] = int(val)
        except ValueError:
            logger.warning("Ignoring dataset cap '%s' (value must be int).", pair)
    return caps


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    split_ratios = make_split_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest {args.manifest} not found. Provide --manifest or create one (see datasets_manifest.example.json)."
        )

    run(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        total_docs=args.total_docs,
        split_ratios=split_ratios,
        seed=args.seed,
        sanity_check=args.sanity_check,
        sanity_limit=args.sanity_limit,
        max_documents_per_dataset=args.max_documents_per_dataset,
        shuffle_buffer_size=args.shuffle_buffer_size,
        no_shuffle_datasets=set(map(str.lower, args.no_shuffle_datasets or [])),
        dataset_max_docs=_parse_dataset_caps(args.dataset_max_docs),
        allow_other_languages=args.allow_other_languages,
        chunking_params={
            "max_tokens": args.max_chunk_tokens,
            "target_chunk_tokens": args.target_chunk_tokens,
            "min_chunk_tokens": args.min_chunk_tokens,
            "chunk_probability": args.chunk_probability,
        },
        truncate_to_tokens=args.truncate_to_tokens,
        defer_sampling=args.defer_sampling,
        write_retrieval_artifacts=not args.skip_retrieval_artifacts,
    )


if __name__ == "__main__":
    main()

    """
    python -m processing.build_benchmark \
    --manifest processing/datasets_manifest.json \
    --output-dir processing/outputs/sanity \
    --truncate-to-tokens 2000 \
    --chunk-probability 0.7 \
    --total-docs 10000 \
    --allow-other-languages \
    --sanity-check --sanity-limit 500
    

    # Official run with total 100K
    python -m processing.build_benchmark \
    --manifest processing/datasets_manifest.json \
    --output-dir processing/outputs/official_shufflebuffer_cap \
    --truncate-to-tokens 2000 \
    --chunk-probability 0.7 \
    --total-docs 100000 \
    --allow-other-languages \
    --max-documents-per-dataset 100000 \
    --shuffle-buffer-size 10000 \
    --seed 42

    # Official run with total 300K (because it will filter out many docs)
    python -m processing.build_benchmark \
    --manifest processing/datasets_manifest.json \
    --output-dir processing/outputs/official_shufflebuffer_cap_exclude \
    --truncate-to-tokens 2000 \
    --chunk-probability 0.7 \
    --total-docs 300000 \
    --allow-other-languages \
    --max-documents-per-dataset 10000000 \
    --shuffle-buffer-size 10000 \
    --no-shuffle-datasets french_pd_books german_pd russian_pd spanish_pd_books \
    --dataset-max-docs french_pd_books=10000 german_pd=10000 russian_pd=10000 spanish_pd_books=10000 \
    --seed 42
    """
