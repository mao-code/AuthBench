#!/usr/bin/env python3
"""
Examples:
  python processing/scripts/publish_authbench_hf.py \
      --release-mode full \
      --allow-mixed-redistribution \
      --repo-id MaoXun/AuthBench \
      --push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "processing" / "outputs" / "authbench"

SPLITS = ("train", "dev", "test")
CONFIGS = ("documents", "queries", "candidates", "ground_truth")
ID_FIELDS = {
    "documents": "doc_id",
    "queries": "query_id",
    "candidates": "candidate_id",
}

TIER_A_SOURCES = {
    "exorde",
    "babel_briefings",
    "spanish_pd_books",
    "french_pd_books",
    "russian_pd",
    "german_pd",
    "stackexchange",
}

SOURCE_LICENSE_NOTES = {
    "exorde": "MIT",
    "babel_briefings": "CC BY-NC-SA 4.0",
    "spanish_pd_books": "Public domain",
    "french_pd_books": "Public domain",
    "russian_pd": "Public domain",
    "german_pd": "Public domain",
    "stackexchange": "CC BY-SA (version depends on post date)",
}

FULL_RELEASE_WARNING = (
    "The full AuthBench export mixes sources that DATASET.md and the paper "
    "classify as Tier B / manifest-only for redistribution safety. "
    "Pass --allow-mixed-redistribution only if you have independently decided "
    "that publishing the full text is acceptable for your release."
)


@dataclass
class ExportStats:
    release_mode: str
    included_sources: list[str]
    excluded_sources: list[str]
    documents_total: int
    unique_authors: int
    queries_total: int
    candidates_total: int
    ground_truth_total: int
    split_counts: dict[str, dict[str, int]]
    language_counts: Counter
    source_counts: Counter
    primary_genre_counts: Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally publish AuthBench to the Hugging Face Hub."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the current AuthBench split export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional staging directory to materialize locally.",
    )
    parser.add_argument(
        "--repo-id",
        help="Target Hugging Face dataset repo id, for example 'user/AuthBench'.",
    )
    parser.add_argument(
        "--release-mode",
        choices=("tier_a", "full"),
        default="tier_a",
        help="tier_a keeps only sources that the paper classifies as redistributable.",
    )
    parser.add_argument(
        "--allow-mixed-redistribution",
        action="store_true",
        help="Required to stage or publish release-mode=full.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repository as private when --push is used.",
    )
    parser.add_argument(
        "--revision",
        help="Optional revision / branch to upload to.",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face token. Falls back to environment or local login.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the staged folder to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output-dir if it already exists.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def detect_all_sources(source_dir: Path) -> list[str]:
    counts: Counter[str] = Counter()
    for split in SPLITS:
        for row in iter_jsonl(source_dir / split / "documents.jsonl"):
            counts[row["source"]] += 1
    return sorted(counts)


def resolve_allowed_sources(release_mode: str, all_sources: list[str]) -> tuple[set[str], list[str]]:
    if release_mode == "tier_a":
        allowed = set(TIER_A_SOURCES)
        label = "Tier A public subset"
    else:
        allowed = set(all_sources)
        label = "Full mixed-source release"
    return allowed, label


def size_category(total_documents: int) -> str:
    if total_documents < 1_000:
        return "n<1K"
    if total_documents < 10_000:
        return "1K<n<10K"
    if total_documents < 100_000:
        return "10K<n<100K"
    if total_documents < 1_000_000:
        return "100K<n<1M"
    if total_documents < 10_000_000:
        return "1M<n<10M"
    return "10M<n<100M"


def format_yaml_list(items: Iterable[str], indent: int = 0) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}- {item}" for item in items)


def build_card(stats: ExportStats) -> str:
    language_tags = sorted(stats.language_counts.keys())
    included_sources = "\n".join(
        f"- `{name}`: {stats.source_counts[name]:,} documents"
        + (f" ({SOURCE_LICENSE_NOTES[name]})" if name in SOURCE_LICENSE_NOTES else "")
        for name in stats.included_sources
    )
    excluded_sources = "\n".join(f"- `{name}`" for name in stats.excluded_sources) or "- None"
    split_table = "\n".join(
        f"| {split} | {counts['documents']:,} | {counts['queries']:,} | {counts['candidates']:,} | {counts['ground_truth']:,} |"
        for split, counts in stats.split_counts.items()
    )
    top_languages = "\n".join(
        f"- `{lang}`: {count:,} documents"
        for lang, count in stats.language_counts.most_common()
    )
    source_table = "\n".join(
        f"| `{source}` | {count:,} | {count / stats.documents_total:.1%} |"
        for source, count in stats.source_counts.most_common()
    )
    genre_table = "\n".join(
        f"| `{genre}` | {count:,} | {count / stats.documents_total:.1%} |"
        for genre, count in stats.primary_genre_counts.most_common()
    )
    release_mode_text = (
        "This Hub export contains only the Tier A subset of AuthBench: the sources that the current "
        "paper and `DATASET.md` classify as safer to redistribute as normalized text."
        if stats.release_mode == "tier_a"
        else "This Hub export contains the full mixed-source AuthBench folder, including sources "
        "that the current paper classifies as Tier B / manifest-only from a redistribution standpoint."
    )
    licensing_text = (
        "This release mixes upstream licenses and terms, including MIT, CC BY-NC-SA 4.0, public-domain material, "
        "and CC BY-SA content. Consumers are responsible for complying with each source's attribution, "
        "share-alike, and non-commercial requirements as applicable."
        if stats.release_mode == "tier_a"
        else "This release mixes upstream licenses and platform terms across both Tier A and Tier B sources. "
        "The paper explicitly recommends conservative manifest-only handling for several included sources. "
        "Do not treat this repository as a blanket relicensing of all component texts."
    )

    return f"""---
pretty_name: AuthBench
license: other
language:
{format_yaml_list(language_tags, indent=0)}
multilinguality: multilingual
size_categories:
- {size_category(stats.documents_total)}
configs:
- config_name: documents
  default: true
  data_files:
  - split: train
    path: train/documents.jsonl
  - split: dev
    path: dev/documents.jsonl
  - split: test
    path: test/documents.jsonl
- config_name: queries
  data_files:
  - split: train
    path: train/queries.jsonl
  - split: dev
    path: dev/queries.jsonl
  - split: test
    path: test/queries.jsonl
- config_name: candidates
  data_files:
  - split: train
    path: train/candidates.jsonl
  - split: dev
    path: dev/candidates.jsonl
  - split: test
    path: test/candidates.jsonl
- config_name: ground_truth
  data_files:
  - split: train
    path: train/ground_truth.jsonl
  - split: dev
    path: dev/ground_truth.jsonl
  - split: test
    path: test/ground_truth.jsonl
---

# AuthBench

AuthBench is a multilingual benchmark for authorship representation across languages, genres, and document lengths. It supports:

- authorship attribution as open-world same-author retrieval
- authorship verification as same-author binary decision

{release_mode_text}

## Release Summary

- Release mode: `{stats.release_mode}`
- Documents: {stats.documents_total:,}
- Authors: {stats.unique_authors:,}
- Queries: {stats.queries_total:,}
- Candidates: {stats.candidates_total:,}
- Ground-truth rows: {stats.ground_truth_total:,}
- Languages: {len(stats.language_counts)}

## Included Sources

{included_sources}

## Excluded Sources

{excluded_sources}

## Repository Layout

This dataset repository exposes four dataset configurations:

- `documents`: union of the query and candidate documents for each split
- `queries`: query-side records used for retrieval / verification evaluation
- `candidates`: candidate-side records used for retrieval / verification evaluation
- `ground_truth`: mapping from `query_id` to its same-author `positive_ids`

Each configuration has `train`, `dev`, and `test` splits.

## Load with `datasets`

```python
from datasets import load_dataset

documents = load_dataset("YOUR_HF_NAMESPACE/AuthBench", "documents", split="train")
queries = load_dataset("YOUR_HF_NAMESPACE/AuthBench", "queries", split="test")
candidates = load_dataset("YOUR_HF_NAMESPACE/AuthBench", "candidates", split="test")
ground_truth = load_dataset("YOUR_HF_NAMESPACE/AuthBench", "ground_truth", split="test")
```

## Split Sizes

| Split | Documents | Queries | Candidates | Ground Truth |
| --- | ---: | ---: | ---: | ---: |
{split_table}

## Schema

`documents`

```json
{{
  "doc_id": "mix_009328",
  "lang": "ar",
  "genre": "social_media/technology",
  "content": "...",
  "source": "exorde",
  "token_length": 51,
  "author_id": "...",
  "retrieval_role": "candidate",
  "phase": "phase1",
  "input_split": "dev",
  "input_doc_type": "query"
}}
```

`queries`

```json
{{
  "query_id": "mix_009332",
  "lang": "ar",
  "genre": "social_media/entertainment",
  "content": "...",
  "source": "exorde",
  "token_length": 50,
  "retrieval_role": "query",
  "phase": "phase1",
  "input_split": "dev",
  "input_doc_type": "candidate"
}}
```

`candidates`

```json
{{
  "candidate_id": "mix_009328",
  "lang": "ar",
  "genre": "social_media/technology",
  "content": "...",
  "source": "exorde",
  "token_length": 51,
  "author_id": "...",
  "retrieval_role": "candidate",
  "phase": "phase1",
  "input_split": "dev",
  "input_doc_type": "query"
}}
```

`ground_truth`

```json
{{
  "query_id": "mix_009332",
  "positive_ids": ["mix_009328", "mix_009330", "mix_009329"],
  "author_id": "..."
}}
```

## Language Coverage

{top_languages}

## Source Distribution

| Source | Documents | Share |
| --- | ---: | ---: |
{source_table}

## Primary Genre Distribution

| Primary Genre | Documents | Share |
| --- | ---: | ---: |
{genre_table}

## Licensing And Redistribution Notes

{licensing_text}

For the benchmark-wide source inventory and the Tier A / Tier B rationale, see:

- `DATASET.md` in the AuthBench repository
- `paper/colm_latex.tex`, especially the appendix licensing table

## Caveats

- `queries` intentionally omit `author_id`; the supervision lives in `ground_truth`.
- `documents` are a convenience union of query and candidate records, not an additional split.
- `input_split` and `input_doc_type` refer to the record's origin before the final combined export.
- Source balance is intentionally skewed; the largest sources dominate the benchmark.

## Citation

If you use AuthBench, cite the accompanying manuscript:

`AuthBench: A Large-Scale Multilingual Benchmark for Authorship Representation across Genres and Lengths`
"""


def stage_release(source_dir: Path, output_dir: Path, release_mode: str) -> ExportStats:
    all_sources = detect_all_sources(source_dir)
    allowed_sources, _ = resolve_allowed_sources(release_mode, all_sources)
    excluded_sources = sorted(set(all_sources) - allowed_sources)

    authors: set[str] = set()
    split_counts: dict[str, dict[str, int]] = {}
    language_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    primary_genre_counts: Counter[str] = Counter()

    for split in SPLITS:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        query_ids: set[str] = set()
        candidate_ids: set[str] = set()

        counts = {"documents": 0, "queries": 0, "candidates": 0, "ground_truth": 0}

        for config in ("documents", "queries", "candidates"):
            src_path = source_dir / split / f"{config}.jsonl"
            dst_path = split_dir / f"{config}.jsonl"
            id_field = ID_FIELDS[config]

            with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["source"] not in allowed_sources:
                        continue
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[config] += 1
                    if config == "documents":
                        authors.add(row["author_id"])
                        language_counts[row["lang"]] += 1
                        source_counts[row["source"]] += 1
                        primary_genre_counts[row["genre"].split("/", 1)[0]] += 1
                    elif config == "queries":
                        query_ids.add(row[id_field])
                    elif config == "candidates":
                        candidate_ids.add(row[id_field])

        src_ground_truth = source_dir / split / "ground_truth.jsonl"
        dst_ground_truth = split_dir / "ground_truth.jsonl"
        with src_ground_truth.open("r", encoding="utf-8") as src, dst_ground_truth.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["query_id"] not in query_ids:
                    continue
                positive_ids = [candidate_id for candidate_id in row["positive_ids"] if candidate_id in candidate_ids]
                if not positive_ids:
                    continue
                row["positive_ids"] = positive_ids
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["ground_truth"] += 1

        split_counts[split] = counts

    if (source_dir / "merge_summary.json").exists():
        shutil.copy2(source_dir / "merge_summary.json", output_dir / "merge_summary.json")

    stats = ExportStats(
        release_mode=release_mode,
        included_sources=sorted(allowed_sources),
        excluded_sources=excluded_sources,
        documents_total=sum(split_counts[split]["documents"] for split in SPLITS),
        unique_authors=len(authors),
        queries_total=sum(split_counts[split]["queries"] for split in SPLITS),
        candidates_total=sum(split_counts[split]["candidates"] for split in SPLITS),
        ground_truth_total=sum(split_counts[split]["ground_truth"] for split in SPLITS),
        split_counts=split_counts,
        language_counts=language_counts,
        source_counts=source_counts,
        primary_genre_counts=primary_genre_counts,
    )
    (output_dir / "README.md").write_text(build_card(stats), encoding="utf-8")
    return stats


def upload_to_hub(
    output_dir: Path,
    repo_id: str,
    token: str | None,
    private: bool,
    revision: str | None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        path_in_repo=".",
        revision=revision,
        commit_message="Upload AuthBench dataset export",
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    if args.release_mode == "full" and not args.allow_mixed_redistribution:
        raise SystemExit(FULL_RELEASE_WARNING)

    if args.push and not args.repo_id:
        raise SystemExit("--repo-id is required when --push is used.")

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if args.output_dir:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup_dir = None
    else:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="authbench_hf_")
        output_dir = Path(cleanup_dir.name)

    stats = stage_release(source_dir=source_dir, output_dir=output_dir, release_mode=args.release_mode)

    print(f"Prepared release mode: {stats.release_mode}")
    print(f"Staging directory: {output_dir}")
    print(f"Documents: {stats.documents_total:,}")
    print(f"Authors: {stats.unique_authors:,}")
    print(f"Queries: {stats.queries_total:,}")
    print(f"Candidates: {stats.candidates_total:,}")
    print(f"Ground truth rows: {stats.ground_truth_total:,}")
    print(f"Included sources: {', '.join(stats.included_sources)}")
    if stats.excluded_sources:
        print(f"Excluded sources: {', '.join(stats.excluded_sources)}")

    if args.push:
        if token is None:
            print(
                "No token found in --token, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN. "
                "If you are already logged in with huggingface-cli login, the hub client may still succeed."
            )
        upload_to_hub(
            output_dir=output_dir,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            revision=args.revision,
        )
        print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if __name__ == "__main__":
    main()
