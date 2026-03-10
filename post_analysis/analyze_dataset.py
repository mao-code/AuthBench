#!/usr/bin/env python3
"""Analyze the final AuthBench dataset and export tabular statistics and plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

DEFAULT_SPLITS: Sequence[str] = ("train", "dev", "test")


def discover_splits(dataset_dir: Path) -> List[str]:
    """Return all split directories that contain query/candidate files."""
    splits = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "queries.jsonl").exists() and (child / "candidates.jsonl").exists():
            splits.append(child.name)
    return sorted(splits)


def extract_primary_genre(genre: str | None) -> str | None:
    if not isinstance(genre, str) or not genre:
        return None
    return genre.split("/")[0]


def read_jsonl(
    path: Path,
    id_field: str,
    doc_type: str,
    split: str,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    "doc_id": row[id_field],
                    "lang": row.get("lang"),
                    "genre": row.get("genre"),
                    "primary_genre": extract_primary_genre(row.get("genre")),
                    "source": row.get("source"),
                    "token_length": row.get("token_length"),
                    "author_id": row.get("author_id"),
                    "doc_type": row.get("retrieval_role") or doc_type,
                    "split": split,
                    "phase": row.get("phase"),
                    "input_split": row.get("input_split"),
                    "input_doc_type": row.get("input_doc_type"),
                }
            )
    return pd.DataFrame(records)


def load_documents(dataset_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        documents_path = split_dir / "documents.jsonl"
        if documents_path.exists():
            frames.append(read_jsonl(documents_path, "doc_id", "document", split))
            continue
        frames.append(read_jsonl(split_dir / "queries.jsonl", "query_id", "query", split))
        frames.append(read_jsonl(split_dir / "candidates.jsonl", "candidate_id", "candidate", split))
    if not frames:
        raise FileNotFoundError(f"No data found in {dataset_dir} for splits {splits}.")
    docs = pd.concat(frames, ignore_index=True)
    docs["token_length"] = pd.to_numeric(docs["token_length"], errors="coerce")
    return docs


def load_ground_truth(dataset_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in splits:
        path = dataset_dir / split / "ground_truth.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                for pos_id in row.get("positive_ids", []):
                    rows.append(
                        {
                            "split": split,
                            "query_id": row["query_id"],
                            "positive_id": pos_id,
                            "author_id": row.get("author_id"),
                        }
                    )
    return pd.DataFrame(rows)


def fill_query_authors(docs: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    author_lookup = (
        ground_truth.dropna(subset=["author_id"])
        .drop_duplicates(subset=["query_id"])
        .set_index("query_id")["author_id"]
    )
    mask = (docs["doc_type"] == "query") & docs["author_id"].isna()
    docs.loc[mask, "author_id"] = docs.loc[mask, "doc_id"].map(author_lookup)
    return docs


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def percent(series: pd.Series) -> pd.Series:
    total = series.sum()
    if total == 0:
        return series * 0
    return series / total * 100


def token_length_bucket(token_length: float | int | None) -> str | None:
    if pd.isna(token_length):
        return None
    try:
        value = float(token_length)
    except (TypeError, ValueError):
        return None
    if value <= 10:
        return "short"
    if value <= 100:
        return "medium"
    if value <= 500:
        return "long"
    return "extra_long"


def language_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    overall = (
        docs.groupby("lang")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    overall["pct_all_docs"] = percent(overall["docs"]).round(2)

    by_type = (
        docs.groupby(["doc_type", "lang"])
        .size()
        .reset_index(name="docs")
        .sort_values(["doc_type", "docs"], ascending=[True, False])
    )
    by_type["pct_within_doc_type"] = (
        by_type["docs"] / by_type.groupby("doc_type")["docs"].transform("sum") * 100
    ).round(2)
    by_type["pct_all_docs"] = percent(by_type["docs"]).round(2)

    by_split = (
        docs.groupby(["split", "doc_type", "lang"])
        .size()
        .reset_index(name="docs")
        .sort_values(["split", "doc_type", "docs"], ascending=[True, True, False])
    )
    by_split["pct_within_split_doc_type"] = (
        by_split["docs"] / by_split.groupby(["split", "doc_type"])["docs"].transform("sum") * 100
    ).round(2)

    save_csv(overall, output_dir / "languages_overall.csv")
    save_csv(by_type, output_dir / "languages_by_doc_type.csv")
    save_csv(by_split, output_dir / "languages_by_split_doc_type.csv")


def genre_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    genre = (
        docs.groupby("genre")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    genre["pct_all_docs"] = percent(genre["docs"]).round(2)

    primary = (
        docs.groupby("primary_genre")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    primary["pct_all_docs"] = percent(primary["docs"]).round(2)

    genre_by_lang = (
        docs.groupby(["lang", "genre"])
        .size()
        .reset_index(name="docs")
        .sort_values(["lang", "docs"], ascending=[True, False])
    )
    genre_by_lang["pct_within_lang"] = (
        genre_by_lang["docs"] / genre_by_lang.groupby("lang")["docs"].transform("sum") * 100
    ).round(2)

    primary_by_lang = (
        docs.groupby(["lang", "primary_genre"])
        .size()
        .reset_index(name="docs")
        .sort_values(["lang", "docs"], ascending=[True, False])
    )
    primary_by_lang["pct_within_lang"] = (
        primary_by_lang["docs"] / primary_by_lang.groupby("lang")["docs"].transform("sum") * 100
    ).round(2)

    save_csv(genre, output_dir / "genres_overall.csv")
    save_csv(primary, output_dir / "primary_genres_overall.csv")
    save_csv(genre_by_lang, output_dir / "genres_by_language.csv")
    save_csv(genre_by_lang, output_dir / "genre_distribution_by_language.csv")
    save_csv(primary_by_lang, output_dir / "primary_genres_by_language.csv")


def phase_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    phase_docs = docs.dropna(subset=["phase"]).copy()
    if phase_docs.empty:
        return

    overall = (
        phase_docs.groupby("phase")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    overall["pct_all_docs"] = percent(overall["docs"]).round(2)

    by_lang = (
        phase_docs.groupby(["lang", "phase"])
        .size()
        .reset_index(name="docs")
        .sort_values(["lang", "docs"], ascending=[True, False])
    )
    by_lang["pct_within_lang"] = (
        by_lang["docs"] / by_lang.groupby("lang")["docs"].transform("sum") * 100
    ).round(2)

    by_split_type = (
        phase_docs.groupby(["split", "doc_type", "phase"])
        .size()
        .reset_index(name="docs")
        .sort_values(["split", "doc_type", "docs"], ascending=[True, True, False])
    )
    by_split_type["pct_within_split_doc_type"] = (
        by_split_type["docs"]
        / by_split_type.groupby(["split", "doc_type"])["docs"].transform("sum")
        * 100
    ).round(2)

    save_csv(overall, output_dir / "phases_overall.csv")
    save_csv(by_lang, output_dir / "phases_by_language.csv")
    save_csv(by_split_type, output_dir / "phases_by_split_doc_type.csv")


def token_length_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    def compute(group_cols: List[str], filename: str) -> None:
        grouped = docs.groupby(group_cols)["token_length"]
        stats = grouped.agg(["count", "mean", "std", "min", "max", "median"]).rename(
            columns={"count": "docs"}
        )
        quantiles = grouped.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).unstack()
        quantiles.columns = [f"p{int(q * 100):02d}" for q in quantiles.columns]
        out = stats.join(quantiles)
        out = out.reset_index().sort_values("docs", ascending=False)
        save_csv(out, output_dir / filename)

    compute(["lang"], "token_lengths_by_language.csv")
    compute(["lang", "doc_type"], "token_lengths_by_language_and_type.csv")
    compute(["primary_genre"], "token_lengths_by_primary_genre.csv")
    compute(["primary_genre", "lang"], "token_lengths_by_primary_genre_and_language.csv")

    bucket_by_lang = (
        docs.dropna(subset=["token_length_bucket"])
        .groupby(["lang", "token_length_bucket"])
        .size()
        .reset_index(name="docs")
        .sort_values(["lang", "docs"], ascending=[True, False])
    )
    bucket_by_lang["pct_within_lang"] = (
        bucket_by_lang["docs"]
        / bucket_by_lang.groupby("lang")["docs"].transform("sum")
        * 100
    ).round(2)
    save_csv(bucket_by_lang, output_dir / "token_length_bucket_distribution_by_language.csv")


def source_table(docs: pd.DataFrame, output_dir: Path) -> None:
    sources = (
        docs.groupby("source")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    sources["pct_all_docs"] = percent(sources["docs"]).round(2)
    save_csv(sources, output_dir / "sources_overall.csv")


def author_split_overlap_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    if author_docs.empty:
        return

    membership = (
        author_docs.groupby("author_id")
        .agg(
            docs=("doc_id", "count"),
            splits=("split", lambda x: ",".join(sorted(set(x)))),
            num_splits=("split", "nunique"),
            dominant_lang=("lang", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        )
        .reset_index()
        .sort_values(["num_splits", "docs"], ascending=[False, False])
    )
    save_csv(membership, output_dir / "author_split_membership.csv")

    membership_dist = (
        membership.groupby("num_splits")
        .size()
        .reset_index(name="authors")
        .sort_values("num_splits")
    )
    membership_dist["pct_authors"] = percent(membership_dist["authors"]).round(2)
    save_csv(membership_dist, output_dir / "author_split_membership_distribution.csv")

    split_names = sorted(author_docs["split"].dropna().unique().tolist())
    overlap_rows: List[Dict[str, object]] = []
    split_sets = author_docs.groupby("author_id")["split"].agg(lambda x: set(x.dropna())).tolist()
    for split_a in split_names:
        for split_b in split_names:
            overlap = sum(1 for split_set in split_sets if split_a in split_set and split_b in split_set)
            overlap_rows.append(
                {
                    "split_a": split_a,
                    "split_b": split_b,
                    "authors": overlap,
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    save_csv(overlap_df, output_dir / "author_split_overlap_matrix.csv")


def author_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    if author_docs.empty:
        return

    authors_by_lang = (
        author_docs.groupby("lang")["author_id"]
        .nunique()
        .reset_index(name="unique_authors")
        .sort_values("unique_authors", ascending=False)
    )
    docs_per_author = (
        author_docs.groupby(["lang", "author_id"])
        .size()
        .reset_index(name="docs_per_author")
    )
    docs_per_author_stats = (
        docs_per_author.groupby("lang")["docs_per_author"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
        .rename(columns={"count": "authors"})
        .sort_values("authors", ascending=False)
    )
    quantiles = docs_per_author.groupby("lang")["docs_per_author"].quantile([0.9, 0.95]).unstack()
    docs_per_author_stats["p90"] = docs_per_author_stats["lang"].map(quantiles[0.9])
    docs_per_author_stats["p95"] = docs_per_author_stats["lang"].map(quantiles[0.95])
    overall_dist = (
        docs_per_author["docs_per_author"]
        .describe(percentiles=[0.5, 0.9, 0.95])
        .to_frame()
        .T.reset_index(drop=True)
    )
    overall_dist.rename(
        columns={
            "50%": "p50",
            "90%": "p90",
            "95%": "p95",
            "25%": "p25",
            "75%": "p75",
        },
        inplace=True,
    )

    save_csv(authors_by_lang, output_dir / "authors_by_language.csv")
    save_csv(docs_per_author_stats, output_dir / "docs_per_author_by_language.csv")
    save_csv(overall_dist, output_dir / "docs_per_author_overall.csv")


def positive_pair_tables(
    ground_truth: pd.DataFrame, docs: pd.DataFrame, output_dir: Path
) -> None:
    if ground_truth.empty:
        return

    queries = (
        docs[docs["doc_type"] == "query"][
            ["doc_id", "lang", "genre", "primary_genre", "token_length", "source", "author_id"]
        ]
        .rename(
            columns={
                "doc_id": "query_id",
                "lang": "lang_query",
                "genre": "genre_query",
                "primary_genre": "primary_genre_query",
                "token_length": "token_length_query",
                "source": "source_query",
                "author_id": "author_query",
            }
        )
    )
    candidates = (
        docs[docs["doc_type"] == "candidate"][
            ["doc_id", "lang", "genre", "primary_genre", "token_length", "source", "author_id"]
        ]
        .rename(
            columns={
                "doc_id": "positive_id",
                "lang": "lang_candidate",
                "genre": "genre_candidate",
                "primary_genre": "primary_genre_candidate",
                "token_length": "token_length_candidate",
                "source": "source_candidate",
                "author_id": "author_candidate",
            }
        )
    )

    merged = ground_truth.merge(queries, on="query_id", how="left").merge(
        candidates, on="positive_id", how="left"
    )
    merged["same_language"] = merged["lang_query"] == merged["lang_candidate"]
    merged["same_primary_genre"] = (
        merged["primary_genre_query"] == merged["primary_genre_candidate"]
    )
    merged["token_length_delta"] = (
        merged["token_length_query"] - merged["token_length_candidate"]
    )

    summary = pd.DataFrame(
        {
            "pairs": [len(merged)],
            "pct_same_language": [merged["same_language"].mean() * 100],
            "pct_same_primary_genre": [merged["same_primary_genre"].mean() * 100],
            "avg_token_length_delta": [merged["token_length_delta"].mean()],
        }
    ).round(2)

    by_lang = (
        merged.groupby("lang_query")
        .agg(
            pairs=("query_id", "count"),
            pct_same_language=("same_language", lambda x: x.mean() * 100),
            pct_same_primary_genre=("same_primary_genre", lambda x: x.mean() * 100),
            avg_token_length_delta=("token_length_delta", "mean"),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
        .round(2)
    )

    save_csv(summary, output_dir / "positive_pairs_summary.csv")
    save_csv(by_lang, output_dir / "positive_pairs_by_language.csv")


def retrieval_structure_tables(
    ground_truth: pd.DataFrame, docs: pd.DataFrame, output_dir: Path
) -> None:
    if ground_truth.empty:
        return

    query_meta = (
        docs[docs["doc_type"] == "query"][
            ["doc_id", "lang", "phase", "split", "author_id"]
        ]
        .rename(columns={"doc_id": "query_id"})
        .drop_duplicates(subset=["query_id"])
    )
    candidate_meta = (
        docs[docs["doc_type"] == "candidate"][
            ["doc_id", "author_id", "lang", "phase", "split"]
        ]
        .rename(
            columns={
                "doc_id": "candidate_id",
                "author_id": "candidate_author_id",
                "lang": "candidate_lang",
                "phase": "candidate_phase",
                "split": "candidate_split",
            }
        )
        .drop_duplicates(subset=["candidate_id"])
    )

    gt = ground_truth.copy()
    positive_counts = gt.groupby("query_id").size().reset_index(name="positive_candidates")
    query_level = positive_counts.merge(query_meta, on="query_id", how="left")

    overall = pd.DataFrame(
        {
            "queries": [int(query_level["query_id"].nunique())],
            "avg_positive_candidates": [float(query_level["positive_candidates"].mean())],
            "median_positive_candidates": [float(query_level["positive_candidates"].median())],
            "min_positive_candidates": [int(query_level["positive_candidates"].min())],
            "max_positive_candidates": [int(query_level["positive_candidates"].max())],
        }
    ).round(2)
    save_csv(overall, output_dir / "retrieval_structure_overall.csv")

    by_lang = (
        query_level.groupby("lang")
        .agg(
            queries=("query_id", "nunique"),
            avg_positive_candidates=("positive_candidates", "mean"),
            median_positive_candidates=("positive_candidates", "median"),
        )
        .reset_index()
        .sort_values("queries", ascending=False)
        .round(2)
    )
    save_csv(by_lang, output_dir / "retrieval_structure_by_language.csv")

    if query_level["phase"].notna().any():
        by_phase = (
            query_level.groupby("phase")
            .agg(
                queries=("query_id", "nunique"),
                avg_positive_candidates=("positive_candidates", "mean"),
                median_positive_candidates=("positive_candidates", "median"),
            )
            .reset_index()
            .sort_values("queries", ascending=False)
            .round(2)
        )
        save_csv(by_phase, output_dir / "retrieval_structure_by_phase.csv")

    candidate_authors = (
        candidate_meta.groupby("candidate_author_id")["candidate_id"]
        .nunique()
        .reset_index(name="candidate_docs")
        .sort_values("candidate_docs", ascending=False)
    )
    save_csv(candidate_authors, output_dir / "candidate_docs_per_author.csv")


def plot_language_distribution(docs: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = docs["lang"].value_counts().index
    plt.figure(figsize=(10, 6))
    sns.countplot(data=docs, x="lang", hue="doc_type", order=order)
    plt.xlabel("Language")
    plt.ylabel("Document count")
    plt.title("Language distribution by document type")
    plt.tight_layout()
    plt.savefig(output_dir / "language_distribution.png", dpi=300)
    plt.close()


def plot_primary_genre_heatmap(docs: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    heat = pd.crosstab(docs["lang"], docs["primary_genre"], normalize="index") * 100
    heat = heat.loc[:, heat.sum().sort_values(ascending=False).index]
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heat,
        cmap="Blues",
        annot=False,
        cbar_kws={"label": "% of docs within language"},
        linewidths=0.3,
    )
    plt.xlabel("Primary genre")
    plt.ylabel("Language")
    plt.title("Primary genre mix within each language")
    plt.tight_layout()
    plt.savefig(output_dir / "primary_genre_heatmap.png", dpi=300)
    plt.close()


def plot_token_length_box(docs: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = docs["lang"].value_counts().index
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=docs,
        x="lang",
        y="token_length",
        hue="doc_type",
        order=order,
        showfliers=False,
    )
    plt.xlabel("Language")
    plt.ylabel("Token length")
    plt.title("Token length distribution by language")
    plt.tight_layout()
    plt.savefig(output_dir / "token_length_boxplot.png", dpi=300)
    plt.close()


def plot_primary_genre_share(docs: pd.DataFrame, output_dir: Path, top_n: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_counts = docs["primary_genre"].value_counts()
    top_genres = primary_counts.head(top_n).index
    filtered = docs[docs["primary_genre"].isin(top_genres)]
    order = filtered["primary_genre"].value_counts().index
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered, y="primary_genre", order=order, hue="doc_type")
    plt.xlabel("Document count")
    plt.ylabel("Primary genre")
    plt.title(f"Top {top_n} primary genres")
    plt.tight_layout()
    plt.savefig(output_dir / "primary_genre_distribution.png", dpi=300)
    plt.close()


def plot_phase_distribution(docs: pd.DataFrame, output_dir: Path) -> None:
    phase_docs = docs.dropna(subset=["phase"]).copy()
    if phase_docs.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    order = phase_docs["lang"].value_counts().index
    plt.figure(figsize=(12, 6))
    sns.countplot(data=phase_docs, x="lang", hue="phase", order=order)
    plt.xlabel("Language")
    plt.ylabel("Document count")
    plt.title("Phase distribution by language")
    plt.tight_layout()
    plt.savefig(output_dir / "phase_distribution_by_language.png", dpi=300)
    plt.close()


def run_analysis(dataset_dir: Path, output_dir: Path, splits: Sequence[str]) -> None:
    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    docs = load_documents(dataset_dir, splits)
    ground_truth = load_ground_truth(dataset_dir, splits)
    docs = fill_query_authors(docs, ground_truth)
    docs["token_length_bucket"] = docs["token_length"].apply(token_length_bucket)

    language_tables(docs, csv_dir)
    genre_tables(docs, csv_dir)
    phase_tables(docs, csv_dir)
    token_length_tables(docs, csv_dir)
    source_table(docs, csv_dir)
    author_tables(docs, csv_dir)
    author_split_overlap_tables(docs, csv_dir)
    positive_pair_tables(ground_truth, docs, csv_dir)
    retrieval_structure_tables(ground_truth, docs, csv_dir)

    plot_language_distribution(docs, fig_dir)
    plot_primary_genre_heatmap(docs, fig_dir)
    plot_token_length_box(docs, fig_dir)
    plot_primary_genre_share(docs, fig_dir)
    plot_phase_distribution(docs, fig_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-analysis on AuthBench outputs.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced"),
        help="Root directory containing train/dev/test splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_analysis/outputs"),
        help="Where to place CSVs and figures.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Which splits to analyze. If omitted or set to 'all', all detected splits are used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_splits = args.splits
    if selected_splits is None or "all" in selected_splits:
        selected_splits = discover_splits(args.dataset_dir)
    if not selected_splits:
        raise FileNotFoundError(f"No valid splits found in {args.dataset_dir}.")

    run_analysis(args.dataset_dir, args.output_dir, selected_splits)

    # Example command:
    """
    python post_analysis/analyze_dataset.py \
    --dataset-dir processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced \
    --output-dir post_analysis/outputs \
    --splits all
    """
