#!/usr/bin/env python3
"""Generate qualitative post-analysis artifacts for an AuthBench dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

sns.set_theme(style="whitegrid")

DEFAULT_SPLITS: Sequence[str] = ("train", "dev", "test")


def discover_splits(dataset_dir: Path) -> list[str]:
    splits: list[str] = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "candidates.jsonl").exists() and (child / "queries.jsonl").exists():
            splits.append(child.name)
    return sorted(splits)


def extract_primary_genre(genre: str | None) -> str:
    if not isinstance(genre, str) or not genre:
        return "unknown"
    return genre.split("/", 1)[0]


def normalize_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def snippet(text: str, max_chars: int = 220) -> str:
    clean = normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def text_hash(text: str | None) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def _mode(values: Iterable[object]) -> object | None:
    items = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not items:
        return None
    counts = Counter(items)
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def load_docs(dataset_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        documents_path = split_dir / "documents.jsonl"
        if documents_path.exists():
            with documents_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            "doc_id": row["doc_id"],
                            "doc_type": row.get("retrieval_role") or "document",
                            "split": split,
                            "author_id": row.get("author_id"),
                            "lang": row.get("lang"),
                            "genre": row.get("genre"),
                            "primary_genre": extract_primary_genre(row.get("genre")),
                            "source": row.get("source"),
                            "phase": row.get("phase"),
                            "input_split": row.get("input_split"),
                            "input_doc_type": row.get("input_doc_type"),
                            "token_length": row.get("token_length"),
                            "content": normalize_text(row.get("content")),
                        }
                    )
            continue

        query_author: dict[str, str | None] = {}
        gt_path = split_dir / "ground_truth.jsonl"
        if gt_path.exists():
            with gt_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    query_author[row["query_id"]] = row.get("author_id")

        cand_path = split_dir / "candidates.jsonl"
        if cand_path.exists():
            with cand_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            "doc_id": row["candidate_id"],
                            "doc_type": "candidate",
                            "split": split,
                            "author_id": row.get("author_id"),
                            "lang": row.get("lang"),
                            "genre": row.get("genre"),
                            "primary_genre": extract_primary_genre(row.get("genre")),
                            "source": row.get("source"),
                            "phase": row.get("phase"),
                            "input_split": row.get("input_split"),
                            "input_doc_type": row.get("input_doc_type"),
                            "token_length": row.get("token_length"),
                            "content": normalize_text(row.get("content")),
                        }
                    )

        query_path = split_dir / "queries.jsonl"
        if query_path.exists():
            with query_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    query_id = row["query_id"]
                    rows.append(
                        {
                            "doc_id": query_id,
                            "doc_type": "query",
                            "split": split,
                            "author_id": query_author.get(query_id),
                            "lang": row.get("lang"),
                            "genre": row.get("genre"),
                            "primary_genre": extract_primary_genre(row.get("genre")),
                            "source": row.get("source"),
                            "phase": row.get("phase"),
                            "input_split": row.get("input_split"),
                            "input_doc_type": row.get("input_doc_type"),
                            "token_length": row.get("token_length"),
                            "content": normalize_text(row.get("content")),
                        }
                    )

    if not rows:
        raise FileNotFoundError(f"No benchmark docs found in {dataset_dir}.")
    docs = pd.DataFrame(rows)
    docs["token_length"] = pd.to_numeric(docs["token_length"], errors="coerce")
    docs["content_len_chars"] = docs["content"].map(len)
    return docs


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def author_basic_tables(docs: pd.DataFrame, out_dir: Path) -> dict:
    docs_with_author = docs.dropna(subset=["author_id"]).copy()
    if docs_with_author.empty:
        empty_summary = {
            "total_docs": int(len(docs)),
            "total_docs_with_author": 0,
            "unique_authors": 0,
            "docs_per_author_min": 0,
            "docs_per_author_avg": 0.0,
            "docs_per_author_max": 0,
        }
        save_json(empty_summary, out_dir / "summary_overview.json")
        return empty_summary

    docs_per_author = (
        docs_with_author.groupby("author_id")
        .size()
        .reset_index(name="docs_per_author")
        .sort_values("docs_per_author", ascending=False)
    )
    save_csv(docs_per_author, out_dir / "docs_per_author_raw.csv")

    by_lang = (
        docs_with_author.groupby("lang")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    by_genre = (
        docs_with_author.groupby("primary_genre")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    by_source = (
        docs_with_author.groupby("source")
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    by_split_type = (
        docs.groupby(["split", "doc_type"])
        .size()
        .reset_index(name="docs")
        .sort_values(["split", "doc_type"])
    )

    save_csv(by_lang, out_dir / "docs_by_language.csv")
    save_csv(by_genre, out_dir / "docs_by_primary_genre.csv")
    save_csv(by_source, out_dir / "docs_by_source.csv")
    save_csv(by_split_type, out_dir / "docs_by_split_and_type.csv")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_docs": int(len(docs)),
        "total_docs_with_author": int(len(docs_with_author)),
        "unique_authors": int(docs_per_author["author_id"].nunique()),
        "docs_per_author_min": int(docs_per_author["docs_per_author"].min()),
        "docs_per_author_avg": float(docs_per_author["docs_per_author"].mean()),
        "docs_per_author_median": float(docs_per_author["docs_per_author"].median()),
        "docs_per_author_p90": float(docs_per_author["docs_per_author"].quantile(0.90)),
        "docs_per_author_p95": float(docs_per_author["docs_per_author"].quantile(0.95)),
        "docs_per_author_max": int(docs_per_author["docs_per_author"].max()),
        "language_count": int(by_lang["lang"].nunique()),
        "primary_genre_count": int(by_genre["primary_genre"].nunique()),
        "source_count": int(by_source["source"].nunique()),
    }
    save_json(summary, out_dir / "summary_overview.json")
    return summary


def _entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def per_author_genre_entropy(docs: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    docs_with_author = docs.dropna(subset=["author_id"]).copy()
    if docs_with_author.empty:
        save_csv(pd.DataFrame(), out_dir / "author_genre_entropy.csv")
        return pd.DataFrame(), {}

    author_genre = (
        docs_with_author.groupby(["author_id", "primary_genre"])
        .size()
        .reset_index(name="docs")
    )
    pivot = author_genre.pivot(index="author_id", columns="primary_genre", values="docs").fillna(0)

    author_meta = (
        docs_with_author.groupby("author_id")
        .agg(
            num_docs=("doc_id", "count"),
            dominant_lang=("lang", _mode),
            num_languages=("lang", "nunique"),
            dominant_source=("source", _mode),
        )
        .reset_index()
    )

    rows: list[dict] = []
    for author_id, counts in pivot.iterrows():
        values = counts[counts > 0].to_numpy(dtype=float)
        genres = counts[counts > 0].index.tolist()
        ent = _entropy_from_counts(values)
        k = len(values)
        norm = ent / math.log2(k) if k > 1 else 0.0
        dominant_genre = genres[int(np.argmax(values))]
        dominant_share = float(values.max() / values.sum())
        rows.append(
            {
                "author_id": author_id,
                "num_primary_genres": int(k),
                "genre_entropy": ent,
                "genre_entropy_normalized": norm,
                "dominant_primary_genre": dominant_genre,
                "dominant_primary_genre_share": dominant_share,
            }
        )

    entropy_df = pd.DataFrame(rows).merge(author_meta, on="author_id", how="left")
    entropy_df = entropy_df.sort_values(
        ["genre_entropy_normalized", "num_docs"], ascending=[False, False]
    )
    save_csv(entropy_df, out_dir / "author_genre_entropy.csv")

    summary_by_lang = (
        entropy_df.groupby("dominant_lang")
        .agg(
            authors=("author_id", "count"),
            avg_entropy=("genre_entropy", "mean"),
            avg_normalized_entropy=("genre_entropy_normalized", "mean"),
            pct_multi_genre=("num_primary_genres", lambda x: (x >= 2).mean() * 100.0),
        )
        .reset_index()
        .sort_values("authors", ascending=False)
    )
    save_csv(summary_by_lang, out_dir / "author_genre_entropy_by_language.csv")
    save_csv(entropy_df.head(200), out_dir / "author_genre_entropy_top200.csv")

    plt.figure(figsize=(9, 5))
    sns.histplot(entropy_df["genre_entropy_normalized"], bins=30, kde=True)
    plt.xlabel("Per-author normalized genre entropy")
    plt.ylabel("Author count")
    plt.title("Author genre entropy distribution")
    plt.tight_layout()
    fig_path = out_dir.parents[1] / "figures" / "author_genre_entropy_histogram.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    summary = {
        "authors_total": int(len(entropy_df)),
        "authors_multi_genre": int((entropy_df["num_primary_genres"] >= 2).sum()),
        "pct_authors_multi_genre": float((entropy_df["num_primary_genres"] >= 2).mean() * 100.0),
        "avg_genre_entropy": float(entropy_df["genre_entropy"].mean()),
        "avg_normalized_genre_entropy": float(entropy_df["genre_entropy_normalized"].mean()),
        "p90_normalized_genre_entropy": float(entropy_df["genre_entropy_normalized"].quantile(0.9)),
    }
    save_json(summary, out_dir / "author_genre_entropy_summary.json")
    return entropy_df, summary


def cross_genre_author_analysis(
    docs: pd.DataFrame,
    entropy_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    docs_with_author = docs.dropna(subset=["author_id"]).copy()
    if docs_with_author.empty or entropy_df.empty:
        save_csv(pd.DataFrame(), out_dir / "cross_genre_author_pairs.csv")
        return {}

    author_genres: dict[str, set[str]] = defaultdict(set)
    for row in docs_with_author[["author_id", "primary_genre"]].itertuples(index=False):
        author_genres[row.author_id].add(row.primary_genre)

    pair_counter: Counter[tuple[str, str]] = Counter()
    for genres in author_genres.values():
        ordered = sorted(genres)
        for a, b in combinations(ordered, 2):
            pair_counter[(a, b)] += 1

    pair_rows = [
        {"primary_genre_a": a, "primary_genre_b": b, "authors_with_both": count}
        for (a, b), count in pair_counter.most_common()
    ]
    pairs_df = pd.DataFrame(
        pair_rows,
        columns=["primary_genre_a", "primary_genre_b", "authors_with_both"],
    )
    save_csv(pairs_df, out_dir / "cross_genre_author_pairs.csv")

    dominant_lang = entropy_df[["author_id", "dominant_lang", "num_primary_genres"]].copy()
    cross_lang = (
        dominant_lang.groupby("dominant_lang")
        .agg(
            authors=("author_id", "count"),
            multi_genre_authors=("num_primary_genres", lambda x: int((x >= 2).sum())),
        )
        .reset_index()
    )
    cross_lang["pct_multi_genre_authors"] = (
        cross_lang["multi_genre_authors"] / cross_lang["authors"] * 100.0
    )
    cross_lang = cross_lang.sort_values("authors", ascending=False)
    save_csv(cross_lang, out_dir / "cross_genre_authors_by_language.csv")

    top_genres = (
        docs_with_author["primary_genre"]
        .value_counts()
        .head(12)
        .index.tolist()
    )
    if top_genres:
        matrix = pd.DataFrame(0, index=top_genres, columns=top_genres, dtype=float)
        authors_per_genre = (
            docs_with_author.groupby("primary_genre")["author_id"].nunique().to_dict()
        )
        for g in top_genres:
            matrix.loc[g, g] = float(authors_per_genre.get(g, 0))
        for row in pairs_df.itertuples(index=False):
            a = row.primary_genre_a
            b = row.primary_genre_b
            if a in matrix.index and b in matrix.columns:
                matrix.loc[a, b] = row.authors_with_both
                matrix.loc[b, a] = row.authors_with_both

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, cmap="YlGnBu", linewidths=0.3)
        plt.title("Cross-genre author overlap (top genres)")
        plt.tight_layout()
        fig_path = out_dir.parents[1] / "figures" / "cross_genre_author_overlap_heatmap.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300)
        plt.close()

    summary = {
        "authors_total": int(len(entropy_df)),
        "authors_multi_genre": int((entropy_df["num_primary_genres"] >= 2).sum()),
        "pct_authors_multi_genre": float((entropy_df["num_primary_genres"] >= 2).mean() * 100.0),
        "genre_pairs_nonzero": int(len(pair_counter)),
        "top_genre_pairs": pair_rows[:20],
    }
    save_json(summary, out_dir / "cross_genre_author_summary.json")
    return summary


def split_leakage_analysis(docs: pd.DataFrame, out_dir: Path) -> dict:
    docs_with_author = docs.dropna(subset=["author_id"]).copy()
    if docs_with_author.empty:
        save_csv(pd.DataFrame(), out_dir / "author_split_membership.csv")
        return {}

    membership = (
        docs_with_author.groupby("author_id")
        .agg(
            num_docs=("doc_id", "count"),
            num_splits=("split", "nunique"),
            splits=("split", lambda x: ",".join(sorted(set(x)))),
            dominant_lang=("lang", _mode),
            num_phases=("phase", lambda x: x.dropna().nunique()),
        )
        .reset_index()
        .sort_values(["num_splits", "num_docs"], ascending=[False, False])
    )
    save_csv(membership, out_dir / "author_split_membership.csv")

    membership_dist = (
        membership.groupby("num_splits")
        .size()
        .reset_index(name="authors")
        .sort_values("num_splits")
    )
    membership_dist["pct_authors"] = membership_dist["authors"] / membership_dist["authors"].sum() * 100.0
    save_csv(membership_dist, out_dir / "author_split_membership_distribution.csv")

    by_lang = (
        membership.groupby("dominant_lang")
        .agg(
            authors=("author_id", "count"),
            multi_split_authors=("num_splits", lambda x: int((x >= 2).sum())),
        )
        .reset_index()
        .sort_values("authors", ascending=False)
    )
    by_lang["pct_multi_split_authors"] = by_lang["multi_split_authors"] / by_lang["authors"] * 100.0
    save_csv(by_lang, out_dir / "author_split_leakage_by_language.csv")

    split_names = sorted(docs_with_author["split"].dropna().unique().tolist())
    split_matrix = pd.DataFrame(0, index=split_names, columns=split_names, dtype=int)
    author_split_sets = (
        docs_with_author.groupby("author_id")["split"]
        .agg(lambda x: sorted(set(x.dropna())))
        .tolist()
    )
    for split_set in author_split_sets:
        for split_a in split_set:
            for split_b in split_set:
                split_matrix.loc[split_a, split_b] += 1
    split_matrix_df = split_matrix.reset_index(names="split")
    save_csv(split_matrix_df, out_dir / "author_split_overlap_matrix.csv")

    fig_dir = out_dir.parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(split_matrix, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.3)
    plt.title("Authors appearing across split pairs")
    plt.tight_layout()
    plt.savefig(fig_dir / "author_split_overlap_heatmap.png", dpi=300)
    plt.close()

    phase_summary: dict[str, object] = {}
    phase_docs = docs_with_author.dropna(subset=["phase"]).copy()
    if not phase_docs.empty:
        phase_names = sorted(phase_docs["phase"].dropna().unique().tolist())
        phase_matrix = pd.DataFrame(0, index=phase_names, columns=phase_names, dtype=int)
        author_phase_sets = (
            phase_docs.groupby("author_id")["phase"]
            .agg(lambda x: sorted(set(x.dropna())))
            .tolist()
        )
        for phase_set in author_phase_sets:
            for phase_a in phase_set:
                for phase_b in phase_set:
                    phase_matrix.loc[phase_a, phase_b] += 1
        save_csv(phase_matrix.reset_index(names="phase"), out_dir / "author_phase_overlap_matrix.csv")

        plt.figure(figsize=(6, 5))
        sns.heatmap(phase_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.3)
        plt.title("Authors appearing across phase pairs")
        plt.tight_layout()
        plt.savefig(fig_dir / "author_phase_overlap_heatmap.png", dpi=300)
        plt.close()

        phase_summary = {
            "authors_multi_phase": int((membership["num_phases"] >= 2).sum()),
            "pct_authors_multi_phase": float((membership["num_phases"] >= 2).mean() * 100.0),
        }

    summary = {
        "authors_total": int(len(membership)),
        "authors_multi_split": int((membership["num_splits"] >= 2).sum()),
        "pct_authors_multi_split": float((membership["num_splits"] >= 2).mean() * 100.0),
        "authors_in_all_three_splits": int((membership["num_splits"] >= 3).sum()),
        **phase_summary,
    }
    save_json(summary, out_dir / "split_leakage_summary.json")
    return summary


def exact_duplicate_analysis(docs: pd.DataFrame, out_dir: Path) -> dict:
    fig_dir = out_dir.parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _write_flag_outputs(flag_counts: dict[str, int]) -> None:
        flag_summary = pd.DataFrame(
            [{"flag": key, "groups": value} for key, value in flag_counts.items()]
        )
        save_csv(flag_summary, out_dir / "exact_duplicate_flag_summary.csv")

        plt.figure(figsize=(8, 5))
        sns.barplot(data=flag_summary, x="flag", y="groups", hue="flag", dodge=False, legend=False)
        plt.xticks(rotation=15, ha="right")
        plt.ylabel("Duplicate groups")
        plt.xlabel("")
        plt.title("Exact duplicate group counts")
        plt.tight_layout()
        plt.savefig(fig_dir / "exact_duplicate_group_flags.png", dpi=300)
        plt.close()

    work = docs[docs["content_len_chars"] > 0].copy()
    if work.empty:
        save_csv(pd.DataFrame(), out_dir / "exact_duplicate_groups.csv")
        _write_flag_outputs(
            {
                "cross_author_groups": 0,
                "cross_split_groups": 0,
                "cross_phase_groups": 0,
            }
        )
        return {}

    work["content_hash"] = work["content"].map(text_hash)
    counts = work["content_hash"].value_counts()
    duplicate_hashes = counts[counts > 1].index
    dup_docs = work[work["content_hash"].isin(duplicate_hashes)].copy()
    if dup_docs.empty:
        save_csv(pd.DataFrame(), out_dir / "exact_duplicate_groups.csv")
        zero_summary = {
            "duplicate_groups": 0,
            "duplicate_docs": 0,
            "cross_author_groups": 0,
            "cross_split_groups": 0,
            "cross_phase_groups": 0,
        }
        _write_flag_outputs(
            {
                "cross_author_groups": 0,
                "cross_split_groups": 0,
                "cross_phase_groups": 0,
            }
        )
        save_json(zero_summary, out_dir / "exact_duplicate_summary.json")
        return zero_summary

    group_rows: list[dict] = []
    for content_hash, group in dup_docs.groupby("content_hash", sort=False):
        authors = sorted({str(v) for v in group["author_id"].dropna().tolist()})
        splits = sorted({str(v) for v in group["split"].dropna().tolist()})
        phases = sorted({str(v) for v in group["phase"].dropna().tolist()})
        langs = sorted({str(v) for v in group["lang"].dropna().tolist()})
        sources = sorted({str(v) for v in group["source"].dropna().tolist()})
        group_rows.append(
            {
                "content_hash": content_hash,
                "duplicate_docs": int(len(group)),
                "unique_authors": int(len(authors)),
                "unique_splits": int(len(splits)),
                "unique_phases": int(len(phases)),
                "unique_languages": int(len(langs)),
                "unique_sources": int(len(sources)),
                "cross_author": len(authors) >= 2,
                "cross_split": len(splits) >= 2,
                "cross_phase": len(phases) >= 2,
                "authors": ",".join(authors[:8]),
                "splits": ",".join(splits),
                "phases": ",".join(phases),
                "langs": ",".join(langs),
                "sources": ",".join(sources[:8]),
                "text_snippet": snippet(group["content"].iloc[0], max_chars=180),
            }
        )

    groups_df = pd.DataFrame(group_rows).sort_values(
        ["duplicate_docs", "unique_authors", "unique_splits"],
        ascending=[False, False, False],
    )
    save_csv(groups_df, out_dir / "exact_duplicate_groups.csv")
    save_csv(groups_df.head(200), out_dir / "exact_duplicate_groups_top200.csv")

    flag_counts = {
        "cross_author_groups": int(groups_df["cross_author"].sum()),
        "cross_split_groups": int(groups_df["cross_split"].sum()),
        "cross_phase_groups": int(groups_df["cross_phase"].sum()),
    }
    _write_flag_outputs(flag_counts)
    flag_summary = pd.DataFrame(
        [{"flag": key, "groups": value} for key, value in flag_counts.items()]
    )

    preview_lines = [
        "# Exact Duplicate Preview",
        "",
        "Top duplicate groups after whitespace/case normalization.",
        "",
    ]
    for idx, row in groups_df.head(80).iterrows():
        preview_lines.extend(
            [
                f"## Group {idx + 1}",
                f"- duplicate_docs: `{row['duplicate_docs']}` | unique_authors: `{row['unique_authors']}` | unique_splits: `{row['unique_splits']}` | unique_phases: `{row['unique_phases']}`",
                f"- splits: `{row['splits']}` | phases: `{row['phases']}` | langs: `{row['langs']}`",
                f"- authors: `{row['authors']}`",
                f"- snippet: {row['text_snippet']}",
                "",
            ]
        )
    (out_dir / "exact_duplicate_preview.md").write_text("\n".join(preview_lines), encoding="utf-8")

    summary = {
        "duplicate_groups": int(len(groups_df)),
        "duplicate_docs": int(dup_docs.shape[0]),
        "cross_author_groups": int(groups_df["cross_author"].sum()),
        "cross_split_groups": int(groups_df["cross_split"].sum()),
        "cross_phase_groups": int(groups_df["cross_phase"].sum()),
        "largest_group_size": int(groups_df["duplicate_docs"].max()),
    }
    save_json(summary, out_dir / "exact_duplicate_summary.json")
    return summary


def _sample_group(df: pd.DataFrame, max_docs: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_docs:
        return df.reset_index(drop=True)
    return df.sample(n=max_docs, random_state=seed).reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, field: str, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = df[field].value_counts()
    raw_quota = counts / counts.sum() * n
    quota = raw_quota.round().astype(int).clip(lower=1)
    diff = int(n - quota.sum())

    if diff != 0:
        order = raw_quota - quota
        order = order.sort_values(ascending=(diff < 0))
        for key in order.index:
            if diff == 0:
                break
            if diff > 0:
                quota[key] += 1
                diff -= 1
            else:
                if quota[key] > 1:
                    quota[key] -= 1
                    diff += 1

    parts: list[pd.DataFrame] = []
    for key, q in quota.items():
        group = df[df[field] == key]
        take = min(len(group), int(q))
        parts.append(group.sample(n=take, random_state=seed))
    sampled = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["doc_id"])
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remaining = df[~df["doc_id"].isin(sampled["doc_id"])]
        extra = min(len(remaining), n - len(sampled))
        if extra > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra, random_state=seed)],
                ignore_index=True,
            )
    return sampled.reset_index(drop=True)


def topic_controlled_same_genre_pairs(
    docs: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    min_group_docs: int = 120,
    max_group_docs: int = 1200,
    max_topic_groups: int = 18,
    max_pairs_per_cluster_per_type: int = 3,
) -> dict:
    work = docs.dropna(subset=["author_id"]).copy()
    work = work[(work["content_len_chars"] >= 80) & work["primary_genre"].notna() & work["lang"].notna()]
    if work.empty:
        save_csv(pd.DataFrame(), out_dir / "topic_controlled_same_genre_pairs.csv")
        return {}

    strata = (
        work.groupby(["lang", "primary_genre"])
        .size()
        .reset_index(name="docs")
        .sort_values("docs", ascending=False)
    )
    strata = strata[strata["docs"] >= min_group_docs].head(max_topic_groups)
    save_csv(strata, out_dir / "topic_group_candidates.csv")

    pair_rows: list[dict] = []
    topic_rows: list[dict] = []

    for stratum in strata.itertuples(index=False):
        lang = stratum.lang
        primary_genre = stratum.primary_genre
        group = work[(work["lang"] == lang) & (work["primary_genre"] == primary_genre)].copy()
        group = _sample_group(group, max_docs=max_group_docs, seed=seed)
        if len(group) < min_group_docs:
            continue

        texts = group["content"].str.slice(0, 1400).tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=7000,
        )
        try:
            x = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        if x.shape[0] < 40 or x.shape[1] < 20:
            continue

        n_clusters = max(3, min(10, int(round(math.sqrt(x.shape[0]) / 2.0))))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=512,
            n_init="auto",
        )
        labels = kmeans.fit_predict(x)
        group = group.reset_index(drop=True)
        group["topic_cluster"] = labels

        for topic_cluster in sorted(group["topic_cluster"].unique().tolist()):
            topic_df = group[group["topic_cluster"] == topic_cluster].copy().reset_index(drop=True)
            if len(topic_df) < 10:
                continue
            cluster_mask = (group["topic_cluster"] == topic_cluster).to_numpy()
            x_topic = x[cluster_mask]
            n_neighbors = min(12, len(topic_df))
            nbrs = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
            nbrs.fit(x_topic)
            distances, indices = nbrs.kneighbors(x_topic)

            seen_pairs: set[tuple[int, int]] = set()
            positive_pairs: list[dict] = []
            negative_pairs: list[dict] = []

            for i in range(len(topic_df)):
                for rank in range(1, n_neighbors):
                    j = int(indices[i, rank])
                    if i == j:
                        continue
                    key = tuple(sorted((i, j)))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    a = topic_df.iloc[i]
                    b = topic_df.iloc[j]
                    if a["author_id"] == b["author_id"] and a["doc_id"] == b["doc_id"]:
                        continue

                    sim = 1.0 - float(distances[i, rank])
                    token_a = float(a["token_length"]) if pd.notna(a["token_length"]) else np.nan
                    token_b = float(b["token_length"]) if pd.notna(b["token_length"]) else np.nan
                    if pd.notna(token_a) and pd.notna(token_b):
                        small = max(min(token_a, token_b), 1.0)
                        length_ratio = max(token_a, token_b) / small
                    else:
                        length_ratio = np.nan
                    if pd.notna(length_ratio) and length_ratio > 1.6:
                        continue

                    base = {
                        "lang": lang,
                        "primary_genre": primary_genre,
                        "topic_cluster": int(topic_cluster),
                        "cosine_similarity": sim,
                        "doc_id_1": a["doc_id"],
                        "doc_id_2": b["doc_id"],
                        "author_id_1": a["author_id"],
                        "author_id_2": b["author_id"],
                        "token_length_1": token_a,
                        "token_length_2": token_b,
                        "length_ratio": length_ratio,
                        "split_1": a["split"],
                        "split_2": b["split"],
                        "doc_type_1": a["doc_type"],
                        "doc_type_2": b["doc_type"],
                        "source_1": a["source"],
                        "source_2": b["source"],
                        "text_snippet_1": snippet(a["content"]),
                        "text_snippet_2": snippet(b["content"]),
                    }
                    if a["author_id"] == b["author_id"]:
                        base["pair_type"] = "same_author_same_genre_same_topic"
                        positive_pairs.append(base)
                    else:
                        base["pair_type"] = "different_author_same_genre_same_topic"
                        negative_pairs.append(base)

            positive_pairs = sorted(positive_pairs, key=lambda r: r["cosine_similarity"], reverse=True)
            negative_pairs = sorted(negative_pairs, key=lambda r: r["cosine_similarity"], reverse=True)

            pair_rows.extend(positive_pairs[:max_pairs_per_cluster_per_type])
            pair_rows.extend(negative_pairs[:max_pairs_per_cluster_per_type])

            topic_rows.append(
                {
                    "lang": lang,
                    "primary_genre": primary_genre,
                    "topic_cluster": int(topic_cluster),
                    "docs_in_cluster": int(len(topic_df)),
                    "candidate_pairs_seen": int(len(seen_pairs)),
                    "positive_pairs_found": int(len(positive_pairs)),
                    "negative_pairs_found": int(len(negative_pairs)),
                }
            )

    pairs_df = pd.DataFrame(
        pair_rows,
        columns=[
            "pair_type",
            "lang",
            "primary_genre",
            "topic_cluster",
            "cosine_similarity",
            "doc_id_1",
            "doc_id_2",
            "author_id_1",
            "author_id_2",
            "token_length_1",
            "token_length_2",
            "length_ratio",
            "split_1",
            "split_2",
            "doc_type_1",
            "doc_type_2",
            "source_1",
            "source_2",
            "text_snippet_1",
            "text_snippet_2",
        ],
    )
    topics_df = pd.DataFrame(
        topic_rows,
        columns=[
            "lang",
            "primary_genre",
            "topic_cluster",
            "docs_in_cluster",
            "candidate_pairs_seen",
            "positive_pairs_found",
            "negative_pairs_found",
        ],
    )
    save_csv(pairs_df, out_dir / "topic_controlled_same_genre_pairs.csv")
    save_csv(topics_df, out_dir / "topic_cluster_summary.csv")

    if not pairs_df.empty:
        summary = (
            pairs_df.groupby(["lang", "primary_genre", "pair_type"])
            .agg(
                pairs=("pair_type", "count"),
                avg_cosine_similarity=("cosine_similarity", "mean"),
                p90_cosine_similarity=("cosine_similarity", lambda x: x.quantile(0.9)),
            )
            .reset_index()
            .sort_values("pairs", ascending=False)
        )
        save_csv(summary, out_dir / "topic_controlled_pairs_summary.csv")
    else:
        summary = pd.DataFrame()
        save_csv(summary, out_dir / "topic_controlled_pairs_summary.csv")

    preview_path = out_dir / "topic_controlled_same_genre_pairs_preview.md"
    lines = [
        "# Topic-Controlled Same-Genre Example Pairs",
        "",
        "Pairs are nearest neighbors inside the same (language, primary_genre, topic_cluster).",
        "",
    ]
    if pairs_df.empty:
        lines.append("No eligible groups were large enough for pair extraction.")
    else:
        for idx, row in pairs_df.head(80).iterrows():
            lines.extend(
                [
                    f"## Pair {idx + 1}: {row['pair_type']}",
                    f"- lang: `{row['lang']}` | primary_genre: `{row['primary_genre']}` | topic_cluster: `{int(row['topic_cluster'])}`",
                    f"- cosine_similarity: `{row['cosine_similarity']:.4f}` | length_ratio: `{row['length_ratio']:.3f}`",
                    f"- doc1: `{row['doc_id_1']}` | author1: `{row['author_id_1']}`",
                    f"- doc2: `{row['doc_id_2']}` | author2: `{row['author_id_2']}`",
                    f"- snippet1: {row['text_snippet_1']}",
                    f"- snippet2: {row['text_snippet_2']}",
                    "",
                ]
            )
    preview_path.write_text("\n".join(lines), encoding="utf-8")

    stats = {
        "topic_groups_processed": int(strata.shape[0]),
        "topic_clusters_summarized": int(topics_df.shape[0]),
        "pairs_exported": int(pairs_df.shape[0]),
        "same_author_pairs": int(
            (pairs_df["pair_type"] == "same_author_same_genre_same_topic").sum()
        )
        if not pairs_df.empty
        else 0,
        "different_author_pairs": int(
            (pairs_df["pair_type"] == "different_author_same_genre_same_topic").sum()
        )
        if not pairs_df.empty
        else 0,
    }
    save_json(stats, out_dir / "topic_controlled_pairs_stats.json")
    return stats


def embedding_visualization(
    docs: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    max_points: int = 3000,
) -> dict:
    work = docs[docs["content_len_chars"] >= 60].copy()
    work = work.dropna(subset=["lang", "primary_genre"])
    if work.empty:
        save_csv(pd.DataFrame(), out_dir / "embedding_projection_points.csv")
        return {}

    sampled = _stratified_sample(work, field="lang", n=max_points, seed=seed)
    if len(sampled) < 80:
        save_csv(pd.DataFrame(), out_dir / "embedding_projection_points.csv")
        return {}

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=12000,
    )
    x = vectorizer.fit_transform(sampled["content"].str.slice(0, 1400).tolist())
    n_components = min(64, x.shape[0] - 1, x.shape[1] - 1)
    if n_components < 2:
        save_csv(pd.DataFrame(), out_dir / "embedding_projection_points.csv")
        return {}

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    x_reduced = svd.fit_transform(x)

    perplexity = max(8, min(35, (len(sampled) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    coords = tsne.fit_transform(x_reduced)
    sampled = sampled.copy()
    sampled["x"] = coords[:, 0]
    sampled["y"] = coords[:, 1]
    save_csv(sampled, out_dir / "embedding_projection_points.csv")

    fig_dir = out_dir.parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=sampled,
        x="x",
        y="y",
        hue="lang",
        s=16,
        alpha=0.75,
        linewidth=0,
    )
    plt.title("2D embedding projection colored by language")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "embedding_projection_by_language.png", dpi=300)
    plt.close()

    top_genres = sampled["primary_genre"].value_counts().head(10).index.tolist()
    sampled["primary_genre_top10"] = sampled["primary_genre"].where(
        sampled["primary_genre"].isin(top_genres),
        "other",
    )
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=sampled,
        x="x",
        y="y",
        hue="primary_genre_top10",
        s=16,
        alpha=0.75,
        linewidth=0,
    )
    plt.title("2D embedding projection colored by primary genre")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "embedding_projection_by_primary_genre.png", dpi=300)
    plt.close()

    summary = {
        "points_used": int(len(sampled)),
        "vectorizer_features": int(x.shape[1]),
        "svd_components": int(n_components),
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "tsne_perplexity": int(perplexity),
    }
    save_json(summary, out_dir / "embedding_projection_summary.json")
    return summary


def write_markdown_report(
    out_dir: Path,
    *,
    dataset_dir: Path,
    splits: Sequence[str],
    basic_summary: dict,
    entropy_summary: dict,
    cross_summary: dict,
    split_summary: dict,
    duplicate_summary: dict,
    topic_summary: dict,
    embedding_summary: dict,
) -> None:
    report_path = out_dir / "qualitative_report.md"
    lines = [
        "# Qualitative Analysis Report",
        "",
        f"- dataset_dir: `{dataset_dir}`",
        f"- splits: `{', '.join(splits)}`",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Basic Statistics",
        f"- total_docs: `{basic_summary.get('total_docs', 0)}`",
        f"- total_docs_with_author: `{basic_summary.get('total_docs_with_author', 0)}`",
        f"- unique_authors: `{basic_summary.get('unique_authors', 0)}`",
        f"- docs_per_author_min: `{basic_summary.get('docs_per_author_min', 0)}`",
        f"- docs_per_author_avg: `{basic_summary.get('docs_per_author_avg', 0):.4f}`",
        f"- docs_per_author_max: `{basic_summary.get('docs_per_author_max', 0)}`",
        "",
        "## Per-Author Genre Entropy",
        f"- authors_total: `{entropy_summary.get('authors_total', 0)}`",
        f"- authors_multi_genre: `{entropy_summary.get('authors_multi_genre', 0)}`",
        f"- pct_authors_multi_genre: `{entropy_summary.get('pct_authors_multi_genre', 0):.2f}%`",
        f"- avg_normalized_genre_entropy: `{entropy_summary.get('avg_normalized_genre_entropy', 0):.4f}`",
        "",
        "## Cross-Genre Author Analysis",
        f"- authors_multi_genre: `{cross_summary.get('authors_multi_genre', 0)}`",
        f"- genre_pairs_nonzero: `{cross_summary.get('genre_pairs_nonzero', 0)}`",
        "",
        "## Split Leakage",
        f"- authors_multi_split: `{split_summary.get('authors_multi_split', 0)}`",
        f"- pct_authors_multi_split: `{split_summary.get('pct_authors_multi_split', 0):.2f}%`",
        f"- authors_in_all_three_splits: `{split_summary.get('authors_in_all_three_splits', 0)}`",
        "",
        "## Exact Duplicates",
        f"- duplicate_groups: `{duplicate_summary.get('duplicate_groups', 0)}`",
        f"- duplicate_docs: `{duplicate_summary.get('duplicate_docs', 0)}`",
        f"- cross_author_groups: `{duplicate_summary.get('cross_author_groups', 0)}`",
        f"- cross_split_groups: `{duplicate_summary.get('cross_split_groups', 0)}`",
        f"- cross_phase_groups: `{duplicate_summary.get('cross_phase_groups', 0)}`",
        "",
        "## Topic-Controlled Same-Genre Pairs",
        f"- topic_groups_processed: `{topic_summary.get('topic_groups_processed', 0)}`",
        f"- topic_clusters_summarized: `{topic_summary.get('topic_clusters_summarized', 0)}`",
        f"- pairs_exported: `{topic_summary.get('pairs_exported', 0)}`",
        f"- same_author_pairs: `{topic_summary.get('same_author_pairs', 0)}`",
        f"- different_author_pairs: `{topic_summary.get('different_author_pairs', 0)}`",
        "",
        "## Embedding Visualization",
        f"- points_used: `{embedding_summary.get('points_used', 0)}`",
        f"- vectorizer_features: `{embedding_summary.get('vectorizer_features', 0)}`",
        f"- svd_components: `{embedding_summary.get('svd_components', 0)}`",
        f"- tsne_perplexity: `{embedding_summary.get('tsne_perplexity', 0)}`",
        "",
        "## Key Output Files",
        "- csv/basic/summary_overview.json",
        "- csv/entropy/author_genre_entropy.csv",
        "- csv/cross_genre/cross_genre_author_pairs.csv",
        "- csv/split_leakage/author_split_membership.csv",
        "- csv/duplicates/exact_duplicate_groups.csv",
        "- csv/topic_pairs/topic_controlled_same_genre_pairs.csv",
        "- csv/embedding/embedding_projection_points.csv",
        "- figures/author_genre_entropy_histogram.png",
        "- figures/author_split_overlap_heatmap.png",
        "- figures/exact_duplicate_group_flags.png",
        "- figures/cross_genre_author_overlap_heatmap.png",
        "- figures/embedding_projection_by_language.png",
        "- figures/embedding_projection_by_primary_genre.png",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(dataset_dir: Path, output_dir: Path, splits: Sequence[str], seed: int) -> None:
    docs = load_docs(dataset_dir, splits)

    csv_root = output_dir / "csv"
    basic_dir = csv_root / "basic"
    entropy_dir = csv_root / "entropy"
    cross_dir = csv_root / "cross_genre"
    split_dir = csv_root / "split_leakage"
    duplicate_dir = csv_root / "duplicates"
    topic_dir = csv_root / "topic_pairs"
    embedding_dir = csv_root / "embedding"

    basic_summary = author_basic_tables(docs, basic_dir)
    entropy_df, entropy_summary = per_author_genre_entropy(docs, entropy_dir)
    cross_summary = cross_genre_author_analysis(docs, entropy_df, cross_dir)
    split_summary = split_leakage_analysis(docs, split_dir)
    duplicate_summary = exact_duplicate_analysis(docs, duplicate_dir)
    topic_summary = topic_controlled_same_genre_pairs(docs, topic_dir, seed=seed)
    embedding_summary = embedding_visualization(docs, embedding_dir, seed=seed)

    write_markdown_report(
        output_dir,
        dataset_dir=dataset_dir,
        splits=splits,
        basic_summary=basic_summary,
        entropy_summary=entropy_summary,
        cross_summary=cross_summary,
        split_summary=split_summary,
        duplicate_summary=duplicate_summary,
        topic_summary=topic_summary,
        embedding_summary=embedding_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qualitative post-analysis on AuthBench outputs.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("processing/outputs/combined_phase1_phase2_1m"),
        help="Root directory containing split folders with candidates/queries/ground_truth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_analysis/outputs/combined_phase1_phase2_1m/qualitative"),
        help="Output directory for qualitative analysis artifacts.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to include. Use 'all' (default behavior) to auto-discover.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and clustering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_splits = args.splits
    if selected_splits is None or "all" in selected_splits:
        selected_splits = discover_splits(args.dataset_dir)
    if not selected_splits:
        raise FileNotFoundError(f"No valid split directories found in {args.dataset_dir}.")

    run(args.dataset_dir, args.output_dir, selected_splits, seed=args.seed)
