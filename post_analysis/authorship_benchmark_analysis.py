#!/usr/bin/env python3
"""Supplementary authorship-benchmark analysis for AuthBench datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from processing.deduplication import (
    DedupConfig,
    _author_signature_source,
    _doc_sort_key,
    _hamming_distance,
    _lsh_keys,
    _normalize_text as dedup_normalize_text,
    _simhash64,
)
from processing.postprocess import _read_stage_documents

from . import analyze_dataset as stats_analysis
from . import qualitative_analysis as qual_analysis

sns.set_theme(style="whitegrid", context="talk")

DEFAULT_SPLITS: Sequence[str] = ("train", "dev", "test")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def format_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def format_pct(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return f"{0.0:.{digits}f}%"


def compact_stage_label(label: str) -> str:
    mapping = {
        "build output": "build\noutput",
        "after filter": "after\nfilter",
        "after dedup": "after\ndedup",
        "after audit": "after\naudit",
        "final benchmark": "final\nbenchmark",
        "merge loaded": "merge\nloaded",
        "after merge dedup": "merge\ndedup",
        "after cross-phase exact": "cross-phase\nexact",
        "selected for combined": "selected",
        "combined exported": "combined\nexported",
    }
    return mapping.get(label, label.replace(" ", "\n"))


def mode_or_none(values: Iterable[object]) -> object | None:
    filtered = [v for v in values if pd.notna(v)]
    if not filtered:
        return None
    counts = Counter(filtered)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def gini_coefficient(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.allclose(arr.sum(), 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n)


def hhi(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    total = arr.sum()
    if total == 0:
        return 0.0
    shares = arr / total
    return float(np.square(shares).sum())


def lighten_color(color: object, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def content_hash(text: str | None) -> str:
    normalized = dedup_normalize_text(str(text or ""))
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def discover_splits(dataset_dir: Path, requested: Sequence[str] | None) -> list[str]:
    if requested is None or "all" in requested:
        splits = stats_analysis.discover_splits(dataset_dir)
    else:
        splits = list(requested)
    if not splits:
        raise FileNotFoundError(f"No valid split directories found in {dataset_dir}.")
    return splits


def load_dataset_docs(dataset_dir: Path, splits: Sequence[str]) -> pd.DataFrame:
    docs = qual_analysis.load_docs(dataset_dir, splits)
    docs["token_length_bucket"] = docs["token_length"].apply(stats_analysis.token_length_bucket)
    docs["genre"] = docs["genre"].fillna("unknown")
    docs["primary_genre"] = docs["primary_genre"].fillna("unknown")
    docs["lang"] = docs["lang"].fillna("unknown")
    docs["source"] = docs["source"].fillna("unknown")
    docs["doc_type"] = docs["doc_type"].fillna("unknown")
    return docs


def summary_overview(docs: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    docs_per_author = (
        author_docs.groupby("author_id").size().rename("docs_per_author")
        if not author_docs.empty
        else pd.Series(dtype=float, name="docs_per_author")
    )
    row = {
        "total_docs": int(len(docs)),
        "docs_with_author": int(len(author_docs)),
        "total_unique_authors": int(author_docs["author_id"].nunique()),
        "avg_docs_per_author": float(docs_per_author.mean()) if not docs_per_author.empty else 0.0,
        "median_docs_per_author": float(docs_per_author.median()) if not docs_per_author.empty else 0.0,
        "max_docs_per_author": int(docs_per_author.max()) if not docs_per_author.empty else 0,
        "language_count": int(docs["lang"].dropna().nunique()),
        "genre_count": int(docs["genre"].dropna().nunique()),
        "primary_genre_count": int(docs["primary_genre"].dropna().nunique()),
        "source_count": int(docs["source"].dropna().nunique()),
        "query_docs": int((docs["doc_type"] == "query").sum()),
        "candidate_docs": int((docs["doc_type"] == "candidate").sum()),
        "avg_token_length": float(docs["token_length"].mean()) if docs["token_length"].notna().any() else 0.0,
    }
    save_csv(pd.DataFrame([row]), output_dir / "summary_overview.csv")
    save_json(row, output_dir / "summary_overview.json")
    return row


def unique_author_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    if author_docs.empty:
        for name in (
            "unique_authors_by_language.csv",
            "unique_authors_by_genre.csv",
            "unique_authors_by_primary_genre.csv",
            "unique_authors_by_length_bucket.csv",
            "unique_authors_by_lang_primary_genre_length.csv",
        ):
            save_csv(pd.DataFrame(), output_dir / name)
        return

    mappings = [
        ("lang", "unique_authors_by_language.csv"),
        ("genre", "unique_authors_by_genre.csv"),
        ("primary_genre", "unique_authors_by_primary_genre.csv"),
        ("token_length_bucket", "unique_authors_by_length_bucket.csv"),
    ]
    for field, filename in mappings:
        counts = (
            author_docs.groupby(field)["author_id"]
            .nunique()
            .reset_index(name="unique_authors")
            .sort_values("unique_authors", ascending=False)
        )
        save_csv(counts, output_dir / filename)

    cell_counts = (
        author_docs.groupby(["lang", "primary_genre", "token_length_bucket"])["author_id"]
        .nunique()
        .reset_index(name="unique_authors")
        .sort_values("unique_authors", ascending=False)
    )
    save_csv(cell_counts, output_dir / "unique_authors_by_lang_primary_genre_length.csv")


def docs_per_author_distributions(docs: pd.DataFrame, output_dir: Path) -> None:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    if author_docs.empty:
        for name in (
            "docs_per_author_distribution_overall.csv",
            "docs_per_author_distribution_by_language.csv",
            "docs_per_author_distribution_by_primary_genre.csv",
            "docs_per_author_distribution_by_length_bucket.csv",
        ):
            save_csv(pd.DataFrame(), output_dir / name)
        return

    overall = author_docs.groupby("author_id").size().reset_index(name="docs_per_author")
    overall_dist = (
        overall.groupby("docs_per_author")
        .size()
        .reset_index(name="authors")
        .sort_values("docs_per_author")
    )
    overall_dist["pct_authors"] = overall_dist["authors"] / overall_dist["authors"].sum() * 100.0
    save_csv(overall_dist, output_dir / "docs_per_author_distribution_overall.csv")

    for field, filename in (
        ("lang", "docs_per_author_distribution_by_language.csv"),
        ("primary_genre", "docs_per_author_distribution_by_primary_genre.csv"),
        ("token_length_bucket", "docs_per_author_distribution_by_length_bucket.csv"),
    ):
        group_counts = (
            author_docs.groupby([field, "author_id"])
            .size()
            .reset_index(name="docs_per_author")
        )
        dist = (
            group_counts.groupby([field, "docs_per_author"])
            .size()
            .reset_index(name="authors")
            .sort_values([field, "docs_per_author"])
        )
        dist["pct_authors_within_group"] = (
            dist["authors"] / dist.groupby(field)["authors"].transform("sum") * 100.0
        )
        save_csv(dist, output_dir / filename)


def author_balance_summary(author_group_df: pd.DataFrame, field_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in author_group_df.groupby(list(field_names), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["docs_per_author"].to_numpy(dtype=float)
        docs_total = float(values.sum())
        sorted_group = group.sort_values("docs_per_author", ascending=False).reset_index(drop=True)
        row = {
            field: key for field, key in zip(field_names, keys)
        }
        row.update(
            {
                "docs": int(docs_total),
                "unique_authors": int(group["author_id"].nunique()),
                "avg_docs_per_author": float(values.mean()) if values.size else 0.0,
                "median_docs_per_author": float(np.median(values)) if values.size else 0.0,
                "max_docs_per_author": int(values.max()) if values.size else 0,
                "top_author_id": sorted_group.loc[0, "author_id"] if not sorted_group.empty else None,
                "top_author_docs": int(sorted_group.loc[0, "docs_per_author"]) if not sorted_group.empty else 0,
                "top_author_share_pct": safe_div(float(values.max()) if values.size else 0.0, docs_total) * 100.0,
                "top_5_author_share_pct": safe_div(float(np.sort(values)[-5:].sum()) if values.size else 0.0, docs_total) * 100.0,
                "gini_docs_per_author": gini_coefficient(values),
                "hhi_docs_per_author": hhi(values),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["top_author_share_pct", "docs"],
        ascending=[False, False],
    )


def author_balance_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    if author_docs.empty:
        for name in (
            "author_balance_by_language.csv",
            "author_balance_by_genre.csv",
            "author_balance_by_primary_genre.csv",
            "author_balance_by_length_bucket.csv",
            "author_balance_by_lang_primary_genre_length.csv",
        ):
            save_csv(pd.DataFrame(), output_dir / name)
        return

    overall_author_counts = (
        author_docs.groupby("author_id").size().reset_index(name="docs_per_author")
    )
    overall = pd.DataFrame(
        [
            {
                "docs": int(overall_author_counts["docs_per_author"].sum()),
                "unique_authors": int(overall_author_counts["author_id"].nunique()),
                "avg_docs_per_author": float(overall_author_counts["docs_per_author"].mean()),
                "median_docs_per_author": float(overall_author_counts["docs_per_author"].median()),
                "max_docs_per_author": int(overall_author_counts["docs_per_author"].max()),
                "top_author_id": overall_author_counts.sort_values("docs_per_author", ascending=False).iloc[0]["author_id"],
                "top_author_docs": int(overall_author_counts["docs_per_author"].max()),
                "top_author_share_pct": safe_div(
                    float(overall_author_counts["docs_per_author"].max()),
                    float(overall_author_counts["docs_per_author"].sum()),
                )
                * 100.0,
                "top_5_author_share_pct": safe_div(
                    float(overall_author_counts["docs_per_author"].sort_values(ascending=False).head(5).sum()),
                    float(overall_author_counts["docs_per_author"].sum()),
                )
                * 100.0,
                "gini_docs_per_author": gini_coefficient(overall_author_counts["docs_per_author"]),
                "hhi_docs_per_author": hhi(overall_author_counts["docs_per_author"]),
            }
        ]
    )
    save_csv(overall, output_dir / "author_balance_overall.csv")

    for fields, filename in (
        (["lang"], "author_balance_by_language.csv"),
        (["genre"], "author_balance_by_genre.csv"),
        (["primary_genre"], "author_balance_by_primary_genre.csv"),
        (["token_length_bucket"], "author_balance_by_length_bucket.csv"),
        (["lang", "primary_genre", "token_length_bucket"], "author_balance_by_lang_primary_genre_length.csv"),
    ):
        group_counts = (
            author_docs.groupby(fields + ["author_id"])
            .size()
            .reset_index(name="docs_per_author")
        )
        summary = author_balance_summary(group_counts, fields)
        save_csv(summary, output_dir / filename)


def distribution_tables(docs: pd.DataFrame, output_dir: Path) -> None:
    for field, filename in (
        ("lang", "language_distribution.csv"),
        ("source", "source_distribution.csv"),
        ("token_length_bucket", "document_length_distribution.csv"),
        ("primary_genre", "primary_genre_distribution.csv"),
        ("genre", "genre_distribution.csv"),
    ):
        counts = (
            docs.groupby(field)
            .size()
            .reset_index(name="docs")
            .sort_values("docs", ascending=False)
        )
        counts["pct_docs"] = counts["docs"] / counts["docs"].sum() * 100.0
        save_csv(counts, output_dir / filename)


def plot_pie_chart(
    counts: pd.Series,
    output_path: Path,
    title: str,
    *,
    max_slices: int | None = None,
) -> pd.DataFrame:
    work = counts[counts > 0].sort_values(ascending=False).copy()
    if max_slices is not None and len(work) > max_slices:
        head = work.iloc[: max_slices - 1]
        tail_sum = float(work.iloc[max_slices - 1 :].sum())
        work = pd.concat([head, pd.Series({"other": tail_sum})])

    plot_df = (
        work.rename_axis("label")
        .reset_index(name="docs")
        .assign(pct_docs=lambda df: df["docs"] / df["docs"].sum() * 100.0)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 8.5))
    wedges, _, autotexts = plt.pie(
        plot_df["docs"],
        labels=None,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        pctdistance=0.75,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1.2},
    )
    for text in autotexts:
        text.set_fontsize(10)
    legend_labels = [
        f"{label} ({int(docs):,}, {pct:.1f}%)"
        for label, docs, pct in plot_df[["label", "docs", "pct_docs"]].itertuples(index=False)
    ]
    plt.legend(
        wedges,
        legend_labels,
        title="Breakdown",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return plot_df


def plot_author_balance_histogram(docs: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    if author_docs.empty:
        save_csv(pd.DataFrame(), output_path.with_suffix(".csv"))
        return pd.DataFrame()

    dist = (
        author_docs.groupby("author_id")
        .size()
        .value_counts()
        .sort_index()
        .rename_axis("docs_per_author")
        .reset_index(name="authors")
    )
    dist["pct_authors"] = dist["authors"] / dist["authors"].sum() * 100.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=dist, x="docs_per_author", y="authors", color="#4C72B0")
    plt.xlabel("Number of documents per author")
    plt.ylabel("Number of authors")
    plt.title("Author balance histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return dist


def plot_nested_genre_donut(
    docs: pd.DataFrame,
    output_path: Path,
    *,
    max_primary_genres: int = 10,
    max_subgenres_per_primary: int = 8,
    legend_subgenres_per_primary: int = 3,
) -> pd.DataFrame:
    work = docs.copy()
    work["primary_genre"] = work["primary_genre"].fillna("unknown")
    work["genre"] = work["genre"].fillna("unknown")

    primary_counts = work["primary_genre"].value_counts()
    if len(primary_counts) > max_primary_genres:
        keep_primary = primary_counts.head(max_primary_genres - 1).index
        work["primary_genre_plot"] = work["primary_genre"].where(
            work["primary_genre"].isin(keep_primary),
            "other",
        )
    else:
        work["primary_genre_plot"] = work["primary_genre"]

    work["genre_plot"] = work["genre"]
    detail_rows: list[dict[str, object]] = []
    total_docs = max(int(work.shape[0]), 1)

    primary_counts = work["primary_genre_plot"].value_counts().sort_values(ascending=False)
    inner_values = primary_counts.tolist()
    inner_labels = primary_counts.index.tolist()

    cmap = plt.get_cmap("tab20")
    inner_colors = [cmap(i % cmap.N) for i in range(len(inner_labels))]
    primary_color_map = dict(zip(inner_labels, inner_colors))

    outer_values: list[float] = []
    outer_labels: list[str] = []
    outer_colors: list[tuple[float, float, float]] = []

    for primary_label in inner_labels:
        subset = work[work["primary_genre_plot"] == primary_label].copy()
        sub_counts = subset["genre_plot"].value_counts().sort_values(ascending=False)
        if len(sub_counts) > max_subgenres_per_primary:
            head = sub_counts.iloc[: max_subgenres_per_primary - 1]
            other_sum = float(sub_counts.iloc[max_subgenres_per_primary - 1 :].sum())
            sub_counts = pd.concat([head, pd.Series({"other": other_sum})])

        base_color = primary_color_map[primary_label]
        for idx, (subgenre, count) in enumerate(sub_counts.items()):
            outer_values.append(float(count))
            outer_labels.append(subgenre)
            outer_colors.append(
                lighten_color(
                    base_color,
                    0.12 + 0.55 * safe_div(idx, max(len(sub_counts) - 1, 1)),
                )
            )
            detail_rows.append(
                {
                    "primary_genre_plot": primary_label,
                    "genre_plot": subgenre,
                    "docs": int(count),
                    "pct_docs": safe_div(count, total_docs) * 100.0,
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["docs", "primary_genre_plot"], ascending=[False, True])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(18, 11),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
    )

    primary_pct = primary_counts / total_docs * 100.0
    inner_wedge_labels = [
        f"{label}\n{primary_pct[label]:.1f}%"
        if primary_pct[label] >= 4.0
        else ""
        for label in inner_labels
    ]

    ax.pie(
        outer_values,
        radius=1.0,
        labels=None,
        startangle=90,
        counterclock=False,
        colors=outer_colors,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.4},
    )
    ax.pie(
        inner_values,
        radius=0.72,
        labels=inner_wedge_labels,
        startangle=90,
        counterclock=False,
        colors=inner_colors,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.0},
        labeldistance=0.62,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    ax.set_title("Primary genre and subgenre distribution")
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    y = 0.96
    primary_summary = (
        detail_df.groupby("primary_genre_plot", as_index=False)
        .agg(primary_docs=("docs", "sum"))
        .sort_values("primary_docs", ascending=False)
    )
    for row in primary_summary.itertuples(index=False):
        primary_label = row.primary_genre_plot
        primary_docs = int(row.primary_docs)
        primary_pct_value = safe_div(primary_docs, total_docs) * 100.0
        primary_color = primary_color_map[primary_label]
        legend_ax.add_patch(
            Rectangle((0.03, y - 0.018), 0.03, 0.03, facecolor=primary_color, edgecolor="none")
        )
        legend_ax.text(
            0.08,
            y,
            f"{primary_label} ({primary_pct_value:.1f}%)",
            fontsize=11,
            fontweight="bold",
            va="center",
        )
        y -= 0.055
        top_subgenres = (
            detail_df[detail_df["primary_genre_plot"] == primary_label]
            .sort_values("docs", ascending=False)
            .head(legend_subgenres_per_primary)
        )
        for sub_idx, sub_row in enumerate(top_subgenres.itertuples(index=False)):
            legend_ax.add_patch(
                Rectangle(
                    (0.08, y - 0.014),
                    0.022,
                    0.022,
                    facecolor=lighten_color(primary_color, 0.18 + 0.18 * sub_idx),
                    edgecolor="none",
                )
            )
            legend_ax.text(
                0.12,
                y,
                f"{sub_row.genre_plot} ({sub_row.pct_docs:.1f}%)",
                fontsize=9.5,
                va="center",
            )
            y -= 0.038
        y -= 0.02
        if y < 0.06:
            break

    legend_ax.text(
        0.03,
        0.02,
        "Legend percentages are shares of the full benchmark.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return detail_df


def plot_stage_document_flow(
    phase1_counts: list[tuple[str, int]],
    phase2_counts: list[tuple[str, int]],
    combined_total: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.9], height_ratios=[1.0, 0.95])
    phase1_ax = fig.add_subplot(gs[0, 0])
    phase2_ax = fig.add_subplot(gs[0, 1])
    merge_ax = fig.add_subplot(gs[0, 2])
    phase1_drop_ax = fig.add_subplot(gs[1, 0])
    phase2_drop_ax = fig.add_subplot(gs[1, 1])
    note_ax = fig.add_subplot(gs[1, 2])

    def draw_retention(ax: plt.Axes, counts: list[tuple[str, int]], title: str, color: str) -> list[dict[str, float]]:
        labels = [compact_stage_label(label) for label, _ in counts]
        values = np.asarray([count for _, count in counts], dtype=float)
        base = max(values[0], 1.0)
        retained_pct = values / base * 100.0
        x = np.arange(len(labels))
        ax.plot(x, retained_pct, color=color, linewidth=2.5, marker="o", markersize=7)
        ax.fill_between(x, retained_pct, np.full_like(retained_pct, retained_pct.min()), color=color, alpha=0.10)
        for idx, (count, pct) in enumerate(zip(values, retained_pct)):
            x_offset = -0.12 if idx % 2 == 0 else 0.12
            y_offset = 0.8 if idx % 2 == 0 else 3.0
            ax.text(
                idx + x_offset,
                pct + y_offset,
                f"{format_int(count)}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        transition_rows: list[dict[str, float]] = []
        for idx in range(len(values) - 1):
            drop = int(values[idx] - values[idx + 1])
            drop_pct = safe_div(drop, values[idx]) * 100.0
            transition_rows.append(
                {
                    "drop": drop,
                    "drop_pct": drop_pct,
                    "from_label": counts[idx][0],
                    "to_label": counts[idx + 1][0],
                }
            )
            if drop > 0:
                midpoint = (x[idx] + x[idx + 1]) / 2
                ax.text(
                    midpoint,
                    min(retained_pct[idx], retained_pct[idx + 1]) - 1.3,
                    f"-{format_int(drop)}\n({drop_pct:.2f}%)",
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#444444",
                )
        ymin = max(0.0, retained_pct.min() - 6.0)
        ymax = min(102.0, retained_pct.max() + 6.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Documents retained vs build output (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        return transition_rows

    def draw_drop_bars(ax: plt.Axes, transitions: list[dict[str, float]], title: str, color: str) -> None:
        labels = [
            f"{compact_stage_label(row['from_label'])}\n->\n{compact_stage_label(row['to_label'])}"
            for row in transitions
        ]
        drops = [row["drop"] for row in transitions]
        drop_pct = [row["drop_pct"] for row in transitions]
        x = np.arange(len(labels))
        bars = ax.bar(x, drops, color=color, alpha=0.85)
        for bar, drop, pct in zip(bars, drops, drop_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(drops) * 0.015 if max(drops) > 0 else 0.1,
                f"{format_int(drop)}\n({pct:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Documents dropped")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    phase1_transitions = draw_retention(phase1_ax, phase1_counts, "Phase 1 document retention", "#4C72B0")
    phase2_transitions = draw_retention(phase2_ax, phase2_counts, "Phase 2 document retention", "#55A868")
    draw_drop_bars(phase1_drop_ax, phase1_transitions, "Phase 1 stage drops", "#4C72B0")
    draw_drop_bars(phase2_drop_ax, phase2_transitions, "Phase 2 stage drops", "#55A868")

    phase1_selected = int(phase1_counts[-1][1])
    phase2_selected = int(phase2_counts[-1][1])
    merge_labels = ["phase1\nselected", "phase2\nselected", "combined\nexported"]
    merge_values = [phase1_selected, phase2_selected, int(combined_total)]
    merge_colors = ["#4C72B0", "#55A868", "#8172B3"]
    merge_x = np.arange(len(merge_labels))
    merge_bars = merge_ax.bar(merge_x, merge_values, color=merge_colors, width=0.62)
    for bar, value in zip(merge_bars, merge_values):
        merge_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(merge_values) * 0.015,
            format_int(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    merge_ax.set_xticks(merge_x)
    merge_ax.set_xticklabels(merge_labels, fontsize=9)
    merge_ax.set_ylabel("Documents")
    merge_ax.set_title("Combined export composition")
    merge_ax.grid(axis="y", alpha=0.25)

    note_ax.axis("off")
    note_ax.text(
        0.0,
        0.95,
        "How to read this figure",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    note_ax.text(
        0.0,
        0.79,
        "Top row: retained-document percentages make small stage losses visible even when absolute counts barely move.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    note_ax.text(
        0.0,
        0.57,
        "Bottom row: absolute drop counts show where documents were removed and how large each transition was relative to the previous stage.",
        fontsize=10,
        va="top",
        wrap=True,
    )
    note_ax.text(
        0.0,
        0.35,
        f"Combined export: {format_int(combined_total)} docs.\nPhase 1 contributes {safe_div(phase1_selected, combined_total) * 100.0:.1f}%.\nPhase 2 contributes {safe_div(phase2_selected, combined_total) * 100.0:.1f}%.",
        fontsize=10,
        va="top",
    )

    fig.suptitle("Stage document retention and drop summary", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_stage_author_counts(author_df: pd.DataFrame, output_path: Path) -> None:
    available = author_df.dropna(subset=["authors"]).copy()
    if available.empty:
        return
    available["authors"] = available["authors"].astype(int)
    order = [
        "build_output",
        "after_original_dedup",
        "final_benchmark",
        "merge_loaded",
        "after_merge_dedup",
        "after_cross_phase_exact",
        "combined_exported",
    ]
    available["stage"] = pd.Categorical(available["stage"], categories=order, ordered=True)
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=available,
        x="stage",
        y="authors",
        hue="phase",
        style="phase",
        markers=True,
        dashes=False,
    )
    plt.xlabel("Stage")
    plt.ylabel("Unique authors")
    plt.title("Available author counts across major stages")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def trace_dedup(
    docs: list,
    *,
    config: DedupConfig | None = None,
    max_examples_per_reason: int = 3,
) -> tuple[list, pd.DataFrame]:
    cfg = config or DedupConfig()
    ordered = sorted(list(docs), key=_doc_sort_key)
    reason_counter: Counter[str] = Counter()
    examples: list[dict[str, object]] = []

    exact_seen: dict[str, int] = {}
    near_index: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    near_fingerprints: list[int] = []
    near_langs: list[str] = []
    kept_after_near: list = []

    near_hamming_max = int(math.floor((1.0 - cfg.near_similarity_threshold) * 64.0))

    for doc in ordered:
        normalized = dedup_normalize_text(doc.text)
        if not normalized:
            continue

        if cfg.exact_text:
            exact_key = content_hash(normalized)
            if exact_key in exact_seen:
                reason = "exact_text_duplicate"
                if reason_counter[reason] < max_examples_per_reason:
                    kept_doc = kept_after_near[exact_seen[exact_key]]
                    examples.append(
                        {
                            "reason": reason,
                            "kept_author_id": kept_doc.author_id,
                            "kept_doc_id": kept_doc.doc_id,
                            "kept_lang": kept_doc.lang,
                            "kept_source": kept_doc.source,
                            "kept_snippet": qual_analysis.snippet(kept_doc.text),
                            "dropped_author_id": doc.author_id,
                            "dropped_doc_id": doc.doc_id,
                            "dropped_lang": doc.lang,
                            "dropped_source": doc.source,
                            "dropped_snippet": qual_analysis.snippet(doc.text),
                        }
                    )
                reason_counter[reason] += 1
                continue

        simhash = 0
        match_idx: int | None = None
        if cfg.near_text and doc.token_length >= cfg.min_tokens_for_near:
            simhash = _simhash64(normalized)
            lang_key = doc.lang if cfg.near_same_language_only else "*"
            candidate_indices: set[int] = set()
            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                bucket = near_index.get((lsh_key[0], lsh_key[1], lang_key))
                if bucket:
                    candidate_indices.update(bucket)
            for idx in sorted(candidate_indices):
                if cfg.near_same_language_only and near_langs[idx] != doc.lang:
                    continue
                dist = _hamming_distance(simhash, near_fingerprints[idx])
                if dist <= near_hamming_max:
                    match_idx = idx
                    break

        if match_idx is not None:
            reason = "near_text_duplicate"
            if reason_counter[reason] < max_examples_per_reason:
                kept_doc = kept_after_near[match_idx]
                examples.append(
                    {
                        "reason": reason,
                        "kept_author_id": kept_doc.author_id,
                        "kept_doc_id": kept_doc.doc_id,
                        "kept_lang": kept_doc.lang,
                        "kept_source": kept_doc.source,
                        "kept_snippet": qual_analysis.snippet(kept_doc.text),
                        "dropped_author_id": doc.author_id,
                        "dropped_doc_id": doc.doc_id,
                        "dropped_lang": doc.lang,
                        "dropped_source": doc.source,
                        "dropped_snippet": qual_analysis.snippet(doc.text),
                    }
                )
            reason_counter[reason] += 1
            continue

        kept_idx = len(kept_after_near)
        kept_after_near.append(doc)
        if cfg.exact_text:
            exact_seen[content_hash(normalized)] = kept_idx
        near_fingerprints.append(simhash)
        near_langs.append(doc.lang)
        if cfg.near_text and doc.token_length >= cfg.min_tokens_for_near:
            lang_key = doc.lang if cfg.near_same_language_only else "*"
            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                bucket = near_index[(lsh_key[0], lsh_key[1], lang_key)]
                if len(bucket) < cfg.max_bucket_size:
                    bucket.append(kept_idx)

    kept_after_author = list(kept_after_near)
    dropped_authors: set[str] = set()

    if cfg.author_similarity:
        docs_by_author: dict[str, list] = defaultdict(list)
        for doc in kept_after_near:
            docs_by_author[doc.author_id].append(doc)

        profile_items: list[tuple[str, str, str, int, int, int, object]] = []
        for author_id, author_docs in docs_by_author.items():
            ordered_docs = sorted(author_docs, key=_doc_sort_key)
            profile_docs = ordered_docs[: max(1, cfg.author_profile_docs)]
            normalized_parts = [dedup_normalize_text(d.text) for d in profile_docs]
            normalized_parts = [part for part in normalized_parts if part]
            if not normalized_parts:
                continue
            profile_text = " ".join(normalized_parts)
            profile_hash = _simhash64(profile_text)
            lang = profile_docs[0].lang
            source = _author_signature_source(author_docs)
            total_tokens = sum(d.token_length for d in author_docs)
            profile_items.append(
                (
                    author_id,
                    lang,
                    source,
                    len(author_docs),
                    total_tokens,
                    profile_hash,
                    profile_docs[0],
                )
            )

        profile_items.sort(key=lambda item: (-item[3], -item[4], item[1], item[2], item[0]))
        author_hamming_max = int(math.floor((1.0 - cfg.author_similarity_threshold) * 64.0))
        author_index: dict[tuple[int, int, str], list[int]] = defaultdict(list)

        for idx, (author_id, lang, source, doc_count, total_tokens, simhash, profile_doc) in enumerate(profile_items):
            if author_id in dropped_authors:
                continue
            lang_key = lang if cfg.author_same_language_only else "*"
            candidate_indices: set[int] = set()
            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                bucket = author_index.get((lsh_key[0], lsh_key[1], lang_key))
                if bucket:
                    candidate_indices.update(bucket)

            drop_current = False
            for cand_idx in sorted(candidate_indices):
                cand_author_id, cand_lang, cand_source, cand_doc_count, cand_total_tokens, cand_simhash, cand_profile_doc = profile_items[cand_idx]
                if cand_author_id in dropped_authors:
                    continue
                if cfg.author_same_language_only and cand_lang != lang:
                    continue
                if cfg.author_cross_source_only and cand_source == source:
                    continue
                dist = _hamming_distance(simhash, cand_simhash)
                if dist > author_hamming_max:
                    continue

                keep_current = (
                    (doc_count > cand_doc_count)
                    or (doc_count == cand_doc_count and total_tokens > cand_total_tokens)
                    or (doc_count == cand_doc_count and total_tokens == cand_total_tokens and author_id < cand_author_id)
                )
                reason = "near_author_duplicate"
                if keep_current:
                    dropped_authors.add(cand_author_id)
                    if reason_counter[reason] < max_examples_per_reason:
                        examples.append(
                            {
                                "reason": reason,
                                "kept_author_id": author_id,
                                "kept_doc_id": profile_doc.doc_id,
                                "kept_lang": profile_doc.lang,
                                "kept_source": profile_doc.source,
                                "kept_snippet": qual_analysis.snippet(profile_doc.text),
                                "dropped_author_id": cand_author_id,
                                "dropped_doc_id": cand_profile_doc.doc_id,
                                "dropped_lang": cand_profile_doc.lang,
                                "dropped_source": cand_profile_doc.source,
                                "dropped_snippet": qual_analysis.snippet(cand_profile_doc.text),
                            }
                        )
                    reason_counter[reason] += 1
                else:
                    dropped_authors.add(author_id)
                    drop_current = True
                    if reason_counter[reason] < max_examples_per_reason:
                        examples.append(
                            {
                                "reason": reason,
                                "kept_author_id": cand_author_id,
                                "kept_doc_id": cand_profile_doc.doc_id,
                                "kept_lang": cand_profile_doc.lang,
                                "kept_source": cand_profile_doc.source,
                                "kept_snippet": qual_analysis.snippet(cand_profile_doc.text),
                                "dropped_author_id": author_id,
                                "dropped_doc_id": profile_doc.doc_id,
                                "dropped_lang": profile_doc.lang,
                                "dropped_source": profile_doc.source,
                                "dropped_snippet": qual_analysis.snippet(profile_doc.text),
                            }
                        )
                    reason_counter[reason] += 1
                    break

            if drop_current:
                continue

            for lsh_key in _lsh_keys(simhash, cfg.near_lsh_bands):
                key = (lsh_key[0], lsh_key[1], lang_key)
                bucket = author_index[key]
                if len(bucket) < cfg.max_bucket_size:
                    bucket.append(idx)

        if dropped_authors:
            kept_after_author = [doc for doc in kept_after_near if doc.author_id not in dropped_authors]

    return kept_after_author, pd.DataFrame(examples)


def quality_filter_examples(
    phase_name: str,
    phase_dir: Path,
    pipeline_data: dict,
    *,
    max_examples: int = 3,
) -> pd.DataFrame:
    log_path = phase_dir / "quality_filter_drops.log"
    if not log_path.exists():
        return pd.DataFrame()

    reason_counts = pipeline_data.get("stages", {}).get("quality_dedup_sampling", {}).get("drop_reasons", {})
    if not reason_counts:
        return pd.DataFrame()

    ranked_families = []
    family_counts: Counter[str] = Counter()
    for reason, count in reason_counts.items():
        family = "dirty_noise" if str(reason).startswith("dirty:") else str(reason)
        family_counts[family] += int(count)
    ranked_families = [family for family, _ in family_counts.most_common(max_examples)]

    rows: list[dict[str, object]] = []
    seen_families: set[str] = set()
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reason = str(row.get("reason", ""))
            family = "dirty_noise" if reason.startswith("dirty:") else reason
            if family not in ranked_families or family in seen_families:
                continue
            seen_families.add(family)
            rows.append(
                {
                    "phase": phase_name,
                    "stage": "quality_filter",
                    "reason_family": family,
                    "reason": reason,
                    "raw_id": row.get("raw_id"),
                    "doc_id": row.get("doc_id"),
                    "lang": row.get("lang"),
                    "source": row.get("source"),
                    "before_text": row.get("snippet"),
                    "after_status": "dropped",
                }
            )
            if len(rows) >= max_examples:
                break
    return pd.DataFrame(rows)


def language_audit_examples(
    phase_name: str,
    phase_dir: Path,
    *,
    max_examples: int = 3,
) -> pd.DataFrame:
    suspects_path = phase_dir / "language_audit_suspects.jsonl"
    if not suspects_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with suspects_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "phase": phase_name,
                    "stage": "language_audit",
                    "raw_id": row.get("raw_id"),
                    "author_id": row.get("author_id"),
                    "doc_id": row.get("doc_id"),
                    "source": row.get("source"),
                    "declared_lang": row.get("lang"),
                    "detected_lang": row.get("detected_lang"),
                    "detected_confidence": row.get("detected_confidence"),
                    "script_ratio": row.get("script_ratio"),
                    "reasons": ",".join(row.get("reasons", [])),
                    "before_text": row.get("text_preview"),
                    "after_status": "flagged_suspect",
                }
            )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["detected_confidence"] = pd.to_numeric(df["detected_confidence"], errors="coerce")
    df = df.sort_values("detected_confidence", ascending=False).drop_duplicates(
        subset=["declared_lang", "detected_lang"],
        keep="first",
    )
    return df.head(max_examples).reset_index(drop=True)


def sampling_deficit_examples(
    phase_name: str,
    pipeline_data: dict,
    *,
    max_examples: int = 3,
) -> pd.DataFrame:
    deficits = pipeline_data.get("stages", {}).get("quality_dedup_sampling", {}).get("sampling_deficits", [])
    if not deficits:
        return pd.DataFrame()
    df = pd.DataFrame(deficits)
    if df.empty:
        return df
    df["deficit"] = pd.to_numeric(df["needed"], errors="coerce") - pd.to_numeric(df["selected"], errors="coerce")
    df["phase"] = phase_name
    return df.sort_values("deficit", ascending=False).head(max_examples).reset_index(drop=True)


def cross_phase_exact_overlap_examples(
    phase1_docs: list,
    phase2_docs: list,
    *,
    max_examples: int = 3,
) -> tuple[pd.DataFrame, int]:
    phase2_lookup: dict[str, object] = {}
    for doc in phase2_docs:
        key = content_hash(doc.text)
        phase2_lookup.setdefault(key, doc)

    rows: list[dict[str, object]] = []
    kept_phase1: list = []
    for doc in phase1_docs:
        key = content_hash(doc.text)
        matched = phase2_lookup.get(key)
        if matched is None:
            kept_phase1.append(doc)
            continue
        if len(rows) < max_examples:
            rows.append(
                {
                    "stage": "cross_phase_exact_overlap",
                    "dropped_phase": "phase1",
                    "dropped_author_id": doc.author_id,
                    "dropped_doc_id": doc.doc_id,
                    "dropped_lang": doc.lang,
                    "dropped_source": doc.source,
                    "dropped_snippet": qual_analysis.snippet(doc.text),
                    "kept_phase": "phase2",
                    "kept_author_id": matched.author_id,
                    "kept_doc_id": matched.doc_id,
                    "kept_lang": matched.lang,
                    "kept_source": matched.source,
                    "kept_snippet": qual_analysis.snippet(matched.text),
                }
            )
    return pd.DataFrame(rows), len({doc.author_id for doc in kept_phase1})


def candidate_author_metadata_span(candidate_docs: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    if candidate_docs.empty:
        save_csv(pd.DataFrame(), output_dir / "candidate_author_metadata_span.csv")
        summary = {}
        save_json(summary, output_dir / "candidate_author_metadata_span_summary.json")
        return summary

    span = (
        candidate_docs.groupby("author_id")
        .agg(
            candidate_docs=("doc_id", "count"),
            candidate_languages=("lang", "nunique"),
            candidate_primary_genres=("primary_genre", "nunique"),
            candidate_genres=("genre", "nunique"),
            candidate_sources=("source", "nunique"),
            candidate_length_buckets=("token_length_bucket", "nunique"),
            dominant_lang=("lang", mode_or_none),
            dominant_primary_genre=("primary_genre", mode_or_none),
        )
        .reset_index()
        .sort_values(["candidate_docs", "candidate_primary_genres"], ascending=[False, False])
    )
    save_csv(span, output_dir / "candidate_author_metadata_span.csv")
    summary = {
        "authors_total": int(span.shape[0]),
        "pct_single_language_authors": float((span["candidate_languages"] == 1).mean() * 100.0),
        "pct_single_primary_genre_authors": float((span["candidate_primary_genres"] == 1).mean() * 100.0),
        "pct_single_fine_genre_authors": float((span["candidate_genres"] == 1).mean() * 100.0),
        "pct_single_source_authors": float((span["candidate_sources"] == 1).mean() * 100.0),
    }
    save_json(summary, output_dir / "candidate_author_metadata_span_summary.json")
    return summary


def positive_pair_metadata_alignment(
    dataset_dir: Path,
    splits: Sequence[str],
    docs: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    ground_truth = stats_analysis.load_ground_truth(dataset_dir, splits)
    if ground_truth.empty:
        save_csv(pd.DataFrame(), output_dir / "positive_pair_metadata_alignment.csv")
        save_csv(pd.DataFrame(), output_dir / "positive_pair_metadata_alignment_by_language.csv")
        summary = {}
        save_json(summary, output_dir / "positive_pair_metadata_alignment_summary.json")
        return summary

    work = docs.copy()
    work["token_length_bucket"] = work["token_length_bucket"].fillna("unknown")
    queries = (
        work[work["doc_type"] == "query"][
            ["doc_id", "lang", "genre", "primary_genre", "source", "token_length_bucket", "author_id"]
        ]
        .rename(
            columns={
                "doc_id": "query_id",
                "lang": "query_lang",
                "genre": "query_genre",
                "primary_genre": "query_primary_genre",
                "source": "query_source",
                "token_length_bucket": "query_length_bucket",
                "author_id": "query_author_id",
            }
        )
        .drop_duplicates(subset=["query_id"])
    )
    candidates = (
        work[work["doc_type"] == "candidate"][
            ["doc_id", "lang", "genre", "primary_genre", "source", "token_length_bucket", "author_id"]
        ]
        .rename(
            columns={
                "doc_id": "positive_id",
                "lang": "candidate_lang",
                "genre": "candidate_genre",
                "primary_genre": "candidate_primary_genre",
                "source": "candidate_source",
                "token_length_bucket": "candidate_length_bucket",
                "author_id": "candidate_author_id",
            }
        )
        .drop_duplicates(subset=["positive_id"])
    )

    aligned = ground_truth.merge(queries, on="query_id", how="left").merge(candidates, on="positive_id", how="left")
    comparisons = {
        "same_language": aligned["query_lang"] == aligned["candidate_lang"],
        "same_primary_genre": aligned["query_primary_genre"] == aligned["candidate_primary_genre"],
        "same_genre": aligned["query_genre"] == aligned["candidate_genre"],
        "same_source": aligned["query_source"] == aligned["candidate_source"],
        "same_length_bucket": aligned["query_length_bucket"] == aligned["candidate_length_bucket"],
    }
    for name, values in comparisons.items():
        aligned[name] = values

    overall = pd.DataFrame(
        [
            {
                "metric": metric,
                "pairs": int(len(aligned)),
                "matches": int(values.sum()),
                "pct_matches": float(values.mean() * 100.0),
            }
            for metric, values in comparisons.items()
        ]
    ).sort_values("pct_matches", ascending=False)
    save_csv(overall, output_dir / "positive_pair_metadata_alignment.csv")

    by_language = (
        aligned.groupby("query_lang")
        .agg(
            pairs=("query_id", "count"),
            pct_same_language=("same_language", lambda x: x.mean() * 100.0),
            pct_same_primary_genre=("same_primary_genre", lambda x: x.mean() * 100.0),
            pct_same_genre=("same_genre", lambda x: x.mean() * 100.0),
            pct_same_source=("same_source", lambda x: x.mean() * 100.0),
            pct_same_length_bucket=("same_length_bucket", lambda x: x.mean() * 100.0),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    save_csv(by_language, output_dir / "positive_pair_metadata_alignment_by_language.csv")

    summary = {
        row["metric"]: float(row["pct_matches"]) for row in overall.to_dict(orient="records")
    }
    summary["pairs_total"] = int(len(aligned))
    save_json(summary, output_dir / "positive_pair_metadata_alignment_summary.json")
    return summary


def metadata_cell_risk(candidate_docs: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if candidate_docs.empty:
        save_csv(pd.DataFrame(), output_dir / "metadata_cell_risk.csv")
        summary = {}
        save_json(summary, output_dir / "metadata_cell_risk_summary.json")
        return pd.DataFrame(), summary

    grouped = (
        candidate_docs.groupby(["lang", "primary_genre", "token_length_bucket", "author_id"])
        .size()
        .reset_index(name="docs_per_author")
    )
    cell_summary = author_balance_summary(grouped, ["lang", "primary_genre", "token_length_bucket"])
    if cell_summary.empty:
        save_csv(cell_summary, output_dir / "metadata_cell_risk.csv")
        summary = {}
        save_json(summary, output_dir / "metadata_cell_risk_summary.json")
        return cell_summary, summary

    def risk_bucket(row: pd.Series) -> str:
        if row["unique_authors"] < 5 or row["top_author_share_pct"] >= 40.0:
            return "high"
        if row["unique_authors"] < 15 or row["top_author_share_pct"] >= 25.0:
            return "medium"
        return "low"

    cell_summary["risk_level"] = cell_summary.apply(risk_bucket, axis=1)
    cell_summary = cell_summary.sort_values(
        ["risk_level", "top_author_share_pct", "docs"],
        ascending=[True, False, False],
    )
    save_csv(cell_summary, output_dir / "metadata_cell_risk.csv")
    summary = {
        "cells_total": int(cell_summary.shape[0]),
        "high_risk_cells": int((cell_summary["risk_level"] == "high").sum()),
        "medium_risk_cells": int((cell_summary["risk_level"] == "medium").sum()),
        "low_risk_cells": int((cell_summary["risk_level"] == "low").sum()),
    }
    save_json(summary, output_dir / "metadata_cell_risk_summary.json")
    return cell_summary, summary


def topic_leakage_report(
    summary_path: Path,
    *,
    alignment_summary: dict[str, object],
    span_summary: dict[str, object],
    cell_summary: dict[str, object],
) -> None:
    pairs_total = int(alignment_summary.get("pairs_total", 0) or 0)
    lines = [
        "# Topic Leakage Risk Report",
        "",
        "## Scope",
        f"- This report audits metadata shortcut risk on `{format_int(pairs_total)}` query-positive pairs and the candidate-author pool used by the combined benchmark.",
        "",
        "## Metric definitions and interpretation",
        f"- `pct_same_language_positive_pairs = {format_pct(alignment_summary.get('same_language', 0))}`",
        "  Method: align each query with its labeled positive candidate and measure the share whose `lang` metadata matches.",
        "  Interpretation: very high values mean the benchmark strongly constrains positives to the same language; this is usually desirable for fairness, but it also means a model can use language as a shortcut unless candidate pools are language-matched.",
        f"- `pct_same_primary_genre_positive_pairs = {format_pct(alignment_summary.get('same_primary_genre', 0))}`",
        "  Method: compare query and positive candidate `primary_genre` labels after pair alignment.",
        "  Interpretation: high values indicate positives mostly stay within the same broad writing domain. This reduces obvious genre confounds, but it also means genre metadata alone can narrow the search space if negatives are not matched.",
        f"- `pct_same_fine_grained_genre_positive_pairs = {format_pct(alignment_summary.get('same_genre', 0))}`",
        "  Method: compare the full fine-grained `genre` label for each positive pair.",
        "  Interpretation: this is the strongest topic-style control among the positive-pair metrics. High values imply better topic control; low values imply genre leakage remains possible inside the positive pool.",
        f"- `pct_same_source_positive_pairs = {format_pct(alignment_summary.get('same_source', 0))}`",
        "  Method: compare `source` metadata for each positive pair.",
        "  Interpretation: high values mean positives often come from the same platform or corpus, which can stabilize writing conditions but can also introduce source-specific artifacts.",
        f"- `pct_same_length_bucket_positive_pairs = {format_pct(alignment_summary.get('same_length_bucket', 0))}`",
        "  Method: bucket token length into `short`, `medium`, `long`, and `extra-long`, then compare bucket equality for each positive pair.",
        "  Interpretation: high values reduce trivial cues from length alone; low values suggest document size may help identify the positive candidate.",
        "",
        f"- `pct_single_language_candidate_authors = {format_pct(span_summary.get('pct_single_language_authors', 0))}`",
        "  Method: for each candidate author, count how many distinct languages appear across that author's candidate documents, then report the share with exactly one language.",
        "  Interpretation: if this is very high, language metadata sharply narrows candidate identity.",
        f"- `pct_single_primary_genre_candidate_authors = {format_pct(span_summary.get('pct_single_primary_genre_authors', 0))}`",
        "  Method: compute the share of candidate authors whose candidate documents appear in only one primary genre.",
        "  Interpretation: high values indicate strong genre specialization, so candidate sampling should be controlled within primary genre whenever possible.",
        f"- `pct_single_fine_genre_candidate_authors = {format_pct(span_summary.get('pct_single_fine_genre_authors', 0))}`",
        "  Method: compute the share of candidate authors whose candidate documents appear in only one fine-grained genre.",
        "  Interpretation: high values indicate even tighter topical concentration, which raises fine-grained topic leakage risk.",
        "",
        f"- `metadata_cells_total = {format_int(cell_summary.get('cells_total', 0))}`",
        "  Method: group candidate documents by `(lang, primary_genre, token_length_bucket)`, then summarize author counts and concentration inside each cell.",
        "  Interpretation: each cell approximates the actual metadata-conditioned candidate pool a model would face.",
        f"- `high_risk_metadata_cells = {format_int(cell_summary.get('high_risk_cells', 0))}`",
        f"- `medium_risk_metadata_cells = {format_int(cell_summary.get('medium_risk_cells', 0))}`",
        "  Risk rule: a cell is `high` if it has fewer than 5 authors or one author owns at least 40% of the documents; `medium` if it has fewer than 15 authors or one author owns at least 25%; otherwise `low`.",
        "  Interpretation: high-risk cells are places where metadata matching can still leave only a few plausible authors, so benchmark difficulty is partly driven by pool construction rather than authorship style alone.",
        "",
        "## Reading this benchmark",
        "- The positive-pair alignment metrics tell you whether the benchmark already controls language, genre, source, and length on the positive side.",
        "- The candidate-author span metrics tell you whether authors are metadata-specialized.",
        "- The metadata-cell risk metrics tell you whether the candidate pool remains balanced after matching on metadata.",
        "",
        "## Recommended standard practice",
        "1. Match positive candidates to queries on language and primary genre whenever possible.",
        "2. When enough data exists, also match or stratify on fine-grained genre and token-length bucket.",
        "3. Report same-language, same-primary-genre, same-fine-genre, same-source, and same-length-bucket rates for positive pairs.",
        "4. Audit candidate pools at the `(language, primary_genre, token_length_bucket)` level, and flag cells with too few authors or large top-author shares.",
        "5. Keep a metadata-only baseline so shortcut leakage can be measured explicitly.",
        "6. Keep a topic-controlled within-genre baseline so topical similarity can be separated from authorship cues.",
    ]
    write_text("\n".join(lines), summary_path)


def build_stage_tables_and_figures(
    dataset_dir: Path,
    docs: pd.DataFrame,
    output_tables: Path,
    output_figures: Path,
    output_reports: Path,
    *,
    phase1_dir: Path | None,
    phase2_dir: Path | None,
) -> None:
    merge_summary = read_json(dataset_dir / "merge_summary.json")
    if not merge_summary:
        return
    if phase1_dir is None or phase2_dir is None:
        return

    phase1_pipeline = read_json(phase1_dir / "pipeline_dynamics.json")
    phase2_pipeline = read_json(phase2_dir / "pipeline_dynamics.json")
    if not phase1_pipeline or not phase2_pipeline:
        return

    phase1_counts = [
        ("build output", int(phase1_pipeline["stages"]["build"]["summary"]["after_sampling"]["total"])),
        ("after filter", int(phase1_pipeline["stages"]["quality_dedup_sampling"]["after_filter"]["total"])),
        ("after dedup", int(phase1_pipeline["stages"]["quality_dedup_sampling"]["after_dedup"]["total"])),
        ("after audit", int(phase1_pipeline["stages"]["quality_dedup_sampling"]["after_language_audit"]["total"])),
        ("final benchmark", int(phase1_pipeline["stages"]["quality_dedup_sampling"]["after_sampling"]["total"])),
    ]
    phase2_counts = [
        ("build output", int(phase2_pipeline["stages"]["build"]["summary"]["after_sampling"]["total"])),
        ("after filter", int(phase2_pipeline["stages"]["quality_dedup_sampling"]["after_filter"]["total"])),
        ("after dedup", int(phase2_pipeline["stages"]["quality_dedup_sampling"]["after_dedup"]["total"])),
        ("after audit", int(phase2_pipeline["stages"]["quality_dedup_sampling"]["after_language_audit"]["total"])),
        ("final benchmark", int(phase2_pipeline["stages"]["quality_dedup_sampling"]["after_sampling"]["total"])),
    ]
    merge_stage_rows = [
        {
            "phase": "phase1",
            "stage_order": 5,
            "stage_label": "merge loaded",
            "docs": int(merge_summary["stage_counts"]["phase1_loaded"]),
        },
        {
            "phase": "phase1",
            "stage_order": 6,
            "stage_label": "after merge dedup",
            "docs": int(merge_summary["stage_counts"]["phase1_after_internal_dedup"]),
        },
        {
            "phase": "phase1",
            "stage_order": 7,
            "stage_label": "after cross-phase exact",
            "docs": int(merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"]),
        },
        {
            "phase": "phase1",
            "stage_order": 8,
            "stage_label": "selected for combined",
            "docs": int(merge_summary["stage_counts"]["phase1_selected"]),
        },
        {
            "phase": "phase2",
            "stage_order": 5,
            "stage_label": "merge loaded",
            "docs": int(merge_summary["stage_counts"]["phase2_loaded"]),
        },
        {
            "phase": "phase2",
            "stage_order": 6,
            "stage_label": "after merge dedup",
            "docs": int(merge_summary["stage_counts"]["phase2_after_internal_dedup"]),
        },
        {
            "phase": "phase2",
            "stage_order": 7,
            "stage_label": "selected for combined",
            "docs": int(merge_summary["stage_counts"]["phase2_selected"]),
        },
        {
            "phase": "combined",
            "stage_order": 9,
            "stage_label": "combined exported",
            "docs": int(merge_summary["stage_counts"]["combined_exported_documents"]),
        },
    ]
    rows = []
    for phase_name, stage_counts in (("phase1", phase1_counts), ("phase2", phase2_counts)):
        for order, (stage_label, count) in enumerate(stage_counts):
            rows.append(
                {
                    "phase": phase_name,
                    "stage_order": order,
                    "stage_label": stage_label,
                    "docs": int(count),
                }
            )
    rows.extend(merge_stage_rows)
    stage_counts_df = pd.DataFrame(rows).sort_values(["phase", "stage_order"])
    save_csv(stage_counts_df, output_tables / "stage_document_counts.csv")

    phase1_full_counts = phase1_counts + [
        ("merge loaded", int(merge_summary["stage_counts"]["phase1_loaded"])),
        ("after merge dedup", int(merge_summary["stage_counts"]["phase1_after_internal_dedup"])),
        ("after cross-phase exact", int(merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"])),
        ("selected for combined", int(merge_summary["stage_counts"]["phase1_selected"])),
    ]
    phase2_full_counts = phase2_counts + [
        ("merge loaded", int(merge_summary["stage_counts"]["phase2_loaded"])),
        ("after merge dedup", int(merge_summary["stage_counts"]["phase2_after_internal_dedup"])),
        ("selected for combined", int(merge_summary["stage_counts"]["phase2_selected"])),
    ]

    phase1_stage_docs = _read_stage_documents(phase1_dir)
    phase2_stage_docs = _read_stage_documents(phase2_dir)
    cross_phase_examples, phase1_after_cross_phase_authors = cross_phase_exact_overlap_examples(
        phase1_stage_docs,
        phase2_stage_docs,
    )
    phase1_after_cross_phase_authors = min(
        int(phase1_after_cross_phase_authors),
        int(merge_summary["phase1_dedup"]["authors_after"]),
    )

    edge_rows = []
    for phase_name, stage_counts in (("phase1", phase1_counts), ("phase2", phase2_counts)):
        for idx in range(len(stage_counts) - 1):
            current_label, current_docs = stage_counts[idx]
            next_label, next_docs = stage_counts[idx + 1]
            edge_rows.append(
                {
                    "phase": phase_name,
                    "from_stage": current_label,
                    "to_stage": next_label,
                    "docs_before": int(current_docs),
                    "docs_kept": int(next_docs),
                    "docs_dropped": int(current_docs - next_docs),
                    "retained_pct": safe_div(next_docs, current_docs) * 100.0,
                    "dropped_pct": safe_div(current_docs - next_docs, current_docs) * 100.0,
                }
            )
    edge_rows.extend(
        [
            {
                "phase": "phase1",
                "from_stage": "final benchmark",
                "to_stage": "merge loaded",
                "docs_before": int(phase1_counts[-1][1]),
                "docs_kept": int(merge_summary["stage_counts"]["phase1_loaded"]),
                "docs_dropped": int(phase1_counts[-1][1] - merge_summary["stage_counts"]["phase1_loaded"]),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_loaded"],
                    phase1_counts[-1][1],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    phase1_counts[-1][1] - merge_summary["stage_counts"]["phase1_loaded"],
                    phase1_counts[-1][1],
                )
                * 100.0,
            },
            {
                "phase": "phase1",
                "from_stage": "merge loaded",
                "to_stage": "after merge dedup",
                "docs_before": int(merge_summary["stage_counts"]["phase1_loaded"]),
                "docs_kept": int(merge_summary["stage_counts"]["phase1_after_internal_dedup"]),
                "docs_dropped": int(
                    merge_summary["stage_counts"]["phase1_loaded"]
                    - merge_summary["stage_counts"]["phase1_after_internal_dedup"]
                ),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_after_internal_dedup"],
                    merge_summary["stage_counts"]["phase1_loaded"],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_loaded"]
                    - merge_summary["stage_counts"]["phase1_after_internal_dedup"],
                    merge_summary["stage_counts"]["phase1_loaded"],
                )
                * 100.0,
            },
            {
                "phase": "phase1",
                "from_stage": "after merge dedup",
                "to_stage": "after cross-phase exact",
                "docs_before": int(merge_summary["stage_counts"]["phase1_after_internal_dedup"]),
                "docs_kept": int(merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"]),
                "docs_dropped": int(
                    merge_summary["stage_counts"]["phase1_after_internal_dedup"]
                    - merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"]
                ),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"],
                    merge_summary["stage_counts"]["phase1_after_internal_dedup"],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_after_internal_dedup"]
                    - merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"],
                    merge_summary["stage_counts"]["phase1_after_internal_dedup"],
                )
                * 100.0,
            },
            {
                "phase": "phase1",
                "from_stage": "after cross-phase exact",
                "to_stage": "selected for combined",
                "docs_before": int(merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"]),
                "docs_kept": int(merge_summary["stage_counts"]["phase1_selected"]),
                "docs_dropped": int(
                    merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"]
                    - merge_summary["stage_counts"]["phase1_selected"]
                ),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_selected"],
                    merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"]
                    - merge_summary["stage_counts"]["phase1_selected"],
                    merge_summary["stage_counts"]["phase1_pool_after_cross_phase_exact"],
                )
                * 100.0,
            },
            {
                "phase": "phase2",
                "from_stage": "final benchmark",
                "to_stage": "merge loaded",
                "docs_before": int(phase2_counts[-1][1]),
                "docs_kept": int(merge_summary["stage_counts"]["phase2_loaded"]),
                "docs_dropped": int(phase2_counts[-1][1] - merge_summary["stage_counts"]["phase2_loaded"]),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase2_loaded"],
                    phase2_counts[-1][1],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    phase2_counts[-1][1] - merge_summary["stage_counts"]["phase2_loaded"],
                    phase2_counts[-1][1],
                )
                * 100.0,
            },
            {
                "phase": "phase2",
                "from_stage": "merge loaded",
                "to_stage": "after merge dedup",
                "docs_before": int(merge_summary["stage_counts"]["phase2_loaded"]),
                "docs_kept": int(merge_summary["stage_counts"]["phase2_after_internal_dedup"]),
                "docs_dropped": int(
                    merge_summary["stage_counts"]["phase2_loaded"]
                    - merge_summary["stage_counts"]["phase2_after_internal_dedup"]
                ),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase2_after_internal_dedup"],
                    merge_summary["stage_counts"]["phase2_loaded"],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    merge_summary["stage_counts"]["phase2_loaded"]
                    - merge_summary["stage_counts"]["phase2_after_internal_dedup"],
                    merge_summary["stage_counts"]["phase2_loaded"],
                )
                * 100.0,
            },
            {
                "phase": "phase2",
                "from_stage": "after merge dedup",
                "to_stage": "selected for combined",
                "docs_before": int(merge_summary["stage_counts"]["phase2_after_internal_dedup"]),
                "docs_kept": int(merge_summary["stage_counts"]["phase2_selected"]),
                "docs_dropped": int(
                    merge_summary["stage_counts"]["phase2_after_internal_dedup"]
                    - merge_summary["stage_counts"]["phase2_selected"]
                ),
                "retained_pct": safe_div(
                    merge_summary["stage_counts"]["phase2_selected"],
                    merge_summary["stage_counts"]["phase2_after_internal_dedup"],
                )
                * 100.0,
                "dropped_pct": safe_div(
                    merge_summary["stage_counts"]["phase2_after_internal_dedup"]
                    - merge_summary["stage_counts"]["phase2_selected"],
                    merge_summary["stage_counts"]["phase2_after_internal_dedup"],
                )
                * 100.0,
            },
            {
                "phase": "combined",
                "from_stage": "selected for combined",
                "to_stage": "combined exported",
                "docs_before": int(
                    merge_summary["stage_counts"]["phase1_selected"]
                    + merge_summary["stage_counts"]["phase2_selected"]
                ),
                "docs_kept": int(merge_summary["stage_counts"]["combined_exported_documents"]),
                "docs_dropped": 0,
                "retained_pct": 100.0,
                "dropped_pct": 0.0,
            },
        ]
    )
    edge_df = pd.DataFrame(edge_rows)
    save_csv(edge_df, output_tables / "stage_document_flow_edges.csv")

    combined_phase1_docs = docs[docs["phase"] == "phase1"].dropna(subset=["author_id"])
    combined_phase2_docs = docs[docs["phase"] == "phase2"].dropna(subset=["author_id"])
    combined_all_docs = docs.dropna(subset=["author_id"])

    author_rows = [
        {
            "phase": "phase1",
            "stage": "build_output",
            "authors": int(phase1_pipeline["stages"]["build"]["summary"]["final_statistics"]["unique_authors"]),
            "count_source": "pipeline_dynamics",
        },
        {
            "phase": "phase1",
            "stage": "after_original_dedup",
            "authors": int(phase1_pipeline["stages"]["quality_dedup_sampling"]["deduplication"]["authors_after"]),
            "count_source": "pipeline_dynamics",
        },
        {
            "phase": "phase1",
            "stage": "final_benchmark",
            "authors": int(phase1_stage_docs and len({doc.author_id for doc in phase1_stage_docs}) or 0),
            "count_source": "reconstructed_from_final_outputs",
        },
        {
            "phase": "phase1",
            "stage": "merge_loaded",
            "authors": int(len({doc.author_id for doc in phase1_stage_docs})),
            "count_source": "reconstructed_from_final_outputs",
        },
        {
            "phase": "phase1",
            "stage": "after_merge_dedup",
            "authors": int(merge_summary["phase1_dedup"]["authors_after"]),
            "count_source": "merge_summary",
        },
        {
            "phase": "phase1",
            "stage": "after_cross_phase_exact",
            "authors": int(phase1_after_cross_phase_authors),
            "count_source": "approx_reconstructed_from_final_outputs",
        },
        {
            "phase": "phase1",
            "stage": "combined_exported",
            "authors": int(combined_phase1_docs["author_id"].nunique()),
            "count_source": "combined_documents",
        },
        {
            "phase": "phase2",
            "stage": "build_output",
            "authors": int(phase2_pipeline["stages"]["build"]["summary"]["final_statistics"]["unique_authors"]),
            "count_source": "pipeline_dynamics",
        },
        {
            "phase": "phase2",
            "stage": "after_original_dedup",
            "authors": int(phase2_pipeline["stages"]["quality_dedup_sampling"]["deduplication"]["authors_after"]),
            "count_source": "pipeline_dynamics",
        },
        {
            "phase": "phase2",
            "stage": "final_benchmark",
            "authors": int(phase2_stage_docs and len({doc.author_id for doc in phase2_stage_docs}) or 0),
            "count_source": "reconstructed_from_final_outputs",
        },
        {
            "phase": "phase2",
            "stage": "merge_loaded",
            "authors": int(len({doc.author_id for doc in phase2_stage_docs})),
            "count_source": "reconstructed_from_final_outputs",
        },
        {
            "phase": "phase2",
            "stage": "after_merge_dedup",
            "authors": int(merge_summary["phase2_dedup"]["authors_after"]),
            "count_source": "merge_summary",
        },
        {
            "phase": "phase2",
            "stage": "after_cross_phase_exact",
            "authors": int(merge_summary["phase2_dedup"]["authors_after"]),
            "count_source": "merge_summary",
        },
        {
            "phase": "phase2",
            "stage": "combined_exported",
            "authors": int(combined_phase2_docs["author_id"].nunique()),
            "count_source": "combined_documents",
        },
        {
            "phase": "combined",
            "stage": "combined_exported",
            "authors": int(combined_all_docs["author_id"].nunique()),
            "count_source": "combined_documents",
        },
    ]
    author_df = pd.DataFrame(author_rows)
    save_csv(author_df, output_tables / "stage_author_counts.csv")

    plot_stage_document_flow(
        phase1_full_counts,
        phase2_full_counts,
        int(merge_summary["stage_counts"]["combined_exported_documents"]),
        output_figures / "stage_document_sankey.png",
    )
    plot_stage_author_counts(author_df, output_figures / "stage_author_counts.png")

    quality_examples = pd.concat(
        [
            quality_filter_examples("phase1", phase1_dir, phase1_pipeline),
            quality_filter_examples("phase2", phase2_dir, phase2_pipeline),
        ],
        ignore_index=True,
    )
    save_csv(quality_examples, output_tables / "quality_filter_examples.csv")

    merge_dedup_summary = pd.DataFrame(
        [
            {
                "phase": "phase1",
                "input_docs": int(merge_summary["phase1_dedup"]["input_docs"]),
                "after_author_dedup": int(merge_summary["phase1_dedup"]["after_author_dedup"]),
                "dropped_total": int(merge_summary["phase1_dedup"]["dropped_total"]),
                "authors_before": int(merge_summary["phase1_dedup"]["authors_before"]),
                "authors_after": int(merge_summary["phase1_dedup"]["authors_after"]),
                "authors_dropped": int(merge_summary["phase1_dedup"]["authors_dropped"]),
                "drop_reasons": json.dumps(merge_summary["phase1_dedup"]["drop_reasons"], ensure_ascii=False),
            },
            {
                "phase": "phase2",
                "input_docs": int(merge_summary["phase2_dedup"]["input_docs"]),
                "after_author_dedup": int(merge_summary["phase2_dedup"]["after_author_dedup"]),
                "dropped_total": int(merge_summary["phase2_dedup"]["dropped_total"]),
                "authors_before": int(merge_summary["phase2_dedup"]["authors_before"]),
                "authors_after": int(merge_summary["phase2_dedup"]["authors_after"]),
                "authors_dropped": int(merge_summary["phase2_dedup"]["authors_dropped"]),
                "drop_reasons": json.dumps(merge_summary["phase2_dedup"]["drop_reasons"], ensure_ascii=False),
            },
        ]
    )
    save_csv(merge_dedup_summary, output_tables / "merge_dedup_summary.csv")

    audit_examples = pd.concat(
        [
            language_audit_examples("phase1", phase1_dir),
            language_audit_examples("phase2", phase2_dir),
        ],
        ignore_index=True,
    )
    save_csv(audit_examples, output_tables / "language_audit_examples.csv")

    sampling_examples = pd.concat(
        [
            sampling_deficit_examples("phase1", phase1_pipeline),
            sampling_deficit_examples("phase2", phase2_pipeline),
        ],
        ignore_index=True,
    )
    save_csv(sampling_examples, output_tables / "sampling_deficit_examples.csv")
    save_csv(cross_phase_examples, output_tables / "cross_phase_exact_overlap_examples.csv")

    stage_lines = [
        "# Stage Examples Report",
        "",
        "## Scope and method",
        "- `stage_document_counts.csv` records the document totals observed at each major pipeline stage.",
        "- `stage_document_flow_edges.csv` records each stage-to-stage transition with `docs_before`, `docs_kept`, `docs_dropped`, `retained_pct`, and `dropped_pct`.",
        "- `stage_author_counts.csv` records unique-author counts where exact monitoring exists; `count_source` tells you whether the number came from pipeline monitoring, merge summaries, or post hoc reconstruction.",
        "- Example snippets are exact when the upstream pipeline persisted the relevant documents. If only aggregate monitoring survived, the report says so explicitly.",
        "",
        "## How to interpret the metrics",
        "- `docs_before`: number of documents entering a stage transition.",
        "- `docs_kept`: number of documents that survive into the next stage.",
        "- `docs_dropped`: number of documents removed at that transition.",
        "- `retained_pct`: `docs_kept / docs_before`, useful when absolute drops are small compared with the full corpus size.",
        "- `dropped_pct`: `docs_dropped / docs_before`, useful for comparing the aggressiveness of different stages.",
        "- `authors_before`, `authors_after`, `authors_dropped`: author-level analogue of the document metrics for merge deduplication.",
        "",
        "## Transition summary",
        "### Phase 1",
    ]
    for row in edge_df[edge_df["phase"] == "phase1"].itertuples(index=False):
        stage_lines.append(
            f"- `{row.from_stage}` -> `{row.to_stage}`: kept `{format_int(row.docs_kept)}` / `{format_int(row.docs_before)}` "
            f"({row.retained_pct:.2f}%), dropped `{format_int(row.docs_dropped)}` ({row.dropped_pct:.2f}%)."
        )
    stage_lines.extend(["", "### Phase 2"])
    for row in edge_df[edge_df["phase"] == "phase2"].itertuples(index=False):
        stage_lines.append(
            f"- `{row.from_stage}` -> `{row.to_stage}`: kept `{format_int(row.docs_kept)}` / `{format_int(row.docs_before)}` "
            f"({row.retained_pct:.2f}%), dropped `{format_int(row.docs_dropped)}` ({row.dropped_pct:.2f}%)."
        )
    stage_lines.extend(
        [
            "",
            "## Build stage",
            "- Interpretation: this is the maximum document pool produced before downstream quality filtering, deduplication, audit, and final sampling.",
            "- Method: counts come directly from `pipeline_dynamics.json` for each phase.",
        ]
    )
    stage_lines.extend(
        [
            f"- phase1 build output docs: `{phase1_counts[0][1]:,}`",
            f"- phase2 build output docs: `{phase2_counts[0][1]:,}`",
            "- Document-level before/after examples are not recoverable post hoc for the build-stage author filter because only aggregated monitoring was persisted.",
            "",
            "## Quality filtering examples",
            "- Interpretation: these examples illustrate documents removed before deduplication because they violated corpus quality rules.",
            "- Method: examples are drawn from persisted pipeline monitoring when available and show the exact pre-drop text snippet.",
        ]
    )
    if quality_examples.empty:
        stage_lines.append("- No quality-filter examples were recovered.")
    else:
        for row in quality_examples.head(6).itertuples(index=False):
            stage_lines.extend(
                [
                    f"- `{row.phase}` | `{row.reason}` | `{row.lang}` | `{row.source}`",
                    f"  before: {row.before_text}",
                    "  after: dropped",
                ]
            )
    stage_lines.extend(
        [
            "",
            "## Language audit examples",
            "- Interpretation: these examples show cases where declared language metadata disagreed with automatic language detection.",
            "- Method: the report compares persisted declared labels with audit-time detector output and confidence.",
        ]
    )
    if audit_examples.empty:
        stage_lines.append("- No language-audit examples were recovered.")
    else:
        for row in audit_examples.head(6).itertuples(index=False):
            stage_lines.extend(
                [
                    f"- `{row.phase}` | declared `{row.declared_lang}` -> detected `{row.detected_lang}` | confidence `{row.detected_confidence:.4f}`",
                    f"  before: {row.before_text}",
                    "  after: flagged by audit",
                ]
            )
    stage_lines.extend(
        [
            "",
            "## Sampling pressure examples",
            "- Interpretation: a large `deficit` means the target benchmark composition could not be met for that metadata slice, so sampling pressure was high.",
            "- Method: deficits are reconstructed from per-cell sampling summaries in the persisted pipeline monitoring.",
        ]
    )
    if sampling_examples.empty:
        stage_lines.append("- No sampling-deficit examples were recovered.")
    else:
        for row in sampling_examples.head(6).itertuples(index=False):
            stage_lines.append(
                f"- `{row.phase}` | lang `{row.lang}` | genre `{row.genre}` | needed `{int(row.needed):,}` | selected `{int(row.selected):,}` | deficit `{int(row.deficit):,}`"
            )
    stage_lines.extend(
        [
            "",
            "## Merge deduplication examples",
            "- Interpretation: these metrics show how many documents and authors were removed when each phase was deduplicated before final merge.",
            "- Method: counts come from `merge_summary.json`; author changes are exact for the merge stage.",
        ]
    )
    for row in merge_dedup_summary.itertuples(index=False):
        stage_lines.append(
            f"- `{row.phase}` | dropped_docs `{row.dropped_total}` | authors_before `{row.authors_before}` | authors_after `{row.authors_after}` | reasons `{row.drop_reasons}`"
        )
    stage_lines.extend(
        [
            "",
            "## Cross-phase exact overlap examples",
            "- Interpretation: these are exact duplicate texts that appeared across phase boundaries; one copy was removed to keep the combined benchmark non-redundant.",
            "- Method: examples are matched by exact normalized-content hash across persisted phase outputs.",
        ]
    )
    if cross_phase_examples.empty:
        stage_lines.append("- No cross-phase exact-overlap examples were recovered.")
    else:
        for row in cross_phase_examples.head(6).itertuples(index=False):
            stage_lines.extend(
                [
                    f"- dropped `{row.dropped_phase}` doc `{row.dropped_doc_id}` vs kept `{row.kept_phase}` doc `{row.kept_doc_id}`",
                    f"  dropped: {row.dropped_snippet}",
                    f"  kept: {row.kept_snippet}",
                ]
            )
    write_text("\n".join(stage_lines), output_reports / "stage_examples_report.md")


def resolve_phase_dirs(
    dataset_dir: Path,
    *,
    phase1_dir: Path | None,
    phase2_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    if phase1_dir is not None and phase2_dir is not None:
        return phase1_dir, phase2_dir

    merge_summary = read_json(dataset_dir / "merge_summary.json")
    inputs = merge_summary.get("inputs", {}) if merge_summary else {}
    resolved_phase1 = phase1_dir or (Path(inputs["phase1_dir"]) if inputs.get("phase1_dir") else None)
    resolved_phase2 = phase2_dir or (Path(inputs["phase2_dir"]) if inputs.get("phase2_dir") else None)
    return resolved_phase1, resolved_phase2


def run(
    dataset_dir: Path,
    output_dir: Path,
    splits: Sequence[str],
    *,
    phase1_dir: Path | None,
    phase2_dir: Path | None,
) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"

    docs = load_dataset_docs(dataset_dir, splits)
    summary_overview(docs, tables_dir)
    distribution_tables(docs, tables_dir)
    unique_author_tables(docs, tables_dir)
    docs_per_author_distributions(docs, tables_dir)
    author_balance_tables(docs, tables_dir)

    language_pie = plot_pie_chart(
        docs["lang"].value_counts(),
        figures_dir / "language_distribution_pie.png",
        "Language distribution",
        max_slices=12,
    )
    save_csv(language_pie, tables_dir / "language_distribution_pie_breakdown.csv")

    length_pie = plot_pie_chart(
        docs["token_length_bucket"].fillna("unknown").value_counts(),
        figures_dir / "document_length_distribution_pie.png",
        "Document length distribution",
        max_slices=8,
    )
    save_csv(length_pie, tables_dir / "document_length_distribution_pie_breakdown.csv")

    source_pie = plot_pie_chart(
        docs["source"].value_counts(),
        figures_dir / "source_distribution_pie.png",
        "Source distribution",
        max_slices=12,
    )
    save_csv(source_pie, tables_dir / "source_distribution_pie_breakdown.csv")

    genre_nested = plot_nested_genre_donut(
        docs,
        figures_dir / "primary_genre_subgenre_nested_donut.png",
    )
    save_csv(genre_nested, tables_dir / "primary_genre_subgenre_nested_donut_breakdown.csv")

    author_balance_dist = plot_author_balance_histogram(
        docs,
        figures_dir / "author_balance_histogram.png",
    )
    save_csv(author_balance_dist, tables_dir / "author_balance_histogram_breakdown.csv")

    candidate_docs = docs[docs["doc_type"] == "candidate"].dropna(subset=["author_id"]).copy()
    span_summary = candidate_author_metadata_span(candidate_docs, tables_dir)
    alignment_summary = positive_pair_metadata_alignment(dataset_dir, splits, docs, tables_dir)
    _, cell_summary = metadata_cell_risk(candidate_docs, tables_dir)
    topic_leakage_report(
        reports_dir / "topic_leakage_report.md",
        alignment_summary=alignment_summary,
        span_summary=span_summary,
        cell_summary=cell_summary,
    )

    build_stage_tables_and_figures(
        dataset_dir,
        docs,
        tables_dir,
        figures_dir,
        reports_dir,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supplementary authorship benchmark analysis and visualization exports."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("processing/outputs/combined_phase1_phase2"),
        help="Benchmark root containing split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_analysis/outputs/combined_phase1_phase2/benchmark_profile"),
        help="Output directory for supplementary benchmark analysis artifacts.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to include. Defaults to all discovered splits.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=None,
        help="Optional explicit phase1 benchmark root used for stage diagnostics.",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=None,
        help="Optional explicit phase2 benchmark root used for stage diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = discover_splits(args.dataset_dir, args.splits)
    phase1_dir, phase2_dir = resolve_phase_dirs(
        args.dataset_dir,
        phase1_dir=args.phase1_dir,
        phase2_dir=args.phase2_dir,
    )
    run(
        args.dataset_dir,
        args.output_dir,
        splits,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
    )


if __name__ == "__main__":
    main()
