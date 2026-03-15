#!/usr/bin/env python3
"""Run side-by-side dataset analysis for two AuthBench benchmark roots."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import analyze_dataset as stats_analysis
from . import qualitative_analysis as qual_analysis

sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    dataset_dir: Path
    splits: Sequence[str]


def _resolve_splits(dataset_dir: Path, requested: Sequence[str] | None) -> list[str]:
    if requested is None or "all" in requested:
        splits = stats_analysis.discover_splits(dataset_dir)
    else:
        splits = list(requested)
    if not splits:
        raise FileNotFoundError(f"No valid splits found in {dataset_dir}.")
    return splits


def _load_docs_and_ground_truth(
    dataset_dir: Path,
    splits: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = stats_analysis.load_documents(dataset_dir, splits)
    ground_truth = stats_analysis.load_ground_truth(dataset_dir, splits)
    docs = stats_analysis.fill_query_authors(docs, ground_truth)
    docs["token_length_bucket"] = docs["token_length"].apply(stats_analysis.token_length_bucket)
    return docs, ground_truth


def _docs_per_author(author_docs: pd.DataFrame) -> pd.Series:
    if author_docs.empty:
        return pd.Series(dtype=float)
    return author_docs.groupby("author_id").size()


def _safe_float(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def benchmark_summary(
    benchmark: BenchmarkConfig,
    docs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, object]:
    author_docs = docs.dropna(subset=["author_id"]).copy()
    docs_per_author = _docs_per_author(author_docs)
    query_count = int((docs["doc_type"] == "query").sum())
    candidate_count = int((docs["doc_type"] == "candidate").sum())

    summary: dict[str, object] = {
        "benchmark": benchmark.name,
        "dataset_dir": str(benchmark.dataset_dir),
        "splits": ",".join(benchmark.splits),
        "docs_total": int(len(docs)),
        "queries": query_count,
        "candidates": candidate_count,
        "unique_authors": int(author_docs["author_id"].nunique()),
        "languages": int(docs["lang"].dropna().nunique()),
        "fine_grained_genres": int(docs["genre"].dropna().nunique()),
        "primary_genres": int(docs["primary_genre"].dropna().nunique()),
        "sources": int(docs["source"].dropna().nunique()),
        "avg_token_length": _safe_float(docs["token_length"].mean()),
        "median_token_length": _safe_float(docs["token_length"].median()),
        "p90_token_length": _safe_float(docs["token_length"].quantile(0.90)),
        "p95_token_length": _safe_float(docs["token_length"].quantile(0.95)),
        "max_token_length": _safe_float(docs["token_length"].max()),
        "docs_per_author_avg": _safe_float(docs_per_author.mean()),
        "docs_per_author_median": _safe_float(docs_per_author.median()),
        "docs_per_author_p90": _safe_float(docs_per_author.quantile(0.90)),
        "docs_per_author_p95": _safe_float(docs_per_author.quantile(0.95)),
        "docs_per_author_max": _safe_float(docs_per_author.max()),
        "positive_pairs": int(len(ground_truth)),
        "avg_positive_candidates_per_query": _safe_float(
            ground_truth.groupby("query_id").size().mean() if not ground_truth.empty else None
        ),
        "median_positive_candidates_per_query": _safe_float(
            ground_truth.groupby("query_id").size().median() if not ground_truth.empty else None
        ),
    }

    if not ground_truth.empty:
        queries = (
            docs[docs["doc_type"] == "query"][
                ["doc_id", "lang", "primary_genre", "token_length"]
            ]
            .rename(
                columns={
                    "doc_id": "query_id",
                    "lang": "lang_query",
                    "primary_genre": "primary_genre_query",
                    "token_length": "token_length_query",
                }
            )
            .drop_duplicates(subset=["query_id"])
        )
        candidates = (
            docs[docs["doc_type"] == "candidate"][
                ["doc_id", "lang", "primary_genre", "token_length"]
            ]
            .rename(
                columns={
                    "doc_id": "positive_id",
                    "lang": "lang_candidate",
                    "primary_genre": "primary_genre_candidate",
                    "token_length": "token_length_candidate",
                }
            )
            .drop_duplicates(subset=["positive_id"])
        )
        aligned = ground_truth.merge(queries, on="query_id", how="left").merge(
            candidates,
            on="positive_id",
            how="left",
        )
        summary["pct_same_language_positive_pairs"] = _safe_float(
            (aligned["lang_query"] == aligned["lang_candidate"]).mean() * 100.0
        )
        summary["pct_same_primary_genre_positive_pairs"] = _safe_float(
            (aligned["primary_genre_query"] == aligned["primary_genre_candidate"]).mean() * 100.0
        )
        summary["avg_positive_token_length_delta"] = _safe_float(
            (aligned["token_length_query"] - aligned["token_length_candidate"]).mean()
        )
    else:
        summary["pct_same_language_positive_pairs"] = None
        summary["pct_same_primary_genre_positive_pairs"] = None
        summary["avg_positive_token_length_delta"] = None

    return summary


def distribution_table(
    docs_map: Dict[str, pd.DataFrame],
    field: str,
    normalize_within_benchmark: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for benchmark_name, docs in docs_map.items():
        counts = (
            docs.groupby(field)
            .size()
            .reset_index(name="docs")
            .rename(columns={field: "bucket"})
            .sort_values("docs", ascending=False)
        )
        counts["benchmark"] = benchmark_name
        if normalize_within_benchmark:
            counts["pct_docs"] = counts["docs"] / counts["docs"].sum() * 100.0
        frames.append(counts)
    if not frames:
        return pd.DataFrame(columns=["benchmark", "bucket", "docs", "pct_docs"])
    return pd.concat(frames, ignore_index=True)


def split_type_table(docs_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for benchmark_name, docs in docs_map.items():
        counts = (
            docs.groupby(["split", "doc_type"])
            .size()
            .reset_index(name="docs")
            .sort_values(["split", "doc_type"])
        )
        counts["benchmark"] = benchmark_name
        counts["pct_docs"] = counts["docs"] / counts["docs"].sum() * 100.0
        frames.append(counts)
    if not frames:
        return pd.DataFrame(columns=["benchmark", "split", "doc_type", "docs", "pct_docs"])
    return pd.concat(frames, ignore_index=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_grouped_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    order: Iterable[str] | None = None,
    rotate_xticks: int = 0,
    horizontal: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    if horizontal:
        sns.barplot(data=df, y=x, x=y, hue=hue, order=list(order) if order is not None else None)
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)
    else:
        sns.barplot(data=df, x=x, y=y, hue=hue, order=list(order) if order is not None else None)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.title(title)
    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_split_type(split_type_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = split_type_df.copy()
    plot_df["split_doc_type"] = plot_df["split"] + ":" + plot_df["doc_type"]
    order = (
        plot_df.groupby("split_doc_type")["docs"].sum().sort_values(ascending=False).index.tolist()
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="split_doc_type", y="docs", hue="benchmark", order=order)
    plt.xlabel("Split and document type")
    plt.ylabel("Document count")
    plt.title("Split/type composition by benchmark")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_report(
    output_dir: Path,
    benchmark_names: Sequence[str],
    summary_df: pd.DataFrame,
    qualitative_summaries: dict[str, dict[str, dict[str, object]]],
) -> None:
    summary_lookup = summary_df.set_index("benchmark").to_dict(orient="index")
    lines = [
        "# Benchmark Comparison",
        "",
        "This report compares the two benchmark roots using the shared `post_analysis`",
        "statistics and qualitative pipelines, orchestrated from `post_analysis/analyze_benchmarks.py`.",
        "",
    ]

    for benchmark_name in benchmark_names:
        stats = summary_lookup.get(benchmark_name, {})
        qualitative = qualitative_summaries.get(benchmark_name, {})
        basic = qualitative.get("basic", {})
        entropy = qualitative.get("entropy", {})
        split = qualitative.get("split", {})
        duplicates = qualitative.get("duplicates", {})
        topic_pairs = qualitative.get("topic_pairs", {})

        lines.extend(
            [
                f"## {benchmark_name}",
                f"- docs_total: `{stats.get('docs_total', 0)}`",
                f"- queries: `{stats.get('queries', 0)}` | candidates: `{stats.get('candidates', 0)}`",
                f"- unique_authors: `{stats.get('unique_authors', 0)}`",
                f"- languages: `{stats.get('languages', 0)}` | primary_genres: `{stats.get('primary_genres', 0)}` | sources: `{stats.get('sources', 0)}`",
                f"- avg_token_length: `{stats.get('avg_token_length', 0):.2f}` | median_token_length: `{stats.get('median_token_length', 0):.2f}`",
                f"- docs_per_author_avg: `{stats.get('docs_per_author_avg', 0):.2f}` | docs_per_author_p95: `{stats.get('docs_per_author_p95', 0):.2f}`",
                f"- pct_same_language_positive_pairs: `{stats.get('pct_same_language_positive_pairs', 0):.2f}`",
                f"- avg_normalized_genre_entropy: `{entropy.get('avg_normalized_genre_entropy', 0):.4f}`",
                f"- pct_authors_multi_genre: `{entropy.get('pct_authors_multi_genre', 0):.2f}`",
                f"- pct_authors_multi_split: `{split.get('pct_authors_multi_split', 0):.2f}`",
                f"- duplicate_groups: `{duplicates.get('duplicate_groups', 0)}` | cross_split_groups: `{duplicates.get('cross_split_groups', 0)}`",
                f"- topic_pairs_total: `{topic_pairs.get('pairs_exported', 0)}`",
                "",
            ]
        )

        if basic:
            lines.append(
                f"  Qualitative summary file: `per_benchmark/{benchmark_name}/qualitative/csv/basic/summary_overview.json`"
            )
            lines.append("")

    if len(benchmark_names) == 2:
        left = summary_lookup.get(benchmark_names[0], {})
        right = summary_lookup.get(benchmark_names[1], {})
        lines.extend(
            [
                "## Headline Differences",
                f"- `{benchmark_names[0]}` has `{left.get('docs_total', 0)}` docs versus `{right.get('docs_total', 0)}` for `{benchmark_names[1]}`.",
                f"- `{benchmark_names[0]}` spans `{left.get('languages', 0)}` languages versus `{right.get('languages', 0)}`.",
                f"- `{benchmark_names[0]}` has `{left.get('unique_authors', 0)}` authors versus `{right.get('unique_authors', 0)}`.",
                f"- Average token length is `{left.get('avg_token_length', 0):.2f}` vs `{right.get('avg_token_length', 0):.2f}`.",
                f"- Same-language positive-pair rate is `{left.get('pct_same_language_positive_pairs', 0):.2f}%` vs `{right.get('pct_same_language_positive_pairs', 0):.2f}%`.",
                "",
            ]
        )

    report_path = output_dir / "benchmark_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    benchmarks: Sequence[BenchmarkConfig],
    output_dir: Path,
) -> None:
    comparison_dir = output_dir / "comparison"
    tables_dir = comparison_dir / "tables"
    figures_dir = comparison_dir / "figures"

    docs_map: Dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    qualitative_summaries: dict[str, dict[str, dict[str, object]]] = {}

    for benchmark in benchmarks:
        docs, ground_truth = _load_docs_and_ground_truth(benchmark.dataset_dir, benchmark.splits)
        docs_map[benchmark.name] = docs
        summary_rows.append(benchmark_summary(benchmark, docs, ground_truth))

        benchmark_output = output_dir / "per_benchmark" / benchmark.name / "qualitative" / "csv"
        qualitative_summaries[benchmark.name] = {
            "basic": _load_json(benchmark_output / "basic" / "summary_overview.json"),
            "entropy": _load_json(benchmark_output / "entropy" / "author_genre_entropy_summary.json"),
            "split": _load_json(benchmark_output / "split_leakage" / "split_leakage_summary.json"),
            "duplicates": _load_json(benchmark_output / "duplicates" / "exact_duplicate_summary.json"),
            "topic_pairs": _load_json(benchmark_output / "topic_pairs" / "topic_controlled_pairs_stats.json"),
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("benchmark")
    _save_csv(summary_df, tables_dir / "benchmark_summary.csv")

    language_df = distribution_table(docs_map, "lang")
    _save_csv(language_df, tables_dir / "language_distribution.csv")

    genre_df = distribution_table(docs_map, "primary_genre")
    _save_csv(genre_df, tables_dir / "primary_genre_distribution.csv")

    length_df = distribution_table(docs_map, "token_length_bucket")
    _save_csv(length_df, tables_dir / "token_length_bucket_distribution.csv")

    source_df = distribution_table(docs_map, "source")
    _save_csv(source_df, tables_dir / "source_distribution.csv")

    split_type_df = split_type_table(docs_map)
    _save_csv(split_type_df, tables_dir / "split_doc_type_distribution.csv")

    order = language_df.groupby("bucket")["docs"].sum().sort_values(ascending=False).index.tolist()
    _plot_grouped_bar(
        language_df,
        x="bucket",
        y="docs",
        hue="benchmark",
        title="Language distribution by benchmark",
        xlabel="Language",
        ylabel="Document count",
        output_path=figures_dir / "language_distribution_comparison.png",
        order=order,
    )

    genre_order = genre_df.groupby("bucket")["docs"].sum().sort_values(ascending=False).head(12).index.tolist()
    _plot_grouped_bar(
        genre_df[genre_df["bucket"].isin(genre_order)],
        x="bucket",
        y="docs",
        hue="benchmark",
        title="Top primary genres by benchmark",
        xlabel="Primary genre",
        ylabel="Document count",
        output_path=figures_dir / "primary_genre_distribution_comparison.png",
        order=genre_order,
        rotate_xticks=25,
    )

    length_order = ["short", "medium", "long", "extra_long"]
    _plot_grouped_bar(
        length_df,
        x="bucket",
        y="docs",
        hue="benchmark",
        title="Token-length bucket distribution by benchmark",
        xlabel="Token-length bucket",
        ylabel="Document count",
        output_path=figures_dir / "token_length_bucket_distribution_comparison.png",
        order=length_order,
    )

    source_order = source_df.groupby("bucket")["docs"].sum().sort_values(ascending=False).head(12).index.tolist()
    _plot_grouped_bar(
        source_df[source_df["bucket"].isin(source_order)],
        x="bucket",
        y="docs",
        hue="benchmark",
        title="Top sources by benchmark",
        xlabel="Source",
        ylabel="Document count",
        output_path=figures_dir / "source_distribution_comparison.png",
        order=source_order,
        rotate_xticks=25,
    )

    _plot_split_type(split_type_df, figures_dir / "split_doc_type_distribution_comparison.png")

    summary_plot_df = summary_df[
        [
            "benchmark",
            "docs_total",
            "unique_authors",
            "avg_token_length",
            "avg_positive_candidates_per_query",
        ]
    ].melt(id_vars="benchmark", var_name="metric", value_name="value")
    _save_csv(summary_plot_df, tables_dir / "summary_metrics_long.csv")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_plot_df, x="metric", y="value", hue="benchmark")
    plt.xlabel("Summary metric")
    plt.ylabel("Value")
    plt.title("Benchmark-level summary metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "benchmark_summary_metrics.png", dpi=300)
    plt.close()

    _comparison_report(
        comparison_dir,
        benchmark_names=[benchmark.name for benchmark in benchmarks],
        summary_df=summary_df,
        qualitative_summaries=qualitative_summaries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistics, qualitative analysis, and comparison plots for two benchmark roots."
    )
    parser.add_argument(
        "--dataset-a-dir",
        type=Path,
        default=Path("processing/outputs/pipeline_phase1_official"),
        help="First benchmark root.",
    )
    parser.add_argument(
        "--dataset-b-dir",
        type=Path,
        default=Path("processing/second_phase_web_crawling/outputs/pipeline_phase2_official"),
        help="Second benchmark root.",
    )
    parser.add_argument(
        "--dataset-a-name",
        default="pipeline_phase1_official",
        help="Label for the first benchmark.",
    )
    parser.add_argument(
        "--dataset-b-name",
        default="pipeline_phase2_official",
        help="Label for the second benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_analysis/outputs/phase1_vs_phase2"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to include for both datasets. Defaults to all detected splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for qualitative analysis sampling.",
    )
    parser.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Skip per-benchmark statistical analysis outputs.",
    )
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help="Skip per-benchmark qualitative analysis outputs.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the side-by-side comparison outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = [
        BenchmarkConfig(
            name=args.dataset_a_name,
            dataset_dir=args.dataset_a_dir,
            splits=_resolve_splits(args.dataset_a_dir, args.splits),
        ),
        BenchmarkConfig(
            name=args.dataset_b_name,
            dataset_dir=args.dataset_b_dir,
            splits=_resolve_splits(args.dataset_b_dir, args.splits),
        ),
    ]

    for benchmark in benchmarks:
        benchmark_root = args.output_dir / "per_benchmark" / benchmark.name
        if not args.skip_statistics:
            stats_analysis.run_analysis(
                benchmark.dataset_dir,
                benchmark_root / "statistics",
                benchmark.splits,
            )
        if not args.skip_qualitative:
            qual_analysis.run(
                benchmark.dataset_dir,
                benchmark_root / "qualitative",
                benchmark.splits,
                seed=args.seed,
            )

    if not args.skip_comparison:
        run_comparison(benchmarks, args.output_dir)


if __name__ == "__main__":
    main()
