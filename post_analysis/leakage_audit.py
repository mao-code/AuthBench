#!/usr/bin/env python3
"""Audit topic and language shortcut risk for an AuthBench benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from . import analyze_dataset as stats_analysis
from . import qualitative_analysis as qual_analysis

DEFAULT_SPLITS: Sequence[str] = ("test",)
UNKNOWN = "unknown"

SCHEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("all_authors", ()),
    ("language_only", ("lang",)),
    ("primary_genre_only", ("primary_genre",)),
    ("fine_genre_only", ("genre",)),
    ("language_primary_genre", ("lang", "primary_genre")),
    ("language_fine_genre", ("lang", "genre")),
    ("full_metadata", ("lang", "genre", "source", "token_length_bucket")),
)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def harmonic_mean_rank_expectation(pool_size: int) -> float:
    if pool_size <= 0:
        return 0.0
    return sum(1.0 / rank for rank in range(1, pool_size + 1)) / float(pool_size)


def discover_splits(dataset_dir: Path, requested: Sequence[str]) -> list[str]:
    if not requested or "all" in requested:
        splits = stats_analysis.discover_splits(dataset_dir)
    else:
        splits = list(requested)
    if not splits:
        raise FileNotFoundError(f"No valid split directories found in {dataset_dir}.")
    return splits


def load_eval_frames(dataset_dir: Path, splits: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    docs = qual_analysis.load_docs(dataset_dir, splits)
    ground_truth = stats_analysis.load_ground_truth(dataset_dir, splits)
    docs = stats_analysis.fill_query_authors(docs, ground_truth)
    docs["token_length_bucket"] = docs["token_length"].apply(stats_analysis.token_length_bucket)
    for column in ("lang", "genre", "primary_genre", "source", "token_length_bucket"):
        docs[column] = docs[column].fillna(UNKNOWN)

    queries = (
        docs[docs["doc_type"] == "query"][
            [
                "doc_id",
                "split",
                "author_id",
                "lang",
                "genre",
                "primary_genre",
                "source",
                "token_length_bucket",
                "content",
            ]
        ]
        .rename(columns={"doc_id": "query_id", "author_id": "query_author_id"})
        .drop_duplicates(subset=["query_id"])
    )
    candidates = (
        docs[docs["doc_type"] == "candidate"][
            [
                "doc_id",
                "split",
                "author_id",
                "lang",
                "genre",
                "primary_genre",
                "source",
                "token_length_bucket",
                "content",
            ]
        ]
        .rename(columns={"doc_id": "candidate_id"})
        .drop_duplicates(subset=["candidate_id"])
    )
    if queries.empty or candidates.empty or ground_truth.empty:
        raise ValueError("Leakage audit requires non-empty queries, candidates, and ground truth.")
    return queries, candidates, ground_truth


def positive_pair_alignment(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    aligned = (
        ground_truth.merge(queries, on="query_id", how="left")
        .merge(
            candidates.rename(
                columns={
                    "candidate_id": "positive_id",
                    "author_id": "candidate_author_id",
                    "lang": "candidate_lang",
                    "genre": "candidate_genre",
                    "primary_genre": "candidate_primary_genre",
                    "source": "candidate_source",
                    "token_length_bucket": "candidate_length_bucket",
                }
            )[
                [
                    "positive_id",
                    "candidate_author_id",
                    "candidate_lang",
                    "candidate_genre",
                    "candidate_primary_genre",
                    "candidate_source",
                    "candidate_length_bucket",
                ]
            ],
            on="positive_id",
            how="left",
        )
    )
    comparisons = {
        "same_language": aligned["lang"] == aligned["candidate_lang"],
        "same_primary_genre": aligned["primary_genre"] == aligned["candidate_primary_genre"],
        "same_fine_genre": aligned["genre"] == aligned["candidate_genre"],
        "same_source": aligned["source"] == aligned["candidate_source"],
        "same_length_bucket": aligned["token_length_bucket"] == aligned["candidate_length_bucket"],
    }
    rows: list[dict[str, object]] = []
    summary: dict[str, float] = {"pairs_total": float(len(aligned))}
    for metric, values in comparisons.items():
        pct = float(values.mean() * 100.0) if len(values) else 0.0
        rows.append(
            {
                "metric": metric,
                "pairs": int(len(aligned)),
                "matches": int(values.sum()),
                "pct_matches": pct,
            }
        )
        summary[metric] = pct
    return pd.DataFrame(rows).sort_values("pct_matches", ascending=False), summary


def author_metadata_span(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    span = (
        candidates.groupby("author_id")
        .agg(
            candidate_docs=("candidate_id", "count"),
            candidate_languages=("lang", "nunique"),
            candidate_primary_genres=("primary_genre", "nunique"),
            candidate_fine_genres=("genre", "nunique"),
            candidate_sources=("source", "nunique"),
            candidate_length_buckets=("token_length_bucket", "nunique"),
        )
        .reset_index()
        .sort_values(["candidate_docs", "candidate_primary_genres"], ascending=[False, False])
    )
    summary = {
        "authors_total": float(len(span)),
        "pct_single_language_authors": float((span["candidate_languages"] == 1).mean() * 100.0),
        "pct_single_primary_genre_authors": float((span["candidate_primary_genres"] == 1).mean() * 100.0),
        "pct_single_fine_genre_authors": float((span["candidate_fine_genres"] == 1).mean() * 100.0),
        "pct_single_source_authors": float((span["candidate_sources"] == 1).mean() * 100.0),
    }
    return span, summary


def build_author_lookup(
    candidates: pd.DataFrame,
    fields: Sequence[str],
) -> tuple[dict[tuple[object, ...], list[str]], dict[tuple[object, ...], dict[str, int]]]:
    totals = candidates.groupby("author_id").size().rename("total_candidate_docs")
    if fields:
        grouped = (
            candidates.groupby(list(fields) + ["author_id"])
            .size()
            .reset_index(name="match_docs")
            .merge(totals.reset_index(), on="author_id", how="left")
        )
    else:
        grouped = totals.reset_index()
        grouped["match_docs"] = grouped["total_candidate_docs"]
        for field in fields:
            grouped[field] = None

    if fields:
        grouped = grouped.sort_values(
            list(fields) + ["match_docs", "total_candidate_docs", "author_id"],
            ascending=[True] * len(fields) + [False, False, True],
        )
    else:
        grouped = grouped.sort_values(
            ["match_docs", "total_candidate_docs", "author_id"],
            ascending=[False, False, True],
        )

    author_lists: dict[tuple[object, ...], list[str]] = {}
    author_ranks: dict[tuple[object, ...], dict[str, int]] = {}
    if fields:
        for values, frame in grouped.groupby(list(fields), dropna=False, sort=False):
            key = values if isinstance(values, tuple) else (values,)
            authors = frame["author_id"].tolist()
            author_lists[key] = authors
            author_ranks[key] = {author_id: idx + 1 for idx, author_id in enumerate(authors)}
    else:
        authors = grouped["author_id"].tolist()
        author_lists[tuple()] = authors
        author_ranks[tuple()] = {author_id: idx + 1 for idx, author_id in enumerate(authors)}
    return author_lists, author_ranks


def evaluate_shortcut_schemes(
    queries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    candidates: pd.DataFrame,
    schemes: Sequence[tuple[str, Sequence[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_rows = (
        ground_truth.rename(columns={"split": "ground_truth_split", "author_id": "true_author_id"})
        .merge(queries, on="query_id", how="left")
        .copy()
    )
    per_query_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    by_lang_rows: list[dict[str, object]] = []

    for scheme_name, fields in schemes:
        author_lists, author_ranks = build_author_lookup(candidates, fields)
        rows: list[dict[str, object]] = []
        for row in query_rows.itertuples(index=False):
            key = tuple(getattr(row, field) for field in fields)
            authors = author_lists.get(key, [])
            rank_map = author_ranks.get(key, {})
            pool_size = len(authors)
            in_pool = row.true_author_id in rank_map
            rank = rank_map.get(row.true_author_id)
            rows.append(
                {
                    "split": row.ground_truth_split,
                    "query_id": row.query_id,
                    "query_author_id": row.true_author_id,
                    "scheme": scheme_name,
                    "pool_size": int(pool_size),
                    "true_author_in_pool": bool(in_pool),
                    "true_author_rank": int(rank) if rank is not None else None,
                    "expected_success@1": (1.0 / pool_size) if in_pool and pool_size else 0.0,
                    "expected_success@5": min(5.0 / pool_size, 1.0) if in_pool and pool_size else 0.0,
                    "expected_mrr": harmonic_mean_rank_expectation(pool_size) if in_pool and pool_size else 0.0,
                    "lang": row.lang,
                    "primary_genre": row.primary_genre,
                    "genre": row.genre,
                    "source": row.source,
                    "token_length_bucket": row.token_length_bucket,
                }
            )

        per_query = pd.DataFrame(rows)
        per_query_frames.append(per_query)
        applicable = per_query["pool_size"] > 0
        covered = per_query["true_author_in_pool"]
        ranks = pd.to_numeric(per_query["true_author_rank"], errors="coerce")

        summary_rows.append(
            {
                "scheme": scheme_name,
                "queries": int(len(per_query)),
                "queries_with_nonempty_pool": int(applicable.sum()),
                "pct_queries_with_nonempty_pool": float(applicable.mean() * 100.0),
                "pct_true_author_covered": float(covered.mean() * 100.0),
                "pool_size_mean": float(per_query["pool_size"].mean()),
                "pool_size_median": float(per_query["pool_size"].median()),
                "pool_size_p90": float(per_query["pool_size"].quantile(0.90)),
                "pool_size_p95": float(per_query["pool_size"].quantile(0.95)),
                "pct_pool_size_le_1": float((per_query["pool_size"] <= 1).mean() * 100.0),
                "pct_pool_size_le_5": float((per_query["pool_size"] <= 5).mean() * 100.0),
                "pct_pool_size_le_10": float((per_query["pool_size"] <= 10).mean() * 100.0),
                "expected_success@1": float(per_query["expected_success@1"].mean()),
                "expected_success@5": float(per_query["expected_success@5"].mean()),
                "expected_mrr": float(per_query["expected_mrr"].mean()),
                "deterministic_success@1": float((ranks == 1).mean()),
                "deterministic_success@5": float((ranks <= 5).mean()),
                "deterministic_mrr": float((1.0 / ranks).fillna(0.0).mean()),
            }
        )

        for lang, group in per_query.groupby("lang", dropna=False):
            lang_ranks = pd.to_numeric(group["true_author_rank"], errors="coerce")
            by_lang_rows.append(
                {
                    "scheme": scheme_name,
                    "lang": lang,
                    "queries": int(len(group)),
                    "pct_true_author_covered": float(group["true_author_in_pool"].mean() * 100.0),
                    "pool_size_median": float(group["pool_size"].median()),
                    "pool_size_p90": float(group["pool_size"].quantile(0.90)),
                    "expected_success@1": float(group["expected_success@1"].mean()),
                    "expected_success@5": float(group["expected_success@5"].mean()),
                    "deterministic_success@1": float((lang_ranks == 1).mean()),
                    "deterministic_success@5": float((lang_ranks <= 5).mean()),
                }
            )

    return (
        pd.concat(per_query_frames, ignore_index=True),
        pd.DataFrame(summary_rows).sort_values("expected_success@1", ascending=False),
        pd.DataFrame(by_lang_rows).sort_values(["scheme", "queries"], ascending=[True, False]),
    )


def assign_topic_clusters(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    min_docs_per_stratum: int,
    docs_per_cluster: int,
    max_clusters: int,
    max_features: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [
            queries.assign(doc_type="query").rename(columns={"query_id": "doc_id"}),
            candidates.assign(doc_type="candidate").rename(columns={"candidate_id": "doc_id"}),
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    stratum_rows: list[dict[str, object]] = []

    for (lang, primary_genre), group in combined.groupby(["lang", "primary_genre"], dropna=False):
        texts = group["content"].fillna("").astype(str)
        usable = texts.str.len() > 0
        group = group.loc[usable].copy()
        texts = texts.loc[usable]
        if len(group) < min_docs_per_stratum:
            stratum_rows.append(
                {
                    "lang": lang,
                    "primary_genre": primary_genre,
                    "docs": int(len(group)),
                    "clusters": 0,
                    "status": "too_small",
                }
            )
            continue

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=max_features,
        )
        try:
            matrix = vectorizer.fit_transform(texts.tolist())
        except ValueError:
            stratum_rows.append(
                {
                    "lang": lang,
                    "primary_genre": primary_genre,
                    "docs": int(len(group)),
                    "clusters": 0,
                    "status": "vectorizer_failed",
                }
            )
            continue
        if matrix.shape[1] < 2:
            stratum_rows.append(
                {
                    "lang": lang,
                    "primary_genre": primary_genre,
                    "docs": int(len(group)),
                    "clusters": 0,
                    "status": "not_enough_features",
                }
            )
            continue

        size_cap = max(3, len(group) // max(docs_per_cluster, 1))
        heuristic_clusters = max(3, int(round(math.sqrt(len(group)) / 2.0)))
        clusters = max(3, min(max_clusters, heuristic_clusters, size_cap))
        clusters = min(clusters, len(group))
        if clusters < 2:
            stratum_rows.append(
                {
                    "lang": lang,
                    "primary_genre": primary_genre,
                    "docs": int(len(group)),
                    "clusters": 0,
                    "status": "too_small_after_cap",
                }
            )
            continue

        model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=seed,
            batch_size=min(4096, max(len(group), 256)),
            n_init=5,
        )
        labels = model.fit_predict(matrix)
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        group = group.copy()
        group["topic_cluster"] = labels
        for row in group[["doc_id", "doc_type", "lang", "primary_genre", "topic_cluster"]].itertuples(index=False):
            rows.append(
                {
                    "doc_id": row.doc_id,
                    "doc_type": row.doc_type,
                    "lang": row.lang,
                    "primary_genre": row.primary_genre,
                    "topic_cluster": int(row.topic_cluster),
                    "topic_cluster_size": int(cluster_sizes.get(row.topic_cluster, 0)),
                }
            )
        stratum_rows.append(
            {
                "lang": lang,
                "primary_genre": primary_genre,
                "docs": int(len(group)),
                "clusters": int(clusters),
                "min_cluster_docs": int(min(cluster_sizes.values())),
                "median_cluster_docs": float(pd.Series(cluster_sizes).median()),
                "max_cluster_docs": int(max(cluster_sizes.values())),
                "status": "clustered",
            }
        )

    assignments = pd.DataFrame(rows)
    strata = pd.DataFrame(stratum_rows).sort_values(["status", "docs"], ascending=[True, False])
    return assignments, strata


def evaluate_topic_shortcuts(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if assignments.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    query_topics = assignments[assignments["doc_type"] == "query"][
        ["doc_id", "topic_cluster", "topic_cluster_size"]
    ].rename(columns={"doc_id": "query_id", "topic_cluster_size": "query_topic_cluster_size"})
    candidate_topics = assignments[assignments["doc_type"] == "candidate"][
        ["doc_id", "topic_cluster", "topic_cluster_size"]
    ].rename(
        columns={
            "doc_id": "candidate_id",
            "topic_cluster_size": "candidate_topic_cluster_size",
        }
    )
    queries_with_topics = queries.merge(query_topics, on="query_id", how="left")
    candidates_with_topics = candidates.merge(candidate_topics, on="candidate_id", how="left")

    covered_queries = queries_with_topics["topic_cluster"].notna().sum()
    aligned = (
        ground_truth.merge(queries_with_topics, on="query_id", how="left")
        .merge(
            candidates_with_topics.rename(
                columns={
                    "candidate_id": "positive_id",
                    "topic_cluster": "positive_topic_cluster",
                }
            )[["positive_id", "positive_topic_cluster"]],
            on="positive_id",
            how="left",
        )
    )
    aligned["same_topic_cluster"] = aligned["topic_cluster"] == aligned["positive_topic_cluster"]
    topic_alignment = (
        aligned.groupby("lang", dropna=False)
        .agg(
            pairs=("query_id", "count"),
            pct_same_topic_cluster=("same_topic_cluster", lambda x: x.mean() * 100.0),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    topic_summary = {
        "queries_total": int(len(queries)),
        "queries_with_topic_cluster": int(covered_queries),
        "pct_queries_with_topic_cluster": float(covered_queries / max(len(queries), 1) * 100.0),
        "pairs_total": int(len(aligned)),
        "pct_same_topic_cluster_positive_pairs": float(aligned["same_topic_cluster"].mean() * 100.0),
    }

    scheme_df, scheme_summary, _ = evaluate_shortcut_schemes(
        queries_with_topics.dropna(subset=["topic_cluster"]).copy(),
        ground_truth.merge(
            queries_with_topics[["query_id", "topic_cluster"]],
            on="query_id",
            how="left",
        ).dropna(subset=["topic_cluster"])[["split", "query_id", "positive_id", "author_id"]],
        candidates_with_topics.dropna(subset=["topic_cluster"]).copy(),
        [("language_primary_genre_topic_cluster", ("lang", "primary_genre", "topic_cluster"))],
    )
    return scheme_df, topic_alignment, {**topic_summary, **scheme_summary.iloc[0].to_dict()}


def build_report(
    report_path: Path,
    *,
    dataset_dir: Path,
    splits: Sequence[str],
    alignment_summary: dict[str, float],
    span_summary: dict[str, float],
    shortcut_summary: pd.DataFrame,
    topic_summary: dict[str, object],
) -> None:
    shortcut_rows = shortcut_summary.set_index("scheme").to_dict(orient="index")
    lang_row = shortcut_rows.get("language_only", {})
    genre_row = shortcut_rows.get("language_fine_genre", {})
    full_row = shortcut_rows.get("full_metadata", {})
    topic_row = topic_summary if topic_summary else {}

    lines = [
        "# Leakage Audit Report",
        "",
        "## What the challenge means",
        "- `language leakage`: a model can narrow the answer mainly from language identity rather than writing style. This happens when positives are strongly same-language and candidate authors are language-specialized.",
        "- `topic leakage`: a model can win by matching subject/domain cues rather than authorial signal. This happens when positives are tightly same-topic and topic-matched pools collapse to only a few authors.",
        "",
        "## Audit scope",
        f"- dataset: `{dataset_dir}`",
        f"- splits: `{', '.join(splits)}`",
        f"- positive pairs audited: `{int(alignment_summary.get('pairs_total', 0)):,}`",
        "",
        "## Positive-pair alignment",
        f"- same language: `{alignment_summary.get('same_language', 0.0):.2f}%`",
        f"- same primary genre: `{alignment_summary.get('same_primary_genre', 0.0):.2f}%`",
        f"- same fine genre: `{alignment_summary.get('same_fine_genre', 0.0):.2f}%`",
        f"- same source: `{alignment_summary.get('same_source', 0.0):.2f}%`",
        f"- same length bucket: `{alignment_summary.get('same_length_bucket', 0.0):.2f}%`",
        "",
        "## Candidate-author specialization",
        f"- single-language authors: `{span_summary.get('pct_single_language_authors', 0.0):.2f}%`",
        f"- single-primary-genre authors: `{span_summary.get('pct_single_primary_genre_authors', 0.0):.2f}%`",
        f"- single-fine-genre authors: `{span_summary.get('pct_single_fine_genre_authors', 0.0):.2f}%`",
        f"- single-source authors: `{span_summary.get('pct_single_source_authors', 0.0):.2f}%`",
        "",
        "## Shortcut strength from metadata only",
        f"- language only: median pool `{lang_row.get('pool_size_median', 0):.1f}`, expected success@1 `{lang_row.get('expected_success@1', 0.0):.4f}`, deterministic success@1 `{lang_row.get('deterministic_success@1', 0.0):.4f}`",
        f"- language + fine genre: median pool `{genre_row.get('pool_size_median', 0):.1f}`, expected success@1 `{genre_row.get('expected_success@1', 0.0):.4f}`, deterministic success@1 `{genre_row.get('deterministic_success@1', 0.0):.4f}`",
        f"- full metadata: median pool `{full_row.get('pool_size_median', 0):.1f}`, expected success@1 `{full_row.get('expected_success@1', 0.0):.4f}`, deterministic success@1 `{full_row.get('deterministic_success@1', 0.0):.4f}`",
    ]
    if topic_summary:
        lines.extend(
            [
                "",
                "## Topic-cluster shortcut audit",
                f"- queries with topic clusters: `{topic_row.get('pct_queries_with_topic_cluster', 0.0):.2f}%`",
                f"- positive pairs in same topic cluster: `{topic_row.get('pct_same_topic_cluster_positive_pairs', 0.0):.2f}%`",
                f"- language + primary genre + topic cluster: median pool `{topic_row.get('pool_size_median', 0):.1f}`, expected success@1 `{topic_row.get('expected_success@1', 0.0):.4f}`, deterministic success@1 `{topic_row.get('deterministic_success@1', 0.0):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## How to use this in a benchmark-quality argument",
            "- If `language_only` or `language_primary_genre` already gives strong success, the benchmark is vulnerable to metadata shortcuts.",
            "- If matched pools stay large and metadata-only success remains low, the benchmark is harder to solve without authorship signal.",
            "- The strongest reviewer-facing evidence is to report these shortcut baselines next to your real model results.",
        ]
    )
    write_text("\n".join(lines), report_path)


def run(
    dataset_dir: Path,
    output_dir: Path,
    splits: Sequence[str],
    *,
    min_topic_docs_per_stratum: int,
    docs_per_topic_cluster: int,
    max_topic_clusters: int,
    max_tfidf_features: int,
    seed: int,
) -> None:
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"

    queries, candidates, ground_truth = load_eval_frames(dataset_dir, splits)

    alignment_df, alignment_summary = positive_pair_alignment(queries, candidates, ground_truth)
    save_csv(alignment_df, tables_dir / "positive_pair_alignment.csv")
    save_json(alignment_summary, tables_dir / "positive_pair_alignment_summary.json")

    span_df, span_summary = author_metadata_span(candidates)
    save_csv(span_df, tables_dir / "candidate_author_metadata_span.csv")
    save_json(span_summary, tables_dir / "candidate_author_metadata_span_summary.json")

    per_query_df, shortcut_summary_df, shortcut_by_lang_df = evaluate_shortcut_schemes(
        queries,
        ground_truth,
        candidates,
        SCHEMES,
    )
    save_csv(per_query_df, tables_dir / "shortcut_pool_sizes_by_query.csv")
    save_csv(shortcut_summary_df, tables_dir / "shortcut_pool_summary.csv")
    save_csv(shortcut_by_lang_df, tables_dir / "shortcut_pool_summary_by_language.csv")

    assignments_df, stratum_df = assign_topic_clusters(
        queries,
        candidates,
        min_docs_per_stratum=min_topic_docs_per_stratum,
        docs_per_cluster=docs_per_topic_cluster,
        max_clusters=max_topic_clusters,
        max_features=max_tfidf_features,
        seed=seed,
    )
    save_csv(stratum_df, tables_dir / "topic_cluster_strata.csv")
    topic_summary: dict[str, object] = {}
    if not assignments_df.empty:
        save_csv(assignments_df, tables_dir / "topic_cluster_assignments.csv")
        topic_per_query_df, topic_alignment_df, topic_summary = evaluate_topic_shortcuts(
            queries,
            candidates,
            ground_truth,
            assignments_df,
        )
        save_csv(topic_per_query_df, tables_dir / "topic_shortcut_pool_sizes_by_query.csv")
        save_csv(topic_alignment_df, tables_dir / "topic_alignment_by_language.csv")
        save_json(topic_summary, tables_dir / "topic_shortcut_summary.json")
    else:
        save_csv(pd.DataFrame(), tables_dir / "topic_cluster_assignments.csv")
        save_csv(pd.DataFrame(), tables_dir / "topic_shortcut_pool_sizes_by_query.csv")
        save_csv(pd.DataFrame(), tables_dir / "topic_alignment_by_language.csv")
        save_json({}, tables_dir / "topic_shortcut_summary.json")

    build_report(
        reports_dir / "leakage_audit_report.md",
        dataset_dir=dataset_dir,
        splits=splits,
        alignment_summary=alignment_summary,
        span_summary=span_summary,
        shortcut_summary=shortcut_summary_df,
        topic_summary=topic_summary,
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "splits": list(splits),
        "queries": int(len(queries)),
        "candidates": int(len(candidates)),
        "candidate_authors": int(candidates["author_id"].nunique()),
        "positive_pairs": int(len(ground_truth)),
        "positive_pair_alignment": alignment_summary,
        "candidate_author_span": span_summary,
        "best_metadata_shortcut_by_expected_success@1": shortcut_summary_df.iloc[0].to_dict()
        if not shortcut_summary_df.empty
        else {},
        "topic_shortcut": topic_summary,
    }
    save_json(summary, output_dir / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit topic and language leakage risk for an AuthBench benchmark."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("processing/outputs/authbench"),
        help="Benchmark root containing split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("post_analysis/outputs/authbench/leakage_audit"),
        help="Where to write leakage audit outputs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Benchmark splits to audit. Use `all` to auto-discover.",
    )
    parser.add_argument(
        "--min-topic-docs-per-stratum",
        type=int,
        default=80,
        help="Minimum docs in a (language, primary_genre) stratum before topic clustering.",
    )
    parser.add_argument(
        "--docs-per-topic-cluster",
        type=int,
        default=250,
        help="Approximate target documents per topic cluster.",
    )
    parser.add_argument(
        "--max-topic-clusters",
        type=int,
        default=24,
        help="Maximum topic clusters per (language, primary_genre) stratum.",
    )
    parser.add_argument(
        "--max-tfidf-features",
        type=int,
        default=15000,
        help="Maximum TF-IDF features per topic-cluster stratum.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for topic clustering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = discover_splits(args.dataset_dir, args.splits)
    run(
        args.dataset_dir,
        args.output_dir,
        splits,
        min_topic_docs_per_stratum=args.min_topic_docs_per_stratum,
        docs_per_topic_cluster=args.docs_per_topic_cluster,
        max_topic_clusters=args.max_topic_clusters,
        max_tfidf_features=args.max_tfidf_features,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
