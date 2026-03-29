from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from eval.data import load_split
from eval.pools import build_topic_candidate_index, build_topic_pool

try:
    from utilities.model_registry import SELF_CONSISTENCY_LLM_MODELS
except Exception:
    SELF_CONSISTENCY_LLM_MODELS = set()


DEFAULT_BASELINE_MODELS = ("tfidf", "ngram", "ppm")
METRIC_ORDER = {
    "success": 0,
    "recall": 1,
    "ndcg": 2,
    "mrr": 3,
    "roc_auc": 4,
    "eer": 5,
}
MODEL_TYPE_COLORS = {
    "embedding": "#3c6e71",
    "embedding-instruct": "#6c9a8b",
    "llm-base": "#d9a441",
    "llm-instruct": "#c75b39",
    "baseline": "#6f4e7c",
    "other": "#5b6572",
}
BASELINE_LINE_COLORS = {
    "tfidf": "#c1121f",
    "ngram": "#7b2cbf",
    "ppm": "#1982c4",
}
GROUPING_LABELS = {
    "language": "language",
    "full_genre": "genre",
    "primary_genre": "primary genre",
    "length_bucket": "length bucket",
}
WIDE_GROUPINGS = ("language", "primary_genre", "length_bucket")
LONG_GROUPINGS = ("language", "full_genre", "primary_genre", "length_bucket")
RESULT_METADATA_KEYS = {
    "by_language",
    "by_genre",
    "by_length_bucket",
    "num_queries",
    "num_candidates",
    "min_num_candidates",
    "max_num_candidates",
    "positive_pairs",
    "negative_pairs",
    "negative_strategy",
    "negatives_per_query",
}
SUM_KEYS = {"num_queries", "positive_pairs", "negative_pairs"}
MIN_KEYS = {"min_num_candidates"}
MAX_KEYS = {"max_num_candidates"}


@dataclass(frozen=True)
class QueryContext:
    query_id: str
    language: str
    full_genre: str
    primary_genre: str
    length_bucket: str
    num_candidates: int
    num_positives: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AuthBench evaluation results and export tables, plots, and reports."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory containing zero-shot model JSON results (top-level *.json only).",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path("eval/results/baselines"),
        help="Directory containing baseline JSON results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results/analysis"),
        help="Root directory for exported tables, plots, and reports.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("processing/outputs/authbench"),
        help="Processed benchmark root used to reconstruct random-guess reference metrics.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name used during evaluation.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=str,
        choices=("all", "topic"),
        default="all",
        help="Candidate-pool setting used during evaluation.",
    )
    parser.add_argument(
        "--max-topic-candidates",
        type=int,
        default=None,
        help="Topic-pool candidate cap, if topic-controlled evaluation was used.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query cap used during evaluation.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional candidate cap used during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Sampling seed used with --max-queries/--max-candidates.",
    )
    parser.add_argument(
        "--topic-seed",
        type=int,
        default=13,
        help="Sampling seed used for topic pools.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to export. Defaults to auto-discovering every metric in the JSON results.",
    )
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        default=list(DEFAULT_BASELINE_MODELS),
        help="Baseline model names to use as dashed reference lines in plots.",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=["png", "pdf"],
        help="Plot formats to export.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation and export tables/reports only.",
    )
    return parser.parse_args()


def clean_label(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def primary_genre_label(value: object) -> str:
    return clean_label(value).split("/", 1)[0]


def length_bucket_from_record(record: Mapping[str, object]) -> str:
    bucket = record.get("length_bucket")
    if bucket:
        return clean_label(bucket)
    token_length = record.get("token_length")
    try:
        token_length_int = int(token_length)
    except Exception:
        return "unknown"
    if token_length_int <= 10:
        return "short"
    if token_length_int <= 100:
        return "medium"
    if token_length_int <= 500:
        return "long"
    return "extra_long"


def metric_sort_key(metric: str) -> tuple[int, int, str]:
    if "@" in metric:
        prefix, _, suffix = metric.partition("@")
        try:
            k = int(suffix)
        except ValueError:
            k = 0
        return (METRIC_ORDER.get(prefix, 99), k, metric)
    return (METRIC_ORDER.get(metric, 99), 0, metric)


def canonical_metric_name(metric: str) -> str:
    normalized = metric.strip().lower().replace("-", "_")
    if normalized.startswith("eer"):
        return "eer"
    if normalized in {"roc_auc", "rocauc"}:
        return "roc_auc"
    return normalized


def metric_task(metric: str) -> str:
    return "attribution" if canonical_metric_name(metric) in {"eer", "roc_auc"} else "representation"


def metric_higher_is_better(metric: str) -> bool:
    return canonical_metric_name(metric) != "eer"


def metric_optimum(metric: str) -> float:
    return 0.0 if canonical_metric_name(metric) == "eer" else 1.0


def numeric_value(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def get_metric_value(block: Mapping[str, object], metric: str) -> Optional[float]:
    canonical = canonical_metric_name(metric)
    if canonical == "eer":
        return numeric_value(block.get("eer"))
    return numeric_value(block.get(canonical))


def load_single_result(path: Path) -> tuple[str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty top-level JSON object in {path}")
    model, payload = next(iter(data.items()))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected model payload to be a JSON object in {path}")
    return str(model), payload


def load_results(
    results_dir: Path,
    baselines_dir: Path,
    baseline_models: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    for path in sorted(results_dir.glob("*.json")):
        model, payload = load_single_result(path)
        if model in results:
            raise ValueError(f"Duplicate result entry for model '{model}'")
        results[model] = payload

    for path in sorted(baselines_dir.glob("*.json")):
        model, payload = load_single_result(path)
        if model in results:
            raise ValueError(f"Duplicate result entry for model '{model}'")
        results[model] = payload

    missing = [name for name in baseline_models if name not in results]
    if missing:
        raise ValueError(
            f"Missing required baseline result(s): {', '.join(missing)}. "
            f"Expected them under {baselines_dir}."
        )

    if not results:
        raise ValueError(
            f"No JSON result files found in {results_dir} or baseline directory {baselines_dir}."
        )
    return results


def discover_metrics(results: Mapping[str, Mapping[str, object]]) -> List[str]:
    metrics: set[str] = set()
    for payload in results.values():
        for task in ("representation", "attribution"):
            block = payload.get(task)
            if not isinstance(block, dict):
                continue
            for key, value in block.items():
                if key in RESULT_METADATA_KEYS or isinstance(value, dict):
                    continue
                if numeric_value(value) is None:
                    continue
                metrics.add(canonical_metric_name(key))
    return sorted(metrics, key=metric_sort_key)


def infer_model_family(model_name: str) -> str:
    name = model_name.lower()
    if name in DEFAULT_BASELINE_MODELS:
        return "baseline"
    families = (
        ("qwen3-embedding", "qwen-embedding"),
        ("qwen3-emb", "qwen-embedding"),
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("deepseek", "deepseek"),
        ("bge", "bge"),
        ("multilingual-e5", "e5"),
        ("e5", "e5"),
        ("gte", "gte"),
        ("jina", "jina"),
        ("snowflake", "snowflake"),
        ("mxbai", "mixedbread"),
        ("nomic", "nomic"),
        ("sfr", "salesforce"),
        ("specter", "specter"),
        ("contriever", "contriever"),
        ("minilm", "sentence-transformers"),
        ("mpnet", "sentence-transformers"),
        ("distiluse", "sentence-transformers"),
        ("distilbert", "sentence-transformers"),
        ("roberta", "sentence-transformers"),
        ("bert", "bert"),
    )
    for prefix, family in families:
        if prefix in name:
            return family
    return name.split("-", 1)[0]


def infer_model_type(model_name: str, payload: Mapping[str, object]) -> str:
    name = model_name.lower()
    if payload.get("baseline") is not None or name in DEFAULT_BASELINE_MODELS:
        return "baseline"

    embedding_markers = (
        "embed",
        "embedding",
        "e5",
        "bge",
        "gte",
        "jina",
        "snowflake",
        "mxbai",
        "nomic",
        "sfr",
        "mpnet",
        "minilm",
        "roberta",
        "distiluse",
        "distilbert",
        "specter",
        "contriever",
        "bert-base-uncased",
    )
    llm_markers = ("llama", "deepseek", "qwen")
    is_embedding = any(marker in name for marker in embedding_markers)
    is_llm = (
        model_name in SELF_CONSISTENCY_LLM_MODELS
        or (any(marker in name for marker in llm_markers) and not is_embedding)
    )
    is_instruct = any(marker in name for marker in ("instruct", "chat"))

    if is_llm and is_instruct:
        return "llm-instruct"
    if is_llm:
        return "llm-base"
    if is_embedding and is_instruct:
        return "embedding-instruct"
    if is_embedding:
        return "embedding"
    return "other"


def build_query_contexts(
    dataset_root: Path,
    split_name: str,
    candidate_pool: str,
    max_topic_candidates: Optional[int],
    max_queries: Optional[int],
    max_candidates: Optional[int],
    seed: int,
    topic_seed: int,
) -> List[QueryContext]:
    split = load_split(dataset_root, split_name).limited(
        max_queries=max_queries,
        max_candidates=max_candidates,
        seed=seed,
    )
    candidate_ids = [candidate["candidate_id"] for candidate in split.candidates]
    candidate_index = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
    topic_candidates = (
        build_topic_candidate_index(split.candidates) if candidate_pool == "topic" else None
    )

    contexts: List[QueryContext] = []
    for query_record in split.queries:
        query_id = query_record["query_id"]
        positives = split.positives_by_query.get(query_id, [])
        positive_indices = [candidate_index[cid] for cid in positives if cid in candidate_index]
        if not positive_indices:
            continue

        if candidate_pool == "all":
            num_candidates = len(candidate_ids)
            num_positives = len(positive_indices)
        else:
            pool_indices = build_topic_pool(
                query_record=query_record,
                query_id=query_id,
                candidate_ids=candidate_ids,
                candidate_indices_by_topic=topic_candidates or {},
                positive_indices=positive_indices,
                max_candidates=max_topic_candidates,
                seed=topic_seed,
            )
            if not pool_indices:
                continue
            pool_set = set(pool_indices)
            num_candidates = len(pool_indices)
            num_positives = sum(1 for idx in positive_indices if idx in pool_set)
            if num_positives <= 0:
                continue

        full_genre = clean_label(query_record.get("genre"))
        contexts.append(
            QueryContext(
                query_id=query_id,
                language=clean_label(query_record.get("lang") or query_record.get("language")),
                full_genre=full_genre,
                primary_genre=primary_genre_label(full_genre),
                length_bucket=length_bucket_from_record(query_record),
                num_candidates=num_candidates,
                num_positives=num_positives,
            )
        )

    if not contexts:
        raise ValueError(
            "No evaluable queries found while reconstructing query-level benchmark stats. "
            "Check dataset-root / split / candidate-pool settings."
        )
    return contexts


@lru_cache(maxsize=None)
def random_success_at_k(num_candidates: int, num_positives: int, k: int) -> float:
    if num_candidates <= 0 or num_positives <= 0:
        return 0.0
    k = min(max(k, 0), num_candidates)
    if k == 0:
        return 0.0
    if num_positives >= num_candidates:
        return 1.0
    prob_no_hit = 1.0
    for i in range(k):
        prob_no_hit *= (num_candidates - num_positives - i) / (num_candidates - i)
    return 1.0 - prob_no_hit


@lru_cache(maxsize=None)
def random_recall_at_k(num_candidates: int, num_positives: int, k: int) -> float:
    if num_candidates <= 0 or num_positives <= 0:
        return 0.0
    return min(max(k, 0), num_candidates) / float(num_candidates)


def ideal_dcg(num_positives: int, k: int) -> float:
    top = min(num_positives, k)
    if top <= 0:
        return 0.0
    return sum(1.0 / math.log2(idx + 2) for idx in range(top))


@lru_cache(maxsize=None)
def random_ndcg_at_k(num_candidates: int, num_positives: int, k: int) -> float:
    if num_candidates <= 0 or num_positives <= 0:
        return 0.0
    k = min(max(k, 0), num_candidates)
    if k == 0:
        return 0.0
    expected_dcg = (num_positives / float(num_candidates)) * sum(
        1.0 / math.log2(idx + 2) for idx in range(k)
    )
    idcg = ideal_dcg(num_positives, k)
    if idcg <= 0.0:
        return 0.0
    return expected_dcg / idcg


@lru_cache(maxsize=None)
def random_mrr(num_candidates: int, num_positives: int) -> float:
    if num_candidates <= 0 or num_positives <= 0:
        return 0.0
    if num_positives >= num_candidates:
        return 1.0
    expectation = 0.0
    prob_no_hit_before = 1.0
    max_rank = num_candidates - num_positives + 1
    for rank in range(1, max_rank + 1):
        remaining_positions = num_candidates - rank + 1
        prob_hit_here = prob_no_hit_before * (num_positives / float(remaining_positions))
        expectation += prob_hit_here / rank
        prob_no_hit_before *= (remaining_positions - num_positives) / float(remaining_positions)
    return expectation


def random_metric_expectation(metric: str, num_candidates: int, num_positives: int) -> Optional[float]:
    canonical = canonical_metric_name(metric)
    if canonical.startswith("success@"):
        _, _, suffix = canonical.partition("@")
        return random_success_at_k(num_candidates, num_positives, int(suffix))
    if canonical.startswith("recall@"):
        _, _, suffix = canonical.partition("@")
        return random_recall_at_k(num_candidates, num_positives, int(suffix))
    if canonical.startswith("ndcg@"):
        _, _, suffix = canonical.partition("@")
        return random_ndcg_at_k(num_candidates, num_positives, int(suffix))
    if canonical == "mrr":
        return random_mrr(num_candidates, num_positives)
    if canonical in {"roc_auc", "eer"}:
        return 0.5
    return None


def chance_adjusted_score(metric: str, value: Optional[float], random_expected: Optional[float]) -> Optional[float]:
    if value is None or random_expected is None:
        return None
    optimum = metric_optimum(metric)
    if metric_higher_is_better(metric):
        denom = optimum - random_expected
        if denom == 0:
            return None
        return (value - random_expected) / denom
    denom = random_expected - optimum
    if denom == 0:
        return None
    return (random_expected - value) / denom


def summarize_query_contexts(
    query_contexts: Sequence[QueryContext],
    metrics: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[QueryContext]]] = {
        "overall": {"overall": list(query_contexts)},
        "language": defaultdict(list),
        "full_genre": defaultdict(list),
        "primary_genre": defaultdict(list),
        "length_bucket": defaultdict(list),
    }
    for context in query_contexts:
        grouped["language"][context.language].append(context)
        grouped["full_genre"][context.full_genre].append(context)
        grouped["primary_genre"][context.primary_genre].append(context)
        grouped["length_bucket"][context.length_bucket].append(context)

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for grouping, buckets in grouped.items():
        summaries[grouping] = {}
        for bucket, rows in buckets.items():
            if not rows:
                continue
            candidate_counts = [row.num_candidates for row in rows]
            positive_counts = [row.num_positives for row in rows]
            summary: Dict[str, float] = {
                "num_queries": float(len(rows)),
                "avg_num_candidates": sum(candidate_counts) / len(candidate_counts),
                "min_num_candidates": float(min(candidate_counts)),
                "max_num_candidates": float(max(candidate_counts)),
                "avg_num_positives": sum(positive_counts) / len(positive_counts),
            }
            for metric in metrics:
                expectations = [
                    random_metric_expectation(metric, row.num_candidates, row.num_positives)
                    for row in rows
                ]
                valid = [value for value in expectations if value is not None]
                if valid:
                    summary[f"random_{canonical_metric_name(metric)}"] = sum(valid) / len(valid)
            summaries[grouping][bucket] = summary
    return summaries


def aggregate_primary_genre_bucket(
    bucket_data: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for genre, metrics in bucket_data.items():
        grouped[primary_genre_label(genre)].append(metrics)

    aggregated: Dict[str, Dict[str, float]] = {}
    for primary, items in grouped.items():
        keys = sorted({key for item in items for key in item.keys()})
        total_queries = sum(int(numeric_value(item.get("num_queries")) or 0) for item in items)
        effective_weight = total_queries if total_queries > 0 else len(items)
        row: Dict[str, float] = {}
        for key in keys:
            numeric_items = [numeric_value(item.get(key)) for item in items]
            numeric_items = [value for value in numeric_items if value is not None]
            if not numeric_items:
                continue
            if key in SUM_KEYS:
                row[key] = sum(numeric_items)
                continue
            if key in MIN_KEYS:
                row[key] = min(numeric_items)
                continue
            if key in MAX_KEYS:
                row[key] = max(numeric_items)
                continue

            weighted_sum = 0.0
            weight_total = 0.0
            for item in items:
                value = numeric_value(item.get(key))
                if value is None:
                    continue
                weight = numeric_value(item.get("num_queries")) or 1.0
                weighted_sum += value * weight
                weight_total += weight
            if weight_total == 0.0:
                row[key] = sum(numeric_items) / len(numeric_items)
            else:
                row[key] = weighted_sum / weight_total

        if "num_queries" not in row:
            row["num_queries"] = float(effective_weight)
        aggregated[primary] = row
    return aggregated


def get_group_bucket_data(
    payload: Mapping[str, object],
    task: str,
    grouping: str,
) -> Dict[str, Dict[str, float]]:
    block = payload.get(task)
    if not isinstance(block, dict):
        return {}
    if grouping == "language":
        raw = block.get("by_language")
    elif grouping == "full_genre":
        raw = block.get("by_genre")
    elif grouping == "primary_genre":
        raw = block.get("by_genre")
        if isinstance(raw, dict):
            return aggregate_primary_genre_bucket(raw)
        return {}
    elif grouping == "length_bucket":
        raw = block.get("by_length_bucket")
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")

    if not isinstance(raw, dict):
        return {}
    output: Dict[str, Dict[str, float]] = {}
    for bucket, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        converted = {
            key: numeric
            for key, value in metrics.items()
            if (numeric := numeric_value(value)) is not None
        }
        if converted:
            output[str(bucket)] = converted
    return output


def overall_metric_map(rows: Sequence[Dict[str, object]], normalized: bool) -> Dict[tuple[str, str], Optional[float]]:
    value_key = "normalized_value" if normalized else "value"
    return {
        (str(row["model"]), str(row["metric"])): numeric_value(row.get(value_key))
        for row in rows
    }


def build_overall_rows(
    results: Mapping[str, Mapping[str, object]],
    metrics: Sequence[str],
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    baseline_models: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    overall_random = query_summaries["overall"]["overall"]
    baseline_set = set(baseline_models)

    for model, payload in sorted(results.items()):
        model_type = infer_model_type(model, payload)
        model_family = infer_model_family(model)
        source = "baseline" if model in baseline_set or model_type == "baseline" else "model"
        hf_repo = payload.get("hf_repo")
        baseline_meta = payload.get("baseline")

        for metric in metrics:
            task = metric_task(metric)
            block = payload.get(task)
            if not isinstance(block, dict):
                continue
            value = get_metric_value(block, metric)
            if value is None:
                continue
            canonical = canonical_metric_name(metric)
            random_expected = numeric_value(overall_random.get(f"random_{canonical}"))
            rows.append(
                {
                    "model": model,
                    "model_type": model_type,
                    "model_family": model_family,
                    "source": source,
                    "is_baseline": source == "baseline",
                    "task": task,
                    "metric": canonical,
                    "value": value,
                    "random_expected": random_expected,
                    "normalized_value": chance_adjusted_score(canonical, value, random_expected),
                    "higher_is_better": metric_higher_is_better(canonical),
                    "hf_repo": hf_repo or "",
                    "num_queries": numeric_value(block.get("num_queries")),
                    "num_candidates": numeric_value(block.get("num_candidates")),
                    "min_num_candidates": numeric_value(block.get("min_num_candidates")),
                    "max_num_candidates": numeric_value(block.get("max_num_candidates")),
                    "positive_pairs": numeric_value(block.get("positive_pairs")),
                    "negative_pairs": numeric_value(block.get("negative_pairs")),
                    "baseline_metadata": json.dumps(baseline_meta, sort_keys=True)
                    if baseline_meta is not None
                    else "",
                }
            )
    return rows


def build_grouped_rows(
    results: Mapping[str, Mapping[str, object]],
    metrics: Sequence[str],
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    baseline_models: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    baseline_set = set(baseline_models)

    for model, payload in sorted(results.items()):
        model_type = infer_model_type(model, payload)
        model_family = infer_model_family(model)
        source = "baseline" if model in baseline_set or model_type == "baseline" else "model"

        for grouping in LONG_GROUPINGS:
            task_groups: Dict[str, Dict[str, Dict[str, float]]] = {}
            for task in ("representation", "attribution"):
                task_groups[task] = get_group_bucket_data(payload, task, grouping)

            for metric in metrics:
                task = metric_task(metric)
                bucket_data = task_groups.get(task, {})
                if not bucket_data:
                    continue
                canonical = canonical_metric_name(metric)
                for bucket, metrics_block in bucket_data.items():
                    value = get_metric_value(metrics_block, canonical)
                    if value is None:
                        continue
                    query_summary = query_summaries.get(grouping, {}).get(bucket, {})
                    random_expected = numeric_value(query_summary.get(f"random_{canonical}"))
                    rows.append(
                        {
                            "model": model,
                            "model_type": model_type,
                            "model_family": model_family,
                            "source": source,
                            "is_baseline": source == "baseline",
                            "grouping": grouping,
                            "bucket": bucket,
                            "task": task,
                            "metric": canonical,
                            "value": value,
                            "random_expected": random_expected,
                            "normalized_value": chance_adjusted_score(canonical, value, random_expected),
                            "higher_is_better": metric_higher_is_better(canonical),
                            "num_queries": numeric_value(metrics_block.get("num_queries")),
                            "num_candidates": numeric_value(metrics_block.get("num_candidates")),
                            "min_num_candidates": numeric_value(metrics_block.get("min_num_candidates")),
                            "max_num_candidates": numeric_value(metrics_block.get("max_num_candidates")),
                            "positive_pairs": numeric_value(metrics_block.get("positive_pairs")),
                            "negative_pairs": numeric_value(metrics_block.get("negative_pairs")),
                        }
                    )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def bucket_order(grouping: str, query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]]) -> List[str]:
    keys = list(query_summaries.get(grouping, {}).keys())
    if grouping == "length_bucket":
        desired = ["short", "medium", "long", "extra_long", "unknown"]
        return [bucket for bucket in desired if bucket in keys] + sorted(
            [bucket for bucket in keys if bucket not in desired]
        )
    return sorted(keys)


def sort_rows_for_metric(rows: Sequence[Dict[str, object]], metric: str, key: str) -> List[Dict[str, object]]:
    higher_better = metric_higher_is_better(metric)

    def score(row: Mapping[str, object]) -> float:
        value = numeric_value(row.get(key))
        if value is None:
            return float("-inf") if higher_better else float("inf")
        return value

    return sorted(
        rows,
        key=lambda row: (
            score(row) if higher_better else -score(row),
            str(row["model"]),
        ),
        reverse=True,
    )


def build_overall_leaderboard_rows(
    overall_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
) -> List[Dict[str, object]]:
    by_model: Dict[str, Dict[str, object]] = {}
    for row in overall_rows:
        model = str(row["model"])
        record = by_model.setdefault(
            model,
            {
                "model": model,
                "model_type": row["model_type"],
                "model_family": row["model_family"],
                "source": row["source"],
                "hf_repo": row["hf_repo"],
                "baseline_metadata": row["baseline_metadata"],
            },
        )
        metric = str(row["metric"])
        record[metric] = row["value"]
        record[f"normalized_{metric}"] = row["normalized_value"]
        record[f"random_{metric}"] = row["random_expected"]
        record[f"{metric}_num_queries"] = row["num_queries"]
        record[f"{metric}_num_candidates"] = row["num_candidates"]
        if row["min_num_candidates"] is not None:
            record[f"{metric}_min_num_candidates"] = row["min_num_candidates"]
        if row["max_num_candidates"] is not None:
            record[f"{metric}_max_num_candidates"] = row["max_num_candidates"]
        if row["positive_pairs"] is not None:
            record[f"{metric}_positive_pairs"] = row["positive_pairs"]
        if row["negative_pairs"] is not None:
            record[f"{metric}_negative_pairs"] = row["negative_pairs"]

    sort_metric = "success@10" if "success@10" in metrics else metrics[0]
    higher_better = metric_higher_is_better(sort_metric)
    return sorted(
        by_model.values(),
        key=lambda row: (
            numeric_value(row.get(sort_metric))
            if higher_better
            else -(numeric_value(row.get(sort_metric)) or float("inf")),
            str(row["model"]),
        ),
        reverse=True,
    )


def build_best_by_metric_rows(
    overall_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        metric_rows = [row for row in overall_rows if row["metric"] == metric]
        if not metric_rows:
            continue
        higher_better = metric_higher_is_better(metric)

        def best_candidate(candidates: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
            if not candidates:
                return None
            return sorted(
                candidates,
                key=lambda row: (
                    numeric_value(row["value"]) if higher_better else -float(row["value"]),
                    str(row["model"]),
                ),
                reverse=True,
            )[0]

        best_all = best_candidate(metric_rows)
        best_model = best_candidate([row for row in metric_rows if not row["is_baseline"]])
        best_baseline = best_candidate([row for row in metric_rows if row["is_baseline"]])
        gap = None
        gap_norm = None
        if best_model is not None and best_baseline is not None:
            direction = 1.0 if higher_better else -1.0
            gap = direction * (float(best_model["value"]) - float(best_baseline["value"]))
            if best_model["normalized_value"] is not None and best_baseline["normalized_value"] is not None:
                gap_norm = direction * (
                    float(best_model["normalized_value"]) - float(best_baseline["normalized_value"])
                )
        rows.append(
            {
                "metric": metric,
                "task": metric_task(metric),
                "best_overall_model": best_all["model"] if best_all else "",
                "best_overall_value": best_all["value"] if best_all else "",
                "best_non_baseline_model": best_model["model"] if best_model else "",
                "best_non_baseline_value": best_model["value"] if best_model else "",
                "best_non_baseline_normalized": best_model["normalized_value"] if best_model else "",
                "best_baseline_model": best_baseline["model"] if best_baseline else "",
                "best_baseline_value": best_baseline["value"] if best_baseline else "",
                "best_baseline_normalized": best_baseline["normalized_value"] if best_baseline else "",
                "improvement_over_best_baseline": gap,
                "normalized_improvement_over_best_baseline": gap_norm,
                "random_expected": best_model["random_expected"] if best_model else "",
            }
        )
    return rows


def build_model_type_summary_rows(
    overall_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        higher_better = metric_higher_is_better(metric)
        metric_rows = [row for row in overall_rows if row["metric"] == metric]
        model_types = sorted({str(row["model_type"]) for row in metric_rows})
        for model_type in model_types:
            candidates = [row for row in metric_rows if row["model_type"] == model_type]
            if not candidates:
                continue
            values = [float(row["value"]) for row in candidates]
            norm_values = [
                float(row["normalized_value"])
                for row in candidates
                if row["normalized_value"] is not None
            ]
            best_row = sorted(
                candidates,
                key=lambda row: (
                    numeric_value(row["value"]) if higher_better else -float(row["value"]),
                    str(row["model"]),
                ),
                reverse=True,
            )[0]
            rows.append(
                {
                    "metric": metric,
                    "task": metric_task(metric),
                    "model_type": model_type,
                    "num_models": len(candidates),
                    "mean_value": sum(values) / len(values),
                    "median_value": median(values),
                    "mean_normalized_value": (sum(norm_values) / len(norm_values)) if norm_values else "",
                    "median_normalized_value": median(norm_values) if norm_values else "",
                    "best_model": best_row["model"],
                    "best_value": best_row["value"],
                    "best_normalized_value": best_row["normalized_value"],
                }
            )
    return rows


def build_slice_summary_rows(
    grouped_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    best_rows: List[Dict[str, object]] = []
    difficulty_rows: List[Dict[str, object]] = []
    for grouping in ("language", "primary_genre", "length_bucket"):
        for metric in metrics:
            candidates = [
                row
                for row in grouped_rows
                if row["grouping"] == grouping and row["metric"] == metric
            ]
            buckets = sorted({str(row["bucket"]) for row in candidates})
            higher_better = metric_higher_is_better(metric)
            direction = 1.0 if higher_better else -1.0
            for bucket in buckets:
                bucket_rows = [row for row in candidates if row["bucket"] == bucket]
                model_rows = [row for row in bucket_rows if not row["is_baseline"]]
                baseline_rows = [row for row in bucket_rows if row["is_baseline"]]
                if model_rows:
                    best_model_row = sorted(
                        model_rows,
                        key=lambda row: (
                            numeric_value(row["value"]) if higher_better else -float(row["value"]),
                            str(row["model"]),
                        ),
                        reverse=True,
                    )[0]
                    mean_value = sum(float(row["value"]) for row in model_rows) / len(model_rows)
                    norm_values = [
                        float(row["normalized_value"])
                        for row in model_rows
                        if row["normalized_value"] is not None
                    ]
                    difficulty_rows.append(
                        {
                            "grouping": grouping,
                            "bucket": bucket,
                            "metric": metric,
                            "task": metric_task(metric),
                            "num_models": len(model_rows),
                            "mean_value": mean_value,
                            "mean_normalized_value": (sum(norm_values) / len(norm_values))
                            if norm_values
                            else "",
                            "best_model": best_model_row["model"],
                            "best_value": best_model_row["value"],
                            "best_normalized_value": best_model_row["normalized_value"],
                        }
                    )
                else:
                    best_model_row = None

                best_baseline_row = None
                if baseline_rows:
                    best_baseline_row = sorted(
                        baseline_rows,
                        key=lambda row: (
                            numeric_value(row["value"]) if higher_better else -float(row["value"]),
                            str(row["model"]),
                        ),
                        reverse=True,
                    )[0]

                gap = ""
                gap_norm = ""
                if best_model_row is not None and best_baseline_row is not None:
                    gap = direction * (float(best_model_row["value"]) - float(best_baseline_row["value"]))
                    if (
                        best_model_row["normalized_value"] is not None
                        and best_baseline_row["normalized_value"] is not None
                    ):
                        gap_norm = direction * (
                            float(best_model_row["normalized_value"])
                            - float(best_baseline_row["normalized_value"])
                        )

                best_rows.append(
                    {
                        "grouping": grouping,
                        "bucket": bucket,
                        "metric": metric,
                        "task": metric_task(metric),
                        "best_non_baseline_model": best_model_row["model"] if best_model_row else "",
                        "best_non_baseline_value": best_model_row["value"] if best_model_row else "",
                        "best_non_baseline_normalized": (
                            best_model_row["normalized_value"] if best_model_row else ""
                        ),
                        "best_baseline_model": best_baseline_row["model"] if best_baseline_row else "",
                        "best_baseline_value": best_baseline_row["value"] if best_baseline_row else "",
                        "best_baseline_normalized": (
                            best_baseline_row["normalized_value"] if best_baseline_row else ""
                        ),
                        "improvement_over_best_baseline": gap,
                        "normalized_improvement_over_best_baseline": gap_norm,
                    }
                )
    return best_rows, difficulty_rows


def build_wide_rows(
    grouped_rows: Sequence[Dict[str, object]],
    overall_rows: Sequence[Dict[str, object]],
    grouping: str,
    metric: str,
    normalized: bool,
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> List[Dict[str, object]]:
    value_key = "normalized_value" if normalized else "value"
    rows_for_metric = [
        row
        for row in grouped_rows
        if row["grouping"] == grouping and row["metric"] == metric and row.get(value_key) is not None
    ]
    if not rows_for_metric:
        return []

    overall_map = overall_metric_map(overall_rows, normalized=normalized)
    buckets = bucket_order(grouping, query_summaries)
    by_model: Dict[str, Dict[str, object]] = {}
    for row in rows_for_metric:
        model = str(row["model"])
        bucket = str(row["bucket"])
        record = by_model.setdefault(
            model,
            {
                "model": model,
                "model_type": row["model_type"],
                "model_family": row["model_family"],
                "source": row["source"],
                "overall": overall_map.get((model, metric)),
            },
        )
        record[bucket] = row[value_key]

    wide_rows: List[Dict[str, object]] = []
    for model, record in by_model.items():
        values = [
            numeric_value(record.get(bucket))
            for bucket in buckets
            if numeric_value(record.get(bucket)) is not None
        ]
        record["macro_avg"] = (sum(values) / len(values)) if values else None
        wide_rows.append(record)

    sort_key = "overall" if all(numeric_value(row.get("overall")) is not None for row in wide_rows) else "macro_avg"
    return sort_rows_for_metric(wide_rows, metric, sort_key)


def export_wide_tables(
    grouped_rows: Sequence[Dict[str, object]],
    overall_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_dir: Path,
) -> None:
    for grouping in WIDE_GROUPINGS:
        buckets = bucket_order(grouping, query_summaries)
        columns = ["model", "model_type", "model_family", "source", "overall", "macro_avg", *buckets]
        for metric in metrics:
            for normalized, mode in ((False, "raw"), (True, "normalized")):
                rows = build_wide_rows(
                    grouped_rows,
                    overall_rows,
                    grouping,
                    metric,
                    normalized=normalized,
                    query_summaries=query_summaries,
                )
                if not rows:
                    continue
                write_csv(
                    output_dir / grouping / mode / f"{metric}.csv",
                    rows,
                    columns,
                )


def export_random_reference_tables(
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    metrics: Sequence[str],
    output_dir: Path,
) -> None:
    columns = [
        "bucket",
        "num_queries",
        "avg_num_candidates",
        "min_num_candidates",
        "max_num_candidates",
        "avg_num_positives",
        *[f"random_{metric}" for metric in metrics],
    ]
    for grouping, buckets in query_summaries.items():
        rows = []
        for bucket, summary in sorted(buckets.items()):
            row: Dict[str, object] = {"bucket": bucket}
            row.update(summary)
            rows.append(row)
        write_csv(output_dir / f"{grouping}.csv", rows, columns)


def export_long_tables(
    overall_rows: Sequence[Dict[str, object]],
    grouped_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    overall_columns = [
        "model",
        "model_type",
        "model_family",
        "source",
        "is_baseline",
        "task",
        "metric",
        "value",
        "random_expected",
        "normalized_value",
        "higher_is_better",
        "hf_repo",
        "num_queries",
        "num_candidates",
        "min_num_candidates",
        "max_num_candidates",
        "positive_pairs",
        "negative_pairs",
        "baseline_metadata",
    ]
    grouped_columns = [
        "model",
        "model_type",
        "model_family",
        "source",
        "is_baseline",
        "grouping",
        "bucket",
        "task",
        "metric",
        "value",
        "random_expected",
        "normalized_value",
        "higher_is_better",
        "num_queries",
        "num_candidates",
        "min_num_candidates",
        "max_num_candidates",
        "positive_pairs",
        "negative_pairs",
    ]
    write_csv(output_dir / "overall_metrics_long.csv", overall_rows, overall_columns)
    write_csv(output_dir / "grouped_metrics_long.csv", grouped_rows, grouped_columns)


def export_summary_tables(
    overall_rows: Sequence[Dict[str, object]],
    grouped_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    output_dir: Path,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    leaderboard_rows = build_overall_leaderboard_rows(overall_rows, metrics)
    leaderboard_columns = [
        "model",
        "model_type",
        "model_family",
        "source",
        "hf_repo",
        "baseline_metadata",
        *[metric for metric in metrics],
        *[f"normalized_{metric}" for metric in metrics],
        *[f"random_{metric}" for metric in metrics],
        *[f"{metric}_num_queries" for metric in metrics],
        *[f"{metric}_num_candidates" for metric in metrics],
        *[f"{metric}_min_num_candidates" for metric in metrics],
        *[f"{metric}_max_num_candidates" for metric in metrics],
        *[f"{metric}_positive_pairs" for metric in metrics],
        *[f"{metric}_negative_pairs" for metric in metrics],
    ]
    write_csv(output_dir / "leaderboard_overall.csv", leaderboard_rows, leaderboard_columns)

    best_by_metric_rows = build_best_by_metric_rows(overall_rows, metrics)
    write_csv(
        output_dir / "best_by_metric.csv",
        best_by_metric_rows,
        [
            "metric",
            "task",
            "best_overall_model",
            "best_overall_value",
            "best_non_baseline_model",
            "best_non_baseline_value",
            "best_non_baseline_normalized",
            "best_baseline_model",
            "best_baseline_value",
            "best_baseline_normalized",
            "improvement_over_best_baseline",
            "normalized_improvement_over_best_baseline",
            "random_expected",
        ],
    )

    model_type_rows = build_model_type_summary_rows(overall_rows, metrics)
    write_csv(
        output_dir / "by_model_type.csv",
        model_type_rows,
        [
            "metric",
            "task",
            "model_type",
            "num_models",
            "mean_value",
            "median_value",
            "mean_normalized_value",
            "median_normalized_value",
            "best_model",
            "best_value",
            "best_normalized_value",
        ],
    )

    best_slice_rows, difficulty_rows = build_slice_summary_rows(grouped_rows, metrics)
    write_csv(
        output_dir / "best_model_by_slice.csv",
        best_slice_rows,
        [
            "grouping",
            "bucket",
            "metric",
            "task",
            "best_non_baseline_model",
            "best_non_baseline_value",
            "best_non_baseline_normalized",
            "best_baseline_model",
            "best_baseline_value",
            "best_baseline_normalized",
            "improvement_over_best_baseline",
            "normalized_improvement_over_best_baseline",
        ],
    )
    write_csv(
        output_dir / "slice_difficulty.csv",
        difficulty_rows,
        [
            "grouping",
            "bucket",
            "metric",
            "task",
            "num_models",
            "mean_value",
            "mean_normalized_value",
            "best_model",
            "best_value",
            "best_normalized_value",
        ],
    )
    return leaderboard_rows, best_by_metric_rows, model_type_rows, difficulty_rows


def choose_anchor_metrics(metrics: Sequence[str]) -> List[str]:
    available = set(metrics)
    anchors: List[str] = []
    preferred = ("success@10", "recall@10", "ndcg@10", "mrr", "roc_auc", "eer")
    for metric in preferred:
        if metric in available:
            anchors.append(metric)

    for prefix in ("success@", "recall@", "ndcg@"):
        if any(metric.startswith(prefix) for metric in available) and not any(
            anchor.startswith(prefix) for anchor in anchors
        ):
            ranked = sorted(
                [metric for metric in available if metric.startswith(prefix)],
                key=metric_sort_key,
            )
            anchors.append(ranked[-1])
    return anchors


def render_report(
    output_path: Path,
    metrics: Sequence[str],
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    best_by_metric_rows: Sequence[Mapping[str, object]],
    difficulty_rows: Sequence[Mapping[str, object]],
) -> None:
    anchors = choose_anchor_metrics(metrics)
    overall_summary = query_summaries["overall"]["overall"]

    lines = [
        "# Evaluation Analysis",
        "",
        "## Benchmark Reference",
        "",
        f"- Evaluated queries: `{int(overall_summary['num_queries'])}`",
        f"- Average candidate pool size: `{overall_summary['avg_num_candidates']:.2f}`",
        f"- Candidate pool size range: `{int(overall_summary['min_num_candidates'])}` to `{int(overall_summary['max_num_candidates'])}`",
        f"- Average positives per query: `{overall_summary['avg_num_positives']:.4f}`",
        "",
        "## Overall Leaders",
        "",
    ]

    best_by_metric = {str(row["metric"]): row for row in best_by_metric_rows}
    for metric in anchors:
        row = best_by_metric.get(metric)
        if row is None:
            continue
        lines.append(
            f"- `{metric}`: best non-baseline = `{row['best_non_baseline_model']}` "
            f"(`{float(row['best_non_baseline_value']):.4f}`), "
            f"best baseline = `{row['best_baseline_model']}` "
            f"(`{float(row['best_baseline_value']):.4f}`), "
            f"improvement = `{float(row['improvement_over_best_baseline']):.4f}`"
        )

    lines.extend(["", "## Slice Difficulty", ""])

    for metric in anchors:
        metric_rows = [row for row in difficulty_rows if row["metric"] == metric]
        if not metric_rows:
            continue
        lines.append(f"### {metric}")
        lines.append("")
        for grouping in ("language", "primary_genre", "length_bucket"):
            grouping_rows = [row for row in metric_rows if row["grouping"] == grouping]
            normalized_rows = [
                row for row in grouping_rows if numeric_value(row.get("mean_normalized_value")) is not None
            ]
            if not normalized_rows:
                continue
            hardest = sorted(
                normalized_rows,
                key=lambda row: (float(row["mean_normalized_value"]), str(row["bucket"])),
            )[:3]
            easiest = sorted(
                normalized_rows,
                key=lambda row: (float(row["mean_normalized_value"]), str(row["bucket"])),
                reverse=True,
            )[:3]
            lines.append(f"- Hardest {GROUPING_LABELS[grouping]}s:")
            for row in hardest:
                lines.append(
                    f"  - `{row['bucket']}`: mean normalized `{float(row['mean_normalized_value']):.4f}` "
                    f"(best `{row['best_model']}` at `{float(row['best_value']):.4f}`)"
                )
            lines.append(f"- Easiest {GROUPING_LABELS[grouping]}s:")
            for row in easiest:
                lines.append(
                    f"  - `{row['bucket']}`: mean normalized `{float(row['mean_normalized_value']):.4f}` "
                    f"(best `{row['best_model']}` at `{float(row['best_value']):.4f}`)"
                )
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_stem: Path, formats: Sequence[str]) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_stem.with_suffix(f".{fmt}"), bbox_inches="tight")


def plot_horizontal_bar(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    title: str,
    xlabel: str,
    output_stem: Path,
    baseline_values: Mapping[str, float],
    formats: Sequence[str],
) -> None:
    filtered = [row for row in rows if numeric_value(row.get("value")) is not None]
    if not filtered:
        return

    higher_better = metric_higher_is_better(metric)
    filtered = sorted(
        filtered,
        key=lambda row: (
            numeric_value(row["value"]) if higher_better else -float(row["value"]),
            str(row["model"]),
        ),
        reverse=True,
    )

    fig_height = max(6.0, 0.28 * len(filtered) + 1.5)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    y_positions = list(range(len(filtered)))
    values = [float(row["value"]) for row in filtered]
    colors = [
        MODEL_TYPE_COLORS.get(str(row["model_type"]), MODEL_TYPE_COLORS["other"])
        for row in filtered
    ]

    ax.barh(y_positions, values, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row["model"]) for row in filtered])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35, linewidth=1)
    ax.set_axisbelow(True)

    min_value = min(values + list(baseline_values.values()) if baseline_values else values)
    max_value = max(values + list(baseline_values.values()) if baseline_values else values)
    span = max(max_value - min_value, 1e-6)
    ax.set_xlim(min_value - 0.05 * span, max_value + 0.12 * span)

    offset = 0.01 * span
    for idx, value in enumerate(values):
        ax.text(value + offset, idx, f"{value:.3f}", va="center", ha="left", fontsize=9)

    legend_handles: List[object] = []
    used_types = []
    for row in filtered:
        model_type = str(row["model_type"])
        if model_type not in used_types:
            used_types.append(model_type)
            legend_handles.append(
                Patch(
                    facecolor=MODEL_TYPE_COLORS.get(model_type, MODEL_TYPE_COLORS["other"]),
                    label=model_type,
                )
            )

    for baseline_name, baseline_value in baseline_values.items():
        line = ax.axvline(
            baseline_value,
            color=BASELINE_LINE_COLORS.get(baseline_name, "#9e2a2b"),
            linestyle="--",
            linewidth=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=line.get_color(),
                linestyle="--",
                linewidth=2,
                label=f"{baseline_name}: {baseline_value:.3f}",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right")

    fig.tight_layout()
    save_figure(fig, output_stem, formats)
    plt.close(fig)


def export_plots(
    overall_rows: Sequence[Dict[str, object]],
    grouped_rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    baseline_models: Sequence[str],
    formats: Sequence[str],
    output_dir: Path,
    query_summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    apply_plot_style()
    baseline_set = set(baseline_models)
    baseline_names = [model for model in baseline_models if model in baseline_set]

    for metric in metrics:
        for normalized, mode in ((False, "raw"), (True, "normalized")):
            value_key = "normalized_value" if normalized else "value"
            baseline_values = {
                str(row["model"]): float(row[value_key])
                for row in overall_rows
                if row["metric"] == metric
                and row["model"] in baseline_set
                and row.get(value_key) is not None
            }
            model_rows = [
                {
                    "model": row["model"],
                    "model_type": row["model_type"],
                    "value": row[value_key],
                }
                for row in overall_rows
                if row["metric"] == metric
                and row["model"] not in baseline_set
                and row.get(value_key) is not None
            ]
            plot_horizontal_bar(
                model_rows,
                metric=metric,
                title=f"{metric_task(metric).title()} overall ({metric}, {mode})",
                xlabel=f"{metric} ({mode})",
                output_stem=output_dir / "overall" / mode / f"{metric}",
                baseline_values=baseline_values,
                formats=formats,
            )

            for grouping in WIDE_GROUPINGS:
                wide_rows = build_wide_rows(
                    grouped_rows,
                    overall_rows,
                    grouping,
                    metric,
                    normalized=normalized,
                    query_summaries=query_summaries,
                )
                if not wide_rows:
                    continue
                macro_baselines = {
                    str(row["model"]): float(row["macro_avg"])
                    for row in wide_rows
                    if row["model"] in baseline_set and row.get("macro_avg") is not None
                }
                macro_model_rows = [
                    {
                        "model": row["model"],
                        "model_type": row["model_type"],
                        "value": row["macro_avg"],
                    }
                    for row in wide_rows
                    if row["model"] not in baseline_set and row.get("macro_avg") is not None
                ]
                plot_horizontal_bar(
                    macro_model_rows,
                    metric=metric,
                    title=f"{metric_task(metric).title()} macro by {GROUPING_LABELS[grouping]} ({metric}, {mode})",
                    xlabel=f"{metric} macro average ({mode})",
                    output_stem=output_dir / grouping / mode / f"{metric}",
                    baseline_values=macro_baselines,
                    formats=formats,
                )


def main() -> None:
    args = parse_args()

    results = load_results(
        results_dir=args.results_dir,
        baselines_dir=args.baselines_dir,
        baseline_models=args.baseline_models,
    )
    metrics = (
        sorted({canonical_metric_name(metric) for metric in args.metrics}, key=metric_sort_key)
        if args.metrics
        else discover_metrics(results)
    )
    if not metrics:
        raise ValueError("No metrics were found to export.")

    query_contexts = build_query_contexts(
        dataset_root=args.dataset_root,
        split_name=args.split,
        candidate_pool=args.candidate_pool,
        max_topic_candidates=args.max_topic_candidates,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        seed=args.seed,
        topic_seed=args.topic_seed,
    )
    query_summaries = summarize_query_contexts(query_contexts, metrics)

    overall_rows = build_overall_rows(
        results=results,
        metrics=metrics,
        query_summaries=query_summaries,
        baseline_models=args.baseline_models,
    )
    grouped_rows = build_grouped_rows(
        results=results,
        metrics=metrics,
        query_summaries=query_summaries,
        baseline_models=args.baseline_models,
    )

    metadata_dir = args.output_dir / "metadata"
    tables_dir = args.output_dir / "tables"
    summary_dir = tables_dir / "summary"
    long_dir = tables_dir / "long"
    wide_dir = tables_dir / "wide"
    plots_dir = args.output_dir / "plots"
    reports_dir = args.output_dir / "reports"

    write_json(
        metadata_dir / "analysis_config.json",
        {
            "results_dir": str(args.results_dir),
            "baselines_dir": str(args.baselines_dir),
            "dataset_root": str(args.dataset_root),
            "split": args.split,
            "candidate_pool": args.candidate_pool,
            "max_topic_candidates": args.max_topic_candidates,
            "max_queries": args.max_queries,
            "max_candidates": args.max_candidates,
            "seed": args.seed,
            "topic_seed": args.topic_seed,
            "metrics": metrics,
            "baseline_models": list(args.baseline_models),
            "plot_formats": list(args.plot_formats),
        },
    )
    export_random_reference_tables(query_summaries, metrics, metadata_dir / "random_reference")
    export_long_tables(overall_rows, grouped_rows, long_dir)
    export_wide_tables(grouped_rows, overall_rows, metrics, query_summaries, wide_dir)
    _, best_by_metric_rows, _, difficulty_rows = export_summary_tables(
        overall_rows,
        grouped_rows,
        metrics,
        summary_dir,
    )
    render_report(
        reports_dir / "fine_grained_analysis.md",
        metrics=metrics,
        query_summaries=query_summaries,
        best_by_metric_rows=best_by_metric_rows,
        difficulty_rows=difficulty_rows,
    )

    if not args.skip_plots:
        export_plots(
            overall_rows=overall_rows,
            grouped_rows=grouped_rows,
            metrics=metrics,
            baseline_models=args.baseline_models,
            formats=args.plot_formats,
            output_dir=plots_dir,
            query_summaries=query_summaries,
        )

    print(f"Analyzed {len(results)} result files.")
    print(f"Metrics exported: {', '.join(metrics)}")
    print(f"Analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
