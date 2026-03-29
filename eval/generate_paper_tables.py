from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
ANALYSIS_DIR = REPO_ROOT / "eval" / "results" / "analysis" / "tables"
MODEL_SIZE_PATH = REPO_ROOT / "eval" / "model_sizes.json"

GROUP_ORDER = [
    "llm-instruct",
    "llm-base",
    "embedding-instruct",
    "embedding",
    "baseline",
]
GROUP_LABELS = {
    "llm-instruct": "LLMs (instruction-tuned)",
    "llm-base": "LLMs (base)",
    "embedding-instruct": "Embedding models (instruction-tuned)",
    "embedding": "Embedding models",
    "baseline": "Lexical / non-neural baselines",
}
MAIN_SELECTIONS = {
    "llm-instruct": ["llama3.1-8b-instruct", "llama3-8b-instruct", "qwen2.5-7b-instruct", "qwen3-4b-instruct"],
    "llm-base": ["llama3-8b", "llama3.1-8b", "deepseek-llm-7b-base", "qwen3-4b"],
    "embedding-instruct": ["gte-qwen2-7b-instruct", "e5-mistral-7b-instruct"],
    "embedding": ["multilingual-e5-large", "multilingual-e5-base", "qwen3-embedding-8b", "sfr-embedding-mistral"],
    "baseline": ["tfidf", "ngram", "ppm"],
}
BASELINE_DESCRIPTIONS = {
    "ngram": "hashed character/word n-gram stylometric baseline with train-split calibrator",
    "ppm": "fixed-order hashed character language-model approximation of PPM-style scoring",
    "tfidf": "scikit-learn character 3--5 gram TF-IDF cosine baseline",
}
FALLBACK_PARAM_COUNTS = {
    "allenai/specter": 110_000_000,
    "BAAI/bge-base-zh-v1.5": 102_000_000,
    "BAAI/bge-large-en-v1.5": 335_000_000,
    "BAAI/bge-large-zh-v1.5": 326_000_000,
    "BAAI/bge-m3": 567_000_000,
    "deepseek-ai/deepseek-llm-7b-base": 7_000_000_000,
    "deepseek-ai/deepseek-llm-7b-chat": 7_000_000_000,
    "intfloat/e5-base-v2": 109_000_000,
    "intfloat/e5-small-v2": 33_000_000,
    "facebook/contriever": 110_000_000,
    "facebook/contriever-msmarco": 110_000_000,
    "Alibaba-NLP/gte-large-en-v1.5": 409_000_000,
    "jinaai/jina-embeddings-v2-small-en": 33_000_000,
    "Qwen/Qwen3-4B": 4_000_000_000,
    "Qwen/Qwen3-Embedding-4B": 4_000_000_000,
}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def wrap_model(model: str) -> str:
    return rf"\texttt{{{tex_escape(model)}}}"


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def humanize_params(count: int | None) -> str:
    if count is None:
        return "--"
    if count >= 1_000_000_000:
        billions = count / 1_000_000_000.0
        rounded = round(billions)
        if abs(billions - rounded) < 0.05:
            return f"{rounded:.0f}B"
        return f"{billions:.1f}B"
    millions = count / 1_000_000.0
    return f"{round(millions):.0f}M"


def fetch_repo_param_count(repo: str) -> tuple[int | None, str]:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    safetensors = payload.get("safetensors") or {}
    total = maybe_float(safetensors.get("total"))
    if total is not None:
        return int(total), "hf_api_safetensors"
    fallback = FALLBACK_PARAM_COUNTS.get(repo)
    if fallback is not None:
        return fallback, "curated_fallback"
    return None, "unresolved"


def build_model_registry() -> Dict[str, Dict[str, str]]:
    leaderboard_rows = load_csv(ANALYSIS_DIR / "summary" / "leaderboard_overall.csv")
    registry: Dict[str, Dict[str, str]] = {}
    for row in leaderboard_rows:
        registry[row["model"]] = {
            "model": row["model"],
            "model_type": row["model_type"],
            "hf_repo": row.get("hf_repo", ""),
            "source": row.get("source", ""),
        }
    return registry


def build_model_size_manifest(registry: Mapping[str, Mapping[str, str]]) -> Dict[str, Dict[str, object]]:
    existing: Dict[str, Dict[str, object]] = {}
    if MODEL_SIZE_PATH.exists():
        with MODEL_SIZE_PATH.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = {str(k): dict(v) for k, v in loaded.items()}

    manifest: Dict[str, Dict[str, object]] = {}
    for model in sorted(registry):
        meta = registry[model]
        repo = str(meta.get("hf_repo", "") or "")
        if meta.get("model_type") == "baseline":
            manifest[model] = {
                "model": model,
                "hf_repo": "",
                "param_count": None,
                "size_label": "--",
                "source": "baseline",
            }
            continue

        cached = existing.get(model)
        if cached and cached.get("hf_repo") == repo and cached.get("size_label"):
            manifest[model] = cached
            continue

        param_count, source = fetch_repo_param_count(repo)
        manifest[model] = {
            "model": model,
            "hf_repo": repo,
            "param_count": param_count,
            "size_label": humanize_params(param_count),
            "source": source,
        }

    MODEL_SIZE_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def sort_models(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    def sort_key(row: Mapping[str, str]) -> tuple[int, float, str]:
        model_type = row["model_type"]
        group_idx = GROUP_ORDER.index(model_type) if model_type in GROUP_ORDER else len(GROUP_ORDER)
        success = maybe_float(row.get("success@5"))
        return (group_idx, -(success or -1.0), row["model"])

    return [dict(row) for row in sorted(rows, key=sort_key)]


def grouped_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {group: [] for group in GROUP_ORDER}
    for row in rows:
        grouped.setdefault(row["model_type"], []).append(dict(row))
    for group in grouped:
        grouped[group] = sort_models(grouped[group])
    return grouped


def compute_top_two(
    rows: Sequence[Mapping[str, str]],
    column: str,
    higher_is_better: bool,
) -> Dict[str, str]:
    sortable = []
    for row in rows:
        value = maybe_float(row.get(column))
        if value is None:
            continue
        sortable.append((row["model"], value))
    sortable.sort(key=lambda item: ((-item[1]) if higher_is_better else item[1], item[0]))
    styles: Dict[str, str] = {}
    if sortable:
        styles[sortable[0][0]] = "best"
    if len(sortable) > 1:
        styles[sortable[1][0]] = "second"
    return styles


def style_metric(value: float, style: str | None) -> str:
    text = format_metric(value)
    if style == "best":
        return rf"\textbf{{{text}}}"
    if style == "second":
        return rf"\underline{{{text}}}"
    return text


def render_grouped_rows(
    rows: Sequence[Mapping[str, str]],
    metric_columns: Sequence[str],
    size_manifest: Mapping[str, Mapping[str, object]],
    metric_directions: Mapping[str, bool],
) -> List[str]:
    lines: List[str] = []
    grouped = grouped_rows(rows)
    for group in GROUP_ORDER:
        group_rows = grouped.get(group, [])
        if not group_rows:
            continue
        lines.append(rf"\multicolumn{{{2 + len(metric_columns)}}}{{l}}{{\textit{{{GROUP_LABELS[group]}}}}} \\")
        lines.append(r"\midrule")
        per_metric_style = {
            column: compute_top_two(group_rows, column, metric_directions[column]) for column in metric_columns
        }
        for row in group_rows:
            model = row["model"]
            size_label = str(size_manifest[model]["size_label"])
            rendered = [wrap_model(model), size_label]
            for column in metric_columns:
                value = maybe_float(row.get(column))
                if value is None:
                    rendered.append("--")
                    continue
                style = per_metric_style[column].get(model)
                rendered.append(style_metric(value, style))
            lines.append(" & ".join(rendered) + r" \\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return lines


def render_slice_rows(
    rows: Sequence[Mapping[str, str]],
    bucket_columns: Sequence[str],
    size_manifest: Mapping[str, Mapping[str, object]],
    higher_is_better: bool,
) -> List[str]:
    lines: List[str] = []
    grouped = grouped_rows(rows)
    for group in GROUP_ORDER:
        group_rows = grouped.get(group, [])
        if not group_rows:
            continue
        lines.append(rf"\multicolumn{{{2 + len(bucket_columns)}}}{{l}}{{\textit{{{GROUP_LABELS[group]}}}}} \\")
        lines.append(r"\midrule")
        per_bucket_style = {
            column: compute_top_two(group_rows, column, higher_is_better) for column in bucket_columns
        }
        for row in group_rows:
            model = row["model"]
            size_label = str(size_manifest[model]["size_label"])
            rendered = [wrap_model(model), size_label]
            for column in bucket_columns:
                value = maybe_float(row.get(column))
                if value is None:
                    rendered.append("--")
                    continue
                style = per_bucket_style[column].get(model)
                rendered.append(style_metric(value, style))
            lines.append(" & ".join(rendered) + r" \\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return lines


def selected_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    by_model = {row["model"]: dict(row) for row in rows}
    for group in GROUP_ORDER:
        for model in MAIN_SELECTIONS[group]:
            selected.append(by_model[model])
    return selected


def render_main_overall_table(registry_rows: Sequence[Mapping[str, str]], size_manifest: Mapping[str, Mapping[str, object]]) -> str:
    metric_columns = ["success@5", "recall@5", "ndcg@5", "mrr", "roc_auc", "eer"]
    metric_directions = {
        "success@5": True,
        "recall@5": True,
        "ndcg@5": True,
        "mrr": True,
        "roc_auc": True,
        "eer": False,
    }
    body = "\n".join(
        render_grouped_rows(selected_rows(registry_rows), metric_columns, size_manifest, metric_directions)
    )
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{5pt}}
{{\color{{editred}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrr}}
\toprule
\textbf{{Model}} & \textbf{{Model Size}} & \textbf{{S@5 $\uparrow$}} & \textbf{{R@5 $\uparrow$}} & \textbf{{nDCG@5 $\uparrow$}} & \textbf{{MRR $\uparrow$}} & \textbf{{ROC-AUC $\uparrow$}} & \textbf{{EER $\downarrow$}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
}}
\caption{{\color{{editred}}Main results on AuthBench (test split). The retrieval leaderboard now reports S@5, R@5, nDCG@5, and MRR; the verification leaderboard reports both ROC-AUC and EER. \textbf{{Bold}} entries mark the best result within each model group for a metric, and \underline{{underlined}} entries mark the second best. The full 50-model evaluation, including all three non-neural baselines, is deferred to Appendix~\ref{{sec:appendix-full-results}}.}}
\label{{tab:overall-leaderboard}}
\end{{table}}
"""


def render_main_slice_table(
    rows: Sequence[Mapping[str, str]],
    bucket_columns: Sequence[str],
    headers: Sequence[str],
    caption: str,
    label: str,
    colspec: str,
    size_manifest: Mapping[str, Mapping[str, object]],
    higher_is_better: bool,
    use_resizebox: bool = True,
) -> str:
    body = "\n".join(render_slice_rows(selected_rows(rows), bucket_columns, size_manifest, higher_is_better))
    header = " & ".join([r"\textbf{Model}", r"\textbf{Model Size}", *headers]) + r" \\"
    if use_resizebox:
        table_body = rf"""\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
}}"""
    else:
        table_body = rf"""\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}"""
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
{{\color{{editred}}
{table_body}
}}
\caption{{\color{{editred}}{caption}}}
\label{{{label}}}
\end{{table}}
"""


def render_models_table(
    registry_rows: Sequence[Mapping[str, str]],
    size_manifest: Mapping[str, Mapping[str, object]],
) -> str:
    rows = sort_models(registry_rows)
    lines = [
        r"{\color{editred}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{longtable}{@{}p{0.27\linewidth}cp{0.53\linewidth}@{}}",
        r"\caption{Models and baselines evaluated in AuthBench. We list every system appearing in the updated leaderboard together with its model size and Hugging Face repository or baseline implementation note.}",
        r"\label{tab:models-evaluated}\\",
        r"\toprule",
        r"\textbf{Model} & \textbf{Model Size} & \textbf{Repository / Implementation} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Models and baselines evaluated in AuthBench (continued)}\\",
        r"\toprule",
        r"\textbf{Model} & \textbf{Model Size} & \textbf{Repository / Implementation} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    grouped = grouped_rows(rows)
    for group in GROUP_ORDER:
        group_rows = grouped.get(group, [])
        if not group_rows:
            continue
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{GROUP_LABELS[group]}}}}} \\")
        lines.append(r"\midrule")
        for row in group_rows:
            model = row["model"]
            size_label = str(size_manifest[model]["size_label"])
            if row["model_type"] == "baseline":
                repo_text = BASELINE_DESCRIPTIONS[model]
            else:
                repo_text = tex_escape(str(row["hf_repo"]))
            lines.append(f"{tex_escape(model)} & {size_label} & {repo_text} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\end{longtable}", r"\endgroup", r"}", ""])
    return "\n".join(lines)


def render_full_results_tables(
    registry_rows: Sequence[Mapping[str, str]],
    size_manifest: Mapping[str, Mapping[str, object]],
) -> str:
    leaderboard_by_model = {row["model"]: dict(row) for row in registry_rows}
    ordered_rows = [leaderboard_by_model[row["model"]] for row in sort_models(registry_rows)]

    language_s5 = load_csv(ANALYSIS_DIR / "wide" / "language" / "raw" / "success@5.csv")
    language_eer = load_csv(ANALYSIS_DIR / "wide" / "language" / "raw" / "eer.csv")
    genre_s5 = load_csv(ANALYSIS_DIR / "wide" / "primary_genre" / "raw" / "success@5.csv")
    genre_eer = load_csv(ANALYSIS_DIR / "wide" / "primary_genre" / "raw" / "eer.csv")
    length_s5 = load_csv(ANALYSIS_DIR / "wide" / "length_bucket" / "raw" / "success@5.csv")
    length_eer = load_csv(ANALYSIS_DIR / "wide" / "length_bucket" / "raw" / "eer.csv")

    language_headers = [
        r"\textbf{Model}",
        r"\textbf{Model Size}",
        r"\textbf{ar}",
        r"\textbf{de}",
        r"\textbf{en}",
        r"\textbf{es}",
        r"\textbf{fr}",
        r"\textbf{hi}",
        r"\textbf{ja}",
        r"\textbf{ko}",
        r"\textbf{ru}",
        r"\textbf{zh}",
    ]
    genre_headers = [
        r"\textbf{Model}",
        r"\textbf{Model Size}",
        r"\textbf{blog}",
        r"\textbf{ecomm}",
        r"\textbf{literature}",
        r"\textbf{media}",
        r"\textbf{news}",
        r"\textbf{poetry}",
        r"\textbf{qna}",
        r"\textbf{research}",
        r"\textbf{social}",
    ]
    length_headers = [
        r"\textbf{Model}",
        r"\textbf{Model Size}",
        r"\textbf{short}",
        r"\textbf{medium}",
        r"\textbf{long}",
        r"\textbf{extra\_long}",
    ]

    overall_body = "\n".join(
        render_grouped_rows(
            ordered_rows,
            ["success@5", "recall@5", "ndcg@5", "mrr", "roc_auc", "eer"],
            size_manifest,
            {
                "success@5": True,
                "recall@5": True,
                "ndcg@5": True,
                "mrr": True,
                "roc_auc": True,
                "eer": False,
            },
        )
    )
    language_s5_body = "\n".join(
        render_slice_rows(language_s5, ["ar", "de", "en", "es", "fr", "hi", "ja", "ko", "ru", "zh"], size_manifest, True)
    )
    language_eer_body = "\n".join(
        render_slice_rows(language_eer, ["ar", "de", "en", "es", "fr", "hi", "ja", "ko", "ru", "zh"], size_manifest, False)
    )
    genre_s5_body = "\n".join(
        render_slice_rows(
            genre_s5,
            ["blog", "ecommerce_reviews", "literature", "media_reviews", "news", "poetry", "qna", "research_paper", "social_media"],
            size_manifest,
            True,
        )
    )
    genre_eer_body = "\n".join(
        render_slice_rows(
            genre_eer,
            ["blog", "ecommerce_reviews", "literature", "media_reviews", "news", "poetry", "qna", "research_paper", "social_media"],
            size_manifest,
            False,
        )
    )
    length_s5_body = "\n".join(
        render_slice_rows(length_s5, ["short", "medium", "long", "extra_long"], size_manifest, True)
    )
    length_eer_body = "\n".join(
        render_slice_rows(length_eer, ["short", "medium", "long", "extra_long"], size_manifest, False)
    )

    return rf"""\subsection{{Overall Leaderboard Full Results}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{longtable}}{{@{{}}p{{0.30\linewidth}}crrrrrr@{{}}}}
\caption{{Updated overall zero-shot results on AuthBench. Authorship attribution is evaluated with Success@5 (S@5), Recall@5 (R@5), nDCG@5, and MRR (higher is better). Authorship verification is evaluated with ROC-AUC (higher is better) and EER (lower is better). All 47 neural models and the three non-neural baselines are included.}}
\label{{tab:overall-leaderboard-full}}\\
\toprule
\textbf{{Model}} & \textbf{{Model Size}} & \textbf{{S@5 $\uparrow$}} & \textbf{{R@5 $\uparrow$}} & \textbf{{nDCG@5 $\uparrow$}} & \textbf{{MRR $\uparrow$}} & \textbf{{ROC-AUC $\uparrow$}} & \textbf{{EER $\downarrow$}} \\
\midrule
\endfirsthead
\caption[]{{Updated overall zero-shot results on AuthBench (continued)}}\\
\toprule
\textbf{{Model}} & \textbf{{Model Size}} & \textbf{{S@5 $\uparrow$}} & \textbf{{R@5 $\uparrow$}} & \textbf{{nDCG@5 $\uparrow$}} & \textbf{{MRR $\uparrow$}} & \textbf{{ROC-AUC $\uparrow$}} & \textbf{{EER $\downarrow$}} \\
\midrule
\endhead
\midrule
\multicolumn{{8}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{overall_body}
\end{{longtable}}
\endgroup

\subsection{{Overall Metric Bar Charts}}
\begin{{figure}}[t]
  \centering
  \safeincludegraphics[width=0.32\linewidth]{{plots/overall_success5_bar.pdf}}\hfill
  \safeincludegraphics[width=0.32\linewidth]{{plots/overall_recall5_bar.pdf}}\hfill
  \safeincludegraphics[width=0.32\linewidth]{{plots/overall_ndcg5_bar.pdf}}\\[4pt]
  \safeincludegraphics[width=0.32\linewidth]{{plots/overall_mrr_bar.pdf}}\hfill
  \safeincludegraphics[width=0.32\linewidth]{{plots/overall_roc_auc_bar.pdf}}\hfill
  \safeincludegraphics[width=0.32\linewidth]{{plots/overall_eer_bar.pdf}}
  \caption{{Updated overall performance bar charts from the new analysis outputs. The top row reports retrieval metrics (S@5, R@5, nDCG@5), and the bottom row reports MRR and the two verification metrics (ROC-AUC and EER).}}
  \label{{fig:overall-performance-bars}}
\end{{figure}}

\subsection{{Language-wise Full Results}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{@{{}}p{{0.25\linewidth}}crrrrrrrrrr@{{}}}}
\caption{{Language-wise Success@5 on AuthBench (full results).}}
\label{{tab:lang-s5-full}}\\
\toprule
{" & ".join(language_headers)} \\
\midrule
\endfirsthead
\caption[]{{Language-wise Success@5 on AuthBench (continued)}}\\
\toprule
{" & ".join(language_headers)} \\
\midrule
\endhead
\midrule
\multicolumn{{12}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{language_s5_body}
\end{{longtable}}
\endgroup

\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{@{{}}p{{0.25\linewidth}}crrrrrrrrrr@{{}}}}
\caption{{Language-wise EER on AuthBench (full results).}}
\label{{tab:lang-eer5-full}}\\
\toprule
{" & ".join(language_headers)} \\
\midrule
\endfirsthead
\caption[]{{Language-wise EER on AuthBench (continued)}}\\
\toprule
{" & ".join(language_headers)} \\
\midrule
\endhead
\midrule
\multicolumn{{12}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{language_eer_body}
\end{{longtable}}
\endgroup

\subsection{{Primary-genre Full Results}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{@{{}}p{{0.22\linewidth}}crrrrrrrrr@{{}}}}
\caption{{Primary-genre Success@5 on AuthBench (full results).}}
\label{{tab:genre-s5-full}}\\
\toprule
{" & ".join(genre_headers)} \\
\midrule
\endfirsthead
\caption[]{{Primary-genre Success@5 on AuthBench (continued)}}\\
\toprule
{" & ".join(genre_headers)} \\
\midrule
\endhead
\midrule
\multicolumn{{11}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{genre_s5_body}
\end{{longtable}}
\endgroup

\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{@{{}}p{{0.22\linewidth}}crrrrrrrrr@{{}}}}
\caption{{Primary-genre EER on AuthBench (full results).}}
\label{{tab:genre-eer5-full}}\\
\toprule
{" & ".join(genre_headers)} \\
\midrule
\endfirsthead
\caption[]{{Primary-genre EER on AuthBench (continued)}}\\
\toprule
{" & ".join(genre_headers)} \\
\midrule
\endhead
\midrule
\multicolumn{{11}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{genre_eer_body}
\end{{longtable}}
\endgroup

\subsection{{Length-bucket Full Results}}
\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{longtable}}{{@{{}}p{{0.30\linewidth}}crrrr@{{}}}}
\caption{{Length-bucket Success@5 on AuthBench (full results).}}
\label{{tab:length-s5-full}}\\
\toprule
{" & ".join(length_headers)} \\
\midrule
\endfirsthead
\caption[]{{Length-bucket Success@5 on AuthBench (continued)}}\\
\toprule
{" & ".join(length_headers)} \\
\midrule
\endhead
\midrule
\multicolumn{{6}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{length_s5_body}
\end{{longtable}}
\endgroup

\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{longtable}}{{@{{}}p{{0.30\linewidth}}crrrr@{{}}}}
\caption{{Length-bucket EER on AuthBench (full results).}}
\label{{tab:length-eer5-full}}\\
\toprule
{" & ".join(length_headers)} \\
\midrule
\endfirsthead
\caption[]{{Length-bucket EER on AuthBench (continued)}}\\
\toprule
{" & ".join(length_headers)} \\
\midrule
\endhead
\midrule
\multicolumn{{6}}{{r}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{length_eer_body}
\end{{longtable}}
\endgroup
"""


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    registry = build_model_registry()
    size_manifest = build_model_size_manifest(registry)

    leaderboard_rows = load_csv(ANALYSIS_DIR / "summary" / "leaderboard_overall.csv")
    leaderboard_rows = sort_models(leaderboard_rows)

    write_text(PAPER_DIR / "generated_main_overall_table.tex", render_main_overall_table(leaderboard_rows, size_manifest))

    language_rows = load_csv(ANALYSIS_DIR / "wide" / "language" / "raw" / "success@5.csv")
    write_text(
        PAPER_DIR / "generated_main_language_table.tex",
        render_main_slice_table(
            language_rows,
            ["ar", "de", "en", "es", "fr", "hi", "ja", "ko", "ru", "zh"],
            [
                r"\textbf{\texttt{ar}}",
                r"\textbf{\texttt{de}}",
                r"\textbf{\texttt{en}}",
                r"\textbf{\texttt{es}}",
                r"\textbf{\texttt{fr}}",
                r"\textbf{\texttt{hi}}",
                r"\textbf{\texttt{ja}}",
                r"\textbf{\texttt{ko}}",
                r"\textbf{\texttt{ru}}",
                r"\textbf{\texttt{zh}}",
            ],
            "Results by language (S@5). The updated slice table keeps a compact representative subset; complete language-wise S@5 and EER tables are reported in Appendix~\\ref{sec:appendix-full-results}.",
            "tab:lang-s10-top",
            "llrrrrrrrrrr",
            size_manifest,
            True,
        ),
    )

    genre_rows = load_csv(ANALYSIS_DIR / "wide" / "primary_genre" / "raw" / "success@5.csv")
    write_text(
        PAPER_DIR / "generated_main_genre_table.tex",
        render_main_slice_table(
            genre_rows,
            ["blog", "ecommerce_reviews", "literature", "media_reviews", "news", "poetry", "qna", "research_paper", "social_media"],
            [
                r"\textbf{\texttt{blog}}",
                r"\textbf{\texttt{ecommerce\_reviews}}",
                r"\textbf{\texttt{literature}}",
                r"\textbf{\texttt{media\_reviews}}",
                r"\textbf{\texttt{news}}",
                r"\textbf{\texttt{poetry}}",
                r"\textbf{\texttt{qna}}",
                r"\textbf{\texttt{research\_paper}}",
                r"\textbf{\texttt{social\_media}}",
            ],
            "Results by primary genre (S@5). Complete genre-wise S@5 and EER tables are reported in Appendix~\\ref{sec:appendix-full-results}.",
            "tab:genre-s10-top",
            "llrrrrrrrrr",
            size_manifest,
            True,
        ),
    )

    length_rows = load_csv(ANALYSIS_DIR / "wide" / "length_bucket" / "raw" / "success@5.csv")
    write_text(
        PAPER_DIR / "generated_main_length_table.tex",
        render_main_slice_table(
            length_rows,
            ["short", "medium", "long", "extra_long"],
            [
                r"\textbf{\texttt{short}}",
                r"\textbf{\texttt{medium}}",
                r"\textbf{\texttt{long}}",
                r"\textbf{\texttt{extra\_long}}",
            ],
            "Results by length bucket (S@5). Complete length-wise S@5 and EER tables are reported in Appendix~\\ref{sec:appendix-full-results}.",
            "tab:length-s10-top",
            "llrrrr",
            size_manifest,
            True,
            use_resizebox=False,
        ),
    )

    write_text(PAPER_DIR / "generated_models_table.tex", render_models_table(leaderboard_rows, size_manifest))
    write_text(PAPER_DIR / "generated_results_tables.tex", render_full_results_tables(leaderboard_rows, size_manifest))


if __name__ == "__main__":
    main()
