import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib import rcParams


METRICS = [
    ("loss", "Loss"),
    ("success@5", "S@5"),
    ("recall@5", "R@5"),
    ("ndcg@5", "nDCG@5"),
    ("eer", "EER"),
]
BASE_GROUPS = ["by_language", "by_genre", "by_length_bucket"]
GROUPS = BASE_GROUPS + ["by_primary_genre"]
ROLLING_WINDOW = 50
EXCLUDED_MODELS = {
    "bge-large-en-v1.5",
    "e5-large-v2",
    "qwen3-embedding-0.6b",
}
ALIGN_MODEL = "multilingual-e5-large"

rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "stix",
        "font.serif": ["Times New Roman"],
    }
)


def style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)


def load_loss_history(summary_path):
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("loss_history", [])
    steps = [item["step"] for item in history if "step" in item and "loss" in item]
    values = [item["loss"] for item in history if "step" in item and "loss" in item]
    return steps, values


def load_eval_metrics(history_path):
    metrics = {name: [] for name, _ in METRICS if name != "loss"}
    grouped = {
        group_name: {name: {} for name, _ in METRICS if name != "loss"}
        for group_name in GROUPS
    }
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("split") != "dev":
                continue
            step = record.get("step")
            if step is None:
                continue
            representation = record.get("representation", {})
            attribution = record.get("attribution", {})
            if "success@5" in representation:
                metrics["success@5"].append((step, representation["success@5"]))
            if "recall@5" in representation:
                metrics["recall@5"].append((step, representation["recall@5"]))
            if "ndcg@5" in representation:
                metrics["ndcg@5"].append((step, representation["ndcg@5"]))
            if "eer" in attribution:
                metrics["eer"].append((step, attribution["eer"]))
            for group_name in BASE_GROUPS:
                rep_group = representation.get(group_name, {})
                for category, values in rep_group.items():
                    if "success@5" in values:
                        grouped[group_name]["success@5"].setdefault(category, []).append(
                            (step, values["success@5"])
                        )
                    if "recall@5" in values:
                        grouped[group_name]["recall@5"].setdefault(category, []).append(
                            (step, values["recall@5"])
                        )
                    if "ndcg@5" in values:
                        grouped[group_name]["ndcg@5"].setdefault(category, []).append(
                            (step, values["ndcg@5"])
                        )
                attr_group = attribution.get(group_name, {})
                for category, values in attr_group.items():
                    if "eer" in values:
                        grouped[group_name]["eer"].setdefault(category, []).append(
                            (step, values["eer"])
                        )
            primary_rep = aggregate_primary_genres(representation.get("by_genre", {}))
            primary_attr = aggregate_primary_genres(attribution.get("by_genre", {}))
            for category, values in primary_rep.items():
                if "success@5" in values:
                    grouped["by_primary_genre"]["success@5"].setdefault(category, []).append(
                        (step, values["success@5"])
                    )
                if "recall@5" in values:
                    grouped["by_primary_genre"]["recall@5"].setdefault(category, []).append(
                        (step, values["recall@5"])
                    )
                if "ndcg@5" in values:
                    grouped["by_primary_genre"]["ndcg@5"].setdefault(category, []).append(
                        (step, values["ndcg@5"])
                    )
            for category, values in primary_attr.items():
                if "eer" in values:
                    grouped["by_primary_genre"]["eer"].setdefault(category, []).append(
                        (step, values["eer"])
                    )
            for metric_key in ("success@5", "recall@5", "ndcg@5"):
                vals = [metrics.get(metric_key) for metrics in primary_rep.values()]
                vals = [value for value in vals if isinstance(value, (int, float))]
                if vals:
                    grouped["by_primary_genre"][metric_key].setdefault("macro_avg", []).append(
                        (step, sum(vals) / len(vals))
                    )
            vals = [metrics.get("eer") for metrics in primary_attr.values()]
            vals = [value for value in vals if isinstance(value, (int, float))]
            if vals:
                grouped["by_primary_genre"]["eer"].setdefault("macro_avg", []).append(
                    (step, sum(vals) / len(vals))
                )
    return metrics, grouped


def ensure_sorted(series):
    series = sorted(series, key=lambda item: item[0])
    by_step = {}
    for step, value in series:
        by_step[step] = value
    steps = sorted(by_step.keys())
    values = [by_step[step] for step in steps]
    return steps, values


def infer_reference_steps(series_map, target_model):
    counts = {}
    for model_name, (steps, _) in series_map.items():
        if model_name == target_model:
            continue
        key = tuple(steps)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return set(max(counts.items(), key=lambda item: item[1])[0])


def filter_series_steps(series, allowed_steps):
    if not allowed_steps:
        return series
    steps, values = series
    filtered = [(step, value) for step, value in zip(steps, values) if step in allowed_steps]
    if not filtered:
        return ([], [])
    return ([step for step, _ in filtered], [value for _, value in filtered])


def common_step_cap(series_map):
    max_steps = [max(steps) for steps, _ in series_map.values() if steps]
    if not max_steps:
        return None
    return min(max_steps)


def clip_series_map(series_map, max_step):
    if max_step is None:
        return series_map
    clipped = {}
    for name, (steps, values) in series_map.items():
        filtered = [(step, value) for step, value in zip(steps, values) if step <= max_step]
        if not filtered:
            continue
        clipped[name] = (
            [step for step, _ in filtered],
            [value for _, value in filtered],
        )
    return clipped


def primary_genre(genre):
    return str(genre).split("/", 1)[0]


def aggregate_primary_genres(genre_block):
    aggregated = {}
    for genre, metrics in genre_block.items():
        if not isinstance(metrics, dict):
            continue
        primary = primary_genre(genre)
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            aggregated.setdefault(primary, {}).setdefault(key, []).append(float(value))
    collapsed = {}
    for primary, metrics in aggregated.items():
        collapsed[primary] = {key: sum(values) / len(values) for key, values in metrics.items()}
    return collapsed


def rolling_mean_std(values, window):
    window = max(1, window)
    means = []
    stds = []
    running = deque()
    sum_vals = 0.0
    sum_sq = 0.0
    for value in values:
        running.append(value)
        sum_vals += value
        sum_sq += value * value
        if len(running) > window:
            removed = running.popleft()
            sum_vals -= removed
            sum_sq -= removed * removed
        count = len(running)
        mean = sum_vals / count
        variance = max((sum_sq / count) - (mean * mean), 0.0)
        std = variance ** 0.5
        means.append(mean)
        stds.append(std)
    return means, stds


def metric_slug(metric_name):
    return metric_name.replace("@", "").replace("/", "_")


def category_slug(category):
    return category.replace("/", "_")


def baseline_from_series(series_map):
    values = [value for _, values in series_map.values() for value in values]
    if not values:
        return 0.0
    min_val = min(values)
    max_val = max(values)
    pad = max((max_val - min_val) * 0.05, 0.001)
    return min_val - pad


def bounds_from_series(series_map):
    values = [value for _, values in series_map.values() for value in values]
    if not values:
        return 0.0, 1.0
    min_val = min(values)
    max_val = max(values)
    pad = max((max_val - min_val) * 0.05, 0.001)
    return min_val - pad, max_val + pad


def step_bounds(series_map):
    steps = [step for steps, _ in series_map.values() for step in steps]
    if not steps:
        return 0.0, 1.0
    min_step = min(steps)
    max_step = max(steps)
    pad = max((max_step - min_step) * 0.02, 1.0)
    return min_step - pad, max_step + pad


def step_ticks(series_map):
    steps = [step for steps, _ in series_map.values() for step in steps]
    if not steps:
        return [0]
    min_step = int(min(steps))
    max_step = int(max(steps))
    if min_step == max_step:
        return [min_step]

    target_ticks = 6
    span = max_step - min_step
    step_size = max(1, int(round(span / (target_ticks - 1))))
    ticks = list(range(min_step, max_step + 1, step_size))
    if ticks[-1] != max_step:
        ticks.append(max_step)
    return ticks


def legend_loc(metric_name):
    if metric_name in {"success@5", "recall@5", "ndcg@5"}:
        return "lower right"
    return "upper right"


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir()
        and (path / "training_summary.json").exists()
        and path.name not in EXCLUDED_MODELS
    ]
    if not model_dirs:
        raise SystemExit("No training summaries found.")

    model_dirs.sort(key=lambda path: path.name)

    series_by_metric = {name: {} for name, _ in METRICS}
    grouped_by_model = {model_dir.name: {} for model_dir in model_dirs}

    for model_dir in model_dirs:
        model_name = model_dir.name
        summary_path = model_dir / "training_summary.json"
        history_path = model_dir / "eval_history.jsonl"

        loss_steps, loss_values = load_loss_history(summary_path)
        if loss_steps:
            series_by_metric["loss"][model_name] = (loss_steps, loss_values)

        if history_path.exists():
            eval_metrics, grouped_metrics = load_eval_metrics(history_path)
            for metric_name, series in eval_metrics.items():
                if series:
                    series_by_metric[metric_name][model_name] = ensure_sorted(series)
            grouped_by_model[model_name] = {
                group_name: {
                    metric_name: {
                        category: ensure_sorted(series)
                        for category, series in categories.items()
                    }
                    for metric_name, categories in metrics.items()
                }
                for group_name, metrics in grouped_metrics.items()
            }

    allowed_steps_by_metric = {}
    for metric_name, _ in METRICS:
        if metric_name == "loss":
            continue
        series_map = series_by_metric.get(metric_name, {})
        if ALIGN_MODEL not in series_map:
            continue
        allowed_steps = infer_reference_steps(series_map, ALIGN_MODEL)
        if not allowed_steps:
            continue
        allowed_steps_by_metric[metric_name] = allowed_steps
        filtered_series = filter_series_steps(series_map[ALIGN_MODEL], allowed_steps)
        if filtered_series[0]:
            series_map[ALIGN_MODEL] = filtered_series
        else:
            del series_map[ALIGN_MODEL]

    if ALIGN_MODEL in grouped_by_model:
        for _, metrics in grouped_by_model[ALIGN_MODEL].items():
            for metric_name, categories in metrics.items():
                allowed_steps = allowed_steps_by_metric.get(metric_name)
                if not allowed_steps:
                    continue
                for category, series in list(categories.items()):
                    filtered_series = filter_series_steps(series, allowed_steps)
                    if filtered_series[0]:
                        categories[category] = filtered_series
                    else:
                        del categories[category]

    panel_color = "#eef7f1"
    metric_step_caps = {
        metric_name: common_step_cap(series_by_metric.get(metric_name, {}))
        for metric_name, _ in METRICS
    }

    for metric_name, title in METRICS:
        data = clip_series_map(series_by_metric.get(metric_name, {}), metric_step_caps[metric_name])
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor(panel_color)
        style_axes(ax)

        baseline, top = bounds_from_series(data)
        x_min, x_max = step_bounds(data)
        x_ticks = step_ticks(data)
        if metric_name == "loss":
            loss_series = {}
            for model_name, (steps, values) in data.items():
                means, stds = rolling_mean_std(values, ROLLING_WINDOW)
                loss_series[model_name] = (steps, means)
                line = ax.plot(steps, means, label=model_name, linewidth=3.8, zorder=3)
                color = line[0].get_color()
                upper = [m + s for m, s in zip(means, stds)]
                lower = [m - s for m, s in zip(means, stds)]
                ax.fill_between(steps, lower, upper, color=color, alpha=0.15, zorder=1)
                line[0].set_path_effects(
                    [pe.Stroke(linewidth=8, foreground=color, alpha=0.35), pe.Normal()]
                )
            baseline, top = bounds_from_series(loss_series)
            ax.set_ylim(bottom=baseline, top=top)
            ax.set_xlim(left=x_min, right=x_max)
        else:
            for model_name, (steps, values) in data.items():
                line = ax.plot(steps, values, label=model_name, linewidth=3.8, zorder=3)
                color = line[0].get_color()
                ax.fill_between(steps, values, baseline, color=color, alpha=0.15, zorder=1)
                line[0].set_path_effects(
                    [pe.Stroke(linewidth=8, foreground=color, alpha=0.35), pe.Normal()]
                )
            ax.set_ylim(bottom=baseline, top=top)
            ax.set_xlim(left=x_min, right=x_max)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=0)
        ax.tick_params(axis="x", labelsize=27)
        ax.tick_params(axis="y", labelsize=27)
        ax.set_xlabel("Training Steps", fontsize=32)
        ax.set_ylabel(title, fontsize=32)
        ax.legend(loc=legend_loc(metric_name), ncol=2, framealpha=0.9, fontsize=20)
        fig.tight_layout()

        output_path = output_dir / f"{metric_slug(metric_name)}_curve.pdf"
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)

    grouped_series = {group: {metric: {} for metric, _ in METRICS if metric != "loss"} for group in GROUPS}
    for model_name, group_metrics in grouped_by_model.items():
        for group_name, metrics in group_metrics.items():
            for metric_name, categories in metrics.items():
                for category, series in categories.items():
                    grouped_series[group_name][metric_name].setdefault(category, {})[model_name] = series

    for group_name, metrics in grouped_series.items():
        for metric_name, categories in metrics.items():
            for category, series_map in categories.items():
                clipped_series_map = clip_series_map(series_map, metric_step_caps[metric_name])
                if not clipped_series_map:
                    continue
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_facecolor(panel_color)
                style_axes(ax)

                baseline, top = bounds_from_series(clipped_series_map)
                x_min, x_max = step_bounds(clipped_series_map)
                x_ticks = step_ticks(clipped_series_map)
                for model_name, (steps, values) in clipped_series_map.items():
                    line = ax.plot(steps, values, label=model_name, linewidth=3.8, zorder=3)
                    color = line[0].get_color()
                    ax.fill_between(steps, values, baseline, color=color, alpha=0.15, zorder=1)
                    line[0].set_path_effects(
                        [pe.Stroke(linewidth=8, foreground=color, alpha=0.35), pe.Normal()]
                    )

                ax.set_ylim(bottom=baseline, top=top)
                ax.set_xlim(left=x_min, right=x_max)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks, rotation=0)
                ax.tick_params(axis="x", labelsize=27)
                ax.tick_params(axis="y", labelsize=27)
                ax.set_xlabel("Training Steps", fontsize=32)
                ax.set_ylabel(metric_name, fontsize=32)
                ax.legend(loc=legend_loc(metric_name), ncol=2, framealpha=0.9, fontsize=20)
                fig.tight_layout()

                group_dir = output_dir / group_name / metric_slug(metric_name)
                group_dir.mkdir(parents=True, exist_ok=True)
                output_path = group_dir / f"{category_slug(category)}.pdf"
                fig.savefig(output_path, facecolor=fig.get_facecolor())
                plt.close(fig)


if __name__ == "__main__":
    main()

    """
    Example usage:
    python eval/results/training_summary/plot_training_curves.py
    """
