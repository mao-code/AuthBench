from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


LOWER_IS_BETTER_METRICS = {"attribution.eer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs self-consistency eval outputs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--self-consistency", type=Path, required=True, dest="self_consistency")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_single_result(path: Path) -> tuple[str, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"Expected exactly one model result in {path}, found {len(payload)}.")
    model_name, result = next(iter(payload.items()))
    return model_name, result


def extract_overall_metrics(result: Dict[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for section in ("representation", "attribution"):
        section_metrics = result.get(section)
        if not isinstance(section_metrics, dict):
            continue
        for key, value in section_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{section}.{key}"] = float(value)
    return metrics


def main() -> int:
    args = parse_args()
    baseline_model, baseline_result = load_single_result(args.baseline)
    self_consistency_model, self_consistency_result = load_single_result(args.self_consistency)
    if baseline_model != self_consistency_model:
        raise ValueError(
            f"Model mismatch: baseline={baseline_model}, self_consistency={self_consistency_model}"
        )

    baseline_metrics = extract_overall_metrics(baseline_result)
    self_consistency_metrics = extract_overall_metrics(self_consistency_result)
    metric_names = sorted(set(baseline_metrics) | set(self_consistency_metrics))

    comparison_metrics: Dict[str, Dict[str, object]] = {}
    for metric_name in metric_names:
        baseline_value = baseline_metrics.get(metric_name)
        self_consistency_value = self_consistency_metrics.get(metric_name)
        if baseline_value is None or self_consistency_value is None:
            comparison_metrics[metric_name] = {
                "baseline": baseline_value,
                "self_consistency": self_consistency_value,
                "delta": None,
                "improved": None,
            }
            continue
        delta = self_consistency_value - baseline_value
        lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
        improved = delta < 0 if lower_is_better else delta > 0
        comparison_metrics[metric_name] = {
            "baseline": baseline_value,
            "self_consistency": self_consistency_value,
            "delta": delta,
            "improved": improved,
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        }

    summary = {
        "model": baseline_model,
        "baseline_file": str(args.baseline),
        "self_consistency_file": str(args.self_consistency),
        "metrics": comparison_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote comparison summary to {args.output}")
    for metric_name, payload in comparison_metrics.items():
        delta = payload.get("delta")
        if delta is None:
            continue
        print(
            f"{baseline_model} | {metric_name}: "
            f"baseline={payload['baseline']:.6f}, "
            f"self_consistency={payload['self_consistency']:.6f}, "
            f"delta={delta:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
