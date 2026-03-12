from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-model comparison summaries.")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison_files = sorted(args.comparison_dir.glob("*.json"))
    rows: List[Dict[str, object]] = []

    for path in comparison_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload["model"]
        self_consistency_details = payload.get("self_consistency_details") or {}
        for metric_name, metric_payload in payload.get("metrics", {}).items():
            row = {
                "model": model,
                "metric": metric_name,
                "baseline": metric_payload.get("baseline"),
                "self_consistency": metric_payload.get("self_consistency"),
                "delta": metric_payload.get("delta"),
                "improved": metric_payload.get("improved"),
                "direction": metric_payload.get("direction"),
                "aggregation_strategy": self_consistency_details.get("aggregation_strategy"),
                "total_votes": self_consistency_details.get("total_votes"),
                "include_original": self_consistency_details.get("include_original"),
                "comparison_file": str(path),
            }
            rows.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "metric",
                "baseline",
                "self_consistency",
                "delta",
                "improved",
                "direction",
                "aggregation_strategy",
                "total_votes",
                "include_original",
                "comparison_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote aggregated comparison JSON to {args.output_json}")
    print(f"Wrote aggregated comparison CSV to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
