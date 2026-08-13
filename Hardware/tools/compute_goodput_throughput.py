#!/usr/bin/env python3
"""Compute goodput throughput from packaged simulator traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("math500", "gsm8k", "gaokao2023en", "olympiadbench")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sum_goodput_from_trace_dir(trace_dir: Path) -> Dict[str, int]:
    examples = 0
    final_draft_tokens = 0
    final_target_tokens = 0
    third_field_tokens = 0
    raw_draft_tokens = 0
    raw_target_tokens = 0

    files = sorted(trace_dir.glob("example_*.json"))
    if not files:
        raise ValueError(f"No example_*.json files found in {trace_dir}")

    for path in files:
        sample = read_json(path)
        examples += 1
        breakdown = sample.get("summary", {}).get("total_tokens_breakdown")
        if not breakdown or len(breakdown) < 2:
            raise ValueError(f"Missing summary.total_tokens_breakdown in {path}")
        final_draft_tokens += int(breakdown[0] or 0)
        final_target_tokens += int(breakdown[1] or 0)
        if len(breakdown) >= 3:
            third_field_tokens += int(breakdown[2] or 0)

        for step in sample.get("execution_trace", []):
            raw_draft_tokens += int(step.get("draft_tokens_spent", 0) or 0)
            raw_target_tokens += int(step.get("target_total_tokens", 0) or 0)

    return {
        "examples": examples,
        "final_draft_tokens": final_draft_tokens,
        "final_target_tokens": final_target_tokens,
        "third_field_tokens": third_field_tokens,
        "goodput_tokens": final_draft_tokens + final_target_tokens,
        "raw_draft_tokens": raw_draft_tokens,
        "raw_target_tokens": raw_target_tokens,
    }


def latency_from_simulator(sim_path: Path, dataset: str) -> Dict[str, float]:
    result = read_json(sim_path)
    if "cases" in result:
        result = result["cases"]["Config1"]["with_bt"]
    if dataset not in result.get("datasets", {}):
        raise ValueError(f"Dataset {dataset!r} missing in {sim_path}")
    metrics = result["datasets"][dataset]
    return {
        "average_latency_s": float(metrics["average_latency_ms"]) / 1000.0,
        "total_latency_s": float(metrics["total_latency_s"]),
    }


def build_row(
    method: str,
    dataset: str,
    token_stats: Dict[str, int],
    latency_stats: Dict[str, float],
) -> Dict[str, Any]:
    total_latency_s = latency_stats["total_latency_s"]
    return {
        "dataset": dataset,
        "method": method,
        "examples": token_stats["examples"],
        "final_draft_tokens": token_stats["final_draft_tokens"],
        "final_target_tokens": token_stats["final_target_tokens"],
        "goodput_tokens": token_stats["goodput_tokens"],
        "total_latency_s": total_latency_s,
        "average_latency_s": latency_stats["average_latency_s"],
        "goodput_tokens_s": token_stats["goodput_tokens"] / total_latency_s,
        "raw_draft_tokens": token_stats["raw_draft_tokens"],
        "raw_draft_tokens_s": token_stats["raw_draft_tokens"] / total_latency_s,
        "raw_target_tokens": token_stats["raw_target_tokens"],
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "examples",
        "final_draft_tokens",
        "final_target_tokens",
        "goodput_tokens",
        "total_latency_s",
        "average_latency_s",
        "goodput_tokens_s",
        "raw_draft_tokens",
        "raw_draft_tokens_s",
        "raw_target_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_rows(rows: List[Dict[str, Any]]) -> None:
    print(
        "Dataset          Method   Goodput tokens   Avg latency (s)   "
        "Goodput tok/s"
    )
    for row in rows:
        print(
            f"{row['dataset']:<16} "
            f"{row['method']:<8} "
            f"{row['goodput_tokens']:>14} "
            f"{row['average_latency_s']:>17.3f} "
            f"{row['goodput_tokens_s']:>15.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute throughput using final trajectory tokens: "
            "summary.total_tokens_breakdown[0] + [1]."
        )
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=HARDWARE_ROOT / "inputs/Config1/with_bt",
    )
    parser.add_argument(
        "--ours-sim",
        type=Path,
        default=HARDWARE_ROOT / "reference_results/simulator_reference.json",
        help=(
            "Simulator output used as the system latency source. "
            "This may also point to reference_results/simulator_reference.json."
        ),
    )
    parser.add_argument(
        "--method",
        default="ours",
        help="Label to write in the method column.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Dataset to report; repeat for multiple datasets. Defaults to all four.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=HARDWARE_ROOT / "generated_results/config1_goodput_throughput.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset or DATASETS
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        token_stats = sum_goodput_from_trace_dir(args.trace_root / dataset)
        rows.append(
            build_row(
                args.method,
                dataset,
                token_stats,
                latency_from_simulator(args.ours_sim, dataset),
            )
        )

    print_rows(rows)
    write_csv(args.csv, rows)
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
