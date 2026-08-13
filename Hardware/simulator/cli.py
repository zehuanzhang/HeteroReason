"""Command-line interface for the portable hardware simulator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .core import PerformanceConfig, SpeculativeSimulator, build_dataset_metrics
from .trace_io import DATASETS, load_trace_dataset


def run_datasets(
    config_path: Path, input_root: Path, datasets: Iterable[str]
) -> Dict[str, Any]:
    config = PerformanceConfig.from_json(config_path)
    simulator = SpeculativeSimulator(config)
    selected_datasets = list(datasets)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "config": config.name,
        "semantics": config.semantics,
        "dataset_order": selected_datasets,
        "parameters_ms": config.parameters_dict(),
        "component_checks_ms": {
            "draft_kv0_n10": simulator.get_draft_lat(0, 10),
            "prm_n1000": simulator.get_prm_lat(1000),
        },
        "datasets": {},
    }

    for dataset in selected_datasets:
        logs, metadata = load_trace_dataset(input_root / dataset)
        simulation = simulator.simulate_dataset(logs)
        result["datasets"][dataset] = build_dataset_metrics(simulation, metadata)

    return result


def write_json(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "config",
        "dataset",
        "examples",
        "events",
        "total_generated_tokens",
        "total_latency_s",
        "average_latency_ms",
        "throughput_tokens_s",
        "backtracking_latency_s",
        "draft_dominated_steps",
        "prm_dominated_steps",
        "hidden_latency_gain_s",
        "stall_overhead_s",
        "target_trace_events_used",
        "target_total_tokens",
        "target_critical_tokens",
        "target_output_prm_events",
        "target_output_prm_latency_s",
        "target_output_prm_critical_latency_s",
        "input_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in result["dataset_order"]:
            metrics = result["datasets"][dataset]
            writer.writerow(
                {
                    "config": result["config"],
                    "dataset": dataset,
                    **{field: metrics.get(field, "") for field in fields[2:]},
                }
            )


def print_summary(result: Dict[str, Any]) -> None:
    print(f"Portable simulator: {result['config']} ({result['semantics']})")
    print(
        "Dataset           Avg latency (s)   Throughput (raw token/s)"
    )
    for dataset in result["dataset_order"]:
        metrics = result["datasets"][dataset]
        print(
            f"{dataset:<17} "
            f"{metrics['average_latency_ms'] / 1000:>15.3f} "
            f"{metrics['throughput_tokens_s']:>26.2f}"
        )
    trace_events = sum(
        result["datasets"][dataset].get("target_trace_events_used", 0)
        for dataset in result["dataset_order"]
    )
    if trace_events:
        print(f"Target trace events used: {trace_events}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the portable HeteroReason hardware simulator."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per dataset.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Run one dataset; repeat the option for multiple datasets.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset if args.dataset else DATASETS
    result = run_datasets(args.config, args.input_root, datasets)

    if args.output is not None:
        write_json(args.output, result)
    if args.csv is not None:
        write_csv(args.csv, result)
    if not args.quiet:
        print_summary(result)
        if args.output is not None:
            print(f"JSON: {args.output}")
        if args.csv is not None:
            print(f"CSV:  {args.csv}")


if __name__ == "__main__":
    main()
