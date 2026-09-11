#!/usr/bin/env python3
"""Summarize communication overhead without modifying the latency simulator.

The script consumes the baseline and communication-enabled simulator outputs.
It reports raw communication shares, exposed latency shares, and the ratio of
the communication-plus-PRM path to the next draft in ordinary overlap pairs.
"""

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path


DEFAULT_DATASETS = ("math500", "gsm8k", "gaokao2023en", "olympiadbench")
DEFAULT_FIXED_LATENCY_MS = 0.060
DEFAULT_BANDWIDTH_GBPS = 5.0
DEFAULT_INDEX_BYTES = 4


def parse_csv_list(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_int_list(value):
    try:
        values = [int(item) for item in parse_csv_list(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("config values must be integers") from exc
    return values


def parse_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return number


def build_parser():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=project_dir,
        help="AE_hardware directory (default: script parent)",
    )
    parser.add_argument("--configs", type=parse_int_list, default=[1, 2])
    parser.add_argument(
        "--datasets",
        type=parse_csv_list,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="baseline simulator_latency directory",
    )
    parser.add_argument(
        "--communication-dir",
        type=Path,
        default=None,
        help="communication_latency directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for summary and detailed outputs",
    )
    parser.add_argument(
        "--fixed-latency-ms",
        type=parse_float,
        default=DEFAULT_FIXED_LATENCY_MS,
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=parse_float,
        default=DEFAULT_BANDWIDTH_GBPS,
    )
    parser.add_argument(
        "--index-bytes",
        type=parse_float,
        default=DEFAULT_INDEX_BYTES,
    )
    return parser


def resolve_path(path, project_dir):
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_dir / path


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_float(row, field, path):
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError("missing {} in {}".format(field, path))
    return float(value)


def communication_time_ms(draft_output_tokens, fixed_latency_ms, bandwidth_gbps, index_bytes):
    token_count = max(float(draft_output_tokens), 0.0)
    if token_count <= 0.0:
        return 0.0
    payload_bytes = token_count * index_bytes
    payload_time_ms = payload_bytes / (bandwidth_gbps * 1e9) * 1000.0
    return fixed_latency_ms + payload_time_ms


def grouped_steps(rows):
    groups = OrderedDict()
    for row in rows:
        problem = row.get("problem_ordinal", "")
        groups.setdefault(problem, []).append(row)
    return groups


def percentile(values, fraction):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values):
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "fraction_gt_1": float("nan"),
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "fraction_gt_1": sum(value > 1.0 for value in values) / len(values),
    }


def read_total(path):
    rows = read_rows(path)
    try:
        return next(row for row in rows if row.get("problem_ordinal") == "TOTAL")
    except StopIteration as exc:
        raise ValueError("TOTAL row not found in {}".format(path)) from exc


def analyze_one(
    config,
    dataset,
    baseline_dir,
    communication_dir,
    fixed_latency_ms,
    bandwidth_gbps,
    index_bytes,
):
    baseline_path = baseline_dir / (
        "config{}_{}_target_overlap_per_problem.csv".format(config, dataset)
    )
    new_problem_path = communication_dir / "config{}".format(config) / (
        "{}_per_problem.csv".format(dataset)
    )
    new_step_path = communication_dir / "config{}".format(config) / (
        "{}_per_step.csv".format(dataset)
    )

    baseline_total = read_total(baseline_path)
    new_total = read_total(new_problem_path)
    step_rows = read_rows(new_step_path)

    communication_times = []
    for row in step_rows:
        communication_times.append(
            communication_time_ms(
                require_float(row, "draft_output_tokens", new_step_path),
                fixed_latency_ms,
                bandwidth_gbps,
                index_bytes,
            )
        )
    communication_total_ms = sum(communication_times)

    new_latency = {
        method: require_float(new_total, method, new_problem_path)
        for method in ("our", "weak", "strong")
    }
    baseline_latency = {
        method: require_float(baseline_total, method, baseline_path)
        for method in ("our", "weak", "strong")
    }

    raw_shares = {
        method: communication_total_ms / new_latency[method]
        if new_latency[method]
        else float("nan")
        for method in ("our", "weak", "strong")
    }
    exposed_shares = {
        method: (new_latency[method] - baseline_latency[method]) / new_latency[method]
        if new_latency[method]
        else float("nan")
        for method in ("our", "weak", "strong")
    }

    ratio_details = []
    groups = grouped_steps(step_rows)
    for problem_ordinal, steps in groups.items():
        for previous, current in zip(steps, steps[1:]):
            # In the ordinary overlap branch, the previous PRM path is
            # communication + PRM and is compared with the next draft.
            if require_float(previous, "target_latency_ms", new_step_path) > 0.0:
                continue

            previous_comm_ms = communication_time_ms(
                require_float(previous, "draft_output_tokens", new_step_path),
                fixed_latency_ms,
                bandwidth_gbps,
                index_bytes,
            )
            previous_prm_ms = require_float(
                previous, "prm_latency_ms", new_step_path
            )
            path_ms = previous_comm_ms + previous_prm_ms
            ours_draft_ms = require_float(current, "fpga_draft_ms", new_step_path)
            strong_draft_ms = require_float(current, "draft_latency_ms", new_step_path)
            if ours_draft_ms <= 0.0 or strong_draft_ms <= 0.0:
                continue

            ratio_details.append({
                "config": config,
                "dataset": dataset,
                "problem_ordinal": problem_ordinal,
                "previous_step": previous.get("step", ""),
                "current_step": current.get("step", ""),
                "communication_ms": previous_comm_ms,
                "previous_prm_ms": previous_prm_ms,
                "communication_plus_prm_ms": path_ms,
                "ours_draft_ms": ours_draft_ms,
                "strong_draft_ms": strong_draft_ms,
                "ours_ratio": path_ms / ours_draft_ms,
                "strong_ratio": path_ms / strong_draft_ms,
            })

    ours_ratios = [row["ours_ratio"] for row in ratio_details]
    strong_ratios = [row["strong_ratio"] for row in ratio_details]
    ours_summary = summarize_values(ours_ratios)
    strong_summary = summarize_values(strong_ratios)

    summary = {
        "config": config,
        "dataset": dataset,
        "problems": len(groups),
        "steps": len(step_rows),
        "communication_total_ms": communication_total_ms,
        "communication_mean_ms_per_step": communication_total_ms / len(step_rows)
        if step_rows
        else float("nan"),
        "ours_latency_ms": new_latency["our"],
        "weak_latency_ms": new_latency["weak"],
        "strong_latency_ms": new_latency["strong"],
        "raw_share_ours": raw_shares["our"],
        "raw_share_weak": raw_shares["weak"],
        "raw_share_strong": raw_shares["strong"],
        "exposed_share_ours": exposed_shares["our"],
        "exposed_share_weak": exposed_shares["weak"],
        "exposed_share_strong": exposed_shares["strong"],
        "weak_delta_check_ms": new_latency["weak"] - baseline_latency["weak"],
        "weak_delta_minus_communication_ms": (
            new_latency["weak"] - baseline_latency["weak"] - communication_total_ms
        ),
        "ordinary_overlap_pairs": len(ratio_details),
        "ours_comm_prm_over_draft_mean": ours_summary["mean"],
        "ours_comm_prm_over_draft_median": ours_summary["median"],
        "ours_comm_prm_over_draft_p95": ours_summary["p95"],
        "ours_bottleneck_fraction": ours_summary["fraction_gt_1"],
        "strong_comm_prm_over_draft_mean": strong_summary["mean"],
        "strong_comm_prm_over_draft_median": strong_summary["median"],
        "strong_comm_prm_over_draft_p95": strong_summary["p95"],
        "strong_bottleneck_fraction": strong_summary["fraction_gt_1"],
    }
    return summary, ratio_details


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_for_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = build_parser()
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    baseline_dir = resolve_path(args.baseline_dir, project_dir)
    communication_dir = resolve_path(args.communication_dir, project_dir)
    output_dir = resolve_path(args.output_dir, project_dir)
    if baseline_dir is None:
        baseline_dir = project_dir / "outputs" / "simulator_latency"
    if communication_dir is None:
        communication_dir = project_dir / "outputs" / "communication_latency"
    if output_dir is None:
        output_dir = project_dir / "outputs" / "communication_analysis"

    summaries = []
    details = []
    for config in args.configs:
        for dataset in args.datasets:
            summary, ratio_details = analyze_one(
                config,
                dataset,
                baseline_dir,
                communication_dir,
                args.fixed_latency_ms,
                args.bandwidth_gbps,
                args.index_bytes,
            )
            summaries.append(summary)
            details.extend(ratio_details)

    summary_fields = list(summaries[0].keys()) if summaries else []
    detail_fields = [
        "config",
        "dataset",
        "problem_ordinal",
        "previous_step",
        "current_step",
        "communication_ms",
        "previous_prm_ms",
        "communication_plus_prm_ms",
        "ours_draft_ms",
        "strong_draft_ms",
        "ours_ratio",
        "strong_ratio",
    ]
    summary_path = output_dir / "communication_overhead_summary.csv"
    detail_path = output_dir / "communication_overlap_details.csv"
    json_path = output_dir / "communication_overhead_summary.json"
    write_csv(summary_path, summaries, summary_fields)
    write_csv(detail_path, details, detail_fields)
    json_payload = [
        {key: clean_for_json(value) for key, value in row.items()}
        for row in summaries
    ]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)

    print("config dataset problems steps comm_ms raw_ours raw_weak raw_strong exposed_ours exposed_weak exposed_strong ours_ratio_gt1 strong_ratio_gt1")
    for row in summaries:
        print(
            "{config} {dataset} {problems} {steps} {communication_total_ms:.3f} "
            "{raw_share_ours:.6f} {raw_share_weak:.6f} {raw_share_strong:.6f} "
            "{exposed_share_ours:.6f} {exposed_share_weak:.6f} "
            "{exposed_share_strong:.6f} {ours_bottleneck_fraction:.6f} "
            "{strong_bottleneck_fraction:.6f}".format(**row)
        )
    print("\nWrote {}".format(summary_path))
    print("Wrote {}".format(detail_path))
    print("Wrote {}".format(json_path))


if __name__ == "__main__":
    main()
