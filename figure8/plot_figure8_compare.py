#!/usr/bin/env python3
"""Plot the normalized four-bar Figure 8 comparison.

The script combines the metrics JSON files produced by the existing Figure 8
script.  Weak and Strong are taken from the U280 run and checked against the
V80 run; the two Ours bars are taken from their respective runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_U280_METRICS = (
    REPO_ROOT
    / "figure8"
    / "inputs"
    / "u280_metrics.json"
)
DEFAULT_V80_METRICS = (
    REPO_ROOT
    / "figure8"
    / "inputs"
    / "v80_metrics.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

DATASETS = ["math500", "gsm8k", "gaokao2023en", "olympiadbench"]
DATASET_LABELS = {
    "math500": "Math500",
    "gsm8k": "GSM8K",
    "gaokao2023en": "Gaokao23",
    "olympiadbench": "Olympiad",
    "average": "Avg",
}
CONFIGS = ["Config1", "Config2"]
METRICS = [
    "avg_latency_s",
    "throughput_tokens_s",
    "avg_energy_kj_per_problem",
]
METRIC_LABELS = {
    "avg_latency_s": ("Latency (s)", "Normalized Latency (Weak=1)", False),
    "throughput_tokens_s": (
        "Goodput Throughput (token/s)",
        "Normalized Goodput (Weak=1)",
        True,
    ),
    "avg_energy_kj_per_problem": (
        "Energy (kJ/problem)",
        "Normalized Energy (Weak=1)",
        False,
    ),
}

METHODS = ["weak_baseline", "strong_baseline", "ours_u280", "ours_v80"]
METHOD_LABELS = {
    "weak_baseline": "Weak baseline",
    "strong_baseline": "Strong baseline",
    "ours_u280": "Ours (U280 projected)",
    "ours_v80": "Ours (V80 projected)",
}
METHOD_COLORS = {
    "weak_baseline": "#d3d3d3",
    "strong_baseline": "#9ecae1",
    "ours_u280": "#5b9bd5",
    "ours_v80": "#2f75b5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u280-metrics", type=Path, default=DEFAULT_U280_METRICS)
    parser.add_argument("--v80-metrics", type=Path, default=DEFAULT_V80_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Normalize every metric to the Weak baseline for each config/dataset.",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not args.normalized:
        parser.error("This packaged comparison entry point only supports --normalized.")
    return args


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for config in CONFIGS:
        if config not in data.get("configs", {}):
            raise ValueError(f"Missing {config} in {path}")
    return data


def metric_record(record: Dict[str, Any]) -> Dict[str, float]:
    return {metric: float(record[metric]) for metric in METRICS}


def compare_baselines(
    u280: Dict[str, float],
    v80: Dict[str, float],
    config: str,
    dataset: str,
    warnings: List[str],
) -> None:
    for method in ("weak_baseline", "strong_baseline"):
        for metric in METRICS:
            left = float(u280[method][metric])
            right = float(v80[method][metric])
            if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
                warnings.append(
                    f"{config}/{dataset}/{method}/{metric}: U280={left} differs from V80={right}"
                )


def build_metrics(u280_path: Path, v80_path: Path) -> Dict[str, Any]:
    u280 = load_metrics(u280_path)
    v80 = load_metrics(v80_path)
    merged: Dict[str, Any] = {
        "schema_version": 1,
        "description": (
            "Figure 8 comparison of Weak baseline, Strong baseline, "
            "Ours (U280 projected), and Ours (V80 projected)."
        ),
        "sources": {"u280": str(u280_path), "v80": str(v80_path)},
        "configs": {},
    }

    for config in CONFIGS:
        config_u280 = u280["configs"][config]
        config_v80 = v80["configs"][config]
        config_metrics: Dict[str, Any] = {
            "title": config_u280["title"],
            "datasets": {},
            "mean_of_datasets": {},
            "pooled_total": {},
            "warnings": [],
        }

        for dataset in DATASETS:
            u280_records = config_u280["datasets"][dataset]
            v80_records = config_v80["datasets"][dataset]
            compare_baselines(
                u280_records,
                v80_records,
                config,
                dataset,
                config_metrics["warnings"],
            )
            config_metrics["datasets"][dataset] = {
                "weak_baseline": metric_record(u280_records["weak_baseline"]),
                "strong_baseline": metric_record(u280_records["strong_baseline"]),
                "ours_u280": metric_record(u280_records["ours"]),
                "ours_v80": metric_record(v80_records["ours"]),
            }

        for scope in ("mean_of_datasets", "pooled_total"):
            u280_scope = config_u280[scope]
            v80_scope = config_v80[scope]
            if scope == "mean_of_datasets":
                compare_baselines(
                    u280_scope,
                    v80_scope,
                    config,
                    "average",
                    config_metrics["warnings"],
                )
            config_metrics[scope] = {
                "weak_baseline": metric_record(u280_scope["weak_baseline"]),
                "strong_baseline": metric_record(u280_scope["strong_baseline"]),
                "ours_u280": metric_record(u280_scope["ours"]),
                "ours_v80": metric_record(v80_scope["ours"]),
            }

        merged["configs"][config] = config_metrics

    return merged


def values_for_plot(
    metrics: Dict[str, Any],
    config: str,
    method: str,
    metric: str,
) -> List[float]:
    values = [
        float(metrics["configs"][config]["datasets"][dataset][method][metric])
        for dataset in DATASETS
    ]
    values.append(
        float(metrics["configs"][config]["mean_of_datasets"][method][metric])
    )
    return values


def normalize_to_weak(method_values: Dict[str, List[float]]) -> Dict[str, List[float]]:
    weak_values = method_values["weak_baseline"]
    return {
        method: [
            value / weak if weak > 0.0 else math.nan
            for value, weak in zip(values, weak_values)
        ]
        for method, values in method_values.items()
    }


def add_ratio_arrows(
    ax: Any,
    x_positions: List[int],
    method_values: Dict[str, List[float]],
    bar_width: float,
    higher_better: bool,
) -> None:
    """Compare each projected Ours result with both Weak and Strong."""
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    text_shift = y_span * 0.012
    target_specs = {
        "ours_u280": {
            "line_left": -1.82,
            "line_right": 0.90,
            "arrow_offsets": (-1.55, -0.65),
        },
        "ours_v80": {
            "line_left": -1.82,
            "line_right": 1.90,
            "arrow_offsets": (0.55, 1.45),
        },
    }
    for index, xpos in enumerate(x_positions):
        weak = method_values["weak_baseline"][index]
        strong = method_values["strong_baseline"][index]
        for target_method, spec in target_specs.items():
            target = method_values[target_method][index]
            if target <= 0.0:
                continue
            ax.hlines(
                target,
                xpos + spec["line_left"] * bar_width,
                xpos + spec["line_right"] * bar_width,
                colors="red",
                linestyles="--",
                linewidth=0.8,
                alpha=0.8,
            )
            for baseline, arrow_offset in zip(
                (weak, strong), spec["arrow_offsets"]
            ):
                if baseline <= 0.0:
                    continue
                arrow_x = xpos + arrow_offset * bar_width
                ratio = target / baseline if higher_better else baseline / target
                ax.annotate(
                    "",
                    xy=(arrow_x, target),
                    xytext=(arrow_x, baseline),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.85),
                )
                mid_y = (baseline + target) / 2.0
                if abs(baseline - target) < y_span * 0.035:
                    mid_y = max(baseline, target) + text_shift
                mid_y = min(max(mid_y, y_min + text_shift), y_max - text_shift)
                ax.text(
                    arrow_x + 0.010,
                    mid_y,
                    f"{ratio:.2f}x",
                    color="red",
                    ha="left",
                    va="center",
                    fontsize=6.0,
                    clip_on=True,
                )


def plot(metrics: Dict[str, Any], output_dir: Path, normalized: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [DATASET_LABELS[dataset] for dataset in DATASETS] + [
        DATASET_LABELS["average"]
    ]
    x_positions = list(range(len(labels)))
    bar_width = 0.19
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.0))

    for row_index, metric in enumerate(METRICS):
        ylabel, normalized_ylabel, higher_better = METRIC_LABELS[metric]
        for col_index, config in enumerate(CONFIGS):
            ax = axes[row_index][col_index]
            method_values = {
                method: values_for_plot(metrics, config, method, metric)
                for method in METHODS
            }
            if normalized:
                method_values = normalize_to_weak(method_values)

            offsets = (-1.5, -0.5, 0.5, 1.5)
            for offset, method in zip(offsets, METHODS):
                ax.bar(
                    [x + offset * bar_width for x in x_positions],
                    method_values[method],
                    bar_width,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    edgecolor="black",
                    linewidth=0.6,
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(normalized_ylabel if normalized else ylabel, fontsize=13)
            ax.tick_params(axis="y", labelsize=10.5)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(metrics["configs"][config]["title"], fontsize=14, fontweight="bold")

            all_values = [value for values in method_values.values() for value in values]
            finite_values = [value for value in all_values if math.isfinite(value)]
            y_top = max(finite_values) * (1.20 if higher_better else 1.16)
            if normalized:
                y_top = max(y_top, 1.18)
            ax.set_ylim(0.0, y_top)
            add_ratio_arrows(
                ax,
                x_positions,
                method_values,
                bar_width,
                higher_better,
            )

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=True,
        edgecolor="black",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = "figure8_projected_compare_normalized" if normalized else "figure8_projected_compare"
    fig.savefig(output_dir / f"{output_stem}.png", dpi=300)
    fig.savefig(output_dir / f"{output_stem}.pdf")
    plt.close(fig)


def metric_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for config in CONFIGS:
        config_metrics = metrics["configs"][config]
        for dataset in DATASETS:
            for method in METHODS:
                record = config_metrics["datasets"][dataset][method]
                rows.append(
                    {
                        "row_type": "dataset",
                        "config": config,
                        "dataset": dataset,
                        "method": method,
                        **{metric: f"{record[metric]:.9f}" for metric in METRICS},
                    }
                )
        for method in METHODS:
            record = config_metrics["mean_of_datasets"][method]
            rows.append(
                {
                    "row_type": "mean_of_datasets",
                    "config": config,
                    "dataset": "average",
                    "method": method,
                    **{metric: f"{record[metric]:.9f}" for metric in METRICS},
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["row_type", "config", "dataset", "method", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(metrics: Dict[str, Any]) -> None:
    for config in CONFIGS:
        print(f"\n{config}")
        print(
            "%22s %12s %12s %14s"
            % ("method", "avg_lat_s", "goodput_t/s", "kJ/problem")
        )
        print("-" * 66)
        for method in METHODS:
            record = metrics["configs"][config]["pooled_total"][method]
            print(
                "%22s %12.4f %12.2f %14.4f"
                % (
                    METHOD_LABELS[method],
                    record["avg_latency_s"],
                    record["throughput_tokens_s"],
                    record["avg_energy_kj_per_problem"],
                )
            )
        warnings = metrics["configs"][config]["warnings"]
        for warning in warnings:
            print(f"WARNING: {warning}")


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args.u280_metrics, args.v80_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(metrics)
    output_stem = "figure8_projected_compare_normalized" if args.normalized else "figure8_projected_compare"
    json_path = args.output_dir / f"{output_stem}_metrics.json"
    csv_path = args.output_dir / f"{output_stem}_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    if not args.no_plot:
        plot(metrics, args.output_dir, args.normalized)
    print_summary(metrics)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    if not args.no_plot:
        print(f"Wrote {args.output_dir / f'{output_stem}.png'}")
        print(f"Wrote {args.output_dir / f'{output_stem}.pdf'}")


if __name__ == "__main__":
    main()
