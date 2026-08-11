#!/usr/bin/env python3
"""Recreate the Figure 9 RSD-relative latency speedup plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    ANALYSIS_ROOT,
    DEFAULT_OUTPUT_DIR,
    dataset_labels,
    load_json,
    require_matplotlib,
    write_csv,
)


SERIES = {
    "three_gpus": "3 GPUs",
    "ours": "Ours",
}
COLORS = {
    "three_gpus": "#a6cee3",
    "ours": "#1f78b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ANALYSIS_ROOT / "reference_results/figure_metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def compute_rows(data: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config, latencies in data["figure9"]["latency_s"].items():
        rsd = latencies["rsd"]
        for method in SERIES:
            for dataset, base_latency, system_latency in zip(
                data["dataset_order"], rsd, latencies[method]
            ):
                rows.append(
                    {
                        "figure": "Figure 9",
                        "config": config,
                        "dataset": dataset,
                        "method": method,
                        "rsd_latency_s": f"{base_latency:.6f}",
                        "system_latency_s": f"{system_latency:.6f}",
                        "speedup_over_rsd": f"{base_latency / system_latency:.6f}",
                    }
                )
    return rows


def speedup_values(rows: list[dict[str, object]], config: str, method: str) -> list[float]:
    return [
        float(row["speedup_over_rsd"])
        for row in rows
        if row["config"] == config and row["method"] == method
    ]


def plot_speedup_case(
    ax,
    x_positions: list[int],
    labels: list[str],
    three_gpu_speedup: list[float],
    ours_speedup: list[float],
    title: str,
    rsd_label_pos: tuple[float, float],
):
    width = 0.35
    b1 = ax.bar(
        [x - width / 2 for x in x_positions],
        three_gpu_speedup,
        width,
        label=SERIES["three_gpus"],
        color=COLORS["three_gpus"],
        edgecolor="black",
        linewidth=0.8,
    )
    b2 = ax.bar(
        [x + width / 2 for x in x_positions],
        ours_speedup,
        width,
        label=SERIES["ours"],
        color=COLORS["ours"],
        edgecolor="black",
        linewidth=0.8,
    )
    line = ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label="RSD Baseline",
    )
    ax.text(
        rsd_label_pos[0],
        rsd_label_pos[1],
        "RSD",
        color="red",
        fontsize=16,
        fontweight="bold",
        ha="left",
    )

    for bars in (b1, b2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=25)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylabel("Speedup over RSD (x)", fontsize=18, fontweight="bold")
    y_max = max(max(three_gpu_speedup), max(ours_speedup), 2.0)
    ax.set_ylim(0, y_max * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return b1, b2, line


def plot(data: dict, rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = require_matplotlib()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    labels = dataset_labels(data)
    x_positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    h1, h2, hl = plot_speedup_case(
        axes[0],
        x_positions,
        labels,
        speedup_values(rows, "Config1", "three_gpus"),
        speedup_values(rows, "Config1", "ours"),
        "(a) Config 1 Speedup Comparison",
        rsd_label_pos=(-0.45, 1.25),
    )
    plot_speedup_case(
        axes[1],
        x_positions,
        labels,
        speedup_values(rows, "Config2", "three_gpus"),
        speedup_values(rows, "Config2", "ours"),
        "(b) Config 2 Speedup Comparison",
        rsd_label_pos=(-0.45, 1.10),
    )

    fig.legend(
        [h1, h2, hl],
        ["3 GPUs", "Ours", "RSD Baseline"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        edgecolor="black",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure9.png", dpi=300)
    fig.savefig(output_dir / "figure9.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_json(args.metrics)
    rows = compute_rows(data)
    write_csv(
        args.output_dir / "figure9_speedup.csv",
        rows,
        [
            "figure",
            "config",
            "dataset",
            "method",
            "rsd_latency_s",
            "system_latency_s",
            "speedup_over_rsd",
        ],
    )
    if not args.no_plot:
        plot(data, rows, args.output_dir)
    print(f"Wrote Figure 9 data to {args.output_dir / 'figure9_speedup.csv'}")
    if not args.no_plot:
        print(f"Wrote Figure 9 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
