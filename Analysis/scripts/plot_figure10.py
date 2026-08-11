#!/usr/bin/env python3
"""Recreate the Figure 10 ablation plot."""

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


STAGES = {
    "baseline": "Baseline",
    "baseline_t2": "Baseline+T2",
    "baseline_t2_t1": "Baseline+T2+T1\n(HeteroReason)",
    "baseline_t2_t1_no_backtracking": "Baseline+T2+T1\n(HeteroReason without backtracking)",
}
COLORS = {
    "baseline": "#808080",
    "baseline_t2": "#a2d9b1",
    "baseline_t2_t1": "#f7d49e",
    "baseline_t2_t1_no_backtracking": "#5b9bd5",
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
    for config, stages in data["figure10"]["latency_s"].items():
        for stage, values in stages.items():
            for dataset, latency in zip(data["dataset_order"], values):
                rows.append(
                    {
                        "figure": "Figure 10",
                        "config": config,
                        "dataset": dataset,
                        "stage": stage,
                        "value_type": "latency_s",
                        "value": f"{latency:.6f}",
                    }
                )
        stage_names = list(STAGES)
        for prev_stage, next_stage in zip(stage_names, stage_names[1:]):
            for dataset, prev_latency, next_latency in zip(
                data["dataset_order"], stages[prev_stage], stages[next_stage]
            ):
                rows.append(
                    {
                        "figure": "Figure 10",
                        "config": config,
                        "dataset": dataset,
                        "stage": f"{prev_stage}_to_{next_stage}",
                        "value_type": "adjacent_speedup",
                        "value": f"{prev_latency / next_latency:.6f}",
                    }
                )
    return rows


def add_speedup_arrow(
    ax,
    x_pos: float,
    y_start: float,
    y_end: float,
    text_offset: float = 0.08,
    y_offset_ratio: float = 0.95,
    fontsize: int = 11,
    text_dy: float = 0.0,
) -> None:
    speedup = y_start / y_end
    min_len = ax.get_ylim()[1] * 0.06
    actual_end = y_end + 0.2
    actual_start = max(y_start * y_offset_ratio, actual_end + min_len)

    ax.annotate(
        "",
        xy=(x_pos, actual_end),
        xytext=(x_pos, actual_start),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5, ls="-"),
    )
    text_y = (actual_start + actual_end) / 2 + text_dy
    ax.text(
        x_pos + text_offset,
        text_y,
        f"{speedup:.2f}x",
        color="red",
        fontsize=fontsize,
        fontweight="bold",
        va="center",
    )


def add_speedup_text(
    ax,
    x_pos: float,
    y_start: float,
    y_end: float,
    text_offset: float = 0.02,
    y_offset: float = 0.45,
    fontsize: int = 10,
) -> None:
    speedup = y_start / y_end
    ax.text(
        x_pos + text_offset,
        max(y_start, y_end) + y_offset,
        f"{speedup:.2f}x",
        color="red",
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
    )


def plot_case4(
    ax,
    x_positions: list[int],
    labels: list[str],
    values: dict[str, list[float]],
    title: str,
) -> tuple:
    width = 0.22
    ax.set_ylim(0, 30)
    b1 = ax.bar(
        [x - 1.5 * width for x in x_positions],
        values["baseline"],
        width,
        color=COLORS["baseline"],
        edgecolor="black",
        linewidth=0.8,
    )
    b2 = ax.bar(
        [x - 0.5 * width for x in x_positions],
        values["baseline_t2"],
        width,
        color=COLORS["baseline_t2"],
        edgecolor="black",
        linewidth=0.8,
    )
    b3 = ax.bar(
        [x + 0.5 * width for x in x_positions],
        values["baseline_t2_t1"],
        width,
        color=COLORS["baseline_t2_t1"],
        edgecolor="black",
        linewidth=0.8,
    )
    b4 = ax.bar(
        [x + 1.5 * width for x in x_positions],
        values["baseline_t2_t1_no_backtracking"],
        width,
        color=COLORS["baseline_t2_t1_no_backtracking"],
        edgecolor="black",
        linewidth=0.8,
    )

    for i, x_value in enumerate(x_positions):
        current_y_offset = 0.85 if i in (0, 1) else 0.95
        arrow_shift = 0.07
        add_speedup_arrow(
            ax,
            x_value - 1.0 * width + arrow_shift,
            values["baseline"][i],
            values["baseline_t2"][i],
            text_offset=0.055,
            y_offset_ratio=current_y_offset,
            fontsize=10,
        )
        mid_text_dy = 0.75 if i == 1 else 0.0
        add_speedup_arrow(
            ax,
            x_value + arrow_shift,
            values["baseline_t2"][i],
            values["baseline_t2_t1"][i],
            text_offset=0.055,
            y_offset_ratio=0.95,
            fontsize=10,
            text_dy=mid_text_dy,
        )
        if i == 1:
            add_speedup_text(
                ax,
                x_value + 1.0 * width + arrow_shift,
                values["baseline_t2_t1"][i],
                values["baseline_t2_t1_no_backtracking"][i],
                text_offset=0.01,
                y_offset=0.25,
                fontsize=10,
            )
        else:
            add_speedup_arrow(
                ax,
                x_value + 1.0 * width + arrow_shift,
                values["baseline_t2_t1"][i],
                values["baseline_t2_t1_no_backtracking"][i],
                text_offset=0.055,
                y_offset_ratio=0.95,
                fontsize=10,
            )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return b1, b2, b3, b4


def plot_case4_compact(
    ax,
    x_positions: list[int],
    labels: list[str],
    values: dict[str, list[float]],
    title: str,
) -> tuple:
    width = 0.22
    ax.set_ylim(0, 20)
    bars = []
    offsets = {
        "baseline": -1.5 * width,
        "baseline_t2": -0.5 * width,
        "baseline_t2_t1": 0.5 * width,
        "baseline_t2_t1_no_backtracking": 1.5 * width,
    }
    for stage in STAGES:
        bars.append(
            ax.bar(
                [x + offsets[stage] for x in x_positions],
                values[stage],
                width,
                color=COLORS[stage],
                edgecolor="black",
                linewidth=0.8,
            )
        )

    for i, x_value in enumerate(x_positions):
        add_speedup_arrow(
            ax,
            x_value - 1.0 * width + 0.07,
            values["baseline"][i],
            values["baseline_t2"][i],
            text_offset=0.055,
            y_offset_ratio=0.90,
            fontsize=10,
        )
        mid_text_dy = 0.35 if i == 1 else 0.0
        final_text_dy = -0.25 if i == 1 else 0.0
        add_speedup_arrow(
            ax,
            x_value + 0.07,
            values["baseline_t2"][i],
            values["baseline_t2_t1"][i],
            text_offset=0.055,
            y_offset_ratio=0.95,
            fontsize=10,
            text_dy=mid_text_dy,
        )
        add_speedup_arrow(
            ax,
            x_value + 1.0 * width + 0.07,
            values["baseline_t2_t1"][i],
            values["baseline_t2_t1_no_backtracking"][i],
            text_offset=0.055,
            y_offset_ratio=0.95,
            fontsize=10,
            text_dy=final_text_dy,
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return tuple(bars)


def plot(data: dict, output_dir: Path) -> None:
    plt = require_matplotlib()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16

    labels = dataset_labels(data)
    x_positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    c1_values = data["figure10"]["latency_s"]["Config1"]
    c2_values = data["figure10"]["latency_s"]["Config2"]
    h1, h2, h3, h4 = plot_case4(
        axes[0],
        x_positions,
        labels,
        c1_values,
        "(a) Config 1 Ablation with Speedup",
    )
    plot_case4_compact(
        axes[1],
        x_positions,
        labels,
        c2_values,
        "(b) Config 2 Ablation with Speedup",
    )
    axes[0].set_ylabel("Latency (s)", fontsize=18, fontweight="bold")
    axes[1].set_ylabel("Latency (s)", fontsize=18, fontweight="bold")

    fig.legend(
        [h1, h2, h3, h4],
        [STAGES[stage] for stage in STAGES],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=True,
        edgecolor="black",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure10.png", dpi=300)
    fig.savefig(output_dir / "figure10.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_json(args.metrics)
    rows = compute_rows(data)
    write_csv(
        args.output_dir / "figure10_ablation.csv",
        rows,
        ["figure", "config", "dataset", "stage", "value_type", "value"],
    )
    if not args.no_plot:
        plot(data, args.output_dir)
    print(f"Wrote Figure 10 data to {args.output_dir / 'figure10_ablation.csv'}")
    if not args.no_plot:
        print(f"Wrote Figure 10 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
