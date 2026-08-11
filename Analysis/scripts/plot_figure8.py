#!/usr/bin/env python3
"""Recreate the Figure 8 latency/throughput/energy comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    ANALYSIS_ROOT,
    DEFAULT_OUTPUT_DIR,
    append_average,
    dataset_labels,
    load_json,
    require_matplotlib,
    write_csv,
)


METHOD_LABELS = {
    "weak_baseline": "Weak baseline",
    "strong_baseline": "Strong baseline",
    "ours": "Ours",
}
METRIC_LABELS = {
    "latency_s": "Latency (s)",
    "throughput_tokens_s": "Throughput (token/s)",
    "energy_j_per_token": "Energy (J/token)",
}
COLORS = {
    "weak_baseline": "#d3d3d3",
    "strong_baseline": "#9ecae1",
    "ours": "#5b9bd5",
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


def csv_rows(data: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    datasets = [*data["dataset_order"], "average"]
    for config, config_data in data["figure8"]["configs"].items():
        for metric, methods in config_data["metrics"].items():
            for method, values in methods.items():
                for dataset, value in zip(datasets, append_average(values)):
                    rows.append(
                        {
                            "figure": "Figure 8",
                            "config": config,
                            "metric": metric,
                            "method": method,
                            "dataset": dataset,
                            "value": f"{value:.6f}",
                        }
                    )
    return rows


def plot_tri_bar(
    ax,
    x_positions: list[int],
    labels: list[str],
    weak_data: list[float],
    strong_data: list[float],
    ours_data: list[float],
    ylabel: str,
    y_max: float,
    is_higher_better: bool = False,
    weak_text_dy: float = 0.0,
    strong_text_position: str = "above",
) -> None:
    tri_w = 0.24
    ax.bar(
        [x - tri_w for x in x_positions],
        weak_data,
        tri_w,
        label=METHOD_LABELS["weak_baseline"],
        color=COLORS["weak_baseline"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x_positions,
        strong_data,
        tri_w,
        label=METHOD_LABELS["strong_baseline"],
        color=COLORS["strong_baseline"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        [x + tri_w for x in x_positions],
        ours_data,
        tri_w,
        label=METHOD_LABELS["ours"],
        color=COLORS["ours"],
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, y_max)

    for i, x_value in enumerate(x_positions):
        weak = weak_data[i]
        strong = strong_data[i]
        ours = ours_data[i]
        ax.hlines(
            ours,
            x_value - tri_w * 1.2,
            x_value + tri_w * 1.2,
            colors="red",
            linestyles="--",
            linewidth=1.0,
            alpha=0.8,
        )

        x1 = x_value - tri_w * 1.15
        ax.annotate(
            "",
            xy=(x1, ours),
            xytext=(x1, weak),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.1),
        )
        weak_ratio = ours / weak if is_higher_better else weak / ours
        strong_ratio = ours / strong if is_higher_better else strong / ours
        ax.text(
            x1 + 0.03,
            (weak + ours) / 2 + weak_text_dy,
            f"{weak_ratio:.2f}x",
            color="red",
            ha="left",
            va="center",
            fontsize=8,
        )

        x2 = x_value + tri_w * 0.1
        ax.annotate(
            "",
            xy=(x2, ours),
            xytext=(x2, strong),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.1),
        )
        y_offset = max(0.35, y_max * 0.025)
        if strong_text_position == "below":
            ax.text(
                x2 + 0.02,
                max(ours - y_offset, y_offset * 0.4),
                f"{strong_ratio:.2f}x",
                color="red",
                ha="left",
                va="top",
                fontsize=8,
            )
        else:
            ax.text(
                x2 + 0.02,
                (strong + ours) / 2,
                f"{strong_ratio:.2f}x",
                color="red",
                ha="left",
                va="center",
                fontsize=8,
            )


def figure8_series(data: dict, config: str, metric: str, method: str) -> list[float]:
    return append_average(data["figure8"]["configs"][config]["metrics"][metric][method])


def plot(data: dict, output_dir: Path) -> None:
    plt = require_matplotlib()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1

    labels = dataset_labels(data, include_average=True)
    x_positions = list(range(len(labels)))
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))

    for col_index, config in enumerate(("Config1", "Config2")):
        lat_y_max = 30 if config == "Config1" else 20
        energy_y_max = 16 if config == "Config1" else 12
        strong_text_position = "below" if config == "Config2" else "above"
        weak_text_dy = 1.2 if config == "Config1" else 0.45

        plot_tri_bar(
            axes[0][col_index],
            x_positions,
            labels,
            figure8_series(data, config, "latency_s", "weak_baseline"),
            figure8_series(data, config, "latency_s", "strong_baseline"),
            figure8_series(data, config, "latency_s", "ours"),
            METRIC_LABELS["latency_s"],
            y_max=lat_y_max,
            weak_text_dy=weak_text_dy,
            strong_text_position=strong_text_position,
        )
        axes[0][col_index].set_title(
            data["figure8"]["configs"][config]["title"], fontsize=16
        )

        plot_tri_bar(
            axes[1][col_index],
            x_positions,
            labels,
            figure8_series(data, config, "throughput_tokens_s", "weak_baseline"),
            figure8_series(data, config, "throughput_tokens_s", "strong_baseline"),
            figure8_series(data, config, "throughput_tokens_s", "ours"),
            "Throughput (t/s)",
            y_max=160,
            is_higher_better=True,
            weak_text_dy=-15.0 if config == "Config1" else -8.0,
        )

        plot_tri_bar(
            axes[2][col_index],
            x_positions,
            labels,
            figure8_series(data, config, "energy_j_per_token", "weak_baseline"),
            figure8_series(data, config, "energy_j_per_token", "strong_baseline"),
            figure8_series(data, config, "energy_j_per_token", "ours"),
            "Energy (J/Tkn)",
            y_max=energy_y_max,
            weak_text_dy=0.7 if config == "Config1" else 0.25,
        )

    handles, labels_for_legend = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_for_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        edgecolor="black",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure8.png", dpi=300)
    fig.savefig(output_dir / "figure8.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_json(args.metrics)
    rows = csv_rows(data)
    write_csv(
        args.output_dir / "figure8_metrics.csv",
        rows,
        ["figure", "config", "metric", "method", "dataset", "value"],
    )
    if not args.no_plot:
        plot(data, args.output_dir)
    print(f"Wrote Figure 8 data to {args.output_dir / 'figure8_metrics.csv'}")
    if not args.no_plot:
        print(f"Wrote Figure 8 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
