#!/usr/bin/env python3
"""Plot the four-bar Figure 8 with reference lines at Weak and Strong."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "plot_figure8_compare.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("figure8_compare_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_baseline_ratio_arrows(
    ax: Any,
    x_positions: List[int],
    method_values: Dict[str, List[float]],
    bar_width: float,
    higher_better: bool,
) -> None:
    """Draw Weak/Strong reference lines and compare both Ours bars to them."""
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    text_shift = y_span * 0.012
    target_specs = {
        # Keep both comparison arrows visually attached to their Ours bar.
        "ours_u280": (0.30, 0.70),
        "ours_v80": (1.30, 1.70),
    }

    for index, xpos in enumerate(x_positions):
        weak = method_values["weak_baseline"][index]
        strong = method_values["strong_baseline"][index]

        for baseline in (weak, strong):
            if baseline <= 0.0 or not math.isfinite(baseline):
                continue
            ax.hlines(
                baseline,
                xpos - 1.82 * bar_width,
                xpos + 1.90 * bar_width,
                colors="red",
                linestyles="--",
                linewidth=0.8,
                alpha=0.8,
            )

        for target_method, arrow_offsets in target_specs.items():
            target = method_values[target_method][index]
            if target <= 0.0 or not math.isfinite(target):
                continue

            for baseline, arrow_offset in zip((weak, strong), arrow_offsets):
                if baseline <= 0.0 or not math.isfinite(baseline):
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
                if target_method == "ours_u280":
                    text_x = arrow_x - 0.010
                    horizontal_alignment = "right"
                else:
                    text_x = arrow_x + 0.010
                    horizontal_alignment = "left"
                ax.text(
                    text_x,
                    mid_y,
                    f"{ratio:.2f}x",
                    color="red",
                    ha=horizontal_alignment,
                    va="center",
                    fontsize=6.0,
                    clip_on=True,
                )


def main() -> None:
    base = load_base_module()
    base.DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
    base.add_ratio_arrows = add_baseline_ratio_arrows
    base.main()


if __name__ == "__main__":
    main()
