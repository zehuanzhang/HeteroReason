"""Small helpers shared by the analysis scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "outputs"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_average(values: list[float]) -> list[float]:
    return [*values, sum(values) / len(values)]


def dataset_labels(data: dict[str, Any], include_average: bool = False) -> list[str]:
    labels = [data["dataset_labels"][name] for name in data["dataset_order"]]
    if include_average:
        labels.append("Avg")
    return labels


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
