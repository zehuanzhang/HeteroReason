#!/usr/bin/env python3
"""Compare portable outputs with the packaged hardware reference metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("Config1", "with_bt", "config1_with_bt.json"),
)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HARDWARE_ROOT / "generated_results",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=HARDWARE_ROOT / "reference_results/simulator_reference.json",
    )
    args = parser.parse_args()

    reference = load_json(args.reference)
    for config, mode, filename in CASES:
        generated = load_json(args.results_dir / filename)
        expected = reference["cases"][config][mode]

        if generated["config"] != config:
            raise SystemExit(f"Config name mismatch in {filename}")
        if generated["semantics"] != expected["semantics"]:
            raise SystemExit(f"Simulator semantics mismatch in {filename}")
        if generated["parameters_ms"] != expected["parameters_ms"]:
            raise SystemExit(f"Parameter mismatch in {filename}")
        if generated["component_checks_ms"] != expected["component_checks_ms"]:
            raise SystemExit(f"Component check mismatch in {filename}")
        if generated["datasets"] != expected["datasets"]:
            raise SystemExit(f"Dataset metric mismatch in {filename}")

        print(f"PASS {config} {mode}: 4 datasets match the reference")

    print("Portable simulator regression passed: all hardware cases match.")


if __name__ == "__main__":
    main()
