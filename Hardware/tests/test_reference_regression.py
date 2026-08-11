from __future__ import annotations

import json
import unittest
from pathlib import Path

from simulator.cli import run_datasets
from simulator.trace_io import DATASETS


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("Config1", "with_bt", "config1.json", "with_bt"),
    ("Config1", "no_bt", "config1_no_bt.json", "no_bt"),
)


class ReferenceRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reference_path = (
            HARDWARE_ROOT / "reference_results/simulator_reference.json"
        )
        cls.reference = json.loads(reference_path.read_text(encoding="utf-8"))

    def test_all_dataset_cases_match_reference(self) -> None:
        for config, mode, config_file, input_mode in CASES:
            with self.subTest(config=config, mode=mode):
                generated = run_datasets(
                    HARDWARE_ROOT / "simulator/configs" / config_file,
                    HARDWARE_ROOT / "inputs" / config / input_mode,
                    DATASETS,
                )
                expected = self.reference["cases"][config][mode]
                self.assertEqual(generated["semantics"], expected["semantics"])
                self.assertEqual(generated["parameters_ms"], expected["parameters_ms"])
                self.assertEqual(
                    generated["component_checks_ms"], expected["component_checks_ms"]
                )
                self.assertEqual(generated["datasets"], expected["datasets"])


if __name__ == "__main__":
    unittest.main()
