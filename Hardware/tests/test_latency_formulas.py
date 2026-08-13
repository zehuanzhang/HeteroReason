from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.core import PerformanceConfig, SpeculativeSimulator


class LatencyFormulaTest(unittest.TestCase):
    def setUp(self) -> None:
        config = PerformanceConfig(
            name="test",
            draft_alpha=1.0,
            draft_beta=0.1,
            prm_alpha=2.0,
            prm_beta=0.2,
            target_pre_beta=0.3,
            target_alpha=4.0,
            target_gen_beta=0.4,
            target_scale=0.5,
            bandwidth_gb_s=1.0,
            bytes_per_token=1,
        )
        self.simulator = SpeculativeSimulator(config)

    def test_draft_token_level_accumulation(self) -> None:
        self.assertAlmostEqual(self.simulator.get_draft_lat(10, 3), 6.6)

    def test_prm_parallel_prefill(self) -> None:
        self.assertAlmostEqual(self.simulator.get_prm_lat(10), 4.0)

    def test_target_prefill_and_decode(self) -> None:
        self.assertAlmostEqual(self.simulator.get_target_lat(2, 2), 11.4)

    def test_communication_latency(self) -> None:
        self.assertAlmostEqual(self.simulator.get_communication_lat(5), 0.000005)


if __name__ == "__main__":
    unittest.main()
