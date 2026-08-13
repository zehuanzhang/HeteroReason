from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.core import PerformanceConfig, SpeculativeSimulator


class SyntheticTraceTest(unittest.TestCase):
    def test_all_status_paths_without_target_trace(self) -> None:
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
        simulator = SpeculativeSimulator(config)
        logs = [
            {
                "example_id": 7,
                "steps": [
                    {"tokens": 2, "status": "accept_pure_draft"},
                    {"tokens": 5, "target_tokens": 4, "status": "target_rescue"},
                    {"tokens": 3, "status": "BACKTRACK_EVENT"},
                ],
            }
        ]

        result = simulator.simulate_dataset(logs)

        self.assertAlmostEqual(result["total"], 57.100008)
        self.assertAlmostEqual(result["backtrack_total"], 27.200003)
        self.assertEqual(result["draft_dominated"], 1)
        self.assertEqual(result["prm_dominated"], 0)
        self.assertAlmostEqual(result["hidden_gain"], 5.099998)
        self.assertEqual(result["stall_overhead"], 0.0)
        self.assertEqual(result["details"], [{"id": 7, "lat": 57.1}])

    def test_v11_critical_tokens_advance_kv_with_total_tokens(self) -> None:
        config = PerformanceConfig(
            name="test",
            semantics="v11_critical_tokens",
            draft_alpha=1.0,
            draft_beta=0.0,
            prm_alpha=0.0,
            prm_beta=1.0,
            target_pre_beta=0.0,
            target_alpha=10.0,
            target_gen_beta=0.0,
            target_scale=1.0,
            bandwidth_gb_s=1.0,
            bytes_per_token=0,
        )
        simulator = SpeculativeSimulator(config)
        logs = [
            {
                "example_id": 8,
                "steps": [
                    {"tokens": 1, "status": "accept_pure_draft"},
                    {
                        "tokens": 1,
                        "status": "target_rescue",
                        "target_total_tokens": 5,
                        "target_critical_gen_tokens": 2,
                        "has_target_trace": True,
                    },
                    {
                        "tokens": 1,
                        "status": "BACKTRACK_EVENT",
                        "target_total_tokens": 4,
                        "target_critical_gen_tokens": 1,
                        "has_target_trace": True,
                    },
                ],
            }
        ]

        result = simulator.simulate_dataset(logs)

        self.assertAlmostEqual(result["total"], 42.0)
        self.assertAlmostEqual(result["backtrack_total"], 17.0)
        self.assertEqual(result["target_trace_events_used"], 2)
        self.assertEqual(result["target_total_tokens"], 9)
        self.assertEqual(result["target_critical_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
