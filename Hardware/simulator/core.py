"""Core latency model for the portable hardware simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


SUPPORTED_SEMANTICS = frozenset({"v11_critical_tokens"})


@dataclass(frozen=True)
class PerformanceConfig:
    """Latency parameters in milliseconds unless otherwise noted."""

    name: str
    draft_alpha: float
    draft_beta: float
    prm_alpha: float
    prm_beta: float
    target_pre_beta: float
    target_alpha: float
    target_gen_beta: float
    target_scale: float
    bandwidth_gb_s: float
    bytes_per_token: int
    semantics: str = "v11_critical_tokens"

    @classmethod
    def from_json(cls, path: Path) -> "PerformanceConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported config schema in {path}")
        parameters = dict(data["parameters_ms"])
        return cls(
            name=str(data["name"]),
            semantics=str(data.get("semantics", "v11_critical_tokens")),
            **parameters,
        )

    def parameters_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("name")
        result.pop("semantics")
        return result


class SpeculativeSimulator:
    """Trace-driven simulator using target critical-token traces."""

    def __init__(self, config: PerformanceConfig):
        if config.semantics not in SUPPORTED_SEMANTICS:
            raise ValueError(f"Unsupported simulator semantics: {config.semantics}")
        self.cfg = config

    def get_draft_lat(self, draft_kv: int, n_gen: int) -> float:
        total_lat = 0.0
        for token_index in range(1, n_gen + 1):
            current_kv = draft_kv + token_index
            total_lat += self.cfg.draft_alpha + self.cfg.draft_beta * current_kv
        return total_lat

    def get_prm_lat(self, n_total: int) -> float:
        return self.cfg.prm_alpha + self.cfg.prm_beta * n_total

    def get_target_lat(self, n_kv: int, n_gen: int) -> float:
        latency = self.cfg.target_pre_beta * n_kv
        for token_index in range(1, n_gen + 1):
            current_kv = n_kv + token_index
            latency += self.cfg.target_alpha + self.cfg.target_gen_beta * current_kv
        return latency

    def get_communication_lat(self, tokens: int) -> float:
        size_bytes = tokens * self.cfg.bytes_per_token
        return (size_bytes / (self.cfg.bandwidth_gb_s * 1e9)) * 1000

    @property
    def _uses_target_trace(self) -> bool:
        return self.cfg.semantics == "v11_critical_tokens"

    def _target_token_counts(
        self, action: Dict[str, Any], status: str, draft_tokens: int
    ) -> Tuple[int, int, bool]:
        if self._uses_target_trace and action.get("has_target_trace", False):
            total_target_tokens = int(action.get("target_total_tokens", 0))
            critical_target_tokens = int(action.get("target_critical_gen_tokens", 0))
            if total_target_tokens < 0 or critical_target_tokens < 0:
                raise ValueError("Target trace token counts must be non-negative")
            if critical_target_tokens > total_target_tokens:
                raise ValueError("Critical target tokens cannot exceed total tokens")
            return total_target_tokens, critical_target_tokens, True

        if status == "BACKTRACK_EVENT":
            return draft_tokens, draft_tokens, False

        target_tokens = action.get("target_tokens", 0) if status == "target_rescue" else 1
        target_tokens = max(1, int(target_tokens))
        return target_tokens, target_tokens, False

    def simulate_dataset(self, dataset_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_latency = 0.0
        backtrack_lat_total = 0.0
        prm_dominated_count = 0
        draft_dominated_count = 0
        total_hidden_gain = 0.0
        total_stall_overhead = 0.0
        target_trace_events_used = 0
        traced_target_total_tokens = 0
        traced_target_critical_tokens = 0
        example_results = []

        for example in dataset_log:
            ex_lat = 0.0
            n_kv = 0
            draft_kv = 0
            trace = example["steps"]
            if not trace:
                continue

            n_init = trace[0]["tokens"]
            ex_lat += self.get_draft_lat(draft_kv, n_init)
            draft_kv += n_init

            for index in range(len(trace)):
                curr_action = trace[index]
                n_curr = curr_action["tokens"]
                status = curr_action.get("status", "")

                communication_lat = self.get_communication_lat(n_curr)
                prm_lat = self.get_prm_lat(n_kv + n_curr)

                if status == "accept_pure_draft":
                    if index + 1 < len(trace):
                        prm_path = prm_lat + communication_lat
                        draft_path = self.get_draft_lat(
                            draft_kv, trace[index + 1]["tokens"]
                        )

                        if prm_path > draft_path:
                            prm_dominated_count += 1
                            total_stall_overhead += prm_path - draft_path
                        else:
                            draft_dominated_count += 1
                            total_hidden_gain += draft_path - prm_path

                        ex_lat += max(prm_path, draft_path)
                    else:
                        ex_lat += prm_lat + communication_lat

                    n_kv += n_curr
                    draft_kv += n_curr

                elif status in ("target_rescue", "accept_via_prefetch"):
                    target_tokens, critical_tokens, used_trace = (
                        self._target_token_counts(curr_action, status, n_curr)
                    )
                    if used_trace:
                        target_trace_events_used += 1
                        traced_target_total_tokens += target_tokens
                        traced_target_critical_tokens += critical_tokens
                    target_rescue_lat = (
                        self.get_target_lat(n_kv, n_gen=critical_tokens)
                        if critical_tokens > 0 or not used_trace else 0.0
                    )
                    ex_lat += (
                        communication_lat
                        + prm_lat
                        + target_rescue_lat * self.cfg.target_scale
                    )

                    if index + 1 < len(trace):
                        ex_lat += self.get_draft_lat(
                            draft_kv, trace[index + 1]["tokens"]
                        )

                    n_kv += target_tokens
                    # Target rescue does not advance the draft-side KV cache.

                elif status == "BACKTRACK_EVENT":
                    target_tokens, critical_tokens, used_trace = (
                        self._target_token_counts(curr_action, status, n_curr)
                    )
                    if used_trace:
                        target_trace_events_used += 1
                        traced_target_total_tokens += target_tokens
                        traced_target_critical_tokens += critical_tokens
                    target_backtrack_lat = (
                        self.get_target_lat(n_kv, n_gen=critical_tokens)
                        if critical_tokens > 0 or not used_trace else 0.0
                    )
                    step_penalty = communication_lat + prm_lat + target_backtrack_lat

                    backtrack_lat_total += step_penalty
                    ex_lat += step_penalty

                    if index + 1 < len(trace):
                        ex_lat += self.get_draft_lat(
                            draft_kv, trace[index + 1]["tokens"]
                        )

                    n_kv += target_tokens
                    # Backtracking state updates follow the recorded target trace.

            total_latency += ex_lat
            example_results.append(
                {"id": example["example_id"], "lat": round(ex_lat, 2)}
            )

        result = {
            "total": total_latency,
            "backtrack_total": backtrack_lat_total,
            "prm_dominated": prm_dominated_count,
            "draft_dominated": draft_dominated_count,
            "hidden_gain": total_hidden_gain,
            "stall_overhead": total_stall_overhead,
            "details": example_results,
        }
        if target_trace_events_used:
            result.update(
                {
                    "target_trace_events_used": target_trace_events_used,
                    "target_total_tokens": traced_target_total_tokens,
                    "target_critical_tokens": traced_target_critical_tokens,
                }
            )
        return result

    # Keep the historical method name for easier cross-checking.
    simulate_backtracking_dataset = simulate_dataset


def build_dataset_metrics(
    simulation: Dict[str, Any], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    total_latency_ms = simulation["total"]
    total_latency_s = total_latency_ms / 1000.0
    example_count = len(simulation["details"])
    if example_count == 0:
        raise ValueError("Trace dataset contains no non-empty examples")
    if total_latency_s <= 0:
        raise ValueError("Simulated latency must be positive")

    metrics = {
        **metadata,
        "total_latency_s": total_latency_s,
        "average_latency_ms": total_latency_ms / example_count,
        "throughput_tokens_s": metadata["total_generated_tokens"]
        / total_latency_s,
        "backtracking_latency_s": simulation["backtrack_total"] / 1000.0,
        "draft_dominated_steps": simulation["draft_dominated"],
        "prm_dominated_steps": simulation["prm_dominated"],
        "hidden_latency_gain_s": simulation["hidden_gain"] / 1000.0,
        "stall_overhead_s": simulation["stall_overhead"] / 1000.0,
    }
    if simulation.get("target_trace_events_used", 0):
        metrics.update(
            {
                "target_trace_events_used": simulation["target_trace_events_used"],
                "target_total_tokens": simulation["target_total_tokens"],
                "target_critical_tokens": simulation["target_critical_tokens"],
            }
        )
    return metrics
