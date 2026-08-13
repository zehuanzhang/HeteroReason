"""Core latency model for the portable hardware simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


SUPPORTED_SEMANTICS = frozenset(
    {
        "v11_critical_tokens",
        "v12_recomputed_overlap",
        "v12_parallel_prefetch",
        "v12_parallel_prefetch_target_prm_overlap",
        "v12_rescue_serial_then_parallel",
    }
)


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
    initial_gpu_prefill_ms: float = 0.0
    initial_kv_transfer_ms: float = 0.0
    draft_prefill_alpha: float = 0.0
    draft_prefill_beta: float = 0.0
    draft_kv_bytes_per_token: int = 0

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
    """Trace-driven simulator for HeteroReason hardware estimates."""

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

    def get_target_output_verification_lat(self, n_kv: int, target_tokens: int) -> float:
        if target_tokens <= 0:
            return 0.0
        return self.get_prm_lat(n_kv + target_tokens) + self.get_communication_lat(
            target_tokens
        )

    @staticmethod
    def get_target_output_prm_extra_lat(
        verification_lat: float,
        target_total_lat: float,
        next_draft_lat: float,
        target_output_prm_lat: float,
    ) -> float:
        target_prm_start = max(verification_lat, target_total_lat)
        base_completion = max(verification_lat, target_total_lat, next_draft_lat)
        return max(0.0, target_prm_start + target_output_prm_lat - base_completion)

    def get_initialization_lat(self, prompt_tokens: int) -> float:
        if prompt_tokens < 0:
            raise ValueError("Prompt token count must be non-negative")
        dynamic_prefill = 0.0
        if prompt_tokens:
            dynamic_prefill = (
                self.cfg.draft_prefill_alpha
                + self.cfg.draft_prefill_beta * prompt_tokens
            )
        transfer_bytes = prompt_tokens * self.cfg.draft_kv_bytes_per_token
        dynamic_transfer = (
            (transfer_bytes / (self.cfg.bandwidth_gb_s * 1e9)) * 1000
            if transfer_bytes else 0.0
        )
        return (
            self.cfg.initial_gpu_prefill_ms
            + self.cfg.initial_kv_transfer_ms
            + dynamic_prefill
            + dynamic_transfer
        )

    @property
    def _uses_target_trace(self) -> bool:
        return self.cfg.semantics == "v11_critical_tokens"

    @property
    def _recomputes_target_overlap(self) -> bool:
        return self.cfg.semantics in {
            "v12_recomputed_overlap",
            "v12_parallel_prefetch",
            "v12_parallel_prefetch_target_prm_overlap",
            "v12_rescue_serial_then_parallel",
        }

    @property
    def _uses_parallel_target_prefetch(self) -> bool:
        return self.cfg.semantics in {
            "v12_parallel_prefetch",
            "v12_parallel_prefetch_target_prm_overlap",
            "v12_rescue_serial_then_parallel",
        }

    @property
    def _overlaps_target_output_prm(self) -> bool:
        return self.cfg.semantics == "v12_parallel_prefetch_target_prm_overlap"

    @property
    def _uses_rescue_serial_then_parallel(self) -> bool:
        return self.cfg.semantics == "v12_rescue_serial_then_parallel"

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
        if self._recomputes_target_overlap:
            return self._simulate_dataset_v12(dataset_log)

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

    def _target_total_tokens(
        self, action: Dict[str, Any], status: str, draft_tokens: int
    ) -> int:
        target_tokens = int(
            action.get(
                "target_total_tokens",
                action.get("target_tokens", action.get("target_tokens_spent", 0)),
            )
        )
        if target_tokens < 0:
            raise ValueError("Target total tokens must be non-negative")
        if target_tokens:
            return target_tokens
        if status == "BACKTRACK_EVENT":
            return max(0, int(draft_tokens))
        if status in ("target_rescue", "accept_via_prefetch"):
            legacy_tokens = int(action.get("target_tokens", 0))
            return max(1, legacy_tokens)
        return 0

    def _target_prefix_latency(self, n_kv: int, n_gen: int) -> float:
        return self.get_target_lat(n_kv, n_gen) * self.cfg.target_scale

    def _target_overlap_result(
        self, n_kv: int, target_tokens: int, overlap_ms: float
    ) -> Dict[str, Any]:
        if target_tokens <= 0:
            return {
                "target_total_latency_ms": 0.0,
                "target_wait_latency_ms": 0.0,
                "target_cached_tokens": 0,
                "target_critical_tokens": 0,
            }

        total_latency = self._target_prefix_latency(n_kv, target_tokens)
        overlap_ms = max(0.0, overlap_ms)
        cached_tokens = 0
        for candidate in range(1, target_tokens + 1):
            if self._target_prefix_latency(n_kv, candidate) <= overlap_ms:
                cached_tokens = candidate
            else:
                break

        return {
            "target_total_latency_ms": total_latency,
            "target_wait_latency_ms": max(0.0, total_latency - overlap_ms),
            "target_cached_tokens": cached_tokens,
            "target_critical_tokens": target_tokens - cached_tokens,
        }

    @staticmethod
    def _safe_subtract(value: int, delta: int, field_name: str) -> int:
        if delta < 0:
            raise ValueError(f"{field_name} rollback must be non-negative")
        if delta > value:
            raise ValueError(f"{field_name} rollback exceeds active KV length")
        return value - delta

    @staticmethod
    def _logical_step(action: Dict[str, Any], fallback: int) -> int:
        return int(action.get("logical_step", fallback))

    def _rollback_from_stack(
        self, path_stack: List[Dict[str, int]], action: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        required_fields = (
            "from_step",
            "to_step",
            "rollback_draft_tokens",
            "rollback_target_tokens",
            "target_total_tokens",
        )
        missing_fields = [
            field for field in required_fields if action.get(field) is None
        ]
        if missing_fields:
            raise ValueError(
                "v12 BACKTRACK_EVENT is missing required fields: "
                + ", ".join(missing_fields)
            )
        raw_rollback_draft = int(action["rollback_draft_tokens"])
        raw_rollback_target = int(action["rollback_target_tokens"])
        keep_step = int(action["to_step"])
        active_rollback_tokens = 0
        while path_stack and path_stack[-1]["logical_step"] > keep_step:
            active_rollback_tokens += int(path_stack.pop()["tokens"])
        return (
            active_rollback_tokens,
            active_rollback_tokens,
            raw_rollback_draft,
            raw_rollback_target,
        )

    def _append_next_draft(
        self, trace: List[Dict[str, Any]], index: int, draft_kv: int
    ) -> Tuple[float, int]:
        if index + 1 >= len(trace):
            return 0.0, draft_kv
        next_tokens = int(trace[index + 1]["tokens"])
        next_draft_lat = self.get_draft_lat(draft_kv, next_tokens)
        return next_draft_lat, draft_kv + next_tokens

    def _simulate_dataset_v12(self, dataset_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_latency = 0.0
        backtrack_lat_total = 0.0
        prm_dominated_count = 0
        draft_dominated_count = 0
        total_hidden_gain = 0.0
        total_stall_overhead = 0.0
        target_trace_events_used = 0
        traced_target_total_tokens = 0
        simulated_target_cached_tokens = 0
        simulated_target_critical_tokens = 0
        simulated_target_wait_ms = 0.0
        rollback_draft_tokens = 0
        rollback_target_tokens = 0
        raw_rollback_draft_tokens = 0
        raw_rollback_target_tokens = 0
        target_output_prm_events = 0
        target_output_prm_latency_ms = 0.0
        target_output_prm_critical_latency_ms = 0.0
        example_results = []

        for example in dataset_log:
            prompt_tokens = int(example.get("prompt_tokens", 0))
            ex_lat = self.get_initialization_lat(prompt_tokens)
            n_kv = 0
            draft_kv = 0
            path_stack: List[Dict[str, int]] = []
            trace = example["steps"]
            if not trace:
                continue

            first_tokens = int(trace[0]["tokens"])
            current_prefetch_window = self.get_draft_lat(draft_kv, first_tokens)
            ex_lat += current_prefetch_window
            draft_kv += first_tokens

            for index, curr_action in enumerate(trace):
                n_curr = int(curr_action["tokens"])
                status = curr_action.get("status", "")
                logical_step = self._logical_step(curr_action, index)

                communication_lat = self.get_communication_lat(n_curr)
                prm_lat = self.get_prm_lat(n_kv + n_curr)
                verification_lat = prm_lat + communication_lat

                if status == "accept_pure_draft":
                    n_kv += n_curr
                    path_stack.append({"logical_step": logical_step, "tokens": n_curr})

                    next_draft_lat, next_draft_kv = self._append_next_draft(
                        trace, index, draft_kv
                    )
                    if index + 1 < len(trace):
                        step_lat = max(verification_lat, next_draft_lat)
                        if verification_lat > next_draft_lat:
                            prm_dominated_count += 1
                            total_stall_overhead += verification_lat - next_draft_lat
                        else:
                            draft_dominated_count += 1
                            total_hidden_gain += next_draft_lat - verification_lat
                        ex_lat += step_lat
                        current_prefetch_window = step_lat
                        draft_kv = next_draft_kv
                    else:
                        ex_lat += verification_lat

                elif status in ("target_rescue", "accept_via_prefetch"):
                    target_tokens = self._target_total_tokens(
                        curr_action, status, n_curr
                    )
                    target_trace_events_used += 1
                    traced_target_total_tokens += target_tokens

                    corrected_kv = n_kv + target_tokens
                    next_draft_lat, next_draft_kv = self._append_next_draft(
                        trace, index, corrected_kv
                    )
                    if self._uses_rescue_serial_then_parallel:
                        overlap = current_prefetch_window + verification_lat
                    elif self._uses_parallel_target_prefetch and index + 1 < len(trace):
                        overlap = max(verification_lat, next_draft_lat)
                    else:
                        overlap = verification_lat
                    target_result = self._target_overlap_result(
                        n_kv, target_tokens, overlap
                    )
                    wait_lat = target_result["target_wait_latency_ms"]
                    target_output_prm_lat = self.get_target_output_verification_lat(
                        n_kv, target_tokens
                    )
                    if target_output_prm_lat:
                        target_output_prm_events += 1
                        target_output_prm_latency_ms += target_output_prm_lat
                    if self._uses_rescue_serial_then_parallel:
                        if index + 1 < len(trace):
                            target_output_prm_extra_lat = max(
                                0.0, target_output_prm_lat - next_draft_lat
                            )
                            target_output_prm_step_lat = max(
                                target_output_prm_lat, next_draft_lat
                            )
                        else:
                            target_output_prm_extra_lat = target_output_prm_lat
                            target_output_prm_step_lat = target_output_prm_lat
                    elif self._overlaps_target_output_prm:
                        target_output_prm_extra_lat = (
                            self.get_target_output_prm_extra_lat(
                                verification_lat,
                                target_result["target_total_latency_ms"],
                                next_draft_lat if index + 1 < len(trace) else 0.0,
                                target_output_prm_lat,
                            )
                        )
                    else:
                        target_output_prm_extra_lat = target_output_prm_lat
                        target_output_prm_step_lat = target_output_prm_lat
                    target_output_prm_critical_latency_ms += target_output_prm_extra_lat
                    simulated_target_cached_tokens += target_result[
                        "target_cached_tokens"
                    ]
                    simulated_target_critical_tokens += target_result[
                        "target_critical_tokens"
                    ]
                    simulated_target_wait_ms += wait_lat
                    if self._uses_rescue_serial_then_parallel:
                        ex_lat += verification_lat + wait_lat + target_output_prm_step_lat
                    else:
                        ex_lat += overlap + wait_lat + target_output_prm_extra_lat

                    n_kv = corrected_kv
                    draft_kv = corrected_kv
                    path_stack.append(
                        {"logical_step": logical_step, "tokens": target_tokens}
                    )

                    if index + 1 < len(trace):
                        if self._uses_rescue_serial_then_parallel:
                            current_prefetch_window = target_output_prm_step_lat
                        elif not self._uses_parallel_target_prefetch:
                            ex_lat += next_draft_lat
                            current_prefetch_window = next_draft_lat
                        else:
                            current_prefetch_window = overlap
                        draft_kv = next_draft_kv

                elif status == "BACKTRACK_EVENT":
                    (
                        dropped_draft,
                        dropped_target,
                        raw_dropped_draft,
                        raw_dropped_target,
                    ) = self._rollback_from_stack(path_stack, curr_action)
                    rollback_draft_tokens += dropped_draft
                    rollback_target_tokens += dropped_target
                    raw_rollback_draft_tokens += raw_dropped_draft
                    raw_rollback_target_tokens += raw_dropped_target
                    draft_kv = self._safe_subtract(
                        draft_kv, dropped_draft, "draft backtracking"
                    )
                    n_kv = self._safe_subtract(
                        n_kv, dropped_target, "target backtracking"
                    )

                    target_tokens = self._target_total_tokens(
                        curr_action, status, n_curr
                    )
                    target_trace_events_used += 1
                    traced_target_total_tokens += target_tokens

                    overlap = current_prefetch_window + verification_lat
                    target_result = self._target_overlap_result(
                        n_kv, target_tokens, overlap
                    )
                    wait_lat = target_result["target_wait_latency_ms"]
                    target_output_prm_lat = self.get_target_output_verification_lat(
                        n_kv, target_tokens
                    )
                    if target_output_prm_lat:
                        target_output_prm_events += 1
                        target_output_prm_latency_ms += target_output_prm_lat
                    simulated_target_cached_tokens += target_result[
                        "target_cached_tokens"
                    ]
                    simulated_target_critical_tokens += target_result[
                        "target_critical_tokens"
                    ]
                    simulated_target_wait_ms += wait_lat

                    next_draft_lat, next_draft_kv = self._append_next_draft(
                        trace, index, draft_kv + target_tokens
                    )
                    if (
                        self._uses_rescue_serial_then_parallel
                        and index + 1 < len(trace)
                    ):
                        target_output_prm_critical_lat = max(
                            target_output_prm_lat, next_draft_lat
                        )
                        target_output_prm_extra_lat = max(
                            0.0, target_output_prm_lat - next_draft_lat
                        )
                    elif self._overlaps_target_output_prm and index + 1 < len(trace):
                        target_output_prm_critical_lat = max(
                            target_output_prm_lat, next_draft_lat
                        )
                        target_output_prm_extra_lat = max(
                            0.0, target_output_prm_lat - next_draft_lat
                        )
                    else:
                        target_output_prm_critical_lat = target_output_prm_lat
                        target_output_prm_extra_lat = target_output_prm_lat
                    target_output_prm_critical_latency_ms += target_output_prm_extra_lat

                    step_penalty = verification_lat + wait_lat
                    backtrack_lat_total += step_penalty
                    ex_lat += step_penalty

                    target_step = int(curr_action.get("to_step", logical_step))
                    draft_kv += target_tokens
                    n_kv += target_tokens
                    path_stack.append(
                        {"logical_step": target_step, "tokens": target_tokens}
                    )

                    draft_kv = next_draft_kv
                    if index + 1 < len(trace):
                        if self._uses_rescue_serial_then_parallel:
                            ex_lat += target_output_prm_critical_lat
                            current_prefetch_window = target_output_prm_critical_lat
                        elif self._overlaps_target_output_prm:
                            ex_lat += target_output_prm_critical_lat
                            current_prefetch_window = next_draft_lat
                        else:
                            ex_lat += target_output_prm_lat + next_draft_lat
                            current_prefetch_window = next_draft_lat
                    else:
                        ex_lat += target_output_prm_lat

            total_latency += ex_lat
            example_results.append(
                {"id": example["example_id"], "lat": round(ex_lat, 2)}
            )

        return {
            "total": total_latency,
            "backtrack_total": backtrack_lat_total,
            "prm_dominated": prm_dominated_count,
            "draft_dominated": draft_dominated_count,
            "hidden_gain": total_hidden_gain,
            "stall_overhead": total_stall_overhead,
            "details": example_results,
            "target_trace_events_used": target_trace_events_used,
            "target_total_tokens": traced_target_total_tokens,
            "target_critical_tokens": simulated_target_critical_tokens,
            "target_cached_tokens": simulated_target_cached_tokens,
            "target_wait_latency_s": simulated_target_wait_ms / 1000.0,
            "rollback_draft_tokens": rollback_draft_tokens,
            "rollback_target_tokens": rollback_target_tokens,
            "raw_rollback_draft_tokens": raw_rollback_draft_tokens,
            "raw_rollback_target_tokens": raw_rollback_target_tokens,
            "target_output_prm_events": target_output_prm_events,
            "target_output_prm_latency_s": target_output_prm_latency_ms / 1000.0,
            "target_output_prm_critical_latency_s": (
                target_output_prm_critical_latency_ms / 1000.0
            ),
        }

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
    for field in (
        "target_cached_tokens",
        "target_wait_latency_s",
        "rollback_draft_tokens",
        "rollback_target_tokens",
        "raw_rollback_draft_tokens",
        "raw_rollback_target_tokens",
        "target_output_prm_events",
        "target_output_prm_latency_s",
        "target_output_prm_critical_latency_s",
    ):
        if field in simulation:
            metrics[field] = simulation[field]
    return metrics
