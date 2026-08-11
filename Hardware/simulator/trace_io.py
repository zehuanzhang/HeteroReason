"""Trace loading and deterministic input provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


DATASETS = ("math500", "gsm8k", "gaokao2023en", "olympiadbench")
TARGET_TRACE_MODES = frozenset(
    {"prefetched_complete", "partial_resume", "direct_generation"}
)


def digest_trace_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("*.json")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_trace_dataset(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {path}")

    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"Trace directory contains no JSON files: {path}")

    logs: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    total_tokens = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        backtrack_target_events: Dict[int, List[Dict[str, Any]]] = {}
        for event in data.get("target_trace_events", []):
            if event.get("phase") != "backtrack_target_intervention":
                continue
            logical_step = int(event["logical_step"])
            backtrack_target_events.setdefault(logical_step, []).append(event)

        steps = []
        for step in data["execution_trace"]:
            tokens = int(step["draft_tokens_spent"])
            target_tokens = int(step.get("target_tokens_spent", 0))
            status = str(step["status"])
            target_trace = step
            if status == "BACKTRACK_EVENT":
                anchor_step = int(step.get("to_step", -1))
                pending_events = backtrack_target_events.get(anchor_step, [])
                if pending_events:
                    target_trace = pending_events.pop(0)

            target_mode = str(target_trace.get("target_mode", "legacy"))
            has_target_trace = target_mode in TARGET_TRACE_MODES
            target_total = int(
                target_trace.get(
                    "target_tokens",
                    target_trace.get("target_tokens_spent", target_tokens),
                )
            )
            target_critical = int(
                target_trace.get("target_critical_gen_tokens", target_total)
            )
            if tokens < 0:
                raise ValueError(f"Negative draft token count in {file_path}")
            if target_tokens < 0:
                raise ValueError(f"Negative target token count in {file_path}")
            if target_total < 0 or target_critical < 0:
                raise ValueError(f"Negative target trace token count in {file_path}")
            total_tokens += tokens
            status_counts[status] += 1
            steps.append(
                {
                    "tokens": tokens,
                    "target_tokens": target_tokens,
                    "target_total_tokens": target_total,
                    "target_critical_gen_tokens": target_critical,
                    "target_mode": target_mode,
                    "has_target_trace": has_target_trace,
                    "status": status,
                }
            )

        unused_backtrack_events = sum(
            len(events) for events in backtrack_target_events.values()
        )
        if unused_backtrack_events:
            raise ValueError(
                f"{file_path} has {unused_backtrack_events} unmapped "
                "backtrack target events"
            )

        logs.append({"example_id": data["idx"], "steps": steps})

    metadata = {
        "examples": len(files),
        "events": sum(status_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "total_generated_tokens": total_tokens,
        "input_sha256": digest_trace_directory(path),
    }
    return logs, metadata
