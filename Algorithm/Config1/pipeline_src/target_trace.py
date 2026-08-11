"""Target-token trace helpers for prefetch-cache runs."""

from __future__ import annotations

from typing import Any, Dict


TARGET_TRACE_MODES = frozenset(
    {
        "none",
        "partial_cache",
        "partial_resume",
        "prefetched_complete",
        "direct_generation",
    }
)


def build_target_trace(
    mode: str,
    *,
    cached_tokens: int = 0,
    continuation_tokens: int = 0,
) -> Dict[str, Any]:
    if mode not in TARGET_TRACE_MODES:
        raise ValueError(f"Unknown target trace mode: {mode}")

    cached_tokens = int(cached_tokens)
    continuation_tokens = int(continuation_tokens)
    if cached_tokens < 0 or continuation_tokens < 0:
        raise ValueError("Target token counts must be non-negative")

    if mode == "none":
        cached_tokens = 0
        continuation_tokens = 0
    elif mode in {"partial_cache", "prefetched_complete"}:
        if continuation_tokens:
            raise ValueError(f"{mode} cannot contain continuation tokens")
    elif mode == "direct_generation":
        if cached_tokens:
            raise ValueError("direct_generation cannot contain cached tokens")

    target_tokens = cached_tokens + continuation_tokens
    critical_tokens = (
        continuation_tokens
        if mode in {"partial_resume", "direct_generation"}
        else 0
    )
    return {
        "target_mode": mode,
        "target_tokens": target_tokens,
        "target_cached_tokens": cached_tokens,
        "target_continuation_tokens": continuation_tokens,
        "target_critical_gen_tokens": critical_tokens,
    }


def attach_target_trace(node: Any, trace: Dict[str, Any]) -> None:
    for key, value in trace.items():
        setattr(node, key, value)


def get_target_trace(node: Any, *, fallback_tokens: int = 0) -> Dict[str, Any]:
    mode = getattr(node, "target_mode", None)
    if mode not in TARGET_TRACE_MODES:
        if fallback_tokens:
            return build_target_trace(
                "direct_generation", continuation_tokens=fallback_tokens
            )
        return build_target_trace("none")

    return build_target_trace(
        mode,
        cached_tokens=getattr(node, "target_cached_tokens", 0),
        continuation_tokens=getattr(node, "target_continuation_tokens", 0),
    )
