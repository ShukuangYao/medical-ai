from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.tools.result import ToolErrorPayload, ToolResult


def tool_result_to_sse_error(
    *,
    res: ToolResult[Any],
    phase: str,
    fallback_message: str = "tool failed",
) -> Optional[Dict[str, Any]]:
    """Convert a failed ToolResult to an SSE `type=error` event.

    Returns None when res.ok=True.
    """
    if res.ok:
        return None
    err: Optional[ToolErrorPayload] = res.error
    msg = (err.message if err else "") or fallback_message
    evt: Dict[str, Any] = {"type": "error", "content": msg, "phase": phase}
    if err is not None:
        evt.update(err.to_event_fields())
    else:
        evt.update({"code": "INTERNAL", "retriable": False})
    return evt


def record_tool_result_perf(*, perf: Dict[str, Any], tool: str, res: ToolResult[Any]) -> None:
    """Best-effort: append tool call perf info for debugging."""
    try:
        rows = perf.get("tool_calls")
        if not isinstance(rows, list):
            rows = []
            perf["tool_calls"] = rows
        hits = perf.get("tool_cache_hits")
        if not isinstance(hits, dict):
            hits = {}
            perf["tool_cache_hits"] = hits
        misses = perf.get("tool_cache_misses")
        if not isinstance(misses, dict):
            misses = {}
            perf["tool_cache_misses"] = misses

        # Cache hit/miss counters per tool
        if bool(getattr(res, "cached", False)):
            hits[tool] = int(hits.get(tool) or 0) + 1
        else:
            misses[tool] = int(misses.get(tool) or 0) + 1
        rows.append(
            {
                "tool": tool,
                "ok": bool(res.ok),
                "cached": bool(getattr(res, "cached", False)),
                "duration_ms": int(res.duration_ms or 0),
                "idempotency_key": str(res.idempotency_key or ""),
                "error_code": (res.error.code if res.error else ""),
            }
        )
        # Keep it bounded
        if len(rows) > 80:
            del rows[: len(rows) - 80]
    except Exception:
        return

