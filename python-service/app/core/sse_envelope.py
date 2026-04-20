from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import Request


@dataclass
class SSEContext:
    request_id: str = ""
    run_id: str = ""
    seq: int = 0


def sse_context_from_request(req: Request) -> SSEContext:
    # FastAPI/Starlette headers are case-insensitive, but we normalize anyway.
    request_id = (req.headers.get("x-request-id") or "").strip()
    run_id = (req.headers.get("x-run-id") or "").strip()
    return SSEContext(request_id=request_id, run_id=run_id, seq=0)


def sse_envelope(
    chunk: Dict[str, Any],
    *,
    ctx: SSEContext,
    mutate: bool = True,
) -> Dict[str, Any]:
    """Attach request/run correlation fields to an SSE event.

    Keeps backward compatibility by preserving existing `type`/`content` fields.
    """
    evt = chunk if mutate else dict(chunk)
    ctx.seq += 1
    evt["seq"] = ctx.seq
    if ctx.request_id:
        evt["request_id"] = ctx.request_id
    if ctx.run_id:
        evt["run_id"] = ctx.run_id
    return evt


def sse_data_line(evt: Dict[str, Any]) -> str:
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

