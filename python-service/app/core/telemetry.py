from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Span:
    name: str
    start: float
    meta: Dict[str, Any]


class Telemetry:
    """Lightweight in-process telemetry.

    - No external deps.
    - Emits structured dict events so routers can optionally stream/log them.
    """

    def __init__(self):
        self._spans: Dict[str, Span] = {}

    def start(self, name: str, **meta: Any) -> None:
        self._spans[name] = Span(name=name, start=time.perf_counter(), meta=dict(meta))

    def end(self, name: str, **more: Any) -> Optional[Dict[str, Any]]:
        span = self._spans.pop(name, None)
        if not span:
            return None
        dur_ms = int((time.perf_counter() - span.start) * 1000)
        out: Dict[str, Any] = {"span": name, "duration_ms": dur_ms}
        out.update(span.meta)
        out.update(more)
        return out

