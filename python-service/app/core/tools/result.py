from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ToolErrorPayload:
    code: str
    message: str
    retriable: bool = False
    detail: Optional[Dict[str, Any]] = None
    upstream_status: Optional[int] = None

    def to_event_fields(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "retriable": bool(self.retriable)}
        if self.upstream_status is not None:
            out["upstream_status"] = int(self.upstream_status)
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class ToolResult(Generic[T]):
    ok: bool
    data: Optional[T] = None
    error: Optional[ToolErrorPayload] = None
    cached: bool = False
    duration_ms: Optional[int] = None
    idempotency_key: Optional[str] = None

