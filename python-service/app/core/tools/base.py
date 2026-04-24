from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel


class ToolError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retriable: bool = False,
        detail: Optional[Dict[str, Any]] = None,
        upstream_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retriable = retriable
        self.detail = detail
        self.upstream_status = upstream_status

    def to_event_fields(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "retriable": bool(self.retriable),
        }
        if self.upstream_status is not None:
            out["upstream_status"] = int(self.upstream_status)
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class ToolContext:
    # Correlation / tracing
    request_id: str = ""
    run_id: str = ""
    # Request identity / authorization hooks
    user_id: str = ""
    session_id: str = ""
    mode: str = ""


ArgsT = TypeVar("ArgsT", bound=BaseModel)
OutT = TypeVar("OutT")


class BaseTool(Generic[ArgsT, OutT]):
    name: str
    ArgsModel: Type[ArgsT]
    # Hard authorization hook. When True, ToolExecutor must authorize the caller.
    auth_required: bool = False
    # Optional scopes for future expansion (kept simple for now).
    required_scopes: Sequence[str] = ()

    async def run(self, *, args: ArgsT, ctx: ToolContext) -> OutT:  # pragma: no cover
        raise NotImplementedError

