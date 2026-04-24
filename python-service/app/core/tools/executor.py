from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langsmith.run_helpers import get_current_run_tree
from pydantic import ValidationError

import os

from app.config import settings
from app.core.tools.auth import AuthPolicy
from app.core.tools.base import BaseTool, ToolContext, ToolError
from app.core.tools.idempotency import IdempotencyStore, create_idempotency_store, InMemoryIdempotencyStore
from app.core.tools.registry import ToolRegistry
from app.core.tools.result import ToolErrorPayload, ToolResult
from app.core.metrics import inc_tool_error


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        idempotency: Optional[IdempotencyStore] = None,
        auth: Optional[AuthPolicy] = None,
        default_idempotency_ttl_s: float = 60.0,
    ) -> None:
        self.registry = registry
        # Default SQLite idempotency DB should be independent from session DB to reduce lock contention.
        default_sqlite = os.path.join(os.path.dirname(str(getattr(settings, "CHAT_DB_PATH", ""))), "tool_idempotency.db")
        self.idempotency: IdempotencyStore = idempotency or create_idempotency_store(
            sqlite_path=str(
                os.getenv(
                    "TOOL_IDEMPOTENCY_SQLITE_PATH",
                    default_sqlite,
                )
            ),
            redis_url=str(getattr(settings, "REDIS_URL", "")),
        )
        self.auth = auth or AuthPolicy.from_env()
        self.default_idempotency_ttl_s = float(default_idempotency_ttl_s)

    async def run_result(
        self,
        name: str,
        *,
        args: Dict[str, Any],
        ctx: ToolContext,
        idempotency_key: Optional[str] = None,
        idempotency_ttl_s: Optional[float] = None,
        trace_inputs: Optional[Dict[str, Any]] = None,
    ) -> ToolResult[Any]:
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error=ToolErrorPayload(code="NOT_FOUND", message=f"Tool not found: {name}", retriable=False),
                cached=False,
                idempotency_key=idempotency_key,
            )

        # Hard auth (Phase 1): best-effort guard for expensive/sensitive tools
        try:
            self.auth.authorize(tool=tool, ctx=ctx)
        except ToolError as e:
            return ToolResult(
                ok=False,
                error=ToolErrorPayload(
                    code=e.code,
                    message=e.message,
                    retriable=bool(e.retriable),
                    detail=e.detail,
                    upstream_status=e.upstream_status,
                ),
                cached=False,
                idempotency_key=idempotency_key,
            )

        # Idempotency cache (best-effort)
        if idempotency_key:
            cached = self.idempotency.get(idempotency_key)
            if cached is not None:
                return ToolResult(ok=True, data=cached, cached=True, duration_ms=0, idempotency_key=idempotency_key)

        parent = get_current_run_tree()
        span = None
        t0 = time.perf_counter()
        if parent is not None:
            span = parent.create_child(
                name=f"tool:{name}",
                run_type="tool",
                inputs=trace_inputs if trace_inputs is not None else {"args": args, "ctx": ctx.__dict__},
            )
            span.post()

        try:
            try:
                model_args = tool.ArgsModel.model_validate(args)  # type: ignore[attr-defined]
            except ValidationError as e:
                raise ToolError(
                    code="VALIDATION_ERROR",
                    message=f"{name} args validation failed",
                    retriable=False,
                    detail={"errors": e.errors()},
                ) from e

            out = await tool.run(args=model_args, ctx=ctx)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            if span is not None:
                span.end(outputs={"ok": True, "duration_ms": duration_ms})
                span.patch()

            if idempotency_key:
                self.idempotency.set(
                    idempotency_key,
                    out,
                    ttl_s=idempotency_ttl_s if idempotency_ttl_s is not None else self.default_idempotency_ttl_s,
                )
            return ToolResult(
                ok=True,
                data=out,
                cached=False,
                duration_ms=duration_ms,
                idempotency_key=idempotency_key,
            )
        except ToolError as e:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            try:
                inc_tool_error(tool=name, code=e.code, retriable=bool(e.retriable))
            except Exception:
                pass
            if span is not None:
                span.end(
                    error=e.message,
                    outputs={
                        "ok": False,
                        "code": e.code,
                        "retriable": e.retriable,
                        "duration_ms": duration_ms,
                    },
                )
                span.patch()
            return ToolResult(
                ok=False,
                error=ToolErrorPayload(
                    code=e.code,
                    message=e.message,
                    retriable=bool(e.retriable),
                    detail=e.detail,
                    upstream_status=e.upstream_status,
                ),
                cached=False,
                duration_ms=duration_ms,
                idempotency_key=idempotency_key,
            )
        except Exception as e:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            try:
                inc_tool_error(tool=name, code="INTERNAL", retriable=False)
            except Exception:
                pass
            if span is not None:
                span.end(
                    error=str(e),
                    outputs={"ok": False, "code": "INTERNAL", "duration_ms": duration_ms},
                )
                span.patch()
            return ToolResult(
                ok=False,
                error=ToolErrorPayload(code="INTERNAL", message=f"{name} failed", retriable=False, detail={"err": str(e)}),
                cached=False,
                duration_ms=duration_ms,
                idempotency_key=idempotency_key,
            )

    async def run(
        self,
        name: str,
        *,
        args: Dict[str, Any],
        ctx: ToolContext,
        idempotency_key: Optional[str] = None,
        idempotency_ttl_s: Optional[float] = None,
        trace_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        res = await self.run_result(
            name,
            args=args,
            ctx=ctx,
            idempotency_key=idempotency_key,
            idempotency_ttl_s=idempotency_ttl_s,
            trace_inputs=trace_inputs,
        )
        if res.ok:
            return res.data
        err = res.error or ToolErrorPayload(code="INTERNAL", message=f"{name} failed", retriable=False)
        raise ToolError(
            code=err.code,
            message=err.message,
            retriable=bool(err.retriable),
            detail=err.detail,
            upstream_status=err.upstream_status,
        )

