from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langsmith.run_helpers import get_current_run_tree
from pydantic import ValidationError

from app.core.tools.base import BaseTool, ToolContext, ToolError
from app.core.tools.idempotency import InMemoryIdempotencyStore
from app.core.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        idempotency: Optional[InMemoryIdempotencyStore] = None,
        default_idempotency_ttl_s: float = 60.0,
    ) -> None:
        self.registry = registry
        self.idempotency = idempotency or InMemoryIdempotencyStore()
        self.default_idempotency_ttl_s = float(default_idempotency_ttl_s)

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
        tool = self.registry.get(name)
        if tool is None:
            raise ToolError(code="NOT_FOUND", message=f"Tool not found: {name}", retriable=False)

        # Idempotency cache (best-effort)
        if idempotency_key:
            cached = self.idempotency.get(idempotency_key)
            if cached is not None:
                return cached

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
            if span is not None:
                span.end(outputs={"ok": True, "duration_ms": int((time.perf_counter() - t0) * 1000)})
                span.patch()

            if idempotency_key:
                self.idempotency.set(
                    idempotency_key,
                    out,
                    ttl_s=idempotency_ttl_s if idempotency_ttl_s is not None else self.default_idempotency_ttl_s,
                )
            return out
        except ToolError as e:
            if span is not None:
                span.end(
                    error=e.message,
                    outputs={
                        "ok": False,
                        "code": e.code,
                        "retriable": e.retriable,
                        "duration_ms": int((time.perf_counter() - t0) * 1000),
                    },
                )
                span.patch()
            raise
        except Exception as e:
            if span is not None:
                span.end(
                    error=str(e),
                    outputs={"ok": False, "code": "INTERNAL", "duration_ms": int((time.perf_counter() - t0) * 1000)},
                )
                span.patch()
            raise ToolError(code="INTERNAL", message=f"{name} failed", retriable=False) from e

