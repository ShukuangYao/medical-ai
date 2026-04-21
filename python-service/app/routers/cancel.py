"""Cooperative run cancellation (Phase 2)."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.run_cancel import request_cancel

router = APIRouter()


class CancelRequest(BaseModel):
    run_id: str = Field(default="", description="Same as X-Run-Id / SSE envelope run_id")


@router.post("/cancel")
async def cancel_generation(body: CancelRequest) -> Dict[str, Any]:
    """Signal the streaming handler for this run_id to stop after the current step."""
    rid = (body.run_id or "").strip()
    if not rid:
        return {"ok": False, "reason": "missing_run_id"}
    request_cancel(rid)
    return {"ok": True, "run_id": rid}


@router.delete("/cancel/{run_id}")
async def cancel_generation_delete(run_id: str) -> Dict[str, Any]:
    """Optional alias: DELETE /api/cancel/{run_id}"""
    rid = (run_id or "").strip()
    if not rid:
        return {"ok": False, "reason": "missing_run_id"}
    request_cancel(rid)
    return {"ok": True, "run_id": rid}
