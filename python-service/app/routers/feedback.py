from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.core.feedback_store import FeedbackStore

router = APIRouter()
store = FeedbackStore(settings.CHAT_DB_PATH)
logger = logging.getLogger("medical-ai.feedback")


class FeedbackRequest(BaseModel):
    run_id: str = Field(default="", description="Same as X-Run-Id / SSE envelope run_id")
    session_id: str = Field(default="", description="Frontend session id")
    message_id: str = Field(default="", description="Frontend assistant message id")
    mode: str = Field(default="rag", description="rag|agent")
    user_id: str = Field(default="anonymous")
    rating: int = Field(default=0, description="+1 useful, -1 not useful, 0 neutral")
    comment: str = Field(default="")
    corrected_answer: str = Field(default="", description="Optional corrected answer text")


@router.post("/feedback")
async def submit_feedback(req: Request) -> Dict[str, Any]:
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json")
    fb = FeedbackRequest.model_validate(body)
    rid = (fb.run_id or "").strip()
    legacy_no_run_id = False
    if not rid:
        # Backward-compatible: old session history messages may not have run_id persisted.
        # Accept feedback anyway (it will still be stored locally), but skip LangSmith writeback.
        rid = f"legacy_{uuid.uuid4()}"
        legacy_no_run_id = True
    sid = (fb.session_id or "").strip()
    mid = (fb.message_id or "").strip()
    if not sid or not mid:
        raise HTTPException(status_code=400, detail="missing_session_or_message_id")

    fid = str(uuid.uuid4())
    store.add_feedback(
        feedback_id=fid,
        run_id=rid,
        session_id=sid,
        message_id=mid,
        mode=(fb.mode or "rag").strip() or "rag",
        user_id=(fb.user_id or "anonymous").strip() or "anonymous",
        rating=int(fb.rating or 0),
        comment=fb.comment or "",
        corrected_answer=fb.corrected_answer or "",
    )

    # Best-effort: send to LangSmith feedback if configured.
    # Skip for legacy feedback without a traceable run id.
    try:
        if legacy_no_run_id:
            raise RuntimeError("legacy_no_run_id")
        from langsmith import Client  # type: ignore

        c = Client()
        score: Optional[float]
        if fb.rating > 0:
            score = 1.0
        elif fb.rating < 0:
            score = 0.0
        else:
            score = None
        correction_obj: Optional[dict] = (
            {"corrected_answer": (fb.corrected_answer or "")[:8000]}
            if (fb.corrected_answer or "").strip()
            else None
        )
        # NOTE: Client.create_feedback uses `extra=` for metadata (not `metadata=`).
        c.create_feedback(  # type: ignore[attr-defined]
            run_id=rid,
            key="user_feedback",
            score=score,
            comment=(fb.comment or "")[:2000],
            value={"rating": int(fb.rating or 0)},
            correction=correction_obj,
            extra={"session_id": sid, "message_id": mid, "mode": fb.mode, "user_id": fb.user_id},
        )
    except Exception as e:
        # Do not fail the request if LangSmith is unavailable/misconfigured.
        if not legacy_no_run_id:
            logger.warning("langsmith_feedback_failed", extra={"run_id": rid, "error": str(e)[:300]})

    return {"ok": True, "feedback_id": fid, "run_id": rid}

