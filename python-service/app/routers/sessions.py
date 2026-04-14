from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.core.session_store import SessionStore

router = APIRouter()
store = SessionStore(settings.CHAT_DB_PATH)
DEFAULT_USER_ID = "anonymous"


@router.post("/sessions")
async def create_session(request: Request):
    data = await request.json()
    user_id = (data.get("userId") or data.get("user_id") or DEFAULT_USER_ID)
    mode = data.get("mode") or "rag"
    session_id = data.get("sessionId") or data.get("session_id")
    title = data.get("title") or "新会话"
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    store.create_session(session_id=session_id, user_id=user_id, mode=mode, title=title)
    return {"sessionId": session_id}


@router.get("/sessions")
async def list_sessions(userId: Optional[str] = None, mode: str = "rag", includeArchived: bool = False):
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    uid = userId or DEFAULT_USER_ID
    rows = store.list_sessions(user_id=uid, mode=mode, include_archived=includeArchived)
    return {
        "sessions": [
            {
                "sessionId": r.session_id,
                "title": r.title,
                "createdAt": r.created_at,
                "updatedAt": r.updated_at,
                "archived": bool(r.archived),
            }
            for r in rows
        ]
    }


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, userId: Optional[str] = None, mode: str = "rag", limit: int = 200):
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    uid = userId or DEFAULT_USER_ID
    rows = store.list_messages(session_id=session_id, user_id=uid, mode=mode, limit=limit)
    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "report": m.report_json,
                "sources": m.sources_json,
                "trace": m.trace_json,
                "thinkingSteps": m.thinking_steps_json,
                "createdAt": m.created_at,
            }
            for m in rows
        ]
    }


@router.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, request: Request):
    data = await request.json()
    user_id = data.get("userId") or data.get("user_id") or DEFAULT_USER_ID
    mode = data.get("mode") or "rag"
    title = data.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    store.rename_session(session_id=session_id, user_id=user_id, mode=mode, title=title)
    return {"ok": True}


@router.post("/sessions/{session_id}/archive")
async def archive_session(session_id: str, request: Request):
    data = await request.json()
    user_id = data.get("userId") or data.get("user_id") or DEFAULT_USER_ID
    mode = data.get("mode") or "rag"
    archived = bool(data.get("archived", True))
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    store.archive_session(session_id=session_id, user_id=user_id, mode=mode, archived=archived)
    return {"ok": True}

