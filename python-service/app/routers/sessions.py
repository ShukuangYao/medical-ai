from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.core.session_store import SessionStore

router = APIRouter()
store = SessionStore(settings.CHAT_DB_PATH)


@router.post("/sessions")
async def create_session(request: Request):
    data = await request.json()
    user_id = data.get("userId") or data.get("user_id")
    mode = data.get("mode") or "rag"
    session_id = data.get("sessionId") or data.get("session_id")
    title = data.get("title") or "新会话"
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    store.create_session(session_id=session_id, user_id=user_id, mode=mode, title=title)
    return {"sessionId": session_id}


@router.get("/sessions")
async def list_sessions(userId: str, mode: str = "rag", includeArchived: bool = False):
    if mode not in ("rag", "agent"):
        raise HTTPException(status_code=400, detail="mode must be rag|agent")
    rows = store.list_sessions(user_id=userId, mode=mode, include_archived=includeArchived)
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
async def get_messages(session_id: str, userId: str, limit: int = 200):
    rows = store.list_messages(session_id=session_id, user_id=userId, limit=limit)
    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "report": m.report_json,
                "sources": m.sources_json,
                "trace": m.trace_json,
                "createdAt": m.created_at,
            }
            for m in rows
        ]
    }


@router.post("/sessions/{session_id}/rename")
async def rename_session(session_id: str, request: Request):
    data = await request.json()
    user_id = data.get("userId") or data.get("user_id")
    title = data.get("title")
    if not user_id or not title:
        raise HTTPException(status_code=400, detail="userId and title are required")
    store.rename_session(session_id=session_id, user_id=user_id, title=title)
    return {"ok": True}


@router.post("/sessions/{session_id}/archive")
async def archive_session(session_id: str, request: Request):
    data = await request.json()
    user_id = data.get("userId") or data.get("user_id")
    archived = bool(data.get("archived", True))
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    store.archive_session(session_id=session_id, user_id=user_id, archived=archived)
    return {"ok": True}

