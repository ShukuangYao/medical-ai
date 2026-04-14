from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    user_id: str
    mode: str  # "rag" | "agent"
    title: str
    created_at: str
    updated_at: str
    archived: int


@dataclass(frozen=True)
class MessageRow:
    id: str
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    report_json: Optional[Dict[str, Any]]
    sources_json: Optional[List[Dict[str, Any]]]
    trace_json: Optional[List[Dict[str, Any]]]
    created_at: str


class SessionStore:
    """SQLite-backed session/message store.

    Thread safety:
    - Create a fresh sqlite3 connection per operation.
    - Use WAL + busy_timeout for concurrent readers/writers.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                  session_id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  title TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  archived INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  report_json TEXT,
                  sources_json TEXT,
                  trace_json TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_mode_updated ON chat_sessions(user_id, mode, updated_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_created ON chat_messages(session_id, created_at ASC);"
            )

    def create_session(self, *, session_id: str, user_id: str, mode: str, title: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            # 不要用 REPLACE：如果 session_id 意外冲突，会覆盖另一条会话（导致跨模式“互相消失”）
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions
                  (session_id, user_id, mode, title, created_at, updated_at, archived)
                VALUES
                  (?, ?, ?, ?, ?, ?, 0)
                """,
                (session_id, user_id, mode, title, now, now),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET title = COALESCE(NULLIF(?, ''), title), updated_at = ?
                WHERE session_id = ? AND user_id = ?
                """,
                (title, now, session_id, user_id),
            )

    def touch_session(self, *, session_id: str, user_id: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE session_id = ? AND user_id = ?
                """,
                (now, session_id, user_id),
            )

    def rename_session(self, *, session_id: str, user_id: str, title: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ?
                """,
                (title, now, session_id, user_id),
            )

    def archive_session(self, *, session_id: str, user_id: str, archived: bool = True) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET archived = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ?
                """,
                (1 if archived else 0, now, session_id, user_id),
            )

    def list_sessions(self, *, user_id: str, mode: str, include_archived: bool = False) -> List[SessionRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, user_id, mode, title, created_at, updated_at, archived
                FROM chat_sessions
                WHERE user_id = ? AND mode = ? AND (? = 1 OR archived = 0)
                ORDER BY updated_at DESC
                LIMIT 200
                """,
                (user_id, mode, 1 if include_archived else 0),
            ).fetchall()
        return [SessionRow(**dict(r)) for r in rows]

    def add_message(
        self,
        *,
        message_id: str,
        session_id: str,
        user_id: str,
        mode: str,
        role: str,
        content: str,
        report: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            # Ensure session exists (idempotent for hybrid client behavior)
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions
                  (session_id, user_id, mode, title, created_at, updated_at, archived)
                VALUES
                  (?, ?, ?, ?, ?, ?, 0)
                """,
                (session_id, user_id, mode, "新会话", now, now),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO chat_messages
                  (id, session_id, role, content, report_json, sources_json, trace_json, created_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(report, ensure_ascii=False) if report is not None else None,
                    json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                    json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ? AND user_id = ?",
                (now, session_id, user_id),
            )

    def list_messages(self, *, session_id: str, user_id: str, limit: int = 200) -> List[MessageRow]:
        with self._connect() as conn:
            # Ownership check via join
            rows = conn.execute(
                """
                SELECT m.*
                FROM chat_messages m
                JOIN chat_sessions s ON s.session_id = m.session_id
                WHERE m.session_id = ? AND s.user_id = ?
                ORDER BY m.created_at ASC
                LIMIT ?
                """,
                (session_id, user_id, limit),
            ).fetchall()

        def _loads(v: Any) -> Any:
            if v is None:
                return None
            try:
                return json.loads(v)
            except Exception:
                return None

        out: List[MessageRow] = []
        for r in rows:
            d = dict(r)
            out.append(
                MessageRow(
                    id=str(d["id"]),
                    session_id=str(d["session_id"]),
                    role=str(d["role"]),
                    content=str(d["content"]),
                    report_json=_loads(d.get("report_json")),
                    sources_json=_loads(d.get("sources_json")),
                    trace_json=_loads(d.get("trace_json")),
                    created_at=str(d["created_at"]),
                )
            )
        return out

