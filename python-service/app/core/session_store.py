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
    thinking_steps_json: Optional[List[str]]
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
            # v2 schema (hard-isolate rag/agent by introducing session_key = "{mode}:{session_id}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions_v2 (
                  session_key TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
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
                CREATE TABLE IF NOT EXISTS chat_messages_v2 (
                  id TEXT PRIMARY KEY,
                  session_key TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  report_json TEXT,
                  sources_json TEXT,
                  trace_json TEXT,
                  thinking_steps_json TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(session_key) REFERENCES chat_sessions_v2(session_key) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_v2_user_mode_updated ON chat_sessions_v2(user_id, mode, updated_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_v2_session_created ON chat_messages_v2(session_key, created_at ASC);"
            )

            # legacy schema (kept for backward compatibility & migration)
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
                  thinking_steps_json TEXT,
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

            self._migrate_to_v2(conn)
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        """Ensure optional columns exist for forward-compatible upgrades."""

        def _has_column(table: str, col: str) -> bool:
            try:
                rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
                return any(str(r["name"]) == col for r in rows)
            except Exception:
                return False

        def _add_column(table: str, col: str, decl: str) -> None:
            if _has_column(table, col):
                return
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl};")
            except Exception:
                return

        _add_column("chat_messages_v2", "thinking_steps_json", "TEXT")
        _add_column("chat_messages", "thinking_steps_json", "TEXT")

    @staticmethod
    def _session_key(*, session_id: str, mode: str) -> str:
        return f"{mode}:{session_id}"

    def _migrate_to_v2(self, conn: sqlite3.Connection) -> None:
        """One-way best-effort migration from legacy schema into v2 tables.

        Safe to run multiple times (INSERT OR IGNORE).
        """
        # If legacy tables don't exist / empty, nothing to do.
        try:
            legacy_sessions = conn.execute(
                "SELECT session_id, user_id, mode, title, created_at, updated_at, archived FROM chat_sessions"
            ).fetchall()
        except Exception:
            return

        for s in legacy_sessions:
            d = dict(s)
            sk = self._session_key(session_id=str(d["session_id"]), mode=str(d["mode"]))
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions_v2
                  (session_key, session_id, user_id, mode, title, created_at, updated_at, archived)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sk,
                    d["session_id"],
                    d["user_id"],
                    d["mode"],
                    d["title"],
                    d["created_at"],
                    d["updated_at"],
                    d["archived"],
                ),
            )

        try:
            legacy_messages = conn.execute(
                "SELECT id, session_id, role, content, report_json, sources_json, trace_json, created_at FROM chat_messages"
            ).fetchall()
        except Exception:
            return

        # Need session mode/user_id from sessions table
        mode_by_session: Dict[str, str] = {}
        try:
            rows = conn.execute("SELECT session_id, mode FROM chat_sessions").fetchall()
            mode_by_session = {str(r["session_id"]): str(r["mode"]) for r in rows}
        except Exception:
            mode_by_session = {}

        for m in legacy_messages:
            d = dict(m)
            mode = mode_by_session.get(str(d["session_id"]), "rag")
            sk = self._session_key(session_id=str(d["session_id"]), mode=mode)
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_messages_v2
                  (id, session_key, session_id, mode, role, content, report_json, sources_json, trace_json, created_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["id"],
                    sk,
                    d["session_id"],
                    mode,
                    d["role"],
                    d["content"],
                    d.get("report_json"),
                    d.get("sources_json"),
                    d.get("trace_json"),
                    d["created_at"],
                ),
            )

    def create_session(self, *, session_id: str, user_id: str, mode: str, title: str) -> None:
        now = _utc_now()
        session_key = self._session_key(session_id=session_id, mode=mode)
        with self._connect() as conn:
            # v2
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions_v2
                  (session_key, session_id, user_id, mode, title, created_at, updated_at, archived)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (session_key, session_id, user_id, mode, title, now, now),
            )
            conn.execute(
                """
                UPDATE chat_sessions_v2
                SET title = COALESCE(NULLIF(?, ''), title), updated_at = ?
                WHERE session_key = ? AND user_id = ?
                """,
                (title, now, session_key, user_id),
            )

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

    def rename_session(self, *, session_id: str, user_id: str, mode: str, title: str) -> None:
        now = _utc_now()
        session_key = self._session_key(session_id=session_id, mode=mode)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions_v2
                SET title = ?, updated_at = ?
                WHERE session_key = ? AND user_id = ?
                """,
                (title, now, session_key, user_id),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ?
                """,
                (title, now, session_id, user_id),
            )

    def archive_session(self, *, session_id: str, user_id: str, mode: str, archived: bool = True) -> None:
        now = _utc_now()
        session_key = self._session_key(session_id=session_id, mode=mode)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions_v2
                SET archived = ?, updated_at = ?
                WHERE session_key = ? AND user_id = ?
                """,
                (1 if archived else 0, now, session_key, user_id),
            )
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
                FROM chat_sessions_v2
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
        thinking_steps: Optional[List[str]] = None,
    ) -> None:
        now = _utc_now()
        session_key = self._session_key(session_id=session_id, mode=mode)
        with self._connect() as conn:
            # v2
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions_v2
                  (session_key, session_id, user_id, mode, title, created_at, updated_at, archived)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (session_key, session_id, user_id, mode, "新会话", now, now),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO chat_messages_v2
                  (id, session_key, session_id, mode, role, content, report_json, sources_json, trace_json, thinking_steps_json, created_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_key,
                    session_id,
                    mode,
                    role,
                    content,
                    json.dumps(report, ensure_ascii=False) if report is not None else None,
                    json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                    json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                    json.dumps(thinking_steps, ensure_ascii=False) if thinking_steps is not None else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions_v2 SET updated_at = ? WHERE session_key = ? AND user_id = ?",
                (now, session_key, user_id),
            )

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
                  (id, session_id, role, content, report_json, sources_json, trace_json, thinking_steps_json, created_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(report, ensure_ascii=False) if report is not None else None,
                    json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                    json.dumps(trace, ensure_ascii=False) if trace is not None else None,
                    json.dumps(thinking_steps, ensure_ascii=False) if thinking_steps is not None else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ? AND user_id = ?",
                (now, session_id, user_id),
            )

    def list_messages(self, *, session_id: str, user_id: str, mode: str, limit: int = 200) -> List[MessageRow]:
        session_key = self._session_key(session_id=session_id, mode=mode)
        with self._connect() as conn:
            # Ownership check via join
            rows = conn.execute(
                """
                SELECT m.*
                FROM chat_messages_v2 m
                JOIN chat_sessions_v2 s ON s.session_key = m.session_key
                WHERE m.session_key = ? AND s.user_id = ?
                ORDER BY m.created_at ASC
                LIMIT ?
                """,
                (session_key, user_id, limit),
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
                    thinking_steps_json=_loads(d.get("thinking_steps_json")),
                    created_at=str(d["created_at"]),
                )
            )
        return out

