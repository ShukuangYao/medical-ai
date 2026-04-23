from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass(frozen=True)
class FeedbackRow:
    feedback_id: str
    run_id: str
    session_id: str
    message_id: str
    mode: str
    user_id: str
    rating: int
    comment: str
    corrected_answer: str
    created_at: str


class FeedbackStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_feedback (
                  feedback_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  message_id TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  rating INTEGER NOT NULL,
                  comment TEXT NOT NULL DEFAULT '',
                  corrected_answer TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run_id ON chat_feedback(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session_msg ON chat_feedback(session_id, message_id);")

    def add_feedback(
        self,
        *,
        feedback_id: str,
        run_id: str,
        session_id: str,
        message_id: str,
        mode: str,
        user_id: str,
        rating: int,
        comment: str = "",
        corrected_answer: str = "",
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chat_feedback
                  (feedback_id, run_id, session_id, message_id, mode, user_id, rating, comment, corrected_answer, created_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    run_id,
                    session_id,
                    message_id,
                    mode,
                    user_id,
                    int(rating),
                    str(comment or ""),
                    str(corrected_answer or ""),
                    now,
                ),
            )

