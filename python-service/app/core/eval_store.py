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
class EvalSampleRow:
    sample_id: str
    run_id: str
    session_id: str
    message_id: str
    mode: str
    user_id: str
    input_text: str
    output_text: str
    report_json: Optional[Dict[str, Any]]
    sources_json: Optional[List[Dict[str, Any]]]
    rating: int
    comment: str
    corrected_answer: str
    created_at: str


class EvalSampleStore:
    """Minimal demo-oriented evaluation sample store (SQLite).

    Each row represents one user feedback event + its associated (input, output, sources, report).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_samples (
                  sample_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  message_id TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  input_text TEXT NOT NULL,
                  output_text TEXT NOT NULL,
                  report_json TEXT,
                  sources_json TEXT,
                  rating INTEGER NOT NULL,
                  comment TEXT NOT NULL DEFAULT '',
                  corrected_answer TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_samples_created ON eval_samples(created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_samples_run_id ON eval_samples(run_id);")

    def add_sample(
        self,
        *,
        sample_id: str,
        run_id: str,
        session_id: str,
        message_id: str,
        mode: str,
        user_id: str,
        input_text: str,
        output_text: str,
        report: Optional[Dict[str, Any]],
        sources: Optional[List[Dict[str, Any]]],
        rating: int,
        comment: str,
        corrected_answer: str,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_samples
                  (sample_id, run_id, session_id, message_id, mode, user_id, input_text, output_text, report_json, sources_json,
                   rating, comment, corrected_answer, created_at)
                VALUES
                  (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    run_id,
                    session_id,
                    message_id,
                    mode,
                    user_id,
                    input_text,
                    output_text,
                    json.dumps(report, ensure_ascii=False) if report is not None else None,
                    json.dumps(sources, ensure_ascii=False) if sources is not None else None,
                    int(rating),
                    str(comment or ""),
                    str(corrected_answer or ""),
                    now,
                ),
            )

    def list_samples(self, *, limit: int = 50) -> List[EvalSampleRow]:
        lim = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM eval_samples
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()

        def _loads(v: Any) -> Any:
            if v is None:
                return None
            try:
                return json.loads(v)
            except Exception:
                return None

        out: List[EvalSampleRow] = []
        for r in rows:
            d = dict(r)
            out.append(
                EvalSampleRow(
                    sample_id=str(d["sample_id"]),
                    run_id=str(d["run_id"]),
                    session_id=str(d["session_id"]),
                    message_id=str(d["message_id"]),
                    mode=str(d["mode"]),
                    user_id=str(d["user_id"]),
                    input_text=str(d["input_text"]),
                    output_text=str(d["output_text"]),
                    report_json=_loads(d.get("report_json")),
                    sources_json=_loads(d.get("sources_json")),
                    rating=int(d.get("rating") or 0),
                    comment=str(d.get("comment") or ""),
                    corrected_answer=str(d.get("corrected_answer") or ""),
                    created_at=str(d.get("created_at") or ""),
                )
            )
        return out

