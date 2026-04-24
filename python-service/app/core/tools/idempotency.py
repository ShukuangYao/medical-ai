from __future__ import annotations

import time
from dataclasses import dataclass
import json
import os
import sqlite3
import threading
from typing import Any, Dict, Optional, Protocol

import redis  # type: ignore[import-untyped]


@dataclass
class _Entry:
    value: Any
    expires_at: float


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._data: Dict[str, _Entry] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        ent = self._data.get(key)
        if ent is None:
            return None
        if ent.expires_at <= now:
            self._data.pop(key, None)
            return None
        return ent.value

    def set(self, key: str, value: Any, *, ttl_s: float) -> None:
        self._data[key] = _Entry(value=value, expires_at=time.time() + max(0.0, float(ttl_s)))


class IdempotencyStore(Protocol):
    def get(self, key: str) -> Optional[Any]: ...

    def set(self, key: str, value: Any, *, ttl_s: float) -> None: ...


class SQLiteIdempotencyStore:
    def __init__(self, *, path: str, table: str = "tool_idempotency_cache") -> None:
        self.path = path
        self.table = table
        self._lock = threading.Lock()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        # check_same_thread=False because we may call from async event loop threads
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init(self) -> None:
        with self._lock:
            con = self._conn()
            try:
                con.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                      key TEXT PRIMARY KEY,
                      value_json TEXT NOT NULL,
                      expires_at REAL NOT NULL
                    )
                    """
                )
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_expires_at ON {self.table}(expires_at)")
                con.commit()
            finally:
                con.close()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            con = self._conn()
            try:
                row = con.execute(
                    f"SELECT value_json, expires_at FROM {self.table} WHERE key=?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                value_json, expires_at = row
                if float(expires_at) <= now:
                    con.execute(f"DELETE FROM {self.table} WHERE key=?", (key,))
                    con.commit()
                    return None
                try:
                    return json.loads(str(value_json))
                except Exception:
                    return None
            finally:
                con.close()

    def set(self, key: str, value: Any, *, ttl_s: float) -> None:
        try:
            value_json = json.dumps(value, ensure_ascii=False)
        except Exception:
            # best-effort: skip caching non-serializable values
            return
        expires_at = time.time() + max(0.0, float(ttl_s))
        with self._lock:
            con = self._conn()
            try:
                con.execute(
                    f"INSERT OR REPLACE INTO {self.table}(key,value_json,expires_at) VALUES(?,?,?)",
                    (key, value_json, float(expires_at)),
                )
                con.commit()
            finally:
                con.close()


class RedisIdempotencyStore:
    def __init__(self, *, redis_url: str, prefix: str = "tool_idem:") -> None:
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = redis.Redis.from_url(redis_url)

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        v = self._client.get(self._k(key))
        if v is None:
            return None
        try:
            return json.loads(v.decode("utf-8"))
        except Exception:
            return None

    def set(self, key: str, value: Any, *, ttl_s: float) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        except Exception:
            return
        self._client.set(self._k(key), payload, ex=max(1, int(float(ttl_s))))


def create_idempotency_store(*, sqlite_path: str, redis_url: str) -> IdempotencyStore:
    backend = (os.getenv("TOOL_IDEMPOTENCY_BACKEND", "auto") or "auto").strip().lower()
    if backend not in {"auto", "memory", "sqlite", "redis"}:
        backend = "auto"
    if backend == "memory":
        return InMemoryIdempotencyStore()
    if backend == "sqlite":
        return SQLiteIdempotencyStore(path=sqlite_path)
    if backend == "redis":
        return RedisIdempotencyStore(redis_url=redis_url)
    # auto: prefer redis; fallback sqlite; fallback memory
    try:
        return RedisIdempotencyStore(redis_url=redis_url)
    except Exception:
        try:
            return SQLiteIdempotencyStore(path=sqlite_path)
        except Exception:
            return InMemoryIdempotencyStore()

