from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


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

