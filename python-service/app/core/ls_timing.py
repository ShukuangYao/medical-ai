from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Tuple


def now_utc() -> datetime:
    """Timezone-aware UTC datetime for LangSmith start/end_time."""
    return datetime.now(timezone.utc)


def perf_ms_since(t0: float) -> int:
    """Milliseconds since perf_counter() timestamp."""
    return int((time.perf_counter() - t0) * 1000)


def span_times() -> Tuple[datetime, float]:
    """Convenience: (start_time_utc, t0_perf_counter)."""
    return now_utc(), time.perf_counter()

