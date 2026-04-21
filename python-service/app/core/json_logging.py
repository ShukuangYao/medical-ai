from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter to ensure `extra` fields are always emitted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Common structured fields we care about (extras are stored on the record).
        for k in (
            "request_id",
            "run_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "service",
            "mode",
            "cancelled",
        ):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v

        # If caller attached more extras, include them (best-effort).
        # Avoid dumping huge built-in attributes.
        blacklist = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
        for k, v in record.__dict__.items():
            if k in blacklist or k in payload:
                continue
            # keep only JSON-serializable extras
            try:
                json.dumps(v, ensure_ascii=False)
            except Exception:
                continue
            payload[k] = v

        if record.exc_info:
            try:
                payload["exc_type"] = str(record.exc_info[0].__name__)
                payload["exc_msg"] = str(record.exc_info[1])
            except Exception:
                pass
        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure root logging to emit JSON consistently."""
    root = logging.getLogger()
    root.setLevel(level)
    fmt = JsonFormatter()
    # Replace existing handlers (uvicorn config can be inconsistent across run modes).
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root.handlers = [handler]
    root.propagate = False

