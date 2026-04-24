from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from app.core.tools.base import ToolContext


def _short_sha256(obj: Any) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def make_idempotency_key(
    *,
    namespace: str,
    tool_name: str,
    ctx: ToolContext,
    args: Dict[str, Any],
    version: int = 1,
    extra: Optional[str] = None,
) -> str:
    ns = (namespace or "tool").strip()
    tn = (tool_name or "unknown").strip()
    uid = (ctx.user_id or "").strip() or "anonymous"
    sid = (ctx.session_id or "").strip() or "default"
    rid = (ctx.run_id or "").strip() or "-"
    ah = _short_sha256(args)
    ex = (extra or "").strip()
    if ex:
        return f"{ns}:{tn}:v{int(version)}:{uid}:{sid}:{rid}:{ex}:{ah}"
    return f"{ns}:{tn}:v{int(version)}:{uid}:{sid}:{rid}:{ah}"

