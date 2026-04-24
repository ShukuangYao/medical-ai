from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Set

from app.core.tools.base import BaseTool, ToolContext, ToolError

DEFAULT_ANONYMOUS_USER_ID = "anonymous"


def _parse_csv(s: str) -> Set[str]:
    items = set()
    for x in (s or "").split(","):
        x2 = x.strip()
        if x2:
            items.add(x2)
    return items


@dataclass(frozen=True)
class AuthPolicy:
    """Tool-level hard authorization.

    Current implementation is intentionally minimal:
    - off: allow all calls
    - allowlist: require ctx.user_id to be in allowlist for tools that set `auth_required=True`
    """

    mode: str = "off"  # off | allowlist
    allow_users: Set[str] = None  # type: ignore[assignment]

    @classmethod
    def from_env(cls) -> "AuthPolicy":
        mode = (os.getenv("TOOL_AUTH_MODE", "off") or "off").strip().lower()
        if mode not in {"off", "allowlist"}:
            mode = "off"
        allow_users = _parse_csv(os.getenv("TOOL_AUTH_ALLOW_USERS", ""))
        return cls(mode=mode, allow_users=allow_users)

    def authorize(self, *, tool: BaseTool, ctx: ToolContext) -> None:
        if not getattr(tool, "auth_required", False):
            return
        if self.mode == "off":
            return
        # allowlist mode
        uid = (ctx.user_id or "").strip() or DEFAULT_ANONYMOUS_USER_ID
        if uid not in (self.allow_users or set()):
            raise ToolError(code="PERMISSION_DENIED", message="tool access denied", retriable=False)

