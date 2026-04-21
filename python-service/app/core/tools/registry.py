from __future__ import annotations

from typing import Dict, Iterable, Optional

from app.core.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> Iterable[str]:
        return self._tools.keys()

