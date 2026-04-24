from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.graph_querier import GraphQuerier
from app.core.tools.base import BaseTool, ToolContext, ToolError


class GraphQueryArgs(BaseModel):
    intent: str = Field(min_length=1)
    entity: str = Field(min_length=1)


class GraphQueryTool(BaseTool[GraphQueryArgs, List[Dict[str, Any]]]):
    name = "graph_query"
    ArgsModel = GraphQueryArgs
    auth_required = True

    def __init__(self, graph_querier: Optional[GraphQuerier]):
        self._graph = graph_querier

    async def run(self, *, args: GraphQueryArgs, ctx: ToolContext) -> List[Dict[str, Any]]:
        if self._graph is None:
            return []
        try:
            return self._graph.query(args.intent, args.entity)
        except Exception as e:
            raise ToolError(code="INTERNAL", message="graph_query failed", retriable=True, detail={"err": str(e)}) from e

