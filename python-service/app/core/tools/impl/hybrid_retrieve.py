from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.core.retriever import ParentRetriever
from app.core.tools.base import BaseTool, ToolContext, ToolError


class HybridRetrieveArgs(BaseModel):
    query: str = Field(min_length=1)


class HybridRetrieveTool(BaseTool[HybridRetrieveArgs, List[Dict[str, Any]]]):
    name = "hybrid_retrieve"
    ArgsModel = HybridRetrieveArgs
    auth_required = True

    def __init__(self, retriever: ParentRetriever):
        self._retriever = retriever

    async def run(self, *, args: HybridRetrieveArgs, ctx: ToolContext) -> List[Dict[str, Any]]:
        try:
            return await self._retriever.retrieve(args.query)
        except Exception as e:
            raise ToolError(code="INTERNAL", message="hybrid_retrieve failed", retriable=True, detail={"err": str(e)}) from e

