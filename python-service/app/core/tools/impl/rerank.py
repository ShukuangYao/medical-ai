from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.core.reranker import BGEReranker
from app.core.tools.base import BaseTool, ToolContext, ToolError


class RerankArgs(BaseModel):
    query: str = Field(min_length=1)
    docs: List[Dict[str, Any]] = Field(default_factory=list)
    top_k: int | None = None


class RerankTool(BaseTool[RerankArgs, List[Dict[str, Any]]]):
    name = "rerank"
    ArgsModel = RerankArgs

    def __init__(self, reranker: BGEReranker):
        self._reranker = reranker

    async def run(self, *, args: RerankArgs, ctx: ToolContext) -> List[Dict[str, Any]]:
        if not args.docs:
            return []
        try:
            # BGEReranker is sync; keep as sync for now.
            return self._reranker.rerank(args.query, args.docs, top_k=args.top_k)
        except Exception as e:
            raise ToolError(code="INTERNAL", message="rerank failed", retriable=True, detail={"err": str(e)}) from e

