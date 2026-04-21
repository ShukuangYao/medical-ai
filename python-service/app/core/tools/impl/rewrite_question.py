from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.rewrite_chain import RewriteQuestionChain
from app.core.tools.base import BaseTool, ToolContext, ToolError


class RewriteQuestionArgs(BaseModel):
    question: str = Field(min_length=1)
    chat_history: Optional[List[Dict[str, str]]] = None


class RewriteQuestionTool(BaseTool[RewriteQuestionArgs, str]):
    name = "rewrite_question"
    ArgsModel = RewriteQuestionArgs

    def __init__(self, chain: RewriteQuestionChain):
        self._chain = chain

    async def run(self, *, args: RewriteQuestionArgs, ctx: ToolContext) -> str:
        try:
            return await self._chain.rewrite(args.question, args.chat_history)
        except Exception as e:
            raise ToolError(code="INTERNAL", message="rewrite_question failed", retriable=True, detail={"err": str(e)}) from e

