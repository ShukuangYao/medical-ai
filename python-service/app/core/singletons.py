"""Process-wide singletons for heavy components.

This avoids re-initializing embedding/reranker/clients on every request or per-router startup.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.core.agent_orchestrator import MedicalAgentOrchestrator
from app.core.graph_querier import GraphQuerier
from app.core.rag_engine import LocalDocQA


rag_engine = LocalDocQA()

_rag_init_lock = asyncio.Lock()
_rag_inited = False


async def ensure_rag_initialized() -> None:
    global _rag_inited
    if _rag_inited:
        return
    async with _rag_init_lock:
        if _rag_inited:
            return
        await rag_engine.initialize()
        _rag_inited = True


_agent_lock = asyncio.Lock()
_agent_orchestrator: Optional[MedicalAgentOrchestrator] = None


async def get_agent_orchestrator() -> MedicalAgentOrchestrator:
    global _agent_orchestrator
    if _agent_orchestrator is not None:
        return _agent_orchestrator
    async with _agent_lock:
        if _agent_orchestrator is None:
            await ensure_rag_initialized()
            graph = GraphQuerier()
            _agent_orchestrator = MedicalAgentOrchestrator(rag_engine, graph)
        return _agent_orchestrator

