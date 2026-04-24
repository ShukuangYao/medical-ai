"""意图路由器 - 根据意图分发到不同处理器"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import hashlib
import time

from app.config import settings
from app.core.graph_querier import GraphQuerier
from app.core.retriever import ParentRetriever
from app.core.reranker import BGEReranker
from app.core.llm_client import OpenAILLM
from app.core.tools.base import ToolContext
from app.core.tools.executor import ToolExecutor


class IntentRouter:
    """意图路由器，根据意图类型选择最优处理策略"""

    def __init__(
        self,
        graph_querier: Optional[GraphQuerier],
        retriever: ParentRetriever,
        reranker: BGEReranker,
        llm: OpenAILLM,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        self.graph_querier = graph_querier
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.tools = tool_executor
        # In-process cache for expensive rerank calls (RAG). key -> (ts, docs)
        self._rerank_cache: Dict[str, Tuple[float, List[Dict]]] = {}

    @staticmethod
    def _doc_id(doc: Dict) -> str:
        return str(doc.get("id") or doc.get("doc_id") or doc.get("source_id") or doc.get("text", "")[:80])

    def _rerank_cache_key(self, *, query: str, docs: List[Dict], top_k: int) -> str:
        # Make key robust to doc ordering jitter across retrieval calls.
        ids = sorted([self._doc_id(d) for d in docs])[:200]
        payload = f"{query}\n{top_k}\n" + "\n".join(ids)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _prefilter_rerank_candidates(self, docs: List[Dict], *, max_n: int) -> List[Dict]:
        """Reduce rerank candidates using cheap heuristics (score/source)."""
        if max_n <= 0 or len(docs) <= max_n:
            return docs

        def score_key(d: Dict) -> float:
            # prefer already-scored results; default 0
            v = d.get("score")
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        # Keep a bit more from each source if present; otherwise just top by score.
        by_source: Dict[str, List[Dict]] = {}
        for d in docs:
            src = str(d.get("retrieval_source") or "unknown")
            by_source.setdefault(src, []).append(d)
        for src, arr in by_source.items():
            arr.sort(key=score_key, reverse=True)

        picked: List[Dict] = []
        # Prefer graph docs if any (they typically are high precision)
        if "graph" in by_source:
            picked.extend(by_source["graph"][: min(6, max_n)])

        # Then take from vector/ES, proportional-ish
        for src in ("vector", "elasticsearch", "unknown"):
            if src in by_source and len(picked) < max_n:
                remaining = max_n - len(picked)
                take = min(remaining, max(4, remaining))
                picked.extend(by_source[src][:take])

        # If still short, fill from global top-by-score
        if len(picked) < max_n:
            rest = [d for d in docs if d not in picked]
            rest.sort(key=score_key, reverse=True)
            picked.extend(rest[: max_n - len(picked)])

        # De-dup by id
        seen = set()
        out: List[Dict] = []
        for d in picked:
            i = self._doc_id(d)
            if i in seen:
                continue
            seen.add(i)
            out.append(d)
            if len(out) >= max_n:
                break
        return out

    async def _maybe_rerank(self, *, query: str, docs: List[Dict], top_k: int, tool_ctx: ToolContext) -> List[Dict]:
        if not docs:
            return []
        if len(docs) <= top_k:
            return docs
        # Prefilter to bound expensive rerank
        max_n = int(getattr(settings, "RAG_RERANK_CANDIDATES_MAX", 24))
        candidates = self._prefilter_rerank_candidates(docs, max_n=max_n)

        ttl = float(getattr(settings, "RAG_RERANK_CACHE_TTL_S", 120.0))
        uid = (tool_ctx.user_id or "").strip() or "anonymous"
        sid = (tool_ctx.session_id or "").strip() or "default"
        rid = (tool_ctx.run_id or "").strip()
        key = f"{uid}:{sid}:{rid}:{self._rerank_cache_key(query=query, docs=candidates, top_k=top_k)}"
        now = time.time()
        cached = self._rerank_cache.get(key)
        if cached and (now - float(cached[0])) <= ttl:
            return cached[1]

        if self.tools is not None:
            out = await self.tools.run(
                "rerank",
                args={"query": query, "docs": candidates, "top_k": top_k},
                ctx=tool_ctx,
                trace_inputs={"query": query, "docs_count": len(candidates), "top_k": top_k, "cached": False},
                idempotency_key=f"rag:{uid}:{sid}:{rid}:rerank:{query}:{len(candidates)}:{top_k}",
                idempotency_ttl_s=ttl,
            )
        else:
            out = self.reranker.rerank(query, candidates, top_k=top_k)
        try:
            self._rerank_cache[key] = (now, out)
        except Exception:
            pass
        return out

    async def route(
        self,
        intent_result: Dict,
        question: str,
        graph_enabled: bool = True,
        tool_ctx: Optional[ToolContext] = None,
    ) -> tuple[List[Dict], str]:
        """
        根据意图路由到不同处理器

        Args:
            intent_result: 意图识别结果
            question: 用户问题

        Returns:
            (检索文档列表, 路由策略说明)
        """
        intent = intent_result["intent"]
        use_graph = bool(intent_result["use_graph"]) and graph_enabled
        entity = intent_result["entity"]
        ctx = tool_ctx or ToolContext(mode="rag")
        print(
            f"[IntentRouter] intent={intent}, use_graph={use_graph}, graph_enabled={graph_enabled}, "
            f"entity={entity!r}, neo4j_ready={self.graph_querier is not None}"
        )

        # 1. 问候/感谢类 - 直接返回空文档，LLM生成礼貌回复
        if intent in ["greeting", "thanks"]:
            print("[IntentRouter] skip retrieval: greeting/thanks")
            return [], "direct_llm"

        # 2. 超出范围 - 返回空文档，LLM拒绝回答
        if intent == "out_of_scope":
            print("[IntentRouter] skip retrieval: out_of_scope")
            return [], "out_of_scope"

        # 3. 图谱查询类 - 优先图谱，补充向量检索
        if use_graph and graph_enabled and self.graph_querier and entity:
            print(f"[IntentRouter] route -> graph_enhanced (entity={entity})")
            return await self._graph_enhanced_retrieve(intent, entity, question, tool_ctx=ctx), "graph_enhanced"

        # 4. 一般医疗问答 - 纯向量+ES混合检索
        if use_graph and not entity:
            print("[IntentRouter] graph requested but entity is empty -> fallback hybrid")
        elif use_graph and not self.graph_querier:
            print("[IntentRouter] graph requested but neo4j not ready -> fallback hybrid")
        else:
            print("[IntentRouter] route -> hybrid_retrieve")
        return await self._hybrid_retrieve(question, tool_ctx=ctx), "hybrid_retrieve"

    async def _graph_enhanced_retrieve(
        self, intent: str, entity: str, question: str, *, tool_ctx: ToolContext
    ) -> List[Dict]:
        """图谱增强检索：图谱结果 + 向量检索补充"""
        graph_docs: List[Dict] = []
        all_docs = []

        # 图谱查询
        uid = (tool_ctx.user_id or "").strip() or "anonymous"
        sid = (tool_ctx.session_id or "").strip() or "default"
        rid = (tool_ctx.run_id or "").strip()

        if self.graph_querier and entity:
            if self.tools is not None:
                graph_docs = await self.tools.run(
                    "graph_query",
                    args={"intent": intent, "entity": entity},
                    ctx=tool_ctx,
                    trace_inputs={"intent": intent, "entity": entity},
                    idempotency_key=f"rag:{uid}:{sid}:{rid}:graph_query:{intent}:{entity}",
                    idempotency_ttl_s=float(getattr(settings, "RAG_RERANK_CACHE_TTL_S", 120.0)),
                )
            else:
                graph_docs = self.graph_querier.query(intent, entity)
            all_docs.extend(graph_docs)
            print(
                f"[IntentRouter] neo4j query done: intent={intent}, "
                f"entity={entity}, graph_docs={len(graph_docs)}"
            )

        # 向量检索补充
        print("[IntentRouter] supplement retrieval: vector+ES only (skip graph duplicate)")
        if self.tools is not None:
            vector_docs = await self.tools.run(
                "hybrid_retrieve",
                args={"query": question},
                ctx=tool_ctx,
                trace_inputs={"query": question},
                idempotency_key=f"rag:{uid}:{sid}:{rid}:hybrid_retrieve:{question}",
                idempotency_ttl_s=float(getattr(settings, "RAG_RERANK_CACHE_TTL_S", 120.0)),
            )
        else:
            vector_docs = await self.retriever.retrieve(question)
        all_docs.extend(vector_docs)
        print(f"[IntentRouter] vector/es supplement docs={len(vector_docs)}")

        # 去重（基于id）
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.get("id", doc.get("text", "")[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        # 图谱结果优先保留，不参与阈值淘汰；仅对补充文档重排序
        graph_ids = {d.get("id", d.get("text", "")[:50]) for d in graph_docs}
        graph_kept = [
            d for d in unique_docs
            if d.get("id", d.get("text", "")[:50]) in graph_ids
        ]
        supplement_docs = [
            d for d in unique_docs
            if d.get("id", d.get("text", "")[:50]) not in graph_ids
        ]
        if len(supplement_docs) > 3:
            supplement_docs = await self._maybe_rerank(
                query=question, docs=supplement_docs, top_k=10, tool_ctx=tool_ctx
            )
        final_docs = graph_kept + supplement_docs

        graph_count = sum(1 for d in final_docs if d.get("retrieval_source") == "graph")
        es_count = sum(1 for d in final_docs if d.get("retrieval_source") == "elasticsearch")
        vector_count = sum(1 for d in final_docs if d.get("retrieval_source") == "vector")
        print(
            f"[IntentRouter] final merged docs: graph={graph_count}, "
            f"vector={vector_count}, es={es_count}, total={len(final_docs)}"
        )

        return final_docs[:10]

    async def _hybrid_retrieve(self, question: str, *, tool_ctx: ToolContext) -> List[Dict]:
        """混合检索：向量+ES+重排序"""
        uid = (tool_ctx.user_id or "").strip() or "anonymous"
        sid = (tool_ctx.session_id or "").strip() or "default"
        rid = (tool_ctx.run_id or "").strip()
        if self.tools is not None:
            docs = await self.tools.run(
                "hybrid_retrieve",
                args={"query": question},
                ctx=tool_ctx,
                trace_inputs={"query": question},
                idempotency_key=f"rag:{uid}:{sid}:{rid}:hybrid_retrieve:{question}",
                idempotency_ttl_s=float(getattr(settings, "RAG_RERANK_CACHE_TTL_S", 120.0)),
            )
        else:
            docs = await self.retriever.retrieve(question)
        if len(docs) > 3:
            docs = await self._maybe_rerank(query=question, docs=docs, top_k=10, tool_ctx=tool_ctx)
        return docs[:10]
