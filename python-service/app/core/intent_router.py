"""意图路由器 - 根据意图分发到不同处理器"""
from typing import Dict, List, Optional, AsyncGenerator
from app.core.graph_querier import GraphQuerier
from app.core.retriever import ParentRetriever
from app.core.reranker import BGEReranker
from app.core.llm_client import OpenAILLM


class IntentRouter:
    """意图路由器，根据意图类型选择最优处理策略"""

    def __init__(
        self,
        graph_querier: Optional[GraphQuerier],
        retriever: ParentRetriever,
        reranker: BGEReranker,
        llm: OpenAILLM,
    ):
        self.graph_querier = graph_querier
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    async def route(
        self, intent_result: Dict, question: str, graph_enabled: bool = True
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
            return await self._graph_enhanced_retrieve(
                intent, entity, question
            ), "graph_enhanced"

        # 4. 一般医疗问答 - 纯向量+ES混合检索
        if use_graph and not entity:
            print("[IntentRouter] graph requested but entity is empty -> fallback hybrid")
        elif use_graph and not self.graph_querier:
            print("[IntentRouter] graph requested but neo4j not ready -> fallback hybrid")
        else:
            print("[IntentRouter] route -> hybrid_retrieve")
        return await self._hybrid_retrieve(question), "hybrid_retrieve"

    async def _graph_enhanced_retrieve(
        self, intent: str, entity: str, question: str
    ) -> List[Dict]:
        """图谱增强检索：图谱结果 + 向量检索补充"""
        graph_docs: List[Dict] = []
        all_docs = []

        # 图谱查询
        if self.graph_querier and entity:
            graph_docs = self.graph_querier.query(intent, entity)
            all_docs.extend(graph_docs)
            print(
                f"[IntentRouter] neo4j query done: intent={intent}, "
                f"entity={entity}, graph_docs={len(graph_docs)}"
            )

        # 向量检索补充
        print("[IntentRouter] supplement retrieval: vector+ES only (skip graph duplicate)")
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
            supplement_docs = self.reranker.rerank(question, supplement_docs)
        final_docs = graph_kept + supplement_docs

        graph_count = sum(1 for d in final_docs if d.get("retrieval_source") == "graph")
        es_count = sum(1 for d in final_docs if d.get("retrieval_source") == "elasticsearch")
        vector_count = sum(1 for d in final_docs if d.get("retrieval_source") == "vector")
        print(
            f"[IntentRouter] final merged docs: graph={graph_count}, "
            f"vector={vector_count}, es={es_count}, total={len(final_docs)}"
        )

        return final_docs[:10]

    async def _hybrid_retrieve(self, question: str) -> List[Dict]:
        """混合检索：向量+ES+重排序"""
        docs = await self.retriever.retrieve(question)
        if len(docs) > 3:
            docs = self.reranker.rerank(question, docs)
        return docs[:10]
