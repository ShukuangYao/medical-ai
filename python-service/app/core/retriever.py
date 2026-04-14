"""统一检索器 - 整合向量检索+关键词检索+图谱检索的混合检索策略"""
from typing import List, Dict, Optional
from app.core.vector_store import VectorStoreMilvusClient
from app.core.es_store import StoreElasticSearchClient
from app.core.embeddings import BGEEmbeddings
from app.config import settings


class ParentRetriever:
    """统一检索器，整合多种检索策略

    检索流程：
    第一阶段：向量检索（MMR算法，兼顾相关+多样）
    第二阶段：全文检索（BM25关键词精确匹配）
    第三阶段：结果合并与去重
    """

    def __init__(
        self,
        vector_store: VectorStoreMilvusClient,
        es_store: StoreElasticSearchClient,
        embeddings: BGEEmbeddings,
    ):
        self.vector_store = vector_store
        self.es_store = es_store
        self.embeddings = embeddings

    async def retrieve(self, query: str) -> List[Dict]:
        """
        混合检索：向量检索 + 关键词检索 + 合并去重

        说明：
        - 图谱检索统一由上层 `IntentRouter` 负责（避免双入口导致策略分叉）。
        - 本检索器专注于向量/ES 两路召回与去重合并。

        Args:
            query: 查询文本

        Returns:
            去重合并后的文档列表
        """
        # 第一阶段：向量检索（MMR）
        vector_docs = await self._vector_retrieve(query)

        # 第二阶段：全文检索（BM25）
        es_docs = await self._es_retrieve(query)

        # 第三阶段：合并去重
        merged = self._merge_and_deduplicate([], vector_docs, es_docs)

        print(
            f"[ParentRetriever] vector={len(vector_docs)}, ES={len(es_docs)}, merged={len(merged)}"
        )
        return merged

    async def _vector_retrieve(self, query: str) -> List[Dict]:
        """第一阶段：向量检索（MMR算法）"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            docs = self.vector_store.mmr_search(
                query_embedding=query_embedding,
                top_k=settings.VECTOR_TOP_K,
                fetch_k=settings.MMR_FETCH_K,
                lambda_mult=settings.MMR_LAMBDA,
            )
            for doc in docs:
                doc["retrieval_source"] = "vector"
            return docs
        except Exception as e:
            print(f"向量检索失败: {e}")
            return []

    async def _es_retrieve(self, query: str) -> List[Dict]:
        """第二阶段：全文检索（BM25关键词匹配）"""
        try:
            docs = self.es_store.search(query, top_k=settings.ES_TOP_K)
            for doc in docs:
                doc["retrieval_source"] = "elasticsearch"
            return docs
        except Exception as e:
            print(f"ES检索失败: {e}")
            return []

    '''
    静态方法
    这个方法不需要 self（实例对象）
    也不需要 cls（类本身）
    只靠传入的参数就能工作
    普通方法：def foo(self, ...)（会用到实例）
    @classmethod：def bar(cls, ...)（会用到类）
    @staticmethod：def baz(... )（用不到 self/cls
    '''
    @staticmethod
    def _merge_and_deduplicate(
        graph_docs: List[Dict], vector_docs: List[Dict], es_docs: List[Dict]
    ) -> List[Dict]:
        """合并三路检索结果并去重（图谱优先级最高）"""
        seen_ids = set()
        merged = []

        # 第一优先级：图谱检索结果
        for doc in graph_docs:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)

        # 第二优先级：向量检索结果
        for doc in vector_docs:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)

        # 第三优先级：ES检索结果
        for doc in es_docs:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(doc)

        return merged
