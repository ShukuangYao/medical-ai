"""Milvus向量库客户端 - 支持MMR算法检索"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.config import settings


class VectorStoreMilvusClient:
    """Milvus向量库客户端，支持MMR算法实现相关+多样检索"""

    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION
        self.collection: Optional[Collection] = None
        self.connected = False

    def connect(self):
        """连接Milvus"""
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
        self.connected = True
        print(f"已连接Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    def create_collection(self, dimension: int = None):
        """创建集合"""
        dim = dimension or settings.EMBEDDING_DIMENSION
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"集合 {self.collection_name} 已存在，已加载")
            return
        '''
          id：主键（is_primary=True），字符串，最长 128
          text：原文/拼接后的文本内容（最长 8192）
          title：标题（这里你后续会用 question[:100]）
          source：来源标记
          page：页码（这里固定 0）
          embedding：向量字段
          dtype=FLOAT_VECTOR
          dim=dim：维度必须和你的 embedding 模型输出一致（你这里是 1024）
        '''
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="医疗知识库向量集合")
        self.collection = Collection(self.collection_name, schema)

        # 创建向量索引（为了加速检索）
        '''
          index_type="IVF_FLAT"：倒排文件（IVF）+ FLAT 近似检索。
          metric_type="IP"：使用 内积 作为相似度度量。
          注释里说“因为向量已归一化，内积≈余弦相似度”——这要求你的 embedding 在插入时确实做了归一化：
          你在 embeddings.py 里 normalize_embeddings=True，所以这一块是匹配的。
          nlist=128：控制 IVF 的聚类数，数值会影响召回/速度。
        '''
        index_params = {
            "metric_type": "IP",  # 内积（因为向量已归一化，等价于余弦相似度）
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
        self.collection.load()
        print(f"集合 {self.collection_name} 创建完成")

    def ensure_collection_loaded(self):
        """确保集合已连接并加载（用于服务启动后延迟恢复）"""
        if not self.connected:
            self.connect()
        if self.collection is None:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                print(f"集合 {self.collection_name} 已加载")
            else:
                raise RuntimeError(
                    f"集合 {self.collection_name} 不存在，请先运行数据加载脚本初始化知识库"
                )

    def drop_collection(self):
        """删除集合（用于重跑时清空旧数据）"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self.collection = None
            print(f"集合 {self.collection_name} 已删除")
        else:
            print(f"集合 {self.collection_name} 不存在，无需删除")

    def insert(self, docs: List[Dict], embeddings: List[List[float]]):
        """批量插入文档和向量"""
        self.ensure_collection_loaded()

        data = [
            [doc["id"] for doc in docs],
            [doc["text"] for doc in docs],
            [doc.get("title", "") for doc in docs],
            [doc.get("source", "") for doc in docs],
            [doc.get("page", 0) for doc in docs],
            embeddings,
        ]
        self.collection.insert(data)
        self.collection.flush()
        print(f"已插入 {len(docs)} 条文档")

    def search(self, query_embedding: List[float], top_k: int = None) -> List[Dict]:
        """普通向量相似度搜索"""
        k = top_k or settings.VECTOR_TOP_K
        self.ensure_collection_loaded()

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=k,
            output_fields=["text", "title", "source", "page"]
            ,
            timeout=float(getattr(settings, "MILVUS_SEARCH_TIMEOUT_S", 2.5)),
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "title": hit.entity.get("title"),
                    "source": hit.entity.get("source"),
                    "page": hit.entity.get("page"),
                    "score": hit.score,
                })
        return docs

    def mmr_search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        fetch_k: int = None,
        lambda_mult: float = None,
    ) -> List[Dict]:
        """
        MMR（Maximal Marginal Relevance）检索
        兼顾相关性和多样性

        Args:
            query_embedding: 查询向量
            top_k: 最终返回数量
            fetch_k: 初始候选集大小
            lambda_mult: 相关性权重（1=纯相关，0=纯多样）
        """
        k = top_k or settings.VECTOR_TOP_K
        fk = fetch_k or settings.MMR_FETCH_K
        lam = lambda_mult or settings.MMR_LAMBDA

        # 第一步：获取较大的候选集
        candidates = self.search(query_embedding, top_k=fk)
        if not candidates:
            return []

        # 第二步：MMR算法选择
        query_vec = np.array(query_embedding)
        selected = []
        selected_indices = set()
        candidate_embeddings = []

        # 获取候选文档的向量（通过重新搜索获取）
        for c in candidates:
            # 使用score作为与query的相似度
            candidate_embeddings.append(c["score"])

        # 简化MMR：基于分数和已选文档的文本去重
        remaining = list(range(len(candidates)))
        # 从候选集力找到MMR分数最高的前几条
        for _ in range(min(k, len(candidates))):
            if not remaining:
                break

            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                # 相关性分数（与查询的相似度）
                relevance = candidates[idx]["score"]

                # 多样性惩罚（与已选文档的最大相似度）
                max_sim = 0.0
                if selected:
                    for sel_idx in selected_indices:
                        # 基于文本重叠度计算相似度
                        sim = self._text_similarity(
                            candidates[idx]["text"],
                            candidates[sel_idx]["text"]
                        )
                        max_sim = max(max_sim, sim)

                # MMR分数 = λ * 相关性 - (1-λ) * 最大已选相似度
                mmr_score = lam * relevance - (1 - lam) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(candidates[best_idx])
                selected_indices.add(best_idx)
                remaining.remove(best_idx)

        return selected

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """基于字符重叠的简单文本相似度"""
        if not text1 or not text2:
            return 0.0
        set1 = set[str](text1)
        set2 = set(text2)
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    def count(self) -> int:
        """返回集合中的文档数量"""
        if self.collection:
            return self.collection.num_entities
        return 0
