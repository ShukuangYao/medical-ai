"""BGE重排序模型 - 使用BAAI/bge-reranker-large"""
import os
from typing import List, Dict
from sentence_transformers import CrossEncoder
from app.config import settings


class BGEReranker:
    """重排序模型，对检索结果进行精确排序

    功能：
    1. CrossEncoder精排打分
    2. 绝对分数过滤：score < 0.28 丢弃
    3. 相对分数过滤：与最高分差 > 50% 丢弃
    4. 容错机制：重排序失败时退化为向量相似度打分
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.model = None
        self.score_threshold = settings.RERANKER_SCORE_THRESHOLD
        self.relative_threshold = settings.RERANKER_RELATIVE_THRESHOLD
        # 避免 MPS 显存不足，默认优先 CPU；可通过环境变量覆盖
        self.device = os.getenv("RERANK_DEVICE", "cpu")
        self.batch_size = int(os.getenv("RERANK_BATCH_SIZE", "8"))

    def load(self):
        """加载重排序模型"""
        print(f"正在加载重排序模型: {self.model_name}")
        # CrossEncoder（交叉编码器）是一类用于文本对相关性打分/重排序（reranking）的模型。它的特点是：把“query 和 candidate 文本拼在一起”一起喂给模型，而不是分别编码再做向量相似度。
        # 输入形如 [CLS] query [SEP] doc [SEP]，让模型直接输出一个相关性分数（准，适合精排/重排序）。
        # 常比向量相似度更“懂语义匹配”（
        self.model = CrossEncoder(self.model_name, max_length=512, device=self.device)
        print(f"重排序模型加载完成（device={self.device}, batch_size={self.batch_size}）")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 候选文档列表，每个文档需包含 'text' 字段
            top_k: 返回前K个结果

        Returns:
            重排序后的文档列表，每个文档增加 'rerank_score' 字段
        """
        k = top_k or settings.RERANK_TOP_K
        if not documents:
            return []

        try:
            return self._rerank_with_model(query, documents, k)
        except Exception as e:
            print(f"重排序失败，退化为向量相似度: {e}")
            return self._fallback_rerank(documents, k)

    def _rerank_with_model(
        self, query: str, documents: List[Dict], top_k: int
    ) -> List[Dict]:
        """使用CrossEncoder模型重排序"""
        if self.model is None:
            self.load()

        # 构建query-document对
        pairs = [(query, doc["text"]) for doc in documents]

        # CrossEncoder打分
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 为每个文档添加重排序分数
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        # 按重排序分数降序排列
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # 过滤1：绝对分数阈值过滤（score < 0.28 丢弃）
        filtered = [d for d in sorted_docs if d["rerank_score"] >= self.score_threshold]

        # 如果全部被过滤，保留最高分的文档
        if not filtered and sorted_docs:
            filtered = [sorted_docs[0]]

        # 过滤2：相对分数过滤（与最高分差 > 50% 丢弃）
        if filtered:
            max_score = filtered[0]["rerank_score"]
            if max_score > 0:
                threshold = max_score * self.relative_threshold
                filtered = [d for d in filtered if d["rerank_score"] >= threshold]

        return filtered[:top_k]

    @staticmethod
    def _fallback_rerank(documents: List[Dict], top_k: int) -> List[Dict]:
        """容错机制：退化为向量相似度打分排序"""
        # 使用原始检索分数排序
        for doc in documents:
            doc["rerank_score"] = doc.get("score", 0.0)

        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_docs[:top_k]
