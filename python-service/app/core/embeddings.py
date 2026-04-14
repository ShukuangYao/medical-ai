"""BGE嵌入模型 - 使用BAAI/bge-large-zh"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings


class BGEEmbeddings:
    """文本嵌入模型，将文本转换为高维向量"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None

    def load(self):
        """加载嵌入模型"""
        print(f"正在加载嵌入模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"嵌入模型加载完成，维度: {self.model.get_sentence_embedding_dimension()}")

    def embed_query(self, text: str) -> List[float]:
        """将单条查询文本转为向量"""
        if self.model is None:
            self.load()
        # BGE模型查询时需要加前缀
        query_text = f"为这个句子生成表示以用于检索相关文章：{text}"
        embedding = self.model.encode(query_text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多条文档文本批量转为向量"""
        if self.model is None:
            self.load()
        # macOS MPS 上 batch_size 太大会触发 out of memory
        # normalize_embeddings=True：对向量做归一化（配合向量数据库的相似度度量更方便）
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=8,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """返回嵌入维度"""
        if self.model is None:
            self.load()
        return self.model.get_sentence_embedding_dimension()
