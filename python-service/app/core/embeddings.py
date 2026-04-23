"""BGE嵌入模型 - 使用BAAI/bge-large-zh"""
import os
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
        # On macOS, MPS can OOM easily for large embedding models (e.g. bge-large-zh).
        # Default to CPU for stability; override via EMBEDDING_DEVICE=cpu|mps|cuda.
        device = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
        if not device:
            device = "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)
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
        # normalize_embeddings=True：对向量做归一化（配合向量数据库的相似度度量更方便）
        # Try a conservative batch_size; if MPS OOM happens, retry with smaller batch / CPU.
        batch = int(os.getenv("EMBEDDING_BATCH_SIZE", "8") or "8")
        batch = max(1, min(batch, 64))
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except RuntimeError as e:
            msg = str(e)
            if "MPS backend out of memory" not in msg and "out of memory" not in msg:
                raise
            # Retry: smaller batch
            try:
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=1,
                    show_progress_bar=False,
                )
                return embeddings.tolist()
            except Exception:
                # Last resort: switch to CPU and retry once.
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=1,
                    show_progress_bar=False,
                )
                return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """返回嵌入维度"""
        if self.model is None:
            self.load()
        return self.model.get_sentence_embedding_dimension()
