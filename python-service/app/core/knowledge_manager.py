"""知识库管理器 - 管理知识库元数据和文档索引"""
from typing import List, Dict
from app.core.vector_store import VectorStoreMilvusClient
from app.core.es_store import StoreElasticSearchClient
from app.core.embeddings import BGEEmbeddings


class KnowledgeBaseManager:
    """知识库管理器，负责文档的索引构建和管理"""

    def __init__(
        self,
        vector_store: VectorStoreMilvusClient,
        es_store: StoreElasticSearchClient,
        embeddings: BGEEmbeddings,
    ):
        self.vector_store = vector_store
        self.es_store = es_store
        self.embeddings = embeddings

    def index_documents(self, documents: List[Dict]):
        """
        将文档同时索引到Milvus和ES

        Args:
            documents: 文档列表，每个文档需包含 id, text, title, source, page
        """
        if not documents:
            return

        # 提取文本用于嵌入
        texts = [doc["text"] for doc in documents]

        # 批量生成嵌入向量
        print(f"正在为 {len(texts)} 条文档生成嵌入向量...")
        embeddings = self.embeddings.embed_documents(texts)

        # 写入Milvus
        print("正在写入Milvus...")
        self.vector_store.insert(documents, embeddings)

        # 写入ES
        print("正在写入ElasticSearch...")
        self.es_store.insert(documents)

        print(f"知识库索引完成: {len(documents)} 条文档")

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "milvus_count": self.vector_store.count(),
            "es_count": self.es_store.count(),
        }
