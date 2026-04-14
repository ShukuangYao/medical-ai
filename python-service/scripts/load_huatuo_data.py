"""加载华佗医疗数据集到Milvus和ElasticSearch"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets
from app.config import settings
from app.core.embeddings import BGEEmbeddings
from app.core.vector_store import VectorStoreMilvusClient
from app.core.es_store import StoreElasticSearchClient
from app.core.knowledge_manager import KnowledgeBaseManager


def load_huatuo_data(max_samples: int = 1000, reset: bool = False):
    """
    从HuggingFace加载华佗医疗数据集并索引

    Args:
        max_samples: 最大加载样本数（演示用，控制数据量）
    """
    print("=" * 60)
    print("华佗医疗数据集加载工具")
    print("=" * 60)

    # 1. 加载数据集
    print("\n[1/5] 正在从HuggingFace加载数据集...")
    huatuo = datasets.load_dataset("FreedomIntelligence/huatuo26M-testdatasets")
    data = huatuo["train"] if "train" in huatuo else list(huatuo.values())[0]
    print(f"数据集总量: {len(data)}")

    # 2. 初始化组件
    print("\n[2/5] 正在初始化组件...")
    embeddings = BGEEmbeddings()
    embeddings.load()

    vector_store = VectorStoreMilvusClient()
    vector_store.connect()

    if reset:
        print("\n[2/5] --reset 启用：清空 Milvus 旧集合...")
        vector_store.drop_collection()
    vector_store.create_collection(embeddings.dimension)

    es_store = StoreElasticSearchClient()
    es_store.connect()

    if reset:
        print("\n[2/5] --reset 启用：清空 ElasticSearch 旧索引...")
        es_store.drop_index()
    es_store.create_index()

    kb_manager = KnowledgeBaseManager(vector_store, es_store, embeddings)

    # 3. 处理数据
    print(f"\n[3/5] 正在处理数据（最多 {max_samples} 条）...")
    documents = []
    for i, item in enumerate(data):
        if i >= max_samples:
            break

        # 提取问答对作为文档
        # 数据集字段在不同版本里可能是单数/复数命名（例如 questions/answers）
        question = item.get("question", item.get("questions", item.get("input", "")))
        answer = item.get("answer", item.get("answers", item.get("output", "")))

        if not question or not answer:
            continue

        doc = {
            "id": f"huatuo_{i:06d}",
            "text": f"问题：{question}\n回答：{answer}",
            "title": question[:100],
            "source": "huatuo26M",
            "page": 0,
        }
        documents.append(doc)

    print(f"有效文档数: {len(documents)}")

    # 4. 批量索引
    print(f"\n[4/5] 正在批量索引...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        kb_manager.index_documents(batch)
        print(f"  已索引: {min(i + batch_size, len(documents))}/{len(documents)}")

    # 5. 验证
    print(f"\n[5/5] 索引完成!")
    stats = kb_manager.get_stats()
    print(f"Milvus文档数: {stats['milvus_count']}")
    print(f"ES文档数: {stats['es_count']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="加载华佗医疗数据集")
    parser.add_argument("--max-samples", type=int, default=1000, help="最大样本数")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重跑前清空 Milvus collection 和 ElasticSearch index（会删除旧数据）",
    )
    args = parser.parse_args()
    load_huatuo_data(args.max_samples, reset=args.reset)
