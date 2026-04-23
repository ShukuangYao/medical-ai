"""加载华佗医疗数据集到Milvus和ElasticSearch"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import datasets
from app.config import settings
from app.core.embeddings import BGEEmbeddings
from app.core.vector_store import VectorStoreMilvusClient
from app.core.es_store import StoreElasticSearchClient
from app.core.knowledge_manager import KnowledgeBaseManager
from pymilvus.exceptions import MilvusException


def wait_for_milvus_ready(vector_store: VectorStoreMilvusClient, timeout: int = 120):
    print("\n⏳ 等待 Milvus 服务完全初始化...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            vector_store.collection.load()
            print("✅ Milvus 服务已就绪！")
            return True
        except Exception as e:
            time.sleep(5)
    raise Exception("❌ Milvus 服务启动超时！")


def load_huatuo_data(max_samples: int = 1000, reset: bool = False):
    print("=" * 60)
    print("华佗医疗数据集加载工具")
    print("=" * 60)

    print("\n[1/5] 正在从HuggingFace加载数据集...")
    huatuo = datasets.load_dataset("FreedomIntelligence/huatuo26M-testdatasets")
    data = huatuo["train"] if "train" in huatuo else list(huatuo.values())[0]
    print(f"数据集总量: {len(data)}")

    print("\n[2/5] 正在初始化组件...")
    embeddings = BGEEmbeddings()
    embeddings.load()

    vector_store = VectorStoreMilvusClient()
    vector_store.connect()

    if reset:
        print("\n[2/5] --reset 启用：清空 Milvus 旧集合...")
        vector_store.drop_collection()
    vector_store.create_collection(embeddings.dimension)

    wait_for_milvus_ready(vector_store)

    es_store = StoreElasticSearchClient()
    es_store.connect()

    if reset:
        print("\n[2/5] --reset 启用：清空 ElasticSearch 旧索引...")
        es_store.drop_index()
    es_store.create_index()

    kb_manager = KnowledgeBaseManager(vector_store, es_store, embeddings)

    print(f"\n[3/5] 正在处理数据...")
    documents = []
    # 🔥 终极截断：4096字符（远小于8192限制）
    MAX_LEN = 4096

    for i, item in enumerate(data):
        if max_samples > 0 and i >= max_samples:
            break

        question = item.get("question", item.get("questions", ""))
        answer = item.get("answer", item.get("answers", ""))
        if not question or not answer:
            continue

        # 🔥 强制截断，不留任何超长可能
        text = f"Q:{question}\nA:{answer}"
        text = text.encode('utf-8')[:MAX_LEN].decode('utf-8', 'ignore')

        doc = {
            "id": f"huatuo_{i:06d}",
            "text": text,
            "title": question[:50],
            "source": "huatuo",
            "page": 0,
        }
        documents.append(doc)

    print(f"有效文档数: {len(documents)}")

    print(f"\n[4/5] 正在批量索引...")
    batch_size = 10  # 缩小批量，更稳定
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        try:
            kb_manager.index_documents(batch)
            print(f"✅ 索引成功: {min(i+batch_size, len(documents))}/{len(documents)}")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  跳过异常批次")
            continue

    print(f"\n[5/5] 索引完成!")
    stats = kb_manager.get_stats()
    print(f"Milvus文档数: {stats['milvus_count']}")
    print(f"ES文档数: {stats['es_count']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    load_huatuo_data(args.max_samples, reset=args.reset)