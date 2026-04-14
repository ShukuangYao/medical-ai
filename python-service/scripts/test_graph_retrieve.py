"""测试图谱增强检索"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.core.rag_engine import LocalDocQA


async def test_graph_retrieve():
    """测试图谱检索功能"""
    print("=" * 60)
    print("图谱增强检索测试")
    print("=" * 60)

    rag = LocalDocQA()
    await rag.initialize()

    # 测试用例
    test_questions = [
        "高血压用什么药？",
        "糖尿病有什么症状？",
        "冠心病挂什么科？",
        "糖尿病患者吃什么好？",
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 60)

        # 意图识别
        intent = await rag.intent_classifier.classify(question)
        print(f"意图: {intent['intent']}, 实体: {intent.get('entity', 'N/A')}")

        # 路由检索
        docs, strategy = await rag.intent_router.route(intent, question)
        print(f"策略: {strategy}")
        print(f"文档数: {len(docs)}")

        # 统计来源
        graph_count = sum(1 for d in docs if d.get("retrieval_source") == "graph")
        vector_count = len(docs) - graph_count
        print(f"图谱: {graph_count}条, 向量: {vector_count}条")

        # 显示前3条
        for i, doc in enumerate(docs[:3], 1):
            source = doc.get("retrieval_source", "unknown")
            text = doc.get("text", "")[:100]
            print(f"  [{i}] [{source}] {text}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_graph_retrieve())
