"""测试混合意图识别系统"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.core.rag_engine import LocalDocQA


async def test_intent_system():
    """测试意图识别和路由"""
    print("=" * 60)
    print("混合意图识别系统测试")
    print("=" * 60)

    # 初始化RAG引擎
    rag = LocalDocQA()
    await rag.initialize()

    # 测试用例
    test_cases = [
        # 问候类
        ("你好", "greeting"),
        ("早上好", "greeting"),

        # 感谢类
        ("谢谢你的帮助", "thanks"),

        # 疾病用药
        ("高血压用什么药？", "disease_drug"),
        ("糖尿病怎么治疗？", "disease_drug"),

        # 疾病症状
        ("高血压有什么症状？", "disease_symptom"),

        # 药品禁忌
        ("阿司匹林有什么禁忌？", "drug_contraindication"),

        # 疾病科室
        ("高血压挂什么科？", "disease_department"),

        # 疾病饮食
        ("糖尿病患者吃什么好？", "disease_food"),

        # 症状反查疾病
        ("出现头痛症状是什么病？", "symptom_disease"),

        # 一般医疗问答
        ("什么是高血压？", "general_medical"),

        # 超出范围
        ("今天天气怎么样？", "out_of_scope"),
    ]

    print("\n测试意图识别：")
    print("-" * 60)

    for question, expected_intent in test_cases:
        result = rag.intent_classifier.classify(question)

        status = "✓" if result["intent"] == expected_intent else "✗"
        print(f"{status} 问题: {question}")
        print(f"  预期意图: {expected_intent}")
        print(f"  识别意图: {result['intent']}")
        print(f"  提取实体: {result.get('entity', 'N/A')}")
        print(f"  识别方法: {result['method']}")
        print(f"  置信度: {result['confidence']:.2f}")
        print()

    # 统计信息
    stats = rag.intent_classifier.get_stats()
    print("-" * 60)
    print(f"统计信息:")
    print(f"  规则匹配: {stats['rule_hit']} 次")
    print(f"  LLM调用: {stats['llm_call']} 次")
    print(f"  规则命中率: {stats['rule_ratio']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_intent_system())
