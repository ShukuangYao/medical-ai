"""测试记忆功能"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.core.memory_manager import MemoryManager


async def test_memory():
    """测试短期和长期记忆"""
    print("=" * 60)
    print("记忆功能测试")
    print("=" * 60)

    memory = MemoryManager()

    # 测试1：短期记忆（会话历史）
    print("\n【测试1：短期记忆（Redis）】")
    session_id = "test_session_001"

    try:
        memory.short_term.add_message(session_id, "user", "我有高血压，能吃布洛芬吗？")
        memory.short_term.add_message(session_id, "assistant", "高血压患者应慎用布洛芬...")
        memory.short_term.add_message(session_id, "user", "那我应该吃什么药？")

        history = memory.short_term.get_history(session_id)
        print(f"✓ 会话历史（共{len(history)}条）：")
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            print(f"  {role}: {msg['content'][:50]}...")
    except Exception as e:
        print(f"✗ Redis未启动，跳过短期记忆测试: {e}")

    # 测试2：长期记忆（用户档案）
    print("\n【测试2：长期记忆 - 用户档案】")
    user_id = "user_001"

    memory.long_term.create_user_profile(user_id, "张三", 45, "男")
    profile = memory.long_term.get_user_profile(user_id)
    print(f"用户档案: {profile}")

    # 测试3：健康档案
    print("\n【测试3：健康档案】")
    memory.long_term.add_health_record(
        user_id,
        disease_history="高血压5年",
        allergy_history="青霉素过敏",
        medication_history="长期服用降压药"
    )

    health = memory.long_term.get_health_record(user_id)
    print(f"健康档案: {health}")

    # 测试4：咨询历史
    print("\n【测试4：咨询历史】")
    memory.long_term.add_consultation(
        user_id,
        "高血压能吃布洛芬吗？",
        "高血压患者应慎用布洛芬...",
        "disease_drug"
    )

    consultations = memory.long_term.get_consultation_history(user_id, limit=5)
    print(f"咨询历史（共{len(consultations)}条）：")
    for c in consultations:
        print(f"  问题: {c['question']}")
        print(f"  意图: {c['intent']}")
        print(f"  时间: {c['created_at']}")

    # 测试5：构建上下文提示词
    print("\n【测试5：构建上下文提示词】")
    context = memory.build_context_prompt(session_id, user_id)
    print("上下文提示词：")
    print(context)

    print("\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(test_memory())
