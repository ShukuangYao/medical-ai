"""测试上下文指代消解功能"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.core.rag_engine import LocalDocQA


async def test_context_resolver():
    """测试指代消解功能"""
    print("=" * 60)
    print("上下文指代消解测试")
    print("=" * 60)

    rag = LocalDocQA()
    await rag.initialize()

    # 模拟多轮对话
    session_id = "test_context_001"

    # 第1轮：问甲状腺结节
    print("\n【第1轮对话】")
    print("用户: 甲状腺结节是什么？")
    chat_history = []

    async for chunk in rag.query_stream("甲状腺结节是什么？", session_id, chat_history):
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "thinking":
            print(f"\n[思考] {chunk['content']}")

    # 更新历史
    chat_history.append({"role": "user", "content": "甲状腺结节是什么？"})
    chat_history.append({"role": "assistant", "content": "甲状腺结节是指甲状腺内的肿块..."})

    # 第2轮：问"挂什么科"（指代"甲状腺结节"）
    print("\n\n【第2轮对话 - 测试指代消解】")
    print("用户: 挂什么科？")

    async for chunk in rag.query_stream("挂什么科？", session_id, chat_history):
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "thinking":
            print(f"\n[思考] {chunk['content']}")

    # 更新历史
    chat_history.append({"role": "user", "content": "挂什么科？"})
    chat_history.append({"role": "assistant", "content": "甲状腺结节应该挂内分泌科..."})

    # 第3轮：问"刚才问的什么病"
    print("\n\n【第3轮对话 - 测试回溯指代】")
    print("用户: 刚才问的什么病？")

    async for chunk in rag.query_stream("刚才问的什么病？", session_id, chat_history):
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "thinking":
            print(f"\n[思考] {chunk['content']}")

    print("\n\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(test_context_resolver())
