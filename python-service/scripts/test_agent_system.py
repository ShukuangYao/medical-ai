"""测试多智能体协作诊断系统"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from app.core.agent_orchestrator import MedicalAgentOrchestrator
from app.core.rag_engine import LocalDocQA
from app.core.graph_querier import GraphQuerier


async def test_agent_diagnosis():
    """测试多智能体协作诊断"""
    print("=" * 60)
    print("多智能体协作诊断测试")
    print("=" * 60)

    # 初始化
    print("\n初始化系统...")
    rag = LocalDocQA()
    await rag.initialize()
    graph = GraphQuerier()
    orchestrator = MedicalAgentOrchestrator(rag, graph)
    print("✓ 系统初始化完成")

    # 模拟病历
    medical_record = """
患者信息：
- 姓名：张三
- 年龄：45岁
- 性别：男

主诉：
反复胸痛3天，伴出汗

现病史：
患者3天前无明显诱因出现胸骨后疼痛，呈压榨性，持续约10-20分钟，
伴大汗、恶心。休息后可缓解。今日凌晨再次发作，疼痛加重，遂来院就诊。

既往史：
高血压病史5年，不规律服药。吸烟史20年，每日1包。

体格检查：
BP 160/95mmHg，HR 98次/分，律齐。双肺呼吸音清，未闻及干湿啰音。
心界不大，心音有力，未闻及杂音。

辅助检查：
心电图：ST段压低，T波倒置
心肌酶：肌钙蛋白I升高
"""

    print("\n" + "=" * 60)
    print("病历内容：")
    print(medical_record)
    print("=" * 60)

    # 开始诊断
    print("\n开始多智能体协作诊断...\n")

    try:
        async for chunk in orchestrator.diagnose_stream(medical_record):
            if chunk["type"] == "agent":
                print(f"\n【{chunk['agent']}】")
                print(chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"])
                print("-" * 60)
            elif chunk["type"] == "result":
                print("\n" + "=" * 60)
                print("最终诊断报告：")
                print("=" * 60)
                result = chunk["content"]
                print(f"\n病历分析：\n{result.get('record_analysis', '无')[:300]}")
                print(f"\n鉴别诊断：\n{result.get('diagnosis', ['无'])[0][:300]}")
                print(f"\n治疗方案：\n{result.get('treatment', '无')[:300]}")
                print(f"\n审核意见：\n{result.get('review', '无')[:300]}")
            elif chunk["type"] == "error":
                print(f"\n❌ 错误：{chunk['content']}")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(test_agent_diagnosis())
