"""验证Neo4j图谱数据"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.graph_store import GraphStoreNeo4jClient

def verify():
    print("=" * 60)
    print("Neo4j图谱数据验证")
    print("=" * 60)

    client = GraphStoreNeo4jClient()
    client.connect()

    # 测试1: 查询高血压用药
    print("\n[测试1] 高血压用药:")
    result = client.query_disease_drugs("高血压")
    print(f"  结果: {result}")

    # 测试2: 查询糖尿病症状
    print("\n[测试2] 糖尿病症状:")
    result = client.query_disease_symptoms("糖尿病")
    print(f"  结果: {result}")

    # 测试3: 查询高血压科室
    print("\n[测试3] 高血压就诊科室:")
    result = client.query_disease_department("高血压")
    print(f"  结果: {result}")

    # 测试4: 查询糖尿病饮食
    print("\n[测试4] 糖尿病饮食建议:")
    result = client.query_disease_foods("糖尿病")
    print(f"  结果: {result}")

    client.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    verify()
