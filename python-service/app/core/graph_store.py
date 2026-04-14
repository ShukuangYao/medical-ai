"""Neo4j图数据库客户端 - 实体关系查询"""
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from app.config import settings


class GraphStoreNeo4jClient:
    """Neo4j图数据库客户端，用于查询实体关系"""

    def __init__(self):
        self.driver = None

    def connect(self):
        """连接Neo4j"""
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        print(f"已连接Neo4j: {settings.NEO4J_URI}")

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def query_disease_drugs(self, disease: str) -> List[Dict]:
        """查询疾病相关药品"""
        if not self.driver:
            self.connect()

        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (d:Disease {name: $disease})-[:USES_DRUG]->(drug:Drug)
                RETURN drug.name AS drug_name
                LIMIT 20
                """,
                disease=disease
            )
            return [{"drug": record["drug_name"]} for record in result]

    def query_disease_symptoms(self, disease: str) -> List[Dict]:
        """查询疾病症状"""
        if not self.driver:
            self.connect()

        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (d:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
                RETURN s.name AS symptom
                LIMIT 20
                """,
                disease=disease
            )
            return [{"symptom": record["symptom"]} for record in result]

    def query_drug_contraindications(self, drug: str) -> List[Dict]:
        """查询药品禁忌（通过疾病反向推断）"""
        if not self.driver:
            self.connect()

        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (drug:Drug {name: $drug})<-[:USES_DRUG]-(d:Disease)
                RETURN d.name AS disease, d.desc AS desc
                LIMIT 10
                """,
                drug=drug
            )
            return [{"disease": record["disease"], "desc": record["desc"]} for record in result]

    def query_disease_department(self, disease: str) -> Optional[str]:
        """查询疾病所属科室"""
        if not self.driver:
            self.connect()

        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (d:Disease {name: $disease})-[:IN_DEPARTMENT]->(dept:Department)
                RETURN dept.name AS department
                LIMIT 1
                """,
                disease=disease
            )
            record = result.single()
            return record["department"] if record else None

    def query_disease_foods(self, disease: str) -> Dict[str, List[str]]:
        """查询疾病饮食建议"""
        if not self.driver:
            self.connect()

        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            # 推荐食物
            recommend = session.run(
                """
                MATCH (d:Disease {name: $disease})-[:RECOMMEND_FOOD]->(f:Food)
                RETURN f.name AS food
                LIMIT 10
                """,
                disease=disease
            )
            recommend_foods = [r["food"] for r in recommend]

            # 禁忌食物
            avoid = session.run(
                """
                MATCH (d:Disease {name: $disease})-[:AVOID_FOOD]->(f:Food)
                RETURN f.name AS food
                LIMIT 10
                """,
                disease=disease
            )
            avoid_foods = [r["food"] for r in avoid]

            return {"recommend": recommend_foods, "avoid": avoid_foods}

    def query_symptom_diseases(self, symptom: str) -> List[Dict]:
        """根据症状查询可能的疾病"""
        if not self.driver:
            self.connect()

        with self.driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom {name: $symptom})
                RETURN d.name AS disease, d.desc AS desc
                LIMIT 10
                """,
                symptom=symptom
            )
            return [{"disease": record["disease"], "desc": record["desc"]} for record in result]
