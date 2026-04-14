"""图谱查询器 - 封装各类Cypher查询"""
from typing import List, Dict, Optional
from app.core.graph_store import GraphStoreNeo4jClient


class GraphQuerier:
    """图谱查询器，根据意图执行相应的图谱查询"""

    def __init__(self, graph_store: Optional[GraphStoreNeo4jClient] = None):
        # 向后兼容：允许无参初始化（旧脚本直接 GraphQuerier()）
        self.graph_store = graph_store or GraphStoreNeo4jClient()
        try:
            self.graph_store.connect()
        except Exception as e:
            # Neo4j 不可用时允许降级，查询时返回空
            print(f"GraphQuerier初始化警告（Neo4j可能未就绪）: {e}")

    def query(self, intent: str, entity: str) -> List[Dict]:
        """
        根据意图和实体执行图谱查询

        Args:
            intent: 意图类型
            entity: 实体名称

        Returns:
            查询结果列表，格式化为文档格式
        """
        if intent == "disease_drug":
            return self._query_disease_drug(entity)
        elif intent == "disease_symptom":
            return self._query_disease_symptom(entity)
        elif intent == "drug_contraindication":
            return self._query_drug_contraindication(entity)
        elif intent == "disease_department":
            return self._query_disease_department(entity)
        elif intent == "disease_food":
            return self._query_disease_food(entity)
        elif intent == "symptom_disease":
            return self._query_symptom_disease(entity)
        else:
            return []

    # 兼容 agent_orchestrator.py 中的工具调用（旧接口）
    def get_disease_symptoms(self, disease: str) -> List[str]:
        results = self.graph_store.query_disease_symptoms(disease)
        return [r["symptom"] for r in results if r.get("symptom")]

    def get_disease_drugs(self, disease: str) -> List[str]:
        results = self.graph_store.query_disease_drugs(disease)
        return [r["drug"] for r in results if r.get("drug")]

    def _query_disease_drug(self, disease: str) -> List[Dict]:
        """查询疾病用药"""
        results = self.graph_store.query_disease_drugs(disease)
        if not results:
            return []

        drugs = [r["drug"] for r in results]
        text = f"{disease}的常用治疗药物包括：{', '.join(drugs)}"
        return [{
            "id": f"graph_disease_drug_{disease}",
            "text": text,
            "title": f"{disease}用药信息",
            "source": "neo4j_graph",
            "page": None,
            "score": 1.0,
            "retrieval_source": "graph",
        }]

    def _query_disease_symptom(self, disease: str) -> List[Dict]:
        """查询疾病症状"""
        results = self.graph_store.query_disease_symptoms(disease)
        if not results:
            return []

        symptoms = [r["symptom"] for r in results]
        text = f"{disease}的主要症状包括：{', '.join(symptoms)}"
        return [{
            "id": f"graph_disease_symptom_{disease}",
            "text": text,
            "title": f"{disease}症状信息",
            "source": "neo4j_graph",
            "page": None,
            "score": 1.0,
            "retrieval_source": "graph",
        }]

    def _query_drug_contraindication(self, drug: str) -> List[Dict]:
        """查询药品禁忌"""
        results = self.graph_store.query_drug_contraindications(drug)
        if not results:
            return []

        diseases = [f"{r['disease']}({r.get('desc', '')})" for r in results]
        text = f"{drug}主要用于治疗：{', '.join(diseases)}"
        return [{
            "id": f"graph_drug_contra_{drug}",
            "text": text,
            "title": f"{drug}适应症信息",
            "source": "neo4j_graph",
            "page": None,
            "score": 1.0,
            "retrieval_source": "graph",
        }]

    def _query_disease_department(self, disease: str) -> List[Dict]:
        """查询疾病科室"""
        dept = self.graph_store.query_disease_department(disease)
        if not dept:
            return []

        text = f"{disease}应该挂{dept}科"
        return [{
            "id": f"graph_disease_dept_{disease}",
            "text": text,
            "title": f"{disease}就诊科室",
            "source": "neo4j_graph",
            "page": None,
            "score": 1.0,
            "retrieval_source": "graph",
        }]

    def _query_disease_food(self, disease: str) -> List[Dict]:
        """查询疾病饮食"""
        foods = self.graph_store.query_disease_foods(disease)
        if not foods["recommend"] and not foods["avoid"]:
            return []

        text_parts = []
        if foods["recommend"]:
            text_parts.append(f"推荐食物：{', '.join(foods['recommend'])}")
        if foods["avoid"]:
            text_parts.append(f"忌口食物：{', '.join(foods['avoid'])}")

        text = f"{disease}的饮食建议：" + "；".join(text_parts)
        return [{
            "id": f"graph_disease_food_{disease}",
            "text": text,
            "title": f"{disease}饮食指南",
            "source": "neo4j_graph",
            "page": None,
            "score": 1.0,
            "retrieval_source": "graph",
        }]

    def _query_symptom_disease(self, symptom: str) -> List[Dict]:
        """根据症状查询疾病"""
        results = self.graph_store.query_symptom_diseases(symptom)
        if not results:
            return []

        diseases = [f"{r['disease']}({r.get('desc', '')})" for r in results[:5]]
        text = f"出现{symptom}症状可能的疾病包括：{', '.join(diseases)}"
        return [{
            "id": f"graph_symptom_disease_{symptom}",
            "text": text,
            "title": f"{symptom}相关疾病",
            "source": "neo4j_graph",
            "page": None,
            "score": 1.0,
            "retrieval_source": "graph",
        }]
