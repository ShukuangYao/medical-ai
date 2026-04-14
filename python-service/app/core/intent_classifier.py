"""意图识别器 - 判断问题是否适合图谱查询"""
import re
from typing import Dict, List, Optional


class IntentClassifier:
    """意图识别器，判断用户问题类型并提取实体"""

    # 图谱查询关键词模式
    GRAPH_PATTERNS = {
        "disease_drug": [
            r"(.+?)(用什么药|吃什么药|治疗药物|用药|药品)",
            r"(.+?)(怎么治疗|如何治疗|治疗方法)",
        ],
        "disease_symptom": [
            r"(.+?)(有什么症状|症状是什么|表现|临床表现)",
        ],
        "drug_contraindication": [
            r"(.+?)(禁忌|不能吃|副作用|注意事项)",
        ],
        "disease_department": [
            r"(.+?)(挂什么科|看什么科|哪个科室|科室)",
        ],
        "disease_food": [
            r"(.+?)(吃什么|饮食|食物|忌口)",
        ],
        "symptom_disease": [
            r"(出现|有)(.+?)(症状|表现).+(什么病|疾病)",
        ],
    }

    def classify(self, question: str) -> Dict:
        """
        分类问题意图并提取实体

        Returns:
            {
                "use_graph": bool,  # 是否使用图谱
                "intent": str,      # 意图类型
                "entity": str,      # 提取的实体
            }
        """
        question = question.strip()

        # 遍历所有模式
        for intent, patterns in self.GRAPH_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    # 提取实体（通常是第一个捕获组）
                    entity = match.group(1).strip()
                    return {
                        "use_graph": True,
                        "intent": intent,
                        "entity": entity,
                    }

        # 默认不使用图谱，走向量检索
        return {
            "use_graph": False,
            "intent": "general",
            "entity": None,
        }
