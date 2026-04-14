"""混合意图识别器 - 规则+LLM双层策略"""
import re
import json
from typing import Dict, Optional, List
from app.core.llm_client import OpenAILLM


class HybridIntentClassifier:
    """
    混合意图识别器

    策略：
    1. 第一层：规则匹配（快速处理高频/明确意图）
    2. 第二层：LLM分类（处理复杂/模糊意图）
    """

    # 高频简单意图的规则模式
    RULE_PATTERNS = {
        # 问候类
        "greeting": [
            r"^(你好|您好|hi|hello|嗨)",
            r"^(早上好|下午好|晚上好)",
        ],
        # 感谢类
        "thanks": [
            r"(谢谢|感谢|多谢)",
        ],
        # 疾病用药（明确关键词）
        "disease_drug": [
            r"(.+?)(用什么药|吃什么药|治疗药物|用药方案|药品推荐)",
            r"(.+?)(怎么治疗|如何治疗|治疗方法).*药",
        ],
        # 疾病症状
        "disease_symptom": [
            r"(.+?)(有什么症状|症状是什么|有哪些表现|临床表现|症状表现)",
        ],
        # 药品禁忌
        "drug_contraindication": [
            r"(.+?)(禁忌|不能吃|副作用|注意事项|能不能吃)",
        ],
        # 疾病科室
        "disease_department": [
            r"(.+?)(挂什么科|看什么科|哪个科室|去哪个科|科室)",
        ],
        # 疾病饮食
        "disease_food": [
            r"(.+?)(吃什么|不能吃什么|饮食|食物|忌口|饮食禁忌)",
        ],
        # 症状反查疾病
        "symptom_disease": [
            r"(出现|有|感觉)(.+?)(症状|不舒服).+(什么病|疾病|可能是)",
            r"(.+?)(是什么病|什么疾病)",
        ],
    }

    # LLM意图分类提示词
    LLM_INTENT_PROMPT = """你是医疗意图识别专家。请分析用户问题，识别意图类型并提取关键实体。

意图类型定义：
1. greeting - 问候打招呼
2. thanks - 感谢
3. disease_drug - 询问疾病用药/治疗方法
4. disease_symptom - 询问疾病症状
5. drug_contraindication - 询问药品禁忌/副作用
6. disease_department - 询问疾病就诊科室
7. disease_food - 询问疾病饮食建议
8. symptom_disease - 根据症状反查疾病
9. general_medical - 一般医疗知识问答（不属于以上类别）
10. out_of_scope - 非医疗相关问题

用户问题：{question}

请以JSON格式返回：
{{
    "intent": "意图类型",
    "entity": "提取的核心实体（疾病名/药品名/症状，若无则为null）",
    "confidence": 0.0-1.0,
    "reasoning": "简短推理说明"
}}

只返回JSON，不要其他内容。"""

    def __init__(self, llm: Optional[OpenAILLM] = None):
        self.llm = llm
        self.rule_hit_count = 0
        self.llm_call_count = 0

    async def classify(self, question: str) -> Dict:
        """
        混合意图识别

        Returns:
            {
                "use_graph": bool,      # 是否使用图谱
                "intent": str,          # 意图类型
                "entity": str,          # 提取的实体
                "confidence": float,    # 置信度
                "method": str,          # 识别方法：rule/llm
            }
        """
        question = question.strip()

        # 第一层：规则匹配
        rule_result = self._rule_classify(question)
        if rule_result:
            self.rule_hit_count += 1
            return rule_result

        # 第二层：LLM分类
        if self.llm:
            llm_result = await self._llm_classify(question)
            self.llm_call_count += 1
            return llm_result

        # 降级：默认为一般医疗问答
        return {
            "use_graph": False,
            "intent": "general_medical",
            "entity": None,
            "confidence": 0.5,
            "method": "fallback",
        }

    def _rule_classify(self, question: str) -> Optional[Dict]:
        """规则匹配分类"""
        def _pick_entity(groups: tuple[str, ...]) -> Optional[str]:
            if not groups:
                return None
            cleaned: List[str] = []
            for g in groups:
                if not g:
                    continue
                s = str(g).strip()
                s = re.sub(r"^(如果|假如|请问|想问)", "", s).strip()
                # drop very common verbs that appear in symptom patterns
                if s in {"出现", "有", "感觉", "就诊", "挂号"}:
                    continue
                # remove trailing generic question suffixes
                s = re.sub(r"(可能是|可能|是什么病|什么病|疾病|怎么回事)$", "", s).strip(" ，,。；;：:")
                cleaned.append(s)
            if not cleaned:
                return None
            # prefer the most informative segment (longest)
            return max(cleaned, key=len)

        for intent, patterns in self.RULE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    # 提取实体（通常是第一个捕获组）
                    entity = None
                    if match.groups():
                        entity = _pick_entity(match.groups())

                    # 判断是否使用图谱
                    use_graph = intent not in ["greeting", "thanks", "general_medical"]

                    return {
                        "use_graph": use_graph,
                        "intent": intent,
                        "entity": entity,
                        "confidence": 0.95,
                        "method": "rule",
                    }
        return None

    async def _llm_classify(self, question: str) -> Dict:
        """LLM分类"""
        try:
            prompt = self.LLM_INTENT_PROMPT.format(question=question)

            # 构建消息并指定JSON格式
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            # 解析JSON
            result = json.loads(response.choices[0].message.content)

            intent = result.get("intent", "general_medical")
            entity = result.get("entity")
            confidence = result.get("confidence", 0.7)

            # 判断是否使用图谱
            graph_intents = [
                "disease_drug", "disease_symptom", "drug_contraindication",
                "disease_department", "disease_food", "symptom_disease"
            ]
            use_graph = intent in graph_intents

            return {
                "use_graph": use_graph,
                "intent": intent,
                "entity": entity,
                "confidence": confidence,
                "method": "llm",
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            print(f"LLM意图识别失败: {e}")
            return {
                "use_graph": False,
                "intent": "general_medical",
                "entity": None,
                "confidence": 0.5,
                "method": "llm_error",
            }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.rule_hit_count + self.llm_call_count
        return {
            "rule_hit": self.rule_hit_count,
            "llm_call": self.llm_call_count,
            "total": total,
            "rule_ratio": self.rule_hit_count / total if total > 0 else 0,
        }
