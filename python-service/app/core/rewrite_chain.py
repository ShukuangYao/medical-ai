"""问题改写链 - 多轮对话时优化问题表达"""
from typing import List, Dict, Optional
from app.core.llm_client import OpenAILLM


class RewriteQuestionChain:
    """问题改写链，多轮对话时将模糊问题转为精确查询

    当用户在多轮对话中使用代词或省略主语时，
    结合对话历史将问题改写为独立、完整的查询。
    """

    REWRITE_PROMPT = """你是一个问题改写助手。请根据对话历史，将用户的最新问题改写为一个独立、完整、适合检索的问题。

规则：
1. 如果最新问题已经是完整的，直接返回原问题
2. 如果问题中有代词（它、这个、那个等），替换为具体指代内容
3. 如果问题省略了主语或上下文，补充完整
4. 改写后的问题应该是一个独立的、不依赖上下文就能理解的问题
5. 只返回改写后的问题，不要添加任何解释
6. 只能使用“对话历史”中已出现过的实体（疾病/症状/药物/检查等），严禁引入新的疾病名或新的实体；如果无法确定指代对象，保持原问题不变

对话历史：
{history}

用户最新问题：{question}

改写后的问题："""

    def __init__(self, llm: OpenAILLM):
        self.llm = llm

    async def rewrite(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        改写问题

        Args:
            question: 用户当前问题
            chat_history: 对话历史

        Returns:
            改写后的问题（如果无需改写则返回原问题）
        """
        # 如果没有对话历史，无需改写
        if not chat_history or len(chat_history) == 0:
            return question

        # 构建历史文本
        history_text = ""
        for msg in chat_history[-6:]:  # 最近3轮
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}: {msg['content'][:200]}\n"

        # 调用LLM改写
        prompt = self.REWRITE_PROMPT.format(
            history=history_text,
            question=question
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            rewritten = await self.llm.generate(messages, temperature=0.1)
            rewritten = rewritten.strip()
            if rewritten:
                print(f"问题改写: '{question}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            print(f"问题改写失败，使用原问题: {e}")

        return question
