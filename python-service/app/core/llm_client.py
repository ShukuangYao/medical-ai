"""LLM客户端 - OpenAI兼容接口（Qwen/DashScope, DeepSeek 等）"""
from __future__ import annotations

from typing import AsyncGenerator, List, Dict, Optional, Literal
from openai import AsyncOpenAI
from app.config import settings


class OpenAILLM:
    """LLM客户端，与大语言模型交互，支持流式生成回答"""

    def __init__(self, *, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    @classmethod
    def from_provider(
        cls,
        *,
        provider: Optional[Literal["qwen", "deepseek"]] = None,
        model_name: Optional[str] = None,
    ) -> "OpenAILLM":
        """Create an LLM client based on selected provider/model.

        - qwen: uses DASHSCOPE_API_KEY + LLM_API_BASE
        - deepseek: uses DEEPSEEK_API_KEY + DEEPSEEK_API_BASE
        """
        p = provider or "qwen"
        if p == "deepseek":
            api_key = settings.DEEPSEEK_API_KEY or settings.DASHSCOPE_API_KEY
            base_url = settings.DEEPSEEK_API_BASE
            model = model_name or "deepseek-chat"
        else:
            api_key = settings.DASHSCOPE_API_KEY
            base_url = settings.LLM_API_BASE
            model = model_name or settings.LLM_MODEL
        return cls(api_key=api_key, base_url=base_url, model=model)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        model: Optional[str] = None,
    ) -> str:
        """非流式生成回答"""
        response = await self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or settings.MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成回答，逐token返回"""
        response = await self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or settings.MAX_OUTPUT_TOKENS,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def build_rag_messages(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建RAG问答的消息列表"""
        system_prompt = """你是一个专业的医疗AI助手。请根据提供的参考资料回答用户的医疗健康问题。

要求：
1. 回答必须基于提供的参考资料，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 使用专业但易懂的语言
4. 在适当位置标注参考来源
5. 在回答末尾添加免责声明：本回答仅供参考，不能替代专业医生的诊断和建议
6. 实体一致性：只讨论用户问题或对话历史中明确出现的疾病/症状/药物/检查等实体，严禁凭空引入新的疾病名称（例如用户只提到“头疼、糖尿病”，就不要额外加入其他疾病）
7. 如果用户问题涉及“这两个/这些”但无法从对话历史确定对应哪些实体，请先反问澄清，不要自行猜测补全

参考资料：
{context}"""

        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
        ]

        # 添加对话历史
        if chat_history:
            for msg in chat_history[-6:]:  # 最多保留最近3轮对话
                messages.append(msg)

        messages.append({"role": "user", "content": question})
        return messages
