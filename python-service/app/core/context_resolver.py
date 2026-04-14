"""上下文指代消解器 - 处理"它"、"这个病"、"刚才的"等指代"""
from typing import Dict, List, Optional


class ContextResolver:
    """上下文指代消解器"""

    def __init__(self, llm):
        # Force using the project OpenAI-compatible client (DashScope/DeepSeek).
        # This avoids hidden dependency on OPENAI_API_KEY.
        self.llm = llm

    async def resolve_query(
        self, current_query: str, chat_history: List[Dict[str, str]]
    ) -> tuple[str, Optional[str]]:
        """
        消解当前查询中的指代关系

        Returns:
            (resolved_query, entity) - 消解后的完整问题和提取的实体
        """
        # 如果没有历史，直接返回
        if not chat_history or len(chat_history) < 2:
            return current_query, None

        # 仅在可能是追问/省略问题时才触发消解，避免无谓LLM调用
        if not self._should_resolve(current_query):
            return current_query, None

        # 使用LLM进行指代消解
        resolved = await self._llm_resolve(current_query, chat_history)

        # 提取实体
        entity = await self._extract_entity(resolved)

        return resolved, entity

    def _should_resolve(self, current_query: str) -> bool:
        """判断当前问题是否需要做上下文消解。"""
        query = (current_query or "").strip()
        if not query:
            return False

        # Fast exit for pure greetings/thanks: they do not need coreference or follow-up completion.
        small_talk = {
            "谢谢", "谢了", "多谢", "感谢", "非常感谢",
            "好的", "好", "嗯", "OK", "ok",
            "你好", "您好", "嗨", "在吗",
        }
        if query in small_talk:
            return False

        pronouns = ["它", "这个", "那个", "刚才", "上面", "之前", "该", "此", "这个病", "这个情况"]
        if any(p in query for p in pronouns):
            return True

        # 常见追问短句：如“挂什么科”“怎么治疗”“严重吗”
        followup_keywords = [
            "挂什么科", "看什么科", "哪个科", "怎么治", "如何治", "怎么办",
            "严重吗", "会好吗", "会复发吗", "能吃什么", "不能吃什么", "要注意什么",
        ]
        if any(k in query for k in followup_keywords):
            return True

        # 非常短的问句，通常依赖上文（例如“要紧吗？”“怎么处理？”）
        if len(query) <= 8:
            return True

        return False

    async def _llm_resolve(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """使用LLM消解指代"""
        # 构建历史上下文：
        # - 常规追问：最近2轮
        # - 涉及“最初/最早/一开始”：首轮用户问题 + 最近2轮，避免被近期话题覆盖
        history_for_resolve = self._build_history_for_resolution(query, history)
        history_text = "\n".join([
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content'][:200]}"
            for h in history_for_resolve
        ])

        prompt = f"""你是一个上下文理解助手。用户在多轮对话中可能使用指代词（如"它"、"这个病"、"刚才的"等），也可能省略主语直接追问（例如“挂什么科”“怎么治疗”“严重吗”）。

对话历史：
{history_text}

当前问题：{query}

请输出“补全后的完整问题”。你需要同时处理两类情况：
1) 指代消解：把指代词替换为历史中明确出现过的具体实体
2) 追问补全：若当前问题是省略主语的追问（例如“挂什么科/怎么治/严重吗”），且历史中只有一个清晰的主题实体（疾病/症状/药物/检查），则把该实体补全进问题（例如把“挂什么科”补全为“高烧挂什么科”）

要求：
1. 只输出消解后的完整问题，不要解释
2. 若当前问题本身已完整且不需要补全，直接输出原问题
3. 确保消解后的问题语义完整、清晰
4. 只能使用“对话历史”中已经明确出现过的疾病/症状/药物/检查等实体，严禁凭空引入新实体或新疾病名称
5. 如果当前问题明确指向多个实体（例如“这两个病/这些病”），请把这些实体都补全在问题里；若历史中无法确定是哪几个实体，则保留原问题并在问题末尾补充“（请明确指的是哪些疾病/症状）”

消解后的问题："""

        messages = [{"role": "user", "content": prompt}]
        response = ""
        async for token in self.llm.generate_stream(messages, temperature=0.3, max_tokens=100):
            response += token
        return response.strip()

    def _build_history_for_resolution(
        self, query: str, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """根据当前问题选择用于消解的历史窗口。"""
        recent_window = history[-4:]  # 默认最近2轮
        q = (query or "").strip()
        initial_keywords = ["最初", "最早", "一开始", "一开始说", "最先", "第一个"]
        if not any(k in q for k in initial_keywords):
            return recent_window

        # 找到首条用户消息并拼接最近窗口（去重）
        first_user = next((h for h in history if h.get("role") == "user" and h.get("content")), None)
        if not first_user:
            return recent_window

        merged: List[Dict[str, str]] = [first_user]
        for h in recent_window:
            if h is first_user:
                continue
            merged.append(h)
        return merged

    async def _extract_entity(self, query: str) -> Optional[str]:
        """从消解后的问题中提取核心实体"""
        prompt = f"""从以下医疗问题中提取核心疾病或药物实体。

问题：{query}

要求：
1. 只输出实体名称，不要解释
2. 如果有多个实体，只输出最主要的一个
3. 如果没有明确实体，输出"无"

实体："""

        if self.llm:
            messages = [{"role": "user", "content": prompt}]
            response = ""
            async for token in self.llm.generate_stream(messages, temperature=0.1, max_tokens=20):
                response += token
            entity = response.strip()

        return entity if entity and entity != "无" else None
