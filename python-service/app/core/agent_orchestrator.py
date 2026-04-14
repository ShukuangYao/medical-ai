"""AutoGen多智能体协作编排器 - 病历分析系统（严格JSON输出）"""
from __future__ import annotations

from typing import Dict, AsyncGenerator, Any, Optional, List, Tuple
import asyncio
import json
import re

from autogen import ConversableAgent

from app.config import settings
from app.core.llm_client import OpenAILLM
from app.core.rag_engine import LocalDocQA
from app.core.graph_querier import GraphQuerier
from app.core.intent_classifier_v2 import HybridIntentClassifier
from app.core.session_store import SessionStore


class MedicalAgentOrchestrator:
    """医疗病历分析多智能体协调器（阶段式编排）"""

    def __init__(self, rag_engine: LocalDocQA, graph_querier: GraphQuerier):
        self.rag = rag_engine
        self.graph = graph_querier
        self.llm = OpenAILLM.from_provider(provider="qwen", model_name=settings.LLM_MODEL)
        self.intent_classifier = HybridIntentClassifier(llm=self.llm)
        self.session_store = SessionStore(settings.CHAT_DB_PATH)

        # 对话历史（用于展示推理过程）
        self.conversation_history = []

    def _llm_config(self, *, provider: Optional[str], model_name: Optional[str]) -> Dict[str, Any]:
        # AutoGen expects OpenAI-compatible config
        p = provider or "qwen"
        if p == "deepseek":
            api_key = settings.DEEPSEEK_API_KEY or settings.DASHSCOPE_API_KEY
            base_url = settings.DEEPSEEK_API_BASE
            model = model_name or "deepseek-chat"
        else:
            api_key = settings.DASHSCOPE_API_KEY
            base_url = settings.LLM_API_BASE
            model = model_name or settings.LLM_MODEL
        return {
            "config_list": [
                {
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    # Optional pricing for AutoGen cost tracking (per 1K tokens).
                    # If unknown, set to 0 to silence "model not found" warnings.
                    "price": [0.0, 0.0],
                }
            ],
            "temperature": 0.2,
            "timeout": 90,
        }

    def _create_agents(self, *, provider: Optional[str], model_name: Optional[str]) -> Dict[str, ConversableAgent]:
        llm_config = self._llm_config(provider=provider, model_name=model_name)
        common_rules = (
            "要求：\n"
            "1) 只输出JSON，不要Markdown、不加解释文字。\n"
            "2) 不要编造：缺失信息要放入 missing_fields/空字符串/空数组。\n"
            "3) 如存在矛盾或不合理处，明确列出。\n"
            "4) 医疗建议必须保守、以就医与检查为导向，避免具体处方与剂量。\n"
            "5) 禁止捏造用户画像：不得凭空补全年龄/性别/妊娠/基础病/用药/检查结果等个体事实；只能使用输入文本或明确给定的结构化字段。\n"
        )

        intent_agent = ConversableAgent(
            name="IntentAgent",
            system_message="你是医疗意图与问题消解专家。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        validator = ConversableAgent(
            name="RecordValidator",
            system_message="你是病历质量审核与纠错专家（字段完整性、矛盾、单位/术语规范化）。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        extractor = ConversableAgent(
            name="SymptomExtractor",
            system_message="你是病历结构化抽取专家（主诉、症状、病史、用药、过敏、检查）。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        analyst = ConversableAgent(
            name="ConditionAnalyst",
            system_message="你是病症分析与鉴别诊断专家（强调不确诊，给出候选与依据）。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        triage = ConversableAgent(
            name="TriageNurse",
            system_message="你是分诊护士，负责紧急程度分级与红旗征提示。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        dept = ConversableAgent(
            name="DepartmentRecommender",
            system_message="你是就医科室推荐助手，给出首选与备选科室及理由。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        planner = ConversableAgent(
            name="NextStepPlanner",
            system_message="你是下一步检查/处置规划助手（先做什么、何时就医、注意事项）。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        safety = ConversableAgent(
            name="SafetyCritic",
            system_message="你是医疗安全审阅专家（禁忌、风险、过度自诊自疗纠偏）。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        coordinator = ConversableAgent(
            name="Coordinator",
            system_message="你是协调者，负责把各模块结果合并成最终严格JSON，并写一段简短summary。" + common_rules,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        return {
            "intent": intent_agent,
            "validator": validator,
            "extractor": extractor,
            "analyst": analyst,
            "triage": triage,
            "dept": dept,
            "planner": planner,
            "safety": safety,
            "coordinator": coordinator,
        }

    async def _agent_reply(self, agent: ConversableAgent, prompt: str) -> str:
        # autogen agent methods are sync; run them in a thread
        def _run() -> str:
            return agent.generate_reply(messages=[{"role": "user", "content": prompt}])  # type: ignore[no-any-return]
        # 把一个同步/耗时的函数 _run 放到线程池里执行
        return await asyncio.to_thread(_run)

    def _safe_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            # try to extract first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return {}
            return {}

    @staticmethod
    def _has_unseen_patient_facts(*, raw: str, text: str) -> bool:
        """Detect common patient-specific facts that appear in text but not in raw input.

        This is a lightweight guardrail to reduce hallucinated demographics/conditions
        from being amplified by downstream prompts.
        """
        r = (raw or "").strip()
        t = (text or "").strip()
        if not r or not t:
            return False

        patterns = [
            r"\b\d{1,3}\s*岁\b",
            r"\b(男|女)\b",
            r"(透析|血液透析|腹膜透析)",
            r"(孕|妊娠|哺乳)",
            r"(高血压|糖尿病|冠心病|肾衰|肾功能不全|心衰|房颤|脑梗|卒中)",
        ]
        for p in patterns:
            if re.search(p, t) and not re.search(p, r):
                return True
        return False

    def _build_fallback_report(
        self,
        *,
        validated_record: Dict[str, Any],
        intent: Dict[str, Any],
        structured_case: Dict[str, Any],
        symptom_analysis: Dict[str, Any],
        triage: Dict[str, Any],
        department: Dict[str, Any],
        next_steps: Dict[str, Any],
        treatment_safety: Dict[str, Any],
        summary: str,
    ) -> Dict[str, Any]:
        def _dict(v: Any) -> Dict[str, Any]:
            return v if isinstance(v, dict) else {}

        def _list(v: Any) -> List[Any]:
            return v if isinstance(v, list) else []

        triage_level = _dict(triage).get("severity_level") or "routine"
        if triage_level not in {"emergency", "urgent", "routine"}:
            triage_level = "routine"

        report = {
            "validated_record": {
                "normalized_text": str(_dict(validated_record).get("normalized_text") or ""),
                "missing_fields": [str(x) for x in _list(_dict(validated_record).get("missing_fields"))],
                "contradictions": [str(x) for x in _list(_dict(validated_record).get("contradictions"))],
                "corrections": [str(x) for x in _list(_dict(validated_record).get("corrections"))],
            },
            "intent": {
                "raw_question": str(_dict(intent).get("raw_question") or ""),
                "resolved_question": str(_dict(intent).get("resolved_question") or ""),
                "intent_type": str(_dict(intent).get("intent_type") or "general_medical"),
                "confidence": float(_dict(intent).get("confidence") or 0.5),
            },
            "structured_case": {
                "chief_complaint": _dict(structured_case).get("chief_complaint"),
                "symptoms": [str(x) for x in _list(_dict(structured_case).get("symptoms"))],
                "duration": _dict(structured_case).get("duration"),
                "vitals": _dict(_dict(structured_case).get("vitals")),
                "history": _dict(_dict(structured_case).get("history")),
                "medications": [str(x) for x in _list(_dict(structured_case).get("medications"))],
                "allergies": [str(x) for x in _list(_dict(structured_case).get("allergies"))],
                "tests": _list(_dict(structured_case).get("tests")),
            },
            "symptom_analysis": {
                "key_findings": [str(x) for x in _list(_dict(symptom_analysis).get("key_findings"))],
                "possible_conditions": [str(x) for x in _list(_dict(symptom_analysis).get("possible_conditions"))],
            },
            "triage": {
                "severity_level": triage_level,
                "red_flags": [str(x) for x in _list(_dict(triage).get("red_flags"))],
                "why": str(_dict(triage).get("why") or ""),
            },
            "department": {
                "recommended": [str(x) for x in _list(_dict(department).get("recommended"))],
                "alternatives": [str(x) for x in _list(_dict(department).get("alternatives"))],
                "reason": str(_dict(department).get("reason") or ""),
            },
            "next_steps": {
                "immediate_actions": [str(x) for x in _list(_dict(next_steps).get("immediate_actions"))],
                "recommended_tests": [str(x) for x in _list(_dict(next_steps).get("recommended_tests"))],
                "when_to_seek_care": [str(x) for x in _list(_dict(next_steps).get("when_to_seek_care"))],
            },
            "treatment_safety": {
                "medication_considerations": [str(x) for x in _list(_dict(treatment_safety).get("medication_considerations"))],
                "contraindications": [str(x) for x in _list(_dict(treatment_safety).get("contraindications"))],
                "cautions": [str(x) for x in _list(_dict(treatment_safety).get("cautions"))],
            },
            "summary": summary,
            "trace": [{"agent": h["agent"], "message": h["message"]} for h in self.conversation_history],
        }
        self._ensure_next_steps_minimum(report)
        return report

    def _ensure_next_steps_minimum(self, report: Dict[str, Any]) -> None:
        """Ensure next_steps has actionable, non-empty defaults for UI."""
        if not isinstance(report, dict):
            return
        triage_level = (
            (report.get("triage") or {}).get("severity_level")
            if isinstance(report.get("triage"), dict)
            else None
        )
        level = triage_level if triage_level in {"emergency", "urgent", "routine"} else "routine"
        ns = report.get("next_steps")
        if not isinstance(ns, dict):
            ns = {}
            report["next_steps"] = ns

        def _get_list(key: str) -> List[str]:
            v = ns.get(key)
            if isinstance(v, list):
                return [str(x) for x in v if str(x).strip()]
            return []

        immediate = _get_list("immediate_actions")
        tests = _get_list("recommended_tests")
        when = _get_list("when_to_seek_care")

        if not immediate:
            immediate = [
                "补充休息与补液，避免剧烈运动与刺激性食物",
                "监测体温/症状变化（如加重及时就医）",
            ]
            if level in {"urgent", "emergency"}:
                immediate.insert(0, "尽快前往就近医院/急诊评估（避免自行用药拖延）")
            ns["immediate_actions"] = immediate

        if not tests:
            # Keep conservative and common, avoid over-specific tests without context
            tests = ["血常规（如未做）", "C反应蛋白/炎症指标（如医生认为需要）"]
            if level in {"urgent", "emergency"}:
                tests.append("血氧饱和度监测/必要时胸部影像检查（由医生决定）")
            ns["recommended_tests"] = tests

        if not when:
            when = [
                "出现呼吸困难/胸痛/意识改变/持续高热不退等红旗征 → 立即急诊",
                "症状持续不缓解或逐渐加重 → 24–48小时内就医",
            ]
            ns["when_to_seek_care"] = when

    def _ensure_summary_minimum(self, report: Dict[str, Any]) -> None:
        if not isinstance(report, dict):
            return
        s = report.get("summary")
        if isinstance(s, str) and s.strip():
            return
        report["summary"] = "已完成病历分析（结果仅供参考，不能替代专业医生的诊断与建议）。"

    async def _retrieve_evidence(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Use existing RAG pipeline to retrieve sources for grounding.
        Returns: (sources, short_text_snippet)
        """
        sources: List[Dict[str, Any]] = []
        brief: List[str] = []
        try:
            async for chunk in self.rag.query_stream(query, session_id="evidence", chat_history=None, use_graph=True):
                if chunk.get("type") == "sources":
                    for src in (chunk.get("content") or [])[:5]:
                        sources.append(src)
                        title = src.get("title", "source")
                        content = (src.get("content") or "")[:180]
                        brief.append(f"- {title}: {content}")
                    break
        except Exception:
            pass
        return sources, "\n".join(brief) if brief else "未检索到可用参考资料"

    async def diagnose_stream(
        self,
        medical_record: str,
        session_id: str = "default",
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        流式输出病历分析过程（严格JSON）

        Yields:
            {"type": "thinking", "content": str} - 过程说明
            {"type": "agent_step", "content": {...}} - 阶段产出（可用于前端过程展示）
            {"type": "sources", "content": [...] } - 参考资料（可选）
            {"type": "result", "content": Dict} - 最终结构化结果
            {"type": "done", "content": ""} - 完成标记
        """
        # 清空历史
        self.conversation_history = []
        thinking_steps: List[str] = []

        def _thinking(content: str) -> Dict[str, Any]:
            thinking_steps.append(content)
            return {"type": "thinking", "content": content}

        # 创建智能体
        agents = self._create_agents(provider=model_provider, model_name=model_name)

        yield _thinking("开始多智能体病历分析...")

        try:
            raw_question = medical_record.strip()

            # Recover recent session history (minimal context engineering; avoid feeding huge history)
            recent_history: List[Dict[str, str]] = []
            if user_id:
                try:
                    rows = self.session_store.list_messages(
                        session_id=session_id,
                        user_id=user_id,
                        mode="agent",
                        limit=12,
                    )
                    # Keep only plain role/content, and keep the tail to reduce drift
                    recent_history = [{"role": m.role, "content": m.content} for m in rows][-8:]
                except Exception:
                    recent_history = []

            # Persist user message
            if user_id:
                try:
                    import uuid
                    self.session_store.add_message(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        mode="agent",
                        role="user",
                        content=raw_question,
                    )
                except Exception as e:
                    yield _thinking(f"⚠️ 会话写入失败: {e}")

            # 1) Intent (fast path: existing classifier)
            yield _thinking("意图识别与问题消解...")
            intent_res = await self.intent_classifier.classify(raw_question)
            intent_payload = {
                "raw_question": raw_question,
                "resolved_question": raw_question,
                "intent_type": intent_res.get("intent", "general_medical"),
                "confidence": float(intent_res.get("confidence", 0.5)),
            }
            yield {"type": "intent", "content": intent_payload}
            self.conversation_history.append({"agent": "IntentAgent", "message": json.dumps(intent_payload, ensure_ascii=False)})

            # Guardrail: refuse non-relevant questions in Agent (medical record analysis) mode.
            # For chit-chat / non-medical, direct users to the RAG tab.
            intent_type = str(intent_res.get("intent") or intent_payload.get("intent_type") or "")
            if intent_type in {"out_of_scope", "greeting", "thanks"}:
                summary = (
                    "当前问题不属于“病历分析”范畴（更像普通问答/闲聊/非医疗问题）。\n"
                    "为避免误导，我不会在病历分析模式下回答。\n\n"
                    "请切换到「普通问答（RAG）」Tab 再提问；如果你想做病历分析，请粘贴病历文本/症状描述/检查结果等。\n"
                    "（结果仅供参考，不能替代专业医生的诊断与建议。）"
                )
                report = self._build_fallback_report(
                    validated_record={},
                    intent=intent_payload,
                    structured_case={},
                    symptom_analysis={},
                    triage={},
                    department={},
                    next_steps={},
                    treatment_safety={},
                    summary=summary,
                )
                self._ensure_summary_minimum(report)
                yield {"type": "result", "content": report}
                if user_id:
                    try:
                        import uuid
                        self.session_store.add_message(
                            message_id=str(uuid.uuid4()),
                            session_id=session_id,
                            user_id=user_id,
                            mode="agent",
                            role="assistant",
                            content=str(report.get("summary") or ""),
                            report=report,
                            sources=None,
                            trace=report.get("trace") if isinstance(report, dict) else None,
                            thinking_steps=thinking_steps if thinking_steps else None,
                        )
                    except Exception:
                        pass
                yield {"type": "done", "content": ""}
                return

            history_brief = ""
            if recent_history:
                lines = []
                for h in recent_history:
                    role = "用户" if h.get("role") == "user" else "助手"
                    content = (h.get("content") or "").strip()
                    if content:
                        lines.append(f"- {role}: {content[:120]}")
                if lines:
                    history_brief = "最近会话摘要（仅供参考，优先以本次输入为准）：\n" + "\n".join(lines) + "\n\n"

            # 2) Validate & normalize record
            yield _thinking("病历验证与纠错...")
            validator_prompt = (
                "输入为用户给出的病历/描述。请输出 JSON：\n"
                "{\n"
                '  "normalized_text": "规范化后的病历文本（尽量保持原意，统一单位/术语）",\n'
                '  "missing_fields": ["缺失字段..."],\n'
                '  "contradictions": ["矛盾/不一致..."],\n'
                '  "corrections": ["纠错/规范化说明..."]\n'
                "}\n\n"
                "约束：\n"
                "- normalized_text 只能对原文做“同义替换/单位规范化/纠错/断句”，不得新增任何个人信息或病史事实（例如年龄、性别、透析、高血压等）。\n\n"
                f"{history_brief}"
                f"原文：\n{raw_question}\n"
            )
            validator_text = await self._agent_reply(agents["validator"], validator_prompt)
            validated_record = self._safe_json(validator_text)
            yield {"type": "agent_step", "content": {"agent": "RecordValidator", "step": "validated_record", "detail": validated_record}}
            self.conversation_history.append({"agent": "RecordValidator", "message": validator_text})

            normalized_text = str(validated_record.get("normalized_text") or raw_question)
            if self._has_unseen_patient_facts(raw=raw_question, text=normalized_text):
                # Prevent downstream amplification: fall back to raw text when validator hallucinates facts.
                yield _thinking("⚠️ 检测到规范化文本疑似新增个人事实，已回退使用原文继续分析。")
                normalized_text = raw_question

            # 3) Structured extraction
            yield _thinking("结构化抽取关键信息...")
            extractor_prompt = (
                "请从病历文本抽取结构化信息，输出 JSON：\n"
                "{\n"
                '  "chief_complaint": "...",\n'
                '  "symptoms": ["..."],\n'
                '  "duration": "...",\n'
                '  "vitals": { "体温": "...", "脉搏": "...", "血压": "...", "血氧": "..." },\n'
                '  "history": { "既往史": "...", "家族史": "...", "个人史": "...", "手术史": "..." },\n'
                '  "medications": ["..."],\n'
                '  "allergies": ["..."],\n'
                '  "tests": [ { "name": "...", "value": "...", "unit": "...", "note": "..." } ]\n'
                "}\n\n"
                "约束：\n"
                "- 只抽取原文中明确出现的信息；原文未提到的字段填空字符串/空数组/空对象。\n\n"
                f"病历文本：\n{normalized_text}\n"
            )
            extractor_text = await self._agent_reply(agents["extractor"], extractor_prompt)
            structured_case = self._safe_json(extractor_text)
            yield {"type": "agent_step", "content": {"agent": "SymptomExtractor", "step": "structured_case", "detail": structured_case}}
            self.conversation_history.append({"agent": "SymptomExtractor", "message": extractor_text})

            # 4) Evidence retrieval (grounding)
            yield _thinking("检索参考资料（用于依据与下一步建议）...")
            # Build a stable, medical-record-grounded retrieval query to avoid drifting to irrelevant docs.
            symptoms = structured_case.get("symptoms") if isinstance(structured_case, dict) else None
            symptom_text = "；".join([s for s in (symptoms or []) if isinstance(s, str) and s.strip()][:6])
            retrieval_query_parts = [
                raw_question,
                str(structured_case.get("chief_complaint") or "").strip() if isinstance(structured_case, dict) else "",
                symptom_text,
                str(intent_res.get("entity") or "").strip(),
            ]
            retrieval_query = "；".join([p for p in retrieval_query_parts if p])
            sources, sources_brief = await self._retrieve_evidence(str(retrieval_query))
            if sources:
                yield {"type": "sources", "content": sources}

            # 5) Symptom analysis / possible conditions
            yield _thinking("病症分析与鉴别诊断候选...")
            analyst_prompt = (
                "请基于结构化病历与参考资料，输出 JSON：\n"
                "{\n"
                '  "key_findings": ["关键发现..."],\n'
                '  "possible_conditions": ["可能疾病/问题（不要确诊）..."]\n'
                "}\n\n"
                "约束：\n"
                "- 不能把参考资料中的“某个病例/某个患者”的人口学信息当作用户事实。\n"
                "- 用户画像（年龄/性别/基础病/用药/检查结果）只能来自原文或结构化病历；缺失则保持未知。\n\n"
                f"结构化病历：\n{json.dumps(structured_case, ensure_ascii=False)}\n\n"
                f"参考资料摘要：\n{sources_brief}\n\n"
                "要求：\n"
                "- 只能基于病历与参考资料，不要凭空引入未出现的新疾病名\n"
            )
            analyst_text = await self._agent_reply(agents["analyst"], analyst_prompt)
            symptom_analysis = self._safe_json(analyst_text)
            yield {"type": "agent_step", "content": {"agent": "ConditionAnalyst", "step": "symptom_analysis", "detail": symptom_analysis}}
            self.conversation_history.append({"agent": "ConditionAnalyst", "message": analyst_text})

            # 6) Triage
            yield _thinking("紧急程度评估与红旗征...")
            triage_prompt = (
                "请根据病历判断紧急程度，输出 JSON：\n"
                "{\n"
                '  "severity_level": "emergency|urgent|routine",\n'
                '  "red_flags": ["红旗征..."],\n'
                '  "why": "分级理由（简短）"\n'
                "}\n\n"
                f"病历文本：\n{normalized_text}\n"
            )
            triage_text = await self._agent_reply(agents["triage"], triage_prompt)
            triage = self._safe_json(triage_text)
            yield {"type": "agent_step", "content": {"agent": "TriageNurse", "step": "triage", "detail": triage}}
            self.conversation_history.append({"agent": "TriageNurse", "message": triage_text})

            # 7) Department
            yield _thinking("推荐就诊科室...")
            dept_prompt = (
                "请输出 JSON：\n"
                "{\n"
                '  "recommended": ["首选科室..."],\n'
                '  "alternatives": ["备选科室..."],\n'
                '  "reason": "理由（简短）"\n'
                "}\n\n"
                f"病历要点：\n{json.dumps(symptom_analysis, ensure_ascii=False)}\n"
            )
            dept_text = await self._agent_reply(agents["dept"], dept_prompt)
            department = self._safe_json(dept_text)
            yield {"type": "agent_step", "content": {"agent": "DepartmentRecommender", "step": "department", "detail": department}}
            self.conversation_history.append({"agent": "DepartmentRecommender", "message": dept_text})

            # 8) Next steps
            yield _thinking("制定下一步举措...")
            planner_prompt = (
                "请输出 JSON：\n"
                "{\n"
                '  "immediate_actions": ["立即能做的措施（非处方）...（至少2条，无法判断也给通用且安全的建议）"],\n'
                '  "recommended_tests": ["建议检查...（至少2条，无法判断也给通用检查项）"],\n'
                '  "when_to_seek_care": ["何时就医/急诊..."]\n'
                "}\n\n"
                f"病历文本：\n{normalized_text}\n\n"
                f"紧急程度：\n{json.dumps(triage, ensure_ascii=False)}\n"
            )
            planner_text = await self._agent_reply(agents["planner"], planner_prompt)
            next_steps = self._safe_json(planner_text)
            # ensure UI won't show empty blocks even if LLM returns empty arrays
            _tmp_report = {"triage": triage, "next_steps": next_steps}
            self._ensure_next_steps_minimum(_tmp_report)
            next_steps = _tmp_report.get("next_steps") or {}
            yield {"type": "agent_step", "content": {"agent": "NextStepPlanner", "step": "next_steps", "detail": next_steps}}
            self.conversation_history.append({"agent": "NextStepPlanner", "message": planner_text})

            # 9) Safety check
            yield _thinking("安全性审阅与禁忌提醒...")
            safety_prompt = (
                "请输出 JSON：\n"
                "{\n"
                '  "medication_considerations": ["用药考虑（不写剂量，不开处方）..."],\n'
                '  "contraindications": ["常见禁忌/不适用情况..."],\n'
                '  "cautions": ["其他安全提醒..."]\n'
                "}\n\n"
                f"病历文本：\n{normalized_text}\n"
            )
            safety_text = await self._agent_reply(agents["safety"], safety_prompt)
            treatment_safety = self._safe_json(safety_text)
            yield {"type": "agent_step", "content": {"agent": "SafetyCritic", "step": "treatment_safety", "detail": treatment_safety}}
            self.conversation_history.append({"agent": "SafetyCritic", "message": safety_text})

            # 10) Coordinator merge
            yield _thinking("汇总生成最终结构化报告...")
            coordinator_prompt = (
                "请把以下模块结果合并为最终 JSON，字段必须完全包含：\n"
                "{\n"
                '  "validated_record": {...},\n'
                '  "intent": {...},\n'
                '  "structured_case": {...},\n'
                '  "symptom_analysis": {...},\n'
                '  "triage": {...},\n'
                '  "department": {...},\n'
                '  "next_steps": {...},\n'
                '  "treatment_safety": {...},\n'
                '  "summary": "一句到三句的结论性总结（包含免责声明：不能替代医生）",\n'
                '  "trace": [ {"agent": "...", "message": "..." } ]\n'
                "}\n\n"
                "请确保：\n"
                "- severity_level 只能是 emergency/urgent/routine\n"
                "- 缺失字段用空字符串/空数组/空对象，不要省略键\n\n"
                "重要约束（防幻觉）：\n"
                "- 不得捏造用户画像：年龄/性别/妊娠/基础病/用药/检查结果等个体事实只能来自【原文】或上游结构化结果中明确来自原文的字段；不确定就不要写。\n"
                "- 参考资料仅用于通用医学依据，不能把其中的“某病例/某患者”信息当作用户事实。\n\n"
                f"原文={raw_question}\n"
                f"validated_record={json.dumps(validated_record, ensure_ascii=False)}\n"
                f"intent={json.dumps(intent_payload, ensure_ascii=False)}\n"
                f"structured_case={json.dumps(structured_case, ensure_ascii=False)}\n"
                f"symptom_analysis={json.dumps(symptom_analysis, ensure_ascii=False)}\n"
                f"triage={json.dumps(triage, ensure_ascii=False)}\n"
                f"department={json.dumps(department, ensure_ascii=False)}\n"
                f"next_steps={json.dumps(next_steps, ensure_ascii=False)}\n"
                f"treatment_safety={json.dumps(treatment_safety, ensure_ascii=False)}\n"
            )
            coordinator_text = await self._agent_reply(agents["coordinator"], coordinator_prompt)
            report = self._safe_json(coordinator_text)

            # If coordinator fails to output a full schema, fall back to deterministic merge of stage outputs.
            if not isinstance(report, dict) or not report.get("validated_record") or not report.get("triage") or not report.get("department") or not report.get("next_steps"):
                # Build a minimal summary if missing
                fallback_summary = ""
                if isinstance(report, dict):
                    fallback_summary = str(report.get("summary") or "")
                if not fallback_summary:
                    fallback_summary = "已完成病历分析（结果仅供参考，不能替代专业医生的诊断与建议）。"
                report = self._build_fallback_report(
                    validated_record=validated_record,
                    intent=intent_payload,
                    structured_case=structured_case,
                    symptom_analysis=symptom_analysis,
                    triage=triage,
                    department=department,
                    next_steps=next_steps,
                    treatment_safety=treatment_safety,
                    summary=fallback_summary,
                )
            else:
                # attach trace from internal history if coordinator didn't include
                if not isinstance(report.get("trace"), list) or not report.get("trace"):
                    report["trace"] = [{"agent": h["agent"], "message": h["message"]} for h in self.conversation_history]
                # normalize next_steps for UI
                self._ensure_next_steps_minimum(report)
                self._ensure_summary_minimum(report)
            # also ensure summary for fallback path
            self._ensure_summary_minimum(report)

            # Final hallucination guard for the user-facing summary.
            try:
                summary_text = str(report.get("summary") or "")
                if summary_text and self._has_unseen_patient_facts(raw=raw_question, text=summary_text):
                    report["summary"] = "病历信息不足，建议补充关键病史与检查结果后再评估（结果仅供参考，不能替代专业医生的诊断与建议）。"
            except Exception:
                pass

            yield {"type": "result", "content": report}
            # Persist assistant message (summary + report)
            if user_id:
                try:
                    import uuid
                    self.session_store.add_message(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        mode="agent",
                        role="assistant",
                        content=str(report.get("summary") or ""),
                        report=report,
                        sources=sources if sources else None,
                        trace=report.get("trace") if isinstance(report, dict) else None,
                        thinking_steps=thinking_steps if thinking_steps else None,
                    )
                except Exception:
                    pass

        except Exception as e:
            yield {"type": "error", "content": f"诊断过程出错: {str(e)}"}

        yield {"type": "done", "content": ""}
