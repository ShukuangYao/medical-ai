"""AutoGen多智能体协作编排器 - 病历分析系统（严格JSON输出）"""
from __future__ import annotations

from typing import Dict, AsyncGenerator, Any, Optional, List, Tuple, Literal
import asyncio
import json
import hashlib
import os
import re
import time

from autogen import ConversableAgent

from app.config import settings
from app.core.llm_client import OpenAILLM
from app.core.rag_engine import LocalDocQA
from app.core.graph_querier import GraphQuerier
from app.core.intent_classifier_v2 import HybridIntentClassifier
from app.core.session_store import SessionStore
from app.core.tools.base import ToolContext, ToolError
from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolRegistry
from app.core.tools.impl import GraphQueryTool, HybridRetrieveTool, RerankTool, RewriteQuestionTool
from app.core.tools.idempotency_key import make_idempotency_key
from app.core.run_cancel import is_cancelled as run_cancelled
from app.core.report_validator import normalize_and_validate_agent_report
from app.core.tools.result import ToolErrorPayload
from app.core.tools.sse_helpers import tool_result_to_sse_error, record_tool_result_perf

from langsmith import RunTree
from langsmith.run_helpers import get_current_run_tree, tracing_context


class MedicalAgentOrchestrator:
    """医疗病历分析多智能体协调器（阶段式编排）"""

    def __init__(self, rag_engine: LocalDocQA, graph_querier: GraphQuerier):
        self.rag = rag_engine
        self.graph = graph_querier
        _ap0 = settings.AGENT_DEFAULT_LLM_PROVIDER
        if _ap0 not in ("qwen", "deepseek"):
            _ap0 = "deepseek"
        if _ap0 == "deepseek":
            self.llm = OpenAILLM.from_provider(provider="deepseek", model_name=None)
        else:
            self.llm = OpenAILLM.from_provider(provider="qwen", model_name=settings.LLM_MODEL)
        self.intent_classifier = HybridIntentClassifier(llm=self.llm)
        self.session_store = SessionStore(settings.CHAT_DB_PATH)
        # Tool boundary: unify validation/error codes/trace spans for evidence retrieval and graph queries.
        self.tools: Optional[ToolExecutor] = None
        try:
            reg = ToolRegistry()
            retriever = getattr(self.rag, "retriever", None)
            reranker = getattr(self.rag, "reranker", None)
            rewrite_chain = getattr(self.rag, "rewrite_chain", None)
            if retriever is not None:
                reg.register(HybridRetrieveTool(retriever))
            if reranker is not None:
                reg.register(RerankTool(reranker))
            reg.register(GraphQueryTool(self.graph))
            if rewrite_chain is not None:
                reg.register(RewriteQuestionTool(rewrite_chain))
            self.tools = ToolExecutor(registry=reg)
        except Exception:
            self.tools = None
        # In-process cache for evidence retrieval to reduce repeated heavy rerank calls.
        # key: query -> (ts, sources, brief)
        self._evidence_cache: Dict[str, Tuple[float, List[Dict[str, Any]], str]] = {}
        # In-process cache for expensive Validator/Extractor results (record -> prepared outputs).
        # key: sha256(record_text) -> (ts, validated_record, structured_case)
        self._record_prepare_cache: Dict[str, Tuple[float, Dict[str, Any], Dict[str, Any]]] = {}

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

    def _agent_llm_config(
        self,
        *,
        default_provider: Optional[str],
        default_model_name: Optional[str],
        override_provider_env: str,
        override_model_env: str,
    ) -> Dict[str, Any]:
        """Build per-agent llm_config with optional env overrides."""
        p = (os.getenv(override_provider_env, "") or "").strip() or (default_provider or None)
        m = (os.getenv(override_model_env, "") or "").strip() or (default_model_name or None)
        return self._llm_config(provider=p, model_name=m)

    def _create_agents(self, *, provider: Optional[str], model_name: Optional[str]) -> Dict[str, ConversableAgent]:
        # When provider=deepseek, all agents share one default: deepseek-chat (latency/JSON 稳定性更好).
        # Per-role overrides: AGENT_MODEL_* / AGENT_PROVIDER_* env vars still win in `_agent_llm_config`.
        p_norm = (provider or "").strip().lower()
        base_model = model_name
        if p_norm == "deepseek":
            base_model = base_model or "deepseek-chat"

        def _model_for(_role: str) -> Optional[str]:
            return base_model

        common_rules = (
            "要求：\n"
            "1) 只输出JSON，不要Markdown、不加解释文字。\n"
            "2) 不要编造：缺失信息要放入 missing_fields/空字符串/空数组。\n"
            "3) 如存在矛盾或不合理处，明确列出。\n"
            "4) 医疗建议必须保守、以就医与检查为导向，避免具体处方与剂量。\n"
            "5) 禁止捏造用户画像：不得凭空补全年龄/性别/妊娠/基础病/用药/检查结果等个体事实；只能使用输入文本或明确给定的结构化字段。\n"
        )
        validator = ConversableAgent(
            name="RecordValidator",
            system_message="你是病历质量审核与纠错专家（字段完整性、矛盾、单位/术语规范化）。" + common_rules,
            llm_config=self._agent_llm_config(
                default_provider=provider,
                default_model_name=_model_for("validator"),
                override_provider_env="AGENT_PROVIDER_VALIDATOR",
                override_model_env="AGENT_MODEL_VALIDATOR",
            ),
            human_input_mode="NEVER",
        )
        extractor = ConversableAgent(
            name="SymptomExtractor",
            system_message="你是病历结构化抽取专家（主诉、症状、病史、用药、过敏、检查）。" + common_rules,
            llm_config=self._agent_llm_config(
                default_provider=provider,
                default_model_name=_model_for("extractor"),
                override_provider_env="AGENT_PROVIDER_EXTRACTOR",
                override_model_env="AGENT_MODEL_EXTRACTOR",
            ),
            human_input_mode="NEVER",
        )
        analyst = ConversableAgent(
            name="ConditionAnalyst",
            system_message="你是病症分析与鉴别诊断专家（强调不确诊，给出候选与依据）。" + common_rules,
            llm_config=self._agent_llm_config(
                default_provider=provider,
                default_model_name=_model_for("analyst"),
                override_provider_env="AGENT_PROVIDER_ANALYST",
                override_model_env="AGENT_MODEL_ANALYST",
            ),
            human_input_mode="NEVER",
        )
        # Merge roles to reduce LLM calls in `full` pipeline:
        # - triage + department recommendation are tightly coupled
        # - next steps planning naturally includes safety review
        triage_dept = ConversableAgent(
            name="TriageDept",
            system_message="你是分诊与就医路径规划助手，负责紧急程度分级（红旗征）与就诊科室推荐。" + common_rules,
            llm_config=self._agent_llm_config(
                default_provider=provider,
                default_model_name=_model_for("triage"),
                override_provider_env="AGENT_PROVIDER_TRIAGE_DEPT",
                override_model_env="AGENT_MODEL_TRIAGE_DEPT",
            ),
            human_input_mode="NEVER",
        )
        planner = ConversableAgent(
            name="Planner",
            system_message="你是下一步检查/处置规划与安全审阅助手（先做什么、何时就医、注意事项、禁忌与风险提示）。" + common_rules,
            llm_config=self._agent_llm_config(
                default_provider=provider,
                default_model_name=_model_for("planner"),
                override_provider_env="AGENT_PROVIDER_PLANNER",
                override_model_env="AGENT_MODEL_PLANNER",
            ),
            human_input_mode="NEVER",
        )
        coordinator = ConversableAgent(
            name="Coordinator",
            system_message="你是协调者，负责把各模块结果合并成最终严格JSON，并写一段简短summary。" + common_rules,
            llm_config=self._agent_llm_config(
                default_provider=provider,
                default_model_name=_model_for("coordinator"),
                override_provider_env="AGENT_PROVIDER_COORDINATOR",
                override_model_env="AGENT_MODEL_COORDINATOR",
            ),
            human_input_mode="NEVER",
        )
        return {
            "validator": validator,
            "extractor": extractor,
            "analyst": analyst,
            "planner": planner,
            # keep key name `triage` for minimal downstream changes; it now outputs triage+department
            "triage": triage_dept,
            "coordinator": coordinator,
        }

    @staticmethod
    def _agent_llm_identity(agent: ConversableAgent) -> Tuple[str, str, str]:
        """Best-effort (model_id, base_url, provider_guess) for LangSmith labeling."""
        cfg = getattr(agent, "llm_config", None) or {}
        cl = cfg.get("config_list") if isinstance(cfg, dict) else None
        first = cl[0] if isinstance(cl, list) and cl and isinstance(cl[0], dict) else {}
        model = str(first.get("model") or "").strip() or "unknown_model"
        base_url = str(first.get("base_url") or "").strip().lower()
        p = "unknown_provider"
        if "deepseek" in base_url:
            p = "deepseek"
        elif "dashscope" in base_url or "compatible-mode" in base_url or "aliyuncs" in base_url:
            p = "qwen"
        return model, base_url, p

    def _agent_llm_client(self, agent: ConversableAgent) -> OpenAILLM:
        """Build an OpenAILLM client from an AutoGen agent's llm_config.

        This lets us run Agent(full) steps through our own llm_client, so cooperative cancel
        can close upstream streams and end LLM spans early.
        """
        model, base_url, provider_guess = self._agent_llm_identity(agent)
        api_key = settings.DASHSCOPE_API_KEY
        if provider_guess == "deepseek":
            api_key = settings.DEEPSEEK_API_KEY or settings.DASHSCOPE_API_KEY
        # Use agent-specific base_url when present; fall back to settings defaults.
        bu = (base_url or "").strip() or (
            settings.DEEPSEEK_API_BASE if provider_guess == "deepseek" else settings.LLM_API_BASE
        )
        cache_key = f"{provider_guess}|{bu}|{model}"
        cache = getattr(self, "_agent_llm_client_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_agent_llm_client_cache", cache)
        llm = cache.get(cache_key)
        if llm is None:
            llm = OpenAILLM(api_key=api_key, base_url=bu, model=model)
            cache[cache_key] = llm
        return llm

    @classmethod
    def _agent_llm_span_name(cls, agent: ConversableAgent, logical_name: str) -> str:
        model, _, p = cls._agent_llm_identity(agent)
        return f"{logical_name}:{p}:{model}"

    async def _agent_reply(self, agent: ConversableAgent, prompt: str) -> str:
        # autogen agent methods are sync; run them in a thread
        def _run() -> str:
            return agent.generate_reply(messages=[{"role": "user", "content": prompt}])  # type: ignore[no-any-return]
        # 把一个同步/耗时的函数 _run 放到线程池里执行
        return await asyncio.to_thread(_run)

    async def _agent_reply_full(
        self,
        agent: ConversableAgent,
        prompt: str,
        *,
        cancel_run_id: Optional[str] = None,
    ) -> str:
        """`full` 管线专用：单步超时避免无限挂起；支持 cooperative cancel（尽量中断上游请求）。"""
        cap = float(getattr(settings, "AGENT_FULL_LLM_STEP_TIMEOUT_S", 150.0))
        llm = self._agent_llm_client(agent)
        messages = [{"role": "user", "content": prompt}]
        task = asyncio.create_task(
            llm.generate(
                messages,
                temperature=0.2,
                max_tokens=getattr(settings, "MAX_OUTPUT_TOKENS", 2048),
                cancel_run_id=cancel_run_id,
            )
        )
        try:
            return await asyncio.wait_for(task, timeout=cap)
        except asyncio.TimeoutError:
            task.cancel()
            return "{}"
        except asyncio.CancelledError:
            task.cancel()
            return "{}"
        except Exception:
            # Keep legacy behavior: downstream parsers expect JSON-ish strings; return "{}" on failure.
            try:
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass
            return "{}"

    async def _agent_reply_capped(self, agent: ConversableAgent, prompt: str, *, cap_s: float) -> str:
        """Run one autogen step with a custom wall-clock cap (seconds)."""
        try:
            return await asyncio.wait_for(self._agent_reply(agent, prompt), timeout=float(cap_s))
        except asyncio.TimeoutError:
            return "{}"

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

    @staticmethod
    def _truncate_for_prompt(text: str, max_chars: int) -> str:
        t = (text or "").strip()
        if max_chars <= 0:
            return ""
        if len(t) <= max_chars:
            return t
        return t[:max_chars] + "..."

    @staticmethod
    def _evidence_tool_error_event(e: ToolError) -> Dict[str, Any]:
        """SSE payload aligned with RAG stream ToolError handling (code/retriable/detail)."""
        return {"type": "error", "content": e.message, "phase": "evidence_retrieval", **e.to_event_fields()}

    @staticmethod
    def _evidence_error_event_from_payload(p: ToolErrorPayload) -> Dict[str, Any]:
        return {"type": "error", "content": p.message, "phase": "evidence_retrieval", **p.to_event_fields()}

    @staticmethod
    def _agent_cancel_requested(cancel_run_id: Optional[str]) -> bool:
        return bool(cancel_run_id) and run_cancelled(cancel_run_id)

    async def _agent_abort_stream(self, run: RunTree, t0: float) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            run.end(
                outputs={"cancelled": True, "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
                metadata={"cancelled": True},
            )
            run.patch()
        except Exception:
            pass
        yield {"type": "thinking", "content": "— 已取消（服务器已停止后续步骤）"}
        yield {"type": "done", "content": ""}

    async def _retrieve_evidence(
        self,
        *,
        query: str,
        intent_result: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        perf: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        """
        Retrieve sources for grounding (retrieval-only; no LLM generation).
        Returns: (sources, short_text_snippet, sse_error_events)
        """
        q = (query or "").strip()
        if not q:
            return [], "未检索到可用参考资料", []

        uid = (user_id or "").strip() or "anonymous"
        sid = (session_id or "").strip() or "default"
        rid = (run_id or "").strip()
        cache_key = f"{uid}:{sid}:{q}"

        # Cache for 5 minutes to avoid repeated rerank on same query.
        try:
            cached = self._evidence_cache.get(cache_key)
            if cached:
                ts, sources, brief = cached
                if time.time() - ts < 300 and sources:
                    return sources, brief, []
        except Exception:
            pass

        _perf = perf if isinstance(perf, dict) else None

        sources: List[Dict[str, Any]] = []
        brief: List[str] = []
        sse_errors: List[Dict[str, Any]] = []
        try:
            ir = intent_result or {}
            intent = str(ir.get("intent") or "general_medical")
            entity = ir.get("entity")
            use_graph = bool(ir.get("use_graph", True))

            docs: List[Dict[str, Any]] = []
            # Prefer tool boundary if available (Phase 1): graph_query + hybrid_retrieve + rerank.
            if self.tools is not None:
                ctx = ToolContext(user_id=uid, session_id=sid, run_id=rid, mode="agent")
                graph_docs: List[Dict[str, Any]] = []
                if use_graph and entity:
                    _args = {"intent": intent, "entity": str(entity)}
                    res = await self.tools.run_result(
                        "graph_query",
                        args=_args,
                        ctx=ctx,
                        trace_inputs={"intent": intent, "entity": str(entity)},
                        idempotency_key=make_idempotency_key(
                            namespace="agent",
                            tool_name="graph_query",
                            ctx=ctx,
                            args=_args,
                            extra=f"{intent}:{entity}",
                        ),
                        idempotency_ttl_s=300.0,
                    )
                    if _perf is not None:
                        record_tool_result_perf(perf=_perf, tool="graph_query", res=res)
                    if res.ok:
                        graph_docs = res.data or []
                    else:
                        evt = tool_result_to_sse_error(res=res, phase="evidence_retrieval", fallback_message="graph_query failed")
                        if evt is not None:
                            sse_errors.append(evt)
                        graph_docs = []
                hybrid_docs: List[Dict[str, Any]] = []
                _args = {"query": q}
                res = await self.tools.run_result(
                    "hybrid_retrieve",
                    args=_args,
                    ctx=ctx,
                    trace_inputs={"query": q},
                    idempotency_key=make_idempotency_key(
                        namespace="agent",
                        tool_name="hybrid_retrieve",
                        ctx=ctx,
                        args=_args,
                        extra=q[:80],
                    ),
                    idempotency_ttl_s=300.0,
                )
                if _perf is not None:
                    record_tool_result_perf(perf=_perf, tool="hybrid_retrieve", res=res)
                if res.ok:
                    hybrid_docs = res.data or []
                else:
                    evt = tool_result_to_sse_error(res=res, phase="evidence_retrieval", fallback_message="hybrid_retrieve failed")
                    if evt is not None:
                        sse_errors.append(evt)
                    hybrid_docs = []

                # Deduplicate: graph first
                all_docs = list(graph_docs) + list(hybrid_docs or [])
                seen = set()
                unique: List[Dict[str, Any]] = []
                for d in all_docs:
                    doc_id = d.get("id", d.get("text", "")[:50])
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    unique.append(d)

                graph_ids = {d.get("id", d.get("text", "")[:50]) for d in graph_docs}
                graph_kept = [d for d in unique if d.get("id", d.get("text", "")[:50]) in graph_ids]
                supplement = [d for d in unique if d.get("id", d.get("text", "")[:50]) not in graph_ids]
                # Evidence retrieval is for grounding snippets, not for perfect ranking.
                # Reduce rerank workload to cut tail latency (CrossEncoder can be slow on CPU).
                # Keep a small candidate set based on existing retrieval score if present.
                if len(supplement) > 8:
                    supplement = sorted(supplement, key=lambda d: float(d.get("score", 0.0)), reverse=True)[:8]
                if len(supplement) > 6:
                    _args = {"query": q, "docs": supplement, "top_k": 8}
                    res = await self.tools.run_result(
                        "rerank",
                        args=_args,
                        ctx=ctx,
                        trace_inputs={"query": q, "docs_count": len(supplement), "top_k": 8},
                        idempotency_key=make_idempotency_key(
                            namespace="agent",
                            tool_name="rerank",
                            ctx=ctx,
                            args={"query": q, "docs_count": len(supplement), "top_k": 8},
                            extra=f"{q[:60]}:{len(supplement)}:8",
                        ),
                        idempotency_ttl_s=300.0,
                    )
                    if _perf is not None:
                        record_tool_result_perf(perf=_perf, tool="rerank", res=res)
                    if res.ok:
                        supplement = res.data or supplement
                    else:
                        evt = tool_result_to_sse_error(res=res, phase="evidence_retrieval", fallback_message="rerank failed")
                        if evt is not None:
                            sse_errors.append(evt)
                        supplement = supplement[:8]
                docs = (graph_kept + supplement)[:10]
            else:
                # Fallback to legacy router (keeps behavior if tools init fails)
                router = getattr(self.rag, "intent_router", None)
                if router is None:
                    return [], "未检索到可用参考资料", sse_errors
                try:
                    docs, _strategy = await router.route(
                        {"intent": intent, "entity": entity, "use_graph": use_graph},
                        q,
                        graph_enabled=True,
                        tool_ctx=ToolContext(user_id=uid, session_id=sid, run_id=rid, mode="agent"),
                    )
                except ToolError as e:
                    sse_errors.append(self._evidence_tool_error_event(e))
                    docs = []

            for doc in (docs or [])[:5]:
                src = {
                    "title": doc.get("title", "未知来源"),
                    "content": (doc.get("text", "") or "")[:300],
                    "page": doc.get("page"),
                    "retrieval_source": doc.get("retrieval_source", "unknown"),
                }
                sources.append(src)
                title = src.get("title", "source")
                content = (src.get("content") or "")[:180]
                brief.append(f"- {title}: {content}")
        except Exception:
            pass
        brief_text = "\n".join(brief) if brief else "未检索到可用参考资料"
        try:
            self._evidence_cache[cache_key] = (time.time(), sources, brief_text)
        except Exception:
            pass
        return sources, brief_text, sse_errors

    async def diagnose_stream(
        self,
        medical_record: str,
        session_id: str = "default",
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_pipeline: Literal["fast", "full"] = "fast",
        cancel_run_id: Optional[str] = None,
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

        # 多智能体默认走 DeepSeek；仅当请求显式传 qwen/deepseek 时覆盖（RAG 接口不受此项影响）
        mp_in = (model_provider or "").strip().lower()
        if mp_in in ("qwen", "deepseek"):
            model_provider = mp_in
        else:
            model_provider = settings.AGENT_DEFAULT_LLM_PROVIDER

        # 创建智能体
        agents = self._create_agents(provider=model_provider, model_name=model_name)

        yield _thinking("开始多智能体病历分析...")

        parent = get_current_run_tree()
        run: RunTree
        if parent is not None:
            run = parent.create_child(
                name="python_agent",
                run_type="chain",
                inputs={
                    "medical_record": medical_record,
                    "session_id": session_id,
                    "user_id": user_id,
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "agent_pipeline": agent_pipeline,
                    "stream": True,
                },
            )
        else:
            run = RunTree(
                name="python_agent",
                run_type="chain",
                inputs={
                    "medical_record": medical_record,
                    "session_id": session_id,
                    "user_id": user_id,
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "agent_pipeline": agent_pipeline,
                    "stream": True,
                },
                project_name=os.getenv("LANGSMITH_PROJECT"),
            )
        with tracing_context(parent=run):
            try:
                run.post()
                t0 = time.perf_counter()
                if self._agent_cancel_requested(cancel_run_id):
                    async for _ev in self._agent_abort_stream(run, t0):
                        yield _ev
                    return
                perf: Dict[str, Any] = {}
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
                            run_id=cancel_run_id,
                        )
                    except Exception as e:
                        yield _thinking(f"⚠️ 会话写入失败: {e}")

                # 1) Intent (fast path: existing classifier)
                yield _thinking("意图识别与问题消解...")
                intent_span = run.create_child(
                    name="intent_classify",
                    run_type="tool",
                    inputs={
                        "raw_question": raw_question,
                        "model_provider": model_provider,
                        "model_name": model_name,
                    },
                )
                intent_span.post()
                intent_llm = OpenAILLM.from_provider(provider=model_provider, model_name=model_name)
                intent_res = await HybridIntentClassifier(llm=intent_llm).classify(raw_question, cancel_run_id=cancel_run_id)
                intent_span.end(outputs=intent_res)
                intent_span.patch()
                intent_payload = {
                    "raw_question": raw_question,
                    "resolved_question": raw_question,
                    "intent_type": intent_res.get("intent", "general_medical"),
                    "confidence": float(intent_res.get("confidence", 0.5)),
                }
                yield {"type": "intent", "content": intent_payload}
                self.conversation_history.append(
                    {"agent": "IntentAgent", "message": json.dumps(intent_payload, ensure_ascii=False)}
                )
                if self._agent_cancel_requested(cancel_run_id):
                    async for _ev in self._agent_abort_stream(run, t0):
                        yield _ev
                    return

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
                    run.end(outputs={"report": report, "elapsed_ms": int((time.perf_counter() - t0) * 1000)})
                    run.patch()
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
                                run_id=cancel_run_id,
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

                validated_record: Dict[str, Any] = {}
                structured_case: Dict[str, Any] = {}
                normalized_text = raw_question
                raw_short = self._truncate_for_prompt(
                    raw_question,
                    int(getattr(settings, "AGENT_FULL_RECORD_MAX_CHARS", 1800)),
                )
                cache_key = hashlib.sha256((raw_question or "").encode("utf-8")).hexdigest()
                cache_ttl = float(getattr(settings, "AGENT_PREPARE_CACHE_TTL_S", 3600.0))
                now_s = time.time()

                if agent_pipeline == "full":
                    # Legacy: separate validator + extractor (more detailed, slower)
                    cached = self._record_prepare_cache.get(cache_key)
                    if cached and (now_s - float(cached[0])) <= cache_ttl:
                        validated_record = cached[1] or {}
                        structured_case = cached[2] or {}
                        normalized_text = str(validated_record.get("normalized_text") or raw_question)
                        yield _thinking("♻️ 命中缓存：复用病历校验/结构化抽取结果")
                        yield {"type": "agent_step", "content": {"agent": "RecordValidator", "step": "validated_record", "detail": validated_record}}
                        yield {"type": "agent_step", "content": {"agent": "SymptomExtractor", "step": "structured_case", "detail": structured_case}}
                    else:
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
                        f"原文（截断）：\n{raw_short}\n"
                    )
                    vm, vbu, vp = self._agent_llm_identity(agents["validator"])
                    validator_span = run.create_child(
                        name=self._agent_llm_span_name(agents["validator"], "RecordValidator"),
                        run_type="llm",
                        inputs={
                            "prompt": validator_prompt,
                            "prompt_chars": len(validator_prompt),
                            "model": vm,
                            "provider_guess": vp,
                            "base_url": vbu,
                        },
                    )
                    validator_span.post()
                    if not (cached and (now_s - float(cached[0])) <= cache_ttl):
                        try:
                            validator_text = await self._agent_reply_full(
                                agents["validator"], validator_prompt, cancel_run_id=cancel_run_id
                            )
                            validator_span.end(outputs={"text": validator_text})
                            validator_span.patch()
                        except asyncio.CancelledError:
                            validator_span.end(error="cancelled", metadata={"cancelled": True})
                            validator_span.patch()
                            raise
                        validated_record = self._safe_json(validator_text)
                        yield {"type": "agent_step", "content": {"agent": "RecordValidator", "step": "validated_record", "detail": validated_record}}
                        self.conversation_history.append({"agent": "RecordValidator", "message": validator_text})

                        normalized_text = str(validated_record.get("normalized_text") or raw_question)
                        if self._has_unseen_patient_facts(raw=raw_question, text=normalized_text):
                            yield _thinking("⚠️ 检测到规范化文本疑似新增个人事实，已回退使用原文继续分析。")
                            normalized_text = raw_question

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
                            f"病历文本（截断）：\n{self._truncate_for_prompt(normalized_text, int(getattr(settings, 'AGENT_FULL_RECORD_MAX_CHARS', 1800)))}\n"
                        )
                        em, ebu, ep = self._agent_llm_identity(agents["extractor"])
                        extractor_span = run.create_child(
                            name=self._agent_llm_span_name(agents["extractor"], "SymptomExtractor"),
                            run_type="llm",
                            inputs={
                                "prompt": extractor_prompt,
                                "prompt_chars": len(extractor_prompt),
                                "model": em,
                                "provider_guess": ep,
                                "base_url": ebu,
                            },
                        )
                        extractor_span.post()
                        try:
                            extractor_text = await self._agent_reply_full(
                                agents["extractor"], extractor_prompt, cancel_run_id=cancel_run_id
                            )
                            extractor_span.end(outputs={"text": extractor_text})
                            extractor_span.patch()
                        except asyncio.CancelledError:
                            extractor_span.end(error="cancelled", metadata={"cancelled": True})
                            extractor_span.patch()
                            raise
                        structured_case = self._safe_json(extractor_text)
                        yield {"type": "agent_step", "content": {"agent": "SymptomExtractor", "step": "structured_case", "detail": structured_case}}
                        self.conversation_history.append({"agent": "SymptomExtractor", "message": extractor_text})

                        try:
                            self._record_prepare_cache[cache_key] = (
                                now_s,
                                validated_record if isinstance(validated_record, dict) else {},
                                structured_case if isinstance(structured_case, dict) else {},
                            )
                        except Exception:
                            pass
                else:
                    # Fast: validate + extract in ONE LLM call
                    yield _thinking("病历规范化与结构化抽取（合并步骤）...")
                    prepare_prompt = (
                        "输入为用户给出的病历/描述。请严格输出一个 JSON 对象（不要 Markdown/解释文字），顶层包含两个键：\n"
                        "{\n"
                        '  "validated_record": {\n'
                        '    "normalized_text": "规范化后的病历文本（尽量保持原意，统一单位/术语）",\n'
                        '    "missing_fields": ["缺失字段..."],\n'
                        '    "contradictions": ["矛盾/不一致..."],\n'
                        '    "corrections": ["纠错/规范化说明..."]\n'
                        "  },\n"
                        '  "structured_case": {\n'
                        '    "chief_complaint": "...",\n'
                        '    "symptoms": ["..."],\n'
                        '    "duration": "...",\n'
                        '    "vitals": { "体温": "...", "脉搏": "...", "血压": "...", "血氧": "..." },\n'
                        '    "history": { "既往史": "...", "家族史": "...", "个人史": "...", "手术史": "..." },\n'
                        '    "medications": ["..."],\n'
                        '    "allergies": ["..."],\n'
                        '    "tests": [ { "name": "...", "value": "...", "unit": "...", "note": "..." } ]\n'
                        "  }\n"
                        "}\n\n"
                        "约束：\n"
                        "- validated_record.normalized_text 只能对原文做同义替换/单位规范化/纠错/断句，不得新增任何个人信息或病史事实。\n"
                        "- structured_case 只抽取原文中明确出现的信息；未提到的字段填空字符串/空数组/空对象。\n"
                        "- structured_case 必须与 normalized_text 语义一致（若冲突以原文为准）。\n\n"
                        f"{history_brief}"
                        f"原文：\n{raw_question}\n"
                    )
                    pm, pbu, pp = self._agent_llm_identity(agents["validator"])
                    prepare_span = run.create_child(
                        name=self._agent_llm_span_name(agents["validator"], "RecordPrepare"),
                        run_type="llm",
                        inputs={
                            "prompt": prepare_prompt,
                            "prompt_chars": len(prepare_prompt),
                            "model": pm,
                            "provider_guess": pp,
                            "base_url": pbu,
                        },
                    )
                    prepare_span.post()
                    prepare_text = await self._agent_reply(agents["validator"], prepare_prompt)
                    prepare_span.end(outputs={"text": prepare_text})
                    prepare_span.patch()
                    prepared = self._safe_json(prepare_text)
                    validated_record = prepared.get("validated_record") if isinstance(prepared, dict) else {}
                    structured_case = prepared.get("structured_case") if isinstance(prepared, dict) else {}
                    if not isinstance(validated_record, dict):
                        validated_record = {}
                    if not isinstance(structured_case, dict):
                        structured_case = {}

                    yield {"type": "agent_step", "content": {"agent": "RecordValidator", "step": "validated_record", "detail": validated_record}}
                    yield {"type": "agent_step", "content": {"agent": "SymptomExtractor", "step": "structured_case", "detail": structured_case}}
                    self.conversation_history.append({"agent": "RecordPrepare", "message": prepare_text})

                    normalized_text = str(validated_record.get("normalized_text") or raw_question)
                    if self._has_unseen_patient_facts(raw=raw_question, text=normalized_text):
                        yield _thinking("⚠️ 检测到规范化文本疑似新增个人事实，已回退使用原文继续分析。")
                        normalized_text = raw_question

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
                perf["evidence_query_chars"] = len(str(retrieval_query))
                retrieve_task: Optional[asyncio.Task] = None
                ev_span = run.create_child(
                    name="retrieve_evidence",
                    run_type="retriever",
                    inputs={"query": str(retrieval_query), "query_chars": len(str(retrieval_query))},
                )
                ev_span.post()
                if self._agent_cancel_requested(cancel_run_id):
                    ev_span.end(outputs={"cancelled": True})
                    ev_span.patch()
                    async for _ev in self._agent_abort_stream(run, t0):
                        yield _ev
                    return
                retrieve_task = asyncio.create_task(
                    self._retrieve_evidence(
                        query=str(retrieval_query),
                        intent_result=intent_res,
                        user_id=user_id or "anonymous",
                        session_id=session_id,
                        run_id=cancel_run_id,
                        perf=perf,
                    )
                )

                # In `full`, we will run merged agents later (after evidence retrieval):
                # - TriageDept: triage + department
                # - Planner: next_steps + treatment_safety
                triage_text_parallel = ""
                safety_text_parallel = ""

                sources: List[Dict[str, Any]] = []
                sources_brief: str = "未检索到可用参考资料"

                async def _await_retrieve_full() -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
                    try:
                        cap = float(getattr(settings, "AGENT_EVIDENCE_TIMEOUT_FULL_S", 25.0))
                        return await asyncio.wait_for(retrieve_task, timeout=cap)
                    except Exception:
                        try:
                            retrieve_task.cancel()
                        except Exception:
                            pass
                        return [], "未检索到可用参考资料", []

                if agent_pipeline == "full":
                    # NOTE: do NOT wait for triage/safety here; they can be slow and would block downstream steps.
                    # We only await retrieval budget here to populate sources early (if available).
                    sources, sources_brief, evidence_sse_errors = await _await_retrieve_full()
                else:
                    try:
                        cap = float(getattr(settings, "AGENT_EVIDENCE_TIMEOUT_FAST_S", 10.0))
                        sources, sources_brief, evidence_sse_errors = await asyncio.wait_for(
                            retrieve_task, timeout=cap
                        )
                    except Exception:
                        try:
                            retrieve_task.cancel()
                        except Exception:
                            pass
                        sources, sources_brief, evidence_sse_errors = [], "未检索到可用参考资料", []
                for _err_evt in evidence_sse_errors:
                    yield _err_evt
                if self._agent_cancel_requested(cancel_run_id):
                    if retrieve_task is not None and not retrieve_task.done():
                        retrieve_task.cancel()
                        try:
                            await retrieve_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    ev_span.end(outputs={"sources": sources, "sources_brief": sources_brief, "cancelled": True})
                    ev_span.patch()
                    async for _ev in self._agent_abort_stream(run, t0):
                        yield _ev
                    return
                ev_span.end(outputs={"sources": sources, "sources_brief": sources_brief})
                ev_span.patch()
                if sources:
                    yield {"type": "sources", "content": sources}

                symptom_analysis: Dict[str, Any] = {}
                triage: Dict[str, Any] = {}
                department: Dict[str, Any] = {}
                next_steps: Dict[str, Any] = {}
                treatment_safety: Dict[str, Any] = {}
                report: Dict[str, Any] = {}

                if agent_pipeline == "full":
                    # Legacy multi-agent path: separate LLM calls + coordinator merge
                    if self._agent_cancel_requested(cancel_run_id):
                        async for _ev in self._agent_abort_stream(run, t0):
                            yield _ev
                        return
                    yield _thinking("病症分析 / 分诊科室 / 下一步安全（并行）...")
                    # Keep prompts short to reduce latency/cost while preserving accuracy:
                    # Condition analysis mainly needs key fields + brief evidence, not the full raw text.
                    brief_cap = 900
                    try:
                        brief_cap = int(getattr(settings, "AGENT_PARALLEL_PROMPT_MAX_CHARS", 1200))
                    except Exception:
                        brief_cap = 900
                    sources_brief_short = self._truncate_for_prompt(sources_brief, max(200, min(brief_cap, 1200)))
                    case_short = self._truncate_for_prompt(
                        json.dumps(structured_case, ensure_ascii=False),
                        max(600, min(brief_cap, 1600)),
                    )
                    analyst_prompt = (
                        "请基于结构化病历与参考资料，输出 JSON：\n"
                        "{\n"
                        '  "key_findings": ["关键发现..."],\n'
                        '  "possible_conditions": ["可能疾病/问题（不要确诊）..."]\n'
                        "}\n\n"
                        "约束：\n"
                        "- 不能把参考资料中的“某个病例/某个患者”的人口学信息当作用户事实。\n"
                        "- 用户画像（年龄/性别/基础病/用药/检查结果）只能来自原文或结构化病历；缺失则保持未知。\n\n"
                        f"结构化病历（截断）：\n{case_short}\n\n"
                        f"参考资料摘要（截断）：\n{sources_brief_short}\n\n"
                        "要求：\n"
                        "- 只能基于病历与参考资料，不要凭空引入未出现的新疾病名\n"
                    )
                    perf["analyst_prompt_chars"] = len(analyst_prompt)
                    perf["analyst_structured_case_chars"] = len(case_short)
                    perf["analyst_sources_brief_chars"] = len(sources_brief_short)
                    am, abu, ap = self._agent_llm_identity(agents["analyst"])
                    analyst_span = run.create_child(
                        name=self._agent_llm_span_name(agents["analyst"], "ConditionAnalyst"),
                        run_type="llm",
                        inputs={
                            "prompt": analyst_prompt,
                            "prompt_chars": len(analyst_prompt),
                            "structured_case_chars": len(case_short),
                            "sources_brief_chars": len(sources_brief_short),
                            "model": am,
                            "provider_guess": ap,
                            "base_url": abu,
                        },
                    )
                    analyst_span.post()
                    norm_short2 = self._truncate_for_prompt(
                        normalized_text,
                        int(getattr(settings, "AGENT_PARALLEL_PROMPT_MAX_CHARS", 1200)),
                    )

                    async def _analyst_llm() -> str:
                        try:
                            txt = await self._agent_reply_full(
                                agents["analyst"], analyst_prompt, cancel_run_id=cancel_run_id
                            )
                        except asyncio.CancelledError:
                            analyst_span.end(error="cancelled", metadata={"cancelled": True})
                            analyst_span.patch()
                            raise
                        except Exception as e:
                            analyst_span.end(error=str(e))
                            analyst_span.patch()
                            raise
                        analyst_span.end(outputs={"text": txt})
                        analyst_span.patch()
                        return txt

                    # Merged roles: one call for triage+department, one call for plan+safe.
                    triage_dept_prompt = (
                        "请严格输出一个 JSON 对象，顶层包含两个键：triage 与 department。\n"
                        "{\n"
                        '  "triage": {\n'
                        '    "severity_level": "emergency|urgent|routine",\n'
                        '    "red_flags": ["红旗征..."],\n'
                        '    "why": "分级理由（简短）"\n'
                        "  },\n"
                        '  "department": {\n'
                        '    "recommended": ["首选科室..."],\n'
                        '    "alternatives": ["备选科室..."],\n'
                        '    "reason": "理由（简短）"\n'
                        "  }\n"
                        "}\n\n"
                        "约束：\n"
                        "- triage 用保守原则：信息不足时优先提示就医与急诊阈值。\n"
                        "- department 只回答科室选择与就医路径，不要下具体诊断结论。\n\n"
                        f"结构化病历要点（截断）：\n{case_short}\n\n"
                        f"病历文本（截断）：\n{norm_short2}\n\n"
                        f"参考资料摘要（截断）：\n{sources_brief_short}\n"
                    )
                    plan_safety_prompt = (
                        "请严格输出一个 JSON 对象，顶层包含两个键：next_steps 与 treatment_safety。\n"
                        "{\n"
                        '  "next_steps": {\n'
                        '    "immediate_actions": ["立即能做的措施（非处方）...（至少2条，无法判断也给通用且安全的建议）"],\n'
                        '    "recommended_tests": ["建议检查...（至少2条，无法判断也给通用检查项）"],\n'
                        '    "when_to_seek_care": ["何时就医/急诊..."]\n'
                        "  },\n"
                        '  "treatment_safety": {\n'
                        '    "medication_considerations": ["用药考虑（不写剂量，不开处方）..."],\n'
                        '    "contraindications": ["常见禁忌/不适用情况..."],\n'
                        '    "cautions": ["其他安全提醒..."]\n'
                        "  }\n"
                        "}\n\n"
                        "约束：\n"
                        "- 不要给具体处方与剂量；建议以检查/就医/观察为主。\n"
                        "- 安全提醒要覆盖常见风险与需线下确认的关键问题。\n\n"
                        f"结构化病历要点（截断）：\n{case_short}\n\n"
                        f"病历文本（截断）：\n{norm_short2}\n\n"
                        f"参考资料摘要（截断）：\n{sources_brief_short}\n"
                    )
                    perf["triage_dept_prompt_chars"] = len(triage_dept_prompt)
                    perf["plan_safety_prompt_chars"] = len(plan_safety_prompt)

                    tm, tbu, tp = self._agent_llm_identity(agents["triage"])
                    triage_dept_span = run.create_child(
                        name=self._agent_llm_span_name(agents["triage"], "TriageDept"),
                        run_type="llm",
                        inputs={
                            "prompt": triage_dept_prompt,
                            "prompt_chars": len(triage_dept_prompt),
                            "model": tm,
                            "provider_guess": tp,
                            "base_url": tbu,
                        },
                    )
                    triage_dept_span.post()
                    plm, plbu, plp = self._agent_llm_identity(agents["planner"])
                    plan_safety_span = run.create_child(
                        name=self._agent_llm_span_name(agents["planner"], "Planner"),
                        run_type="llm",
                        inputs={
                            "prompt": plan_safety_prompt,
                            "prompt_chars": len(plan_safety_prompt),
                            "model": plm,
                            "provider_guess": plp,
                            "base_url": plbu,
                        },
                    )
                    plan_safety_span.post()

                    if self._agent_cancel_requested(cancel_run_id):
                        async for _ev in self._agent_abort_stream(run, t0):
                            yield _ev
                        return

                    async def _triage_dept_llm() -> str:
                        try:
                            txt = await self._agent_reply_full(
                                agents["triage"], triage_dept_prompt, cancel_run_id=cancel_run_id
                            )
                        except asyncio.CancelledError:
                            triage_dept_span.end(error="cancelled", metadata={"cancelled": True})
                            triage_dept_span.patch()
                            raise
                        except Exception as e:
                            triage_dept_span.end(error=str(e))
                            triage_dept_span.patch()
                            raise
                        triage_dept_span.end(outputs={"text": txt})
                        triage_dept_span.patch()
                        return txt

                    async def _plan_safety_llm() -> str:
                        try:
                            txt = await self._agent_reply_full(
                                agents["planner"], plan_safety_prompt, cancel_run_id=cancel_run_id
                            )
                        except asyncio.CancelledError:
                            plan_safety_span.end(error="cancelled", metadata={"cancelled": True})
                            plan_safety_span.patch()
                            raise
                        except Exception as e:
                            plan_safety_span.end(error=str(e))
                            plan_safety_span.patch()
                            raise
                        plan_safety_span.end(outputs={"text": txt})
                        plan_safety_span.patch()
                        return txt

                    t_analyst = asyncio.create_task(_analyst_llm())
                    t_td = asyncio.create_task(_triage_dept_llm())
                    t_ps = asyncio.create_task(_plan_safety_llm())
                    try:
                        analyst_text, triage_dept_text, plan_safety_text = await asyncio.gather(t_analyst, t_td, t_ps)
                    except asyncio.CancelledError:
                        for t in (t_analyst, t_td, t_ps):
                            if not t.done():
                                t.cancel()
                        raise
                    symptom_analysis = self._safe_json(analyst_text)
                    yield {"type": "agent_step", "content": {"agent": "ConditionAnalyst", "step": "symptom_analysis", "detail": symptom_analysis}}
                    self.conversation_history.append({"agent": "ConditionAnalyst", "message": analyst_text})

                    merged_td = self._safe_json(triage_dept_text)
                    triage = merged_td.get("triage") if isinstance(merged_td, dict) else {}
                    department = merged_td.get("department") if isinstance(merged_td, dict) else {}
                    if not isinstance(triage, dict):
                        triage = {}
                    if not isinstance(department, dict):
                        department = {}
                    merged_ps = self._safe_json(plan_safety_text)
                    next_steps = merged_ps.get("next_steps") if isinstance(merged_ps, dict) else {}
                    treatment_safety = merged_ps.get("treatment_safety") if isinstance(merged_ps, dict) else {}
                    if not isinstance(next_steps, dict):
                        next_steps = {}
                    if not isinstance(treatment_safety, dict):
                        treatment_safety = {}

                    yield {"type": "agent_step", "content": {"agent": "TriageNurse", "step": "triage", "detail": triage}}
                    self.conversation_history.append({"agent": "TriageDept", "message": triage_dept_text})
                    _tmp_report = {"triage": triage, "next_steps": next_steps}
                    self._ensure_next_steps_minimum(_tmp_report)
                    next_steps = _tmp_report.get("next_steps") or {}
                    yield {"type": "agent_step", "content": {"agent": "DepartmentRecommender", "step": "department", "detail": department}}
                    self.conversation_history.append({"agent": "TriageDept", "message": triage_dept_text})
                    yield {"type": "agent_step", "content": {"agent": "NextStepPlanner", "step": "next_steps", "detail": next_steps}}
                    self.conversation_history.append({"agent": "Planner", "message": plan_safety_text})
                    yield {"type": "agent_step", "content": {"agent": "SafetyCritic", "step": "treatment_safety", "detail": treatment_safety}}
                    self.conversation_history.append({"agent": "Planner", "message": plan_safety_text})

                    yield _thinking("汇总生成最终结构化报告...")
                    # Optimization (accuracy-first): coordinator should not re-generate the entire JSON.
                    # We merge module outputs deterministically and only ask LLM for a short summary.
                    report = {
                        "validated_record": validated_record,
                        "intent": intent_payload,
                        "structured_case": structured_case,
                        "symptom_analysis": symptom_analysis,
                        "triage": triage,
                        "department": department,
                        "next_steps": next_steps,
                        "treatment_safety": treatment_safety,
                        "summary": "",
                        "trace": [{"agent": h["agent"], "message": h["message"]} for h in self.conversation_history],
                    }
                    self._ensure_next_steps_minimum(report)

                    summary_prompt = (
                        "请基于以下信息输出一段 1-3 句中文总结，并在末尾包含免责声明：不能替代专业医生的诊断与建议。\n"
                        "要求：不要捏造用户画像，不要给具体处方与剂量。\n\n"
                        f"原文（截断）：\n{self._truncate_for_prompt(raw_question, 800)}\n\n"
                        f"紧急程度：{json.dumps(triage, ensure_ascii=False)}\n"
                        f"就诊科室：{json.dumps(department, ensure_ascii=False)}\n"
                        f"下一步建议：{json.dumps(next_steps, ensure_ascii=False)}\n"
                    )
                    perf["coordinator_prompt_chars"] = len(summary_prompt)
                    cm, cbu, cp = self._agent_llm_identity(agents["coordinator"])
                    coord_span = run.create_child(
                        name=self._agent_llm_span_name(agents["coordinator"], "Coordinator"),
                        run_type="llm",
                        inputs={
                            "prompt": summary_prompt,
                            "prompt_chars": len(summary_prompt),
                            "mode": "summary_only",
                            "model": cm,
                            "provider_guess": cp,
                            "base_url": cbu,
                        },
                    )
                    coord_span.post()
                    try:
                        summary_text = await self._agent_reply_full(
                            agents["coordinator"], summary_prompt, cancel_run_id=cancel_run_id
                        )
                        coord_span.end(outputs={"text": summary_text})
                        coord_span.patch()
                    except asyncio.CancelledError:
                        coord_span.end(error="cancelled", metadata={"cancelled": True})
                        coord_span.patch()
                        raise
                    report["summary"] = (summary_text or "").strip()
                    self._ensure_summary_minimum(report)

                    if (
                        not isinstance(report, dict)
                        or not report.get("validated_record")
                        or not report.get("triage")
                        or not report.get("department")
                        or not report.get("next_steps")
                    ):
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
                        if not isinstance(report.get("trace"), list) or not report.get("trace"):
                            report["trace"] = [{"agent": h["agent"], "message": h["message"]} for h in self.conversation_history]
                        self._ensure_next_steps_minimum(report)
                        self._ensure_summary_minimum(report)
                    self._ensure_summary_minimum(report)
                else:
                    # Fast path: one-shot analysis + triage + plan + safety + summary
                    if self._agent_cancel_requested(cancel_run_id):
                        async for _ev in self._agent_abort_stream(run, t0):
                            yield _ev
                        return
                    yield _thinking("综合分析与生成结构化报告（加速模式）...")
                    one_shot_prompt = (
                        "你是医疗病历分析专家。请严格输出 JSON（不要 Markdown/解释文字）。\n"
                        "你必须生成完整字段：\n"
                        "{\n"
                        '  "symptom_analysis": { "key_findings": ["..."], "possible_conditions": ["..."] },\n'
                        '  "triage": { "severity_level": "emergency|urgent|routine", "red_flags": ["..."], "why": "..." },\n'
                        '  "department": { "recommended": ["..."], "alternatives": ["..."], "reason": "..." },\n'
                        '  "next_steps": { "immediate_actions": ["..."], "recommended_tests": ["..."], "when_to_seek_care": ["..."] },\n'
                        '  "treatment_safety": { "medication_considerations": ["..."], "contraindications": ["..."], "cautions": ["..."] },\n'
                        '  "summary": "一句到三句的结论性总结（必须包含免责声明：不能替代医生）"\n'
                        "}\n\n"
                        "硬约束（防幻觉）：\n"
                        "- 不得捏造用户画像/病史事实；只能使用【原文】与【结构化病历】中明确出现的信息。\n"
                        "- 参考资料仅用于通用医学依据，不得把其中其他病例信息当作用户事实。\n"
                        "- possible_conditions 只能给候选，不要确诊。\n"
                        "- immediate_actions/recommended_tests 至少各2条（无法判断也给通用且安全的建议）。\n\n"
                        f"【原文】\n{raw_question}\n\n"
                        f"【结构化病历】\n{json.dumps(structured_case, ensure_ascii=False)}\n\n"
                        f"【参考资料摘要】\n{sources_brief}\n"
                    )
                    om, obu, op = self._agent_llm_identity(agents["coordinator"])
                    one_span = run.create_child(
                        name=self._agent_llm_span_name(agents["coordinator"], "OneShotReport"),
                        run_type="llm",
                        inputs={
                            "prompt": one_shot_prompt,
                            "prompt_chars": len(one_shot_prompt),
                            "model": om,
                            "provider_guess": op,
                            "base_url": obu,
                        },
                    )
                    one_span.post()
                    one_text = await self._agent_reply(agents["coordinator"], one_shot_prompt)
                    one_span.end(outputs={"text": one_text})
                    one_span.patch()
                    one = self._safe_json(one_text)

                    symptom_analysis = one.get("symptom_analysis") if isinstance(one, dict) else {}
                    triage = one.get("triage") if isinstance(one, dict) else {}
                    department = one.get("department") if isinstance(one, dict) else {}
                    next_steps = one.get("next_steps") if isinstance(one, dict) else {}
                    treatment_safety = one.get("treatment_safety") if isinstance(one, dict) else {}
                    summary = str(one.get("summary") or "") if isinstance(one, dict) else ""

                    yield {"type": "agent_step", "content": {"agent": "ConditionAnalyst", "step": "symptom_analysis", "detail": symptom_analysis}}
                    yield {"type": "agent_step", "content": {"agent": "TriageNurse", "step": "triage", "detail": triage}}
                    yield {"type": "agent_step", "content": {"agent": "DepartmentRecommender", "step": "department", "detail": department}}
                    yield {"type": "agent_step", "content": {"agent": "NextStepPlanner", "step": "next_steps", "detail": next_steps}}
                    yield {"type": "agent_step", "content": {"agent": "SafetyCritic", "step": "treatment_safety", "detail": treatment_safety}}

                    report = self._build_fallback_report(
                        validated_record=validated_record,
                        intent=intent_payload,
                        structured_case=structured_case,
                        symptom_analysis=symptom_analysis,
                        triage=triage,
                        department=department,
                        next_steps=next_steps,
                        treatment_safety=treatment_safety,
                        summary=summary or "已完成病历分析（结果仅供参考，不能替代专业医生的诊断与建议）。",
                    )

                    if (
                        not isinstance(report, dict)
                        or not report.get("validated_record")
                        or not report.get("triage")
                        or not report.get("department")
                        or not report.get("next_steps")
                    ):
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
                        if not isinstance(report.get("trace"), list) or not report.get("trace"):
                            report["trace"] = [{"agent": h["agent"], "message": h["message"]} for h in self.conversation_history]
                        self._ensure_next_steps_minimum(report)
                        self._ensure_summary_minimum(report)
                    self._ensure_summary_minimum(report)

                # Final hallucination guard for the user-facing summary.
                try:
                    summary_text = str(report.get("summary") or "")
                    if summary_text and self._has_unseen_patient_facts(raw=raw_question, text=summary_text):
                        report["summary"] = "病历信息不足，建议补充关键病史与检查结果后再评估（结果仅供参考，不能替代专业医生的诊断与建议）。"
                except Exception:
                    pass

                # Phase 5: schema validation + normalize + unified fallback path
                validated_report, validation_err, validation_tags = normalize_and_validate_agent_report(
                    report if isinstance(report, dict) else {}
                )
                if validation_tags:
                    try:
                        perf["report_validation_tags"] = list(validation_tags)
                    except Exception:
                        pass
                    # Surface normalization warnings for debugging/QA; safe for clients to ignore.
                    yield {
                        "type": "agent_step",
                        "content": {"agent": "ReportValidator", "step": "warnings", "detail": {"tags": validation_tags}},
                    }
                if validation_err is not None:
                    # Keep stream contract: emit an error event, then a safe fallback report.
                    yield {
                        "type": "error",
                        "content": "结构化报告校验失败，已降级为安全输出。",
                        "phase": "report_validation",
                        "code": "VALIDATION_ERROR",
                        "retriable": False,
                        "detail": validation_err,
                    }
                    report = self._build_fallback_report(
                        validated_record=validated_record,
                        intent=intent_payload,
                        structured_case=structured_case,
                        symptom_analysis=symptom_analysis,
                        triage=triage,
                        department=department,
                        next_steps=next_steps,
                        treatment_safety=treatment_safety,
                        summary=str((validated_report or {}).get("summary") or report.get("summary") or ""),
                    )
                    validated_report, _, _ = normalize_and_validate_agent_report(report)
                report = validated_report

                if self._agent_cancel_requested(cancel_run_id):
                    async for _ev in self._agent_abort_stream(run, t0):
                        yield _ev
                    return

                yield {"type": "result", "content": report}
                run.end(
                    outputs={
                        "report": report,
                        "sources": sources,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "perf": perf,
                    }
                )
                run.patch()
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
                            run_id=cancel_run_id,
                            report=report,
                            sources=sources if sources else None,
                            trace=report.get("trace") if isinstance(report, dict) else None,
                            thinking_steps=thinking_steps if thinking_steps else None,
                        )
                    except Exception:
                        pass
            except Exception as e:
                run.end(error=str(e))
                run.patch()
                yield {"type": "error", "content": f"诊断过程出错: {str(e)}"}

        yield {"type": "done", "content": ""}
