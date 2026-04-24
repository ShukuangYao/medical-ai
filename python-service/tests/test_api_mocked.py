import pytest
import httpx
import sys
import types


def _ensure_autogen_stub():
    """Some environments don't have `autogen` installed; stub minimal API for import-time wiring."""
    if "autogen" in sys.modules:
        return
    m = types.ModuleType("autogen")

    class _ConversableAgent:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            pass

        def generate_reply(self, *args, **kwargs):
            return "{}"

    m.ConversableAgent = _ConversableAgent
    sys.modules["autogen"] = m


def _ensure_singletons_stub():
    """Stub `app.core.singletons` to avoid importing heavy ML dependencies in tests."""
    if "app.core.singletons" in sys.modules:
        return
    m = types.ModuleType("app.core.singletons")

    class _RagEngine:
        async def query_stream(self, *args, **kwargs):
            # default empty stream; tests will monkeypatch per-case
            if False:  # pragma: no cover
                yield {}

    async def _ensure_rag_initialized():
        return None

    async def _get_agent_orchestrator():
        class _Dummy:
            async def diagnose_stream(self, *args, **kwargs):
                if False:  # pragma: no cover
                    yield {}

        return _Dummy()

    m.rag_engine = _RagEngine()
    m.ensure_rag_initialized = _ensure_rag_initialized
    m.get_agent_orchestrator = _get_agent_orchestrator
    sys.modules["app.core.singletons"] = m


@pytest.mark.asyncio
async def test_api_rag_mocked(monkeypatch):
    _ensure_autogen_stub()
    _ensure_singletons_stub()
    from app.main import app
    import app.routers.rag as rag_router

    async def _mock_query_stream(
        question: str,
        session_id: str = "default",
        chat_history=None,
        use_graph: bool = True,
        user_id=None,
        model_provider=None,
        model_name=None,
        cancel_run_id=None,
    ):
        yield {"type": "token", "content": "hello"}
        yield {"type": "token", "content": " world"}
        yield {"type": "sources", "content": [{"title": "t", "content": "c", "page": 1}]}
        yield {"type": "done", "content": ""}

    monkeypatch.setattr(rag_router.rag_engine, "query_stream", _mock_query_stream)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/api/rag",
            json={"message": "hi", "session_id": "demo", "user_id": "anonymous"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data and data["answer"] == "hello world"
        assert "sources" in data and isinstance(data["sources"], list) and len(data["sources"]) == 1
        assert "errors" in data and isinstance(data["errors"], list)


@pytest.mark.asyncio
async def test_api_agent_mocked(monkeypatch):
    _ensure_autogen_stub()
    _ensure_singletons_stub()
    from app.main import app
    import app.routers.agent as agent_router

    class _MockOrchestrator:
        async def diagnose_stream(
            self,
            medical_record: str,
            session_id: str = "default",
            model_provider=None,
            model_name=None,
            user_id=None,
            agent_pipeline: str = "fast",
            cancel_run_id=None,
        ):
            # minimal agent_step trace
            yield {"type": "agent_step", "content": {"agent": "X", "step": "y", "detail": {"k": "v"}}}
            # minimal report satisfying AgentReport schema
            report = {
                "validated_record": {
                    "normalized_text": medical_record,
                    "missing_fields": [],
                    "contradictions": [],
                    "corrections": [],
                },
                "intent": {
                    "raw_question": medical_record,
                    "resolved_question": medical_record,
                    "intent_type": "general_medical",
                    "confidence": 0.5,
                },
                "structured_case": {
                    "chief_complaint": None,
                    "symptoms": [],
                    "duration": None,
                    "vitals": {},
                    "history": {},
                    "medications": [],
                    "allergies": [],
                    "tests": [],
                },
                "symptom_analysis": {"key_findings": [], "possible_conditions": []},
                "triage": {"severity_level": "routine", "red_flags": [], "why": ""},
                "department": {"recommended": [], "alternatives": [], "reason": ""},
                "next_steps": {"immediate_actions": [], "recommended_tests": [], "when_to_seek_care": []},
                "treatment_safety": {
                    "medication_considerations": [],
                    "contraindications": [],
                    "cautions": [],
                },
                "summary": "ok（结果仅供参考，不能替代专业医生的诊断与建议）。",
                "trace": [],
            }
            yield {"type": "result", "content": report}
            yield {"type": "done", "content": ""}

    async def _mock_get_agent_orchestrator():
        return _MockOrchestrator()

    monkeypatch.setattr(agent_router, "get_agent_orchestrator", _mock_get_agent_orchestrator)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/api/agent",
            json={"message": "hi", "session_id": "demo", "user_id": "anonymous", "agent_pipeline": "fast"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("report") is not None
        assert isinstance(data.get("trace"), list)

