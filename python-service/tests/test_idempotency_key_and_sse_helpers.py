from app.core.tools.base import ToolContext
from app.core.tools.idempotency_key import make_idempotency_key
from app.core.tools.result import ToolErrorPayload, ToolResult
from app.core.tools.sse_helpers import record_tool_result_perf, tool_result_to_sse_error


def test_make_idempotency_key_stable():
    ctx = ToolContext(user_id="u", session_id="s", run_id="r", mode="rag")
    k1 = make_idempotency_key(namespace="rag", tool_name="rerank", ctx=ctx, args={"a": 1, "b": 2}, extra="x")
    k2 = make_idempotency_key(namespace="rag", tool_name="rerank", ctx=ctx, args={"b": 2, "a": 1}, extra="x")
    assert k1 == k2
    assert k1.startswith("rag:rerank:v1:u:s:r:x:")


def test_tool_result_to_sse_error():
    res = ToolResult(
        ok=False,
        error=ToolErrorPayload(code="PERMISSION_DENIED", message="denied", retriable=False),
        cached=False,
        duration_ms=12,
        idempotency_key="k",
    )
    evt = tool_result_to_sse_error(res=res, phase="evidence_retrieval", fallback_message="fallback")
    assert evt is not None
    assert evt["type"] == "error"
    assert evt["phase"] == "evidence_retrieval"
    assert evt["code"] == "PERMISSION_DENIED"


def test_record_tool_result_perf_aggregates_hits_misses():
    perf = {}
    record_tool_result_perf(perf=perf, tool="hybrid_retrieve", res=ToolResult(ok=True, data=[1], cached=False, duration_ms=5, idempotency_key="k1"))
    record_tool_result_perf(perf=perf, tool="hybrid_retrieve", res=ToolResult(ok=True, data=[1], cached=True, duration_ms=0, idempotency_key="k1"))
    record_tool_result_perf(perf=perf, tool="rerank", res=ToolResult(ok=False, error=ToolErrorPayload(code="INTERNAL", message="x"), cached=False, duration_ms=9, idempotency_key="k2"))

    assert perf["tool_cache_hits"]["hybrid_retrieve"] == 1
    assert perf["tool_cache_misses"]["hybrid_retrieve"] == 1
    assert perf["tool_cache_misses"]["rerank"] == 1
    assert isinstance(perf["tool_calls"], list) and len(perf["tool_calls"]) == 3

