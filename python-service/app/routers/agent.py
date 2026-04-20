import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.models import ChatRequest, ChatResponse, TraceItem, Source
import json
import time
from datetime import datetime, timezone

from app.core.singletons import get_agent_orchestrator

from langsmith import RunTree
from langsmith.run_helpers import get_current_run_tree, tracing_context

router = APIRouter()
DEFAULT_USER_ID = "anonymous"

@router.on_event("startup")
async def startup():
    try:
        _ = await get_agent_orchestrator()
        print("多智能体编排器已初始化")
    except Exception as e:
        print(f"Agent编排器初始化失败: {e}")

@router.post("/agent/stream")
async def agent_diagnose_stream(request: ChatRequest):
    """多智能体协作诊断（流式输出）"""
    agent_orchestrator = await get_agent_orchestrator()

    parent = get_current_run_tree()
    inputs = {
        "message": request.message,
        "session_id": request.session_id or "default",
        "user_id": request.user_id or DEFAULT_USER_ID,
        "model_provider": request.model_provider,
        "model_name": request.model_name,
        "agent_pipeline": request.agent_pipeline or "fast",
        "stream": True,
    }
    start_time = datetime.now(timezone.utc)
    root = (
        parent.create_child(name="http_agent_stream", run_type="chain", inputs=inputs, start_time=start_time)  # type: ignore[union-attr]
        if parent is not None
        else RunTree(name="http_agent_stream", run_type="chain", inputs=inputs, start_time=start_time)
    )
    root.post()
    t0 = time.perf_counter()

    async def generate():
        ended = False
        try:
            last_report = None
            with tracing_context(parent=root):
                uid = request.user_id or DEFAULT_USER_ID
                async for chunk in agent_orchestrator.diagnose_stream(
                    request.message,
                    request.session_id or "default",
                    model_provider=request.model_provider,
                    model_name=request.model_name,
                    user_id=uid,
                    agent_pipeline=request.agent_pipeline or "fast",
                ):
                    if chunk.get("type") == "result":
                        last_report = chunk.get("content")
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            root.end(
                outputs={"report": last_report, "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
                end_time=datetime.now(timezone.utc),
            )
            root.patch()
            ended = True
        except asyncio.CancelledError:
            try:
                if not ended:
                    root.end(error="cancelled", end_time=datetime.now(timezone.utc))
                    root.patch()
            finally:
                raise
        except Exception as e:
            try:
                if not ended:
                    root.end(error=str(e), end_time=datetime.now(timezone.utc))
                    root.patch()
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if not ended:
                try:
                    root.end(
                        outputs={"elapsed_ms": int((time.perf_counter() - t0) * 1000), "ended_in_finally": True},
                        end_time=datetime.now(timezone.utc),
                    )
                    root.patch()
                except Exception:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/agent", response_model=ChatResponse)
async def agent_diagnose(request: ChatRequest):
    """多智能体协作诊断（非流式）"""
    agent_orchestrator = await get_agent_orchestrator()

    try:
        parent = get_current_run_tree()
        inputs = {
            "message": request.message,
            "session_id": request.session_id or "default",
            "user_id": request.user_id or DEFAULT_USER_ID,
            "model_provider": request.model_provider,
            "model_name": request.model_name,
            "agent_pipeline": request.agent_pipeline or "fast",
            "stream": False,
        }
        start_time = datetime.now(timezone.utc)
        root = (
            parent.create_child(name="http_agent", run_type="chain", inputs=inputs, start_time=start_time)  # type: ignore[union-attr]
            if parent is not None
            else RunTree(name="http_agent", run_type="chain", inputs=inputs, start_time=start_time)
        )
        root.post()
        t0 = time.perf_counter()

        trace = []
        report = None
        stream_errors: list[str] = []
        uid = request.user_id or DEFAULT_USER_ID

        with tracing_context(parent=root):
            async for chunk in agent_orchestrator.diagnose_stream(
                request.message,
                request.session_id or "default",
                model_provider=request.model_provider,
                model_name=request.model_name,
                user_id=uid,
                agent_pipeline=request.agent_pipeline or "fast",
            ):
                ct = chunk.get("type")
                if ct == "agent_step":
                    c = chunk.get("content") or {}
                    trace.append(TraceItem(agent=str(c.get("agent", "agent")), message=json.dumps(c, ensure_ascii=False)))
                elif ct == "result":
                    report = chunk.get("content")
                elif ct == "error":
                    err = chunk.get("content")
                    if err is not None:
                        stream_errors.append(str(err))

        if not report:
            detail = stream_errors[-1] if stream_errors else "诊断失败（未收到结构化结果；请查看 Python 日志或 LangSmith trace）"
            raise HTTPException(status_code=500, detail=detail)

        # Prefer report trace if present
        report_trace = report.get("trace") if isinstance(report, dict) else None
        if isinstance(report_trace, list) and report_trace:
            trace = [TraceItem(agent=str(t.get("agent", "agent")), message=str(t.get("message", ""))) for t in report_trace]

        summary_out = (report.get("summary") if isinstance(report, dict) else None) or "病历分析完成"
        resp = ChatResponse(
            answer=summary_out,
            sources=[],
            trace=trace
            ,
            report=report
        )
        root.end(
            outputs={"answer": resp.answer, "report": report, "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
            end_time=datetime.now(timezone.utc),
        )
        root.patch()
        return resp
    except HTTPException:
        raise
    except Exception as e:
        try:
            root.end(error=str(e))  # type: ignore[name-defined]
            root.patch()  # type: ignore[name-defined]
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
