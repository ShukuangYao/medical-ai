from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models import ChatRequest, ChatResponse, TraceItem, Source
from app.core.agent_orchestrator import MedicalAgentOrchestrator
from app.core.rag_engine import LocalDocQA
from app.core.graph_querier import GraphQuerier
import json

router = APIRouter()
agent_orchestrator = None

@router.on_event("startup")
async def startup():
    global agent_orchestrator
    try:
        rag = LocalDocQA()
        await rag.initialize()
        graph = GraphQuerier()
        agent_orchestrator = MedicalAgentOrchestrator(rag, graph)
        print("多智能体编排器已初始化")
    except Exception as e:
        print(f"Agent编排器初始化失败: {e}")

@router.post("/agent/stream")
async def agent_diagnose_stream(request: ChatRequest):
    """多智能体协作诊断（流式输出）"""
    if agent_orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent编排器未初始化")

    async def generate():
        try:
            async for chunk in agent_orchestrator.diagnose_stream(
                request.message,
                request.session_id or "default",
                model_provider=request.model_provider,
                model_name=request.model_name,
                user_id=request.user_id,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/agent", response_model=ChatResponse)
async def agent_diagnose(request: ChatRequest):
    """多智能体协作诊断（非流式）"""
    if agent_orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent编排器未初始化")

    try:
        trace = []
        report = None

        async for chunk in agent_orchestrator.diagnose_stream(
            request.message, request.session_id or "default"
        ):
            if chunk["type"] == "agent_step":
                c = chunk.get("content") or {}
                trace.append(TraceItem(agent=str(c.get("agent", "agent")), message=json.dumps(c, ensure_ascii=False)))
            elif chunk["type"] == "result":
                report = chunk["content"]

        if not report:
            raise HTTPException(status_code=500, detail="诊断失败")

        # Prefer report trace if present
        report_trace = report.get("trace") if isinstance(report, dict) else None
        if isinstance(report_trace, list) and report_trace:
            trace = [TraceItem(agent=str(t.get("agent", "agent")), message=str(t.get("message", ""))) for t in report_trace]

        return ChatResponse(
            answer=report.get("summary", "病历分析完成") if isinstance(report, dict) else "病历分析完成",
            sources=[],
            trace=trace
            ,
            report=report
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
