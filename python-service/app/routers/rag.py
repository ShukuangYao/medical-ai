"""RAG路由 - 支持SSE流式输出"""
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.core.rag_engine import LocalDocQA

router = APIRouter()

# 全局RAG引擎实例
rag_engine = LocalDocQA()


@router.post("/rag")
async def rag_query(request: Request):
    """RAG问答（非流式）"""
    try:
        data = await request.json()
        question = data.get("question", data.get("message", ""))
        session_id = data.get("session_id", "default")
        chat_history = data.get("chat_history", None)
        use_graph = data.get("use_graph", data.get("useGraph", True))
        model_provider = data.get("model_provider")
        model_name = data.get("model_name")

        if not question:
            return {"answer": "请输入您的健康问题", "sources": []}

        result = await rag_engine.query(
            question,
            session_id,
            chat_history,
            use_graph=bool(use_graph),
            model_provider=model_provider,
            model_name=model_name,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/stream")
async def rag_query_stream(request: Request):
    """RAG问答（SSE流式输出）"""
    try:
        data = await request.json()
        question = data.get("question", data.get("message", ""))
        session_id = data.get("session_id", "default")
        chat_history = data.get("chat_history", None)
        use_graph = data.get("use_graph", data.get("useGraph", True))
        model_provider = data.get("model_provider")
        model_name = data.get("model_name")

        if not question:
            return {"answer": "请输入您的健康问题", "sources": []}

        async def event_generator():
            # 先产出一个事件，确保客户端尽快收到响应头并开始渲染（避免初始化耗时导致“无输出”假象）
            yield f"data: {json.dumps({'type': 'thinking', 'content': '🚀 请求已接收，正在准备检索与生成...'}, ensure_ascii=False)}\n\n"
            async for chunk in rag_engine.query_stream(
                question,
                session_id,
                chat_history,
                use_graph=bool(use_graph),
                model_provider=model_provider,
                model_name=model_name,
                user_id=data.get("user_id") or data.get("userId"),
            ):
                event_data = json.dumps(chunk, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
