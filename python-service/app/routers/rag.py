"""RAG路由 - 支持SSE流式输出"""
import asyncio
import json
import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.core.singletons import rag_engine, ensure_rag_initialized

from langsmith import RunTree
from langsmith.run_helpers import get_current_run_tree, tracing_context

router = APIRouter()

DEFAULT_USER_ID = "anonymous"


@router.on_event("startup")
async def startup():
    # Warm up heavy components (embeddings/reranker/DB clients) once per process.
    try:
        await ensure_rag_initialized()
        print("RAG 引擎已初始化（startup 预热）")
    except Exception as e:
        # Allow service to start; requests will retry init path and/or degrade.
        print(f"RAG 引擎初始化失败（startup 预热）: {e}")


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
        user_id = data.get("user_id") or data.get("userId") or DEFAULT_USER_ID

        if not question:
            return {"answer": "请输入您的健康问题", "sources": []}

        parent = get_current_run_tree()
        inputs = {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
            "use_graph": bool(use_graph),
            "model_provider": model_provider,
            "model_name": model_name,
            "stream": False,
        }
        # If TracingMiddleware already created/continued a run from incoming headers,
        # attach to it; otherwise create a standalone root run.
        start_time = datetime.now(timezone.utc)
        root = (
            parent.create_child(name="http_rag", run_type="chain", inputs=inputs, start_time=start_time)  # type: ignore[union-attr]
            if parent is not None
            else RunTree(name="http_rag", run_type="chain", inputs=inputs, start_time=start_time)
        )
        root.post()
        t0 = time.perf_counter()

        # For non-stream calls, we also persist to session storage.
        # We reuse the stream pipeline (persist=True) but consume events to build the final response.
        answer_parts = []
        sources = []
        with tracing_context(parent=root):
            async for evt in rag_engine.query_stream(
                question,
                session_id,
                chat_history,
                use_graph=bool(use_graph),
                user_id=user_id,
                model_provider=model_provider,
                model_name=model_name,
            ):
                if evt.get("type") == "token":
                    answer_parts.append(evt.get("content") or "")
                elif evt.get("type") == "sources":
                    sources = evt.get("content") or []

        answer = "".join(answer_parts)
        root.end(
            outputs={"answer": answer, "sources": sources, "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
            end_time=datetime.now(timezone.utc),
        )
        root.post()
        return {"answer": answer, "sources": sources}
    except Exception as e:
        try:
            root.end(error=str(e))  # type: ignore[name-defined]
            root.post()  # type: ignore[name-defined]
        except Exception:
            pass
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

        parent = get_current_run_tree()
        inputs = {
            "question": question,
            "session_id": session_id,
            "use_graph": bool(use_graph),
            "model_provider": model_provider,
            "model_name": model_name,
            "stream": True,
        }
        start_time = datetime.now(timezone.utc)
        tags = [f"mode:rag", f"stream:true", f"session:{session_id}"]
        root = (
            parent.create_child(  # type: ignore[union-attr]
                name="http_rag_stream",
                run_type="chain",
                inputs=inputs,
                start_time=start_time,
                tags=tags,
                extra={"metadata": {"service": "python", "endpoint": "/api/rag/stream"}},
            )
            if parent is not None
            else RunTree(
                name="http_rag_stream",
                run_type="chain",
                inputs=inputs,
                start_time=start_time,
                tags=tags,
                extra={"metadata": {"service": "python", "endpoint": "/api/rag/stream"}},
            )
        )
        root.post()
        t0 = time.perf_counter()

        async def event_generator():
            full = ""
            sources = []
            ended = False
            try:
                with tracing_context(parent=root):
                    # 先产出一个事件，确保客户端尽快收到响应头并开始渲染（避免初始化耗时导致“无输出”假象）
                    yield f"data: {json.dumps({'type': 'thinking', 'content': '🚀 请求已接收，正在准备检索与生成...'}, ensure_ascii=False)}\n\n"
                    uid = data.get("user_id") or data.get("userId") or DEFAULT_USER_ID
                    async for chunk in rag_engine.query_stream(
                        question,
                        session_id,
                        chat_history,
                        use_graph=bool(use_graph),
                        model_provider=model_provider,
                        model_name=model_name,
                        user_id=uid,
                    ):
                        if chunk.get("type") == "token":
                            full += chunk.get("content") or ""
                        elif chunk.get("type") == "sources":
                            sources = chunk.get("content") or []
                        event_data = json.dumps(chunk, ensure_ascii=False)
                        yield f"data: {event_data}\n\n"
                root.end(
                    outputs={
                        "answer": full,
                        "sources": sources,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "duration_ms": int((time.perf_counter() - t0) * 1000),
                        "build_id": "ls-timing-fix-2026-04-14",
                    },
                    end_time=datetime.now(timezone.utc),
                )
                root.patch()
                ended = True
            except asyncio.CancelledError:
                # Client disconnected / request cancelled; make sure the run is closed.
                if not ended:
                    try:
                        root.end(error="cancelled", end_time=datetime.now(timezone.utc))
                        root.patch()
                    except Exception:
                        pass
                    ended = True
                return
            except Exception as e:
                if not ended:
                    try:
                        root.end(error=str(e), end_time=datetime.now(timezone.utc))
                        root.patch()
                    except Exception:
                        pass
                    ended = True
                # Best-effort: tell client we failed, but do not crash the ASGI app (so traces can flush cleanly).
                try:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
                except Exception:
                    pass
                return
            finally:
                # Double-safety: close the run if we somehow exit without ending.
                if not ended:
                    try:
                        root.end(
                            outputs={
                                "answer": full,
                                "sources": sources,
                                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                                "ended_in_finally": True,
                            }
                            ,
                            end_time=datetime.now(timezone.utc),
                        )
                        root.patch()
                    except Exception:
                        pass

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
        try:
            root.end(error=str(e))  # type: ignore[name-defined]
            root.patch()  # type: ignore[name-defined]
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
