"""RAG路由 - 支持SSE流式输出"""
import asyncio
import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.core.singletons import rag_engine, ensure_rag_initialized
from app.core.sse_envelope import sse_context_from_request, sse_envelope, sse_data_line
from app.core.run_cancel import clear_run

from langsmith import RunTree
from langsmith.run_helpers import get_current_run_tree, tracing_context

router = APIRouter()

DEFAULT_USER_ID = "anonymous"

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

        sse_ctx_ns = sse_context_from_request(request)

        # For non-stream calls, we also persist to session storage.
        # We reuse the stream pipeline (persist=True) but consume events to build the final response.
        answer_parts = []
        sources = []
        stream_errors = []
        with tracing_context(parent=root):
            async for evt in rag_engine.query_stream(
                question,
                session_id,
                chat_history,
                use_graph=bool(use_graph),
                user_id=user_id,
                model_provider=model_provider,
                model_name=model_name,
                cancel_run_id=(sse_ctx_ns.run_id or None),
            ):
                et = evt.get("type")
                if et == "token":
                    answer_parts.append(evt.get("content") or "")
                elif et == "sources":
                    sources = evt.get("content") or []
                elif et == "error":
                    # ToolError / pipeline errors include `code`, `retriable`, optional `detail` (see ToolError.to_event_fields).
                    stream_errors.append({k: v for k, v in evt.items() if k != "type"})

        answer = "".join(answer_parts)
        root.end(
            outputs={
                "answer": answer,
                "sources": sources,
                "errors": stream_errors,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            },
            end_time=datetime.now(timezone.utc),
        )
        root.post()
        return {"answer": answer, "sources": sources, "errors": stream_errors}
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
        sse_ctx = sse_context_from_request(request)

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
            "request_id": sse_ctx.request_id or None,
            "run_id": sse_ctx.run_id or None,
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
            rid = (sse_ctx.run_id or "").strip()
            try:
                with tracing_context(parent=root):
                    # 先产出一个事件，确保客户端尽快收到响应头并开始渲染（避免初始化耗时导致“无输出”假象）
                    meta_evt = sse_envelope(
                        {"type": "thinking", "content": "🚀 请求已接收，正在准备检索与生成..."},
                        ctx=sse_ctx,
                        mutate=False,
                    )
                    yield sse_data_line(meta_evt)
                    uid = data.get("user_id") or data.get("userId") or DEFAULT_USER_ID
                    async for chunk in rag_engine.query_stream(
                        question,
                        session_id,
                        chat_history,
                        use_graph=bool(use_graph),
                        model_provider=model_provider,
                        model_name=model_name,
                        user_id=uid,
                        cancel_run_id=rid or None,
                    ):
                        if chunk.get("type") == "token":
                            full += chunk.get("content") or ""
                        elif chunk.get("type") == "sources":
                            sources = chunk.get("content") or []
                        evt = sse_envelope(chunk, ctx=sse_ctx, mutate=True)
                        yield sse_data_line(evt)
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
                        root.end(error="cancelled", metadata={"cancelled": True}, end_time=datetime.now(timezone.utc))
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
                    err_evt = sse_envelope({"type": "error", "content": str(e)}, ctx=sse_ctx, mutate=False)
                    yield sse_data_line(err_evt)
                except Exception:
                    pass
                return
            finally:
                if rid:
                    clear_run(rid)
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
