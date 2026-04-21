import os

# Hard default: keep all traces in a single LangSmith project.
# Avoid accidental drift to "medicine-ai" from a stale shell/env.
if (not os.getenv("LANGSMITH_PROJECT")) or os.getenv("LANGSMITH_PROJECT") == "medicine-ai":
    os.environ["LANGSMITH_PROJECT"] = "medical-ai"
if not os.getenv("LANGSMITH_TRACING_V2"):
    os.environ["LANGSMITH_TRACING_V2"] = "true"

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from langsmith.middleware import TracingMiddleware
from app.routers import rag, agent, sessions, cancel
import time
import logging
from app.core import metrics as prom_metrics
from app.core.json_logging import configure_json_logging

app = FastAPI(title="医疗AI辅助诊断系统 - Python服务")

# Phase 3: stable JSON logs (ensures `extra={...}` fields are always emitted)
configure_json_logging()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangSmith distributed tracing (reads `langsmith-trace` / `baggage` headers)
app.add_middleware(TracingMiddleware)

# Phase 3: request_id/run_id structured request logs (minimal SLO fields).
logger = logging.getLogger("medical-ai.http")

@app.middleware("http")
async def request_logging_middleware(request, call_next):
    t0 = time.perf_counter()
    request_id = (request.headers.get("x-request-id") or "").strip()
    run_id = (request.headers.get("x-run-id") or "").strip()
    try:
        response = await call_next(request)
        return response
    finally:
        try:
            status_code = getattr(locals().get("response", None), "status_code", None)
        except Exception:
            status_code = None
        # Metrics (Prometheus) - best effort, no hard dependency.
        try:
            if prom_metrics.http_requests_total is not None:
                prom_metrics.http_requests_total.labels(
                    service="python",
                    method=request.method,
                    path=request.url.path,
                    status_code=str(status_code or 0),
                ).inc()
            if prom_metrics.http_request_duration_ms is not None:
                prom_metrics.http_request_duration_ms.labels(
                    service="python",
                    method=request.method,
                    path=request.url.path,
                    status_code=str(status_code or 0),
                ).observe((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass
        logger.info(
            "http_request",
            extra={
                "request_id": request_id or None,
                "run_id": run_id or None,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": int((time.perf_counter() - t0) * 1000),
            },
        )

# 注册路由
app.include_router(rag.router, prefix="/api", tags=["RAG"])
app.include_router(agent.router, prefix="/api", tags=["Agent"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(cancel.router, prefix="/api", tags=["Cancel"])

@app.get("/metrics")
async def metrics() -> Response:
    data = prom_metrics.render_latest()
    return Response(content=data, media_type=prom_metrics.content_type())

@app.get("/")
async def root():
    return {"message": "医疗AI辅助诊断系统 Python服务运行中"}

@app.get("/health")
async def health():
    return {"status": "ok"}
