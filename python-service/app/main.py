import os

# Hard default: keep all traces in a single LangSmith project.
# Avoid accidental drift to "medicine-ai" from a stale shell/env.
if (not os.getenv("LANGSMITH_PROJECT")) or os.getenv("LANGSMITH_PROJECT") == "medicine-ai":
    os.environ["LANGSMITH_PROJECT"] = "medical-ai"
if not os.getenv("LANGSMITH_TRACING_V2"):
    os.environ["LANGSMITH_TRACING_V2"] = "true"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langsmith.middleware import TracingMiddleware
from app.routers import rag, agent, sessions, cancel

app = FastAPI(title="医疗AI辅助诊断系统 - Python服务")

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

# 注册路由
app.include_router(rag.router, prefix="/api", tags=["RAG"])
app.include_router(agent.router, prefix="/api", tags=["Agent"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(cancel.router, prefix="/api", tags=["Cancel"])

@app.get("/")
async def root():
    return {"message": "医疗AI辅助诊断系统 Python服务运行中"}

@app.get("/health")
async def health():
    return {"status": "ok"}
