from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import rag, agent, sessions

app = FastAPI(title="医疗AI辅助诊断系统 - Python服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(rag.router, prefix="/api", tags=["RAG"])
app.include_router(agent.router, prefix="/api", tags=["Agent"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])

@app.get("/")
async def root():
    return {"message": "医疗AI辅助诊断系统 Python服务运行中"}

@app.get("/health")
async def health():
    return {"status": "ok"}
