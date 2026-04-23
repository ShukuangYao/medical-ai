# 医疗AI辅助诊断系统 - 安装与运行指南

## 系统要求

- Node.js 18+
- Python 3.10+
- Docker & Docker Compose（推荐）
- pnpm 或 npm

## 快速启动（Docker方式 - 推荐）

### 1. 配置环境变量

```bash
cd medical-ai
cp .env.example .env
```

编辑 `.env` 文件，填入必要的配置：
```bash
# LLM 配置（RAG / 普通问答常用：DashScope 的 OpenAI 兼容接口）
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 病历分析（多 Agent）默认使用 DeepSeek；请至少配置密钥（否则 Agent 模式会调不通）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# DEEPSEEK_API_BASE=https://api.deepseek.com

# 病历分析：请求未传 model_provider 时的默认提供方（deepseek | qwen，默认 deepseek）
# AGENT_DEFAULT_LLM_PROVIDER=deepseek

# 按角色覆盖 Agent 使用的模型（可选；不配则 DeepSeek 下统一 deepseek-chat）
# AGENT_MODEL_ANALYST=deepseek-chat
# AGENT_MODEL_PLANNER=deepseek-chat

# LangSmith（可选，用于可观测性/Tracing）
# 说明：Node 后端与 Python 服务都会读取这些环境变量；启用后可在 LangSmith UI 查看全链路 trace。
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=medical-ai

# Neo4j密码
NEO4J_PASSWORD=medical_demo_2025
```

### 2. 启动所有服务

```bash
cd medical-ai
docker compose up --build
# 若本机仍是旧版 Docker CLI，也可使用：docker-compose up --build
```

### 3. 访问系统

- 前端界面：http://localhost:3000（Docker Compose 方式的端口映射）
- Node后端API：http://localhost:3001
- Python AI服务：http://localhost:8000
- Neo4j 浏览器（宿主机映射）：http://localhost:7475（容器内为 7474，映射为 **7475:7474** 以避免与本机 Neo4j Desktop 默认 7474 冲突）
- Neo4j Bolt（宿主机映射）：`bolt://localhost:7688`（映射为 **7688:7687**）

**数据说明**：Compose 中 Neo4j 数据目录使用 **Docker 命名卷 `neo4j-data`**，不再绑定到仓库下的 `./data/neo4j`。若你本地曾有旧的 `./data/neo4j` 数据，需要自行迁移进 volume（启动后是新库）。

## 本地开发模式

### 1. 启动Python AI服务

```bash
cd medical-ai/python-service

# 推荐：使用 Conda + Python 3.11（更容易安装 torch/sentence-transformers 等依赖）
conda env create -f conda-env.yml
conda activate medical-ai-py311

# 安装 Python 依赖（如遇 pip SSL EOF，可临时切换镜像）
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
pip install -r requirements.txt -r requirements-dev.txt

# 启动服务
uvicorn app.main:app --reload --port 8000
```

### 2. 启动Node后端

```bash
cd medical-ai/node-backend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm run dev
```

### 3. 启动前端

```bash
cd medical-ai/frontend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm run dev
```

前端将在 http://localhost:3002 启动（本机 Vite dev server，见 `frontend/vite.config.ts`；`3000` 仅对应 compose 的容器端口映射）

## 功能说明

### 普通问答模式（RAG）
- 基于医疗知识库的问答
- 支持多轮对话
- 提供参考文献溯源

### 病历分析模式（多智能体）
- 智能病历分析
- 鉴别诊断建议
- 展示 AI 推理过程（SSE：`thinking` / `intent` / `agent_step` / `sources` / `result` / `done`）
- **详细分析**（仅病历分析 Tab）：打开后走 `agent_pipeline=full`（分角色多轮 LLM，更细但更慢）；关闭为默认 `fast`（合并步骤，更快）
- **默认大模型**：Python 侧在未传 `model_provider` 时默认 **DeepSeek**，各角色在 DeepSeek 下统一 **`deepseek-chat`**；需要整条 Agent 链路用 Qwen 时，请在请求中传 `model_provider: "qwen"`，或设置 `AGENT_DEFAULT_LLM_PROVIDER=qwen`（详见 `docs/design.md` 与 `README.md`）

### 停止生成（可控取消）

- 前端 Stop 按钮会先调用 **`POST /api/cancel`**（携带 `run_id`），再中断 SSE 连接。
- 取消是**协作式**（cooperative）：服务端会尽快停止后续阶段/并发任务；若某个上游 LLM 请求已在网络中飞行，可能需要等待其返回或超时才能完全结束。

### 反馈闭环（评分/纠错）

- 每条 AI 回答支持 `👍 有用 / 👎 无用 / 纠错（可选文本）`
- 网关接口：`POST http://localhost:3001/api/feedback`
- Python 本地落库：SQLite 表 `chat_feedback`（默认 DB：`medical-ai/python-service/app/data/chat_sessions.db`）

查看最近反馈（SQLite）：

```bash
sqlite3 medical-ai/python-service/app/data/chat_sessions.db \
"SELECT created_at, run_id, session_id, message_id, mode, user_id, rating, comment, corrected_answer
 FROM chat_feedback
 ORDER BY created_at DESC
 LIMIT 50;"
```

## 注意事项

⚠️ **重要提示**
- 本系统仅用于演示和学习目的
- 不可用于实际临床诊断
- 所有数据均为模拟或公开脱敏数据

## 故障排除

### 端口冲突
如果端口被占用，可以修改 `docker-compose.yml` 中的端口映射。

### Python依赖安装失败
确保使用Python 3.10+版本，某些依赖可能需要编译工具。

### 前端TypeScript错误
运行 `pnpm install` 确保所有依赖已安装。

### 反馈接口返回 502（python_feedback_failed）

说明 Node 网关转发到 Python 失败。请检查：

- Python 服务是否在 `http://localhost:8000` 启动（或 `PYTHON_SERVICE_URL` 是否指向正确地址）
- Python 日志是否报错（尤其是 JSON body 校验：需要 `run_id/session_id/message_id`）

### 指标（/metrics）无数据或 404

- Node：`GET http://localhost:3001/metrics`
- Python：`GET http://localhost:8000/metrics`

若 Python 未安装 `prometheus-client`，会降级为空实现（不影响主流程）。

## 技术支持

如遇问题，请查看：
- 项目README.md
- 开发设计文档
- 各服务的日志输出
