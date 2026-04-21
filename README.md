# 医疗 AI 辅助诊断系统

一个包含 **RAG 医疗问答** 与 **多智能体病历分析** 的端到端演示系统，采用三层架构：

- **Frontend**：React + Ant Design（Vite）
- **API 网关**：Node.js（Fastify）
- **AI 服务**：Python（FastAPI），内含 RAG 引擎与 Agent 编排器

> 本仓库中还有一个历史子项目 `RAGQnASystem/`，本文档聚焦 `medical-ai/` 这一套可运行的前后端联调 Demo。

## 快速开始（本机开发）

### 端口约定

- **Frontend**：`http://localhost:3002`（Vite dev server，见 `frontend/vite.config.ts`）
- **Node Backend**：`http://localhost:3001`
  - 健康检查：`GET /health`
  - 前端通过 Vite proxy 将 `/api/`* 转发到 Node
- **Python Service**：`http://localhost:8000`
  - 健康检查：`GET /health`

### 环境要求

- Node.js（建议 18+）
- pnpm
- Python（建议 3.10/3.11；与深度学习依赖兼容更好）
-（可选）Docker / Docker Compose：用于一键拉起 Milvus/ES/Neo4j/Redis 等依赖

### 方式 A：用 Docker Compose 拉起全部依赖（推荐）

1. 进入目录并准备环境变量：

```bash
cd medical-ai
# 云服务器 / 纯 compose 部署（容器互联用 service 名）：
cp .env.docker.example .env
#
# 本机分别启动（连本机 Desktop/本机依赖）：
# cp .env.example .env
```

1. 编辑 `.env`，至少填好：

- `DASHSCOPE_API_KEY`（**RAG / 普通问答**等仍可用 DashScope 的 OpenAI 兼容接口）
- `DEEPSEEK_API_KEY`（**病历分析（多 Agent）**默认走 DeepSeek；未配置会导致 Agent 调用失败）
- `NEO4J_PASSWORD`（供 compose 内的 neo4j 使用）

（可选）启用 LangSmith Tracing（全链路可观测性）：在 `.env` 中加入

- `LANGSMITH_TRACING_V2=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=medical-ai`

1. 启动依赖与服务：

```bash
docker compose up -d
```

#### 云服务器部署注意事项

- **容器内不要用 `127.0.0.1` 连接依赖**：应使用 compose service 名（如 `neo4j`/`redis`/`elasticsearch`/`milvus-standalone`）。
- **尽量不要把数据库端口暴露到公网**：一般只需要暴露前端/网关端口；数据库端口仅在内网/容器网络可达即可。
- **数据持久化**：ES/Milvus/etcd/minio/redis 等在 compose 中已使用 volume；**Neo4j 在 compose 中使用命名卷 `neo4j-data`**（避免 macOS 下宿主机 bind mount 带来的文件锁问题）。云上可将 Docker volume 数据目录放到挂载的云盘；若你曾使用旧的 `./data/neo4j` 绑定挂载，需要自行迁移数据到 volume（compose 不会自动迁移）。

1. 打开前端：

- `http://localhost:3000`（注意：compose 中前端映射是 `3000:3000`，与本机 `pnpm dev` 的 `3002` 不同）

> 说明：`docker-compose.yml` 中前端容器端口为 3000；本机 `pnpm dev` 时是 3002。

### 方式 B：本机分别启动（更适合开发调试）

#### 1) 启动 Python Service（8000）

```bash
cd medical-ai/python-service
conda env create -f conda-env.yml
conda activate medical-ai-py311

# 如遇 pip SSL EOF，可临时切换镜像
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

#### 2) 启动 Node Backend（3001）

```bash
cd medical-ai/node-backend
pnpm install
pnpm dev
```

Node 默认会把请求转发到 `PYTHON_SERVICE_URL`（默认 `http://localhost:8000`，见 `node-backend/src/config.ts`）。

#### 3) 启动 Frontend（3002）

```bash
cd medical-ai/frontend
pnpm install
pnpm dev
```

打开：`http://localhost:3002`。

## 使用说明

系统 UI 顶部有两个 Tab（路由页）：

- **普通问答**（RAG）
- **病历分析**（Agent，多智能体协作）

### 普通问答（RAG）

- **输入**：医疗相关问题（自然语言）
- **输出**：回答 +（可选）参考文献 sources
- **流式输出**：前端支持 SSE 流式显示（“流式输出”开关）
- **图谱检索**：前端可切换“图谱检索”开关（后端会将 `useGraph` 传给 Python RAG）

### 病历分析（Agent）

- **输入**：病历文本/症状描述（文字）
- **输出**：结构化病历分析报告（`report`，严格 JSON）+ 简短总结（`answer`/`report.summary`）+（可选）推理过程（`trace`）
- **流式输出**：支持 SSE 流式（`thinking / intent / agent_step / sources / result / done`），前端会实时更新“思考过程”，并在 `result` 到达后渲染结构化卡片。
- **默认大模型**：请求未传 `model_provider` 时，Python 侧默认 **`deepseek`**；在 DeepSeek 下各 Agent 角色统一使用 **`deepseek-chat`**（可通过 `AGENT_MODEL_*` 环境变量按角色覆盖）。若需整条链路仍用 Qwen，请在请求中显式传 `model_provider: "qwen"`，或设置环境变量 `AGENT_DEFAULT_LLM_PROVIDER=qwen`。

#### 快速模式 vs 详细模式（多 Agent 仍在）

为兼顾 **耗时** 与 **分析粒度**，病历分析支持两种流水线（默认 **快速**）：

| 模式 | UI | API（经 Node 网关） | Python 字段 | 说明 |
|------|----|---------------------|---------------|------|
| **快速**（默认） | 「详细分析」关闭 | `agentPipeline: "fast"` 或不传 | `agent_pipeline: "fast"` | 合并部分 LLM 步骤，响应更快；`agent_step` 仍会推送以便展示过程 |
| **详细**（原多步） | 「详细分析」打开 | `agentPipeline: "full"` | `agent_pipeline: "full"` | 恢复分角色多轮 LLM + Coordinator 汇总，**更细但更慢** |

说明：**多 Agent 代码与角色仍在**；`fast` 是编排上的“少轮次合并”，不是删除 Agent。

#### 病历分析输出 JSON（`report: AgentReport`）

`report` 的核心字段如下（前端按模块卡片渲染）：

- `validated_record`: 规范化与纠错（`normalized_text / missing_fields / contradictions / corrections`）
- `intent`: 意图与问题消解（`raw_question / resolved_question / intent_type / confidence`）
- `structured_case`: 病历结构化抽取（`chief_complaint / symptoms / duration / vitals / history / medications / allergies / tests`）
- `symptom_analysis`: 病症分析（`key_findings / possible_conditions`）
- `triage`: 紧急程度（`severity_level: emergency|urgent|routine / red_flags / why`）
- `department`: 科室推荐（`recommended / alternatives / reason`）
- `next_steps`: 下一步举措（`immediate_actions / recommended_tests / when_to_seek_care`）
- `treatment_safety`: 安全审阅（`medication_considerations / contraindications / cautions`）
- `summary`: 1–3 句总结（含免责声明）
- `trace`: 多 agent 过程记录（用于前端折叠面板展示/调试）

## 总体架构

```mermaid
flowchart TB
  user[UserBrowser] --> frontend[Frontend_Vite_React]
  frontend -->|"HTTP /api/* (proxy)"| node[Node_Fastify_3001]
  node -->|"HTTP" | python[Python_FastAPI_8000]

  python --> rag[RAG_Engine]
  python --> agent[Agent_Orchestrator]

  rag --> es[Elasticsearch_9200]
  rag --> milvus[Milvus_19530]
  rag --> neo4j[Neo4j]
  python --> redis[Redis_6379]
```



## 关键流程图

### 1) RAG：SSE 流式问答

对应代码：

- 前端：`frontend/src/services/api.ts` → `POST /api/chat/stream`
- Node：`node-backend/src/routes/chat.ts` → 转发到 Python `POST /api/rag/stream`
- Python：`python-service/app/routers/rag.py` → `rag_engine.query_stream(...)`

```mermaid
sequenceDiagram
  participant B as Browser
  participant F as Frontend
  participant N as Node(3001)
  participant P as Python(8000)
  participant R as RAGEngine

  B->>F: 输入问题
  F->>N: POST /api/chat/stream {mode:rag,...}
  N->>P: POST /api/rag/stream
  P->>R: query_stream(question,session_id,...)
  R-->>P: SSE data: {type:token,...}
  P-->>N: SSE转发
  N-->>F: SSE转发
  F-->>B: 打字机渲染token
```



### 2) Agent：SSE 流式病历分析（多 Agent）

对应代码：

- Node：`node-backend/src/routes/chat.ts` 中 `mode === 'agent'` → Python `POST /api/agent/stream`（流式）或 `POST /api/agent`（非流式）；可选携带 `agentPipeline` → `agent_pipeline`
- Python：`python-service/app/routers/agent.py` 汇总 `diagnose_stream`，通过 SSE 持续输出事件流（`thinking / intent / agent_step / sources / result / done`）

```mermaid
sequenceDiagram
  participant B as Browser
  participant F as Frontend
  participant N as Node(3001)
  participant P as Python(8000)
  participant A as AgentOrchestrator

  B->>F: 输入病历文本
  F->>N: POST /api/chat/stream {mode:agent,message,...}
  N->>P: POST /api/agent/stream
  P->>A: diagnose_stream(message,session_id,model_provider,model_name,agent_pipeline)
  A-->>P: SSE data: {type:thinking/intent/agent_step/sources/result/done}
  P-->>N: SSE转发
  N-->>F: SSE转发
  F-->>B: 实时展示过程 + 最终结构化卡片(report)
```



## 各模块功能图

### Frontend（`medical-ai/frontend`）

```mermaid
flowchart LR
  pages[pages] --> home[Home_普通问答]
  pages --> analysis[Analysis_病历分析]
  components[components] --> chatBox[ChatBox]
  components --> messageList[MessageList]
  store[store] --> chatStore[chatStore_zustand]
  services[services] --> api[api_chatAPI]
  services --> sse[sseClient_createSSEConnection]

  home --> chatBox
  analysis --> chatBox
  chatBox --> chatStore
  chatBox --> api
  api --> sse
  chatBox --> messageList
```



### Node Backend（`medical-ai/node-backend`）

```mermaid
flowchart LR
  server[server.ts] --> routes[routes/*]
  routes --> chat[chat.ts_/api/chat_/api/chat/stream]
  routes --> health[health.ts_/health]
  routes --> cfg[config.ts_/api/config]
  chat --> pythonSvc[PythonService_URL]
```



### Python Service（`medical-ai/python-service`）

```mermaid
flowchart LR
  main[app/main.py] --> ragRouter[routers/rag.py]
  main --> agentRouter[routers/agent.py]

  ragRouter --> ragEngine[core/rag_engine.py_LocalDocQA]
  agentRouter --> orchestrator[core/agent_orchestrator.py_MedicalAgentOrchestrator]

  ragEngine --> retrievers[ES/Milvus/Neo4j]
  orchestrator --> tools[工具:检索/图谱/总结]
  orchestrator --> ragEngine
```



## 常见问题（FAQ）

### 1) 日志提示 Milvus 连接失败，但仍能回答？

RAG 可能会降级走 **Elasticsearch** 或其它检索来源；是否可用取决于你本机/容器中哪些依赖服务已启动。

### 2) 本机 dev 端口与 docker compose 端口不一致

- 本机前端：Vite `3002`（`frontend/vite.config.ts`）
- compose 前端：映射 `3000:3000`（`docker-compose.yml`）

如果你更偏向开发调试，建议使用“方式 B”分别启动；如果只想一键跑通，用“方式 A”即可。

## 开发者调试

- Node 健康检查：`GET http://localhost:3001/health`
- Python 健康检查：`GET http://localhost:8000/health`
- Node 配置回显：`GET http://localhost:3001/api/config`

### 配置与环境变量

- **Node → Python 转发**：`PYTHON_SERVICE_URL`（默认 `http://localhost:8000`）
- **LLM（OpenAI 兼容接口）**：Python 侧通过以下环境变量连接（见 `python-service/app/config.py`）：
  - `DASHSCOPE_API_KEY`: OpenAI 兼容接口的 `api_key`
  - `LLM_API_BASE`: OpenAI 兼容接口的 `base_url`（如 DashScope 兼容地址、OneAPI 地址、本地 vLLM/ollama 的兼容地址）
  - `LLM_MODEL`: 模型名（默认 `qwen3.5-flash`；主要用于 **RAG** 等 Qwen 路径）
  - `DEEPSEEK_API_KEY` / `DEEPSEEK_API_BASE`: **病历分析（Agent）** 默认提供方
  - `AGENT_DEFAULT_LLM_PROVIDER`: 病历分析在请求未带 `model_provider` 时的默认提供方，可选 `deepseek`（默认）或 `qwen`
  - `AGENT_MODEL_VALIDATOR`、`AGENT_MODEL_EXTRACTOR`、`AGENT_MODEL_ANALYST`、`AGENT_MODEL_TRIAGE_DEPT`、`AGENT_MODEL_PLANNER`、`AGENT_MODEL_COORDINATOR` 等：按角色覆盖模型 id（仍优先于默认的 `deepseek-chat`；完整列表见 `docs/design.md`）
- **LangSmith**：启用后，Python 侧 LLM 子 run 的 **name** 会包含模型 id（如 `openai_chat_completions:deepseek-chat`、`RecordValidator:deepseek:deepseek-chat`），便于在 Traces 列表中区分；详见 `docs/design.md`。
- **依赖服务（Python 侧）**：Milvus / Elasticsearch / Neo4j / Redis 的地址通常通过环境变量读取；参考：
  - `medical-ai/.env.example`
  - `medical-ai/docker-compose.yml`

## 变更记录（最近一次）

- **Agent 流式链路对齐**：Node 的 `POST /api/chat/stream` 在 `mode=agent` 时已转发到 Python `POST /api/agent/stream`（SSE）。
- **病历分析严格 JSON**：Python 输出 `report: AgentReport`（含意图、病历校验纠错、结构化抽取、紧急程度、科室、下一步举措、安全审阅等模块）。
- **前端卡片展示**：病历分析结果会以卡片展示（紧急程度/科室/下一步举措），并保留过程日志与 sources。
- **病历分析快速/详细**：支持 `agent_pipeline`（`fast` | `full`）；前端病历分析 Tab 提供「详细分析」开关；非流式请求前端超时已放宽，避免长分析被误判失败。
- **Neo4j（Compose）**：数据使用命名卷 `neo4j-data`；宿主机访问 HTTP `7475`、Bolt `7688`（避免与本机 Neo4j Desktop 默认端口冲突）。
- **病历分析默认 LLM**：未传 `model_provider` 时默认 **DeepSeek**；各角色在 DeepSeek 下统一 **`deepseek-chat`**（`AGENT_DEFAULT_LLM_PROVIDER` / `AGENT_MODEL_*` 可覆盖）；意图识别与本轮 `model_provider` 对齐。
- **LangSmith**：Agent / RAG 的 LLM 子 span 命名包含 **provider + model**，便于在 Traces 中区分（Node 侧根 trace 亦可带动态 name/metadata）。

## 会话（Sessions）

系统已支持会话管理与持久化（用于历史恢复与多轮对话连续性）：

- **RAG / Agent 两套独立会话列表**：两种模式分别维护会话，互不干扰
- **会话重命名 / 归档**：便于整理长期对话
- **SQLite 会话持久化与历史恢复**：刷新页面或重启服务后仍可恢复历史
- **默认自动创建会话**：首次进入对应模式会自动生成一个会话，无需手动新建

涉及文件（节选）：

- `node-backend/src/routes/chat.ts`
- `python-service/app/core/agent_orchestrator.py`
- `python-service/app/models/request.py`
- `python-service/app/models/response.py`
- `python-service/app/routers/agent.py`
- `frontend/src/services/api.ts`
- `frontend/src/components/ChatBox.tsx`
- `frontend/src/components/MessageList.tsx`
- `frontend/src/types/shared.ts`

