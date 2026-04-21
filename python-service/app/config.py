import os
from pathlib import Path

from dotenv import load_dotenv

# 从仓库根目录加载 .env（避免在 python-service/ 下运行脚本时读不到上层配置）
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
load_dotenv()


def _effective_neo4j_uri(uri: str) -> str:
    """Neo4j Aura 控制台给的 neo4j+s:// 会走路由协议，在部分网络下路由表拉取失败；对 *.databases.neo4j.io 默认改为 bolt+s:// 直连。
    需要集群路由时设 NEO4J_USE_ROUTING=1；任意主机强制直连可设 NEO4J_USE_DIRECT_BOLT=1。
    """
    if os.getenv("NEO4J_USE_ROUTING", "").lower() in ("1", "true", "yes"):
        return uri
    force_direct = os.getenv("NEO4J_USE_DIRECT_BOLT", "").lower() in ("1", "true", "yes")
    aura = ".databases.neo4j.io" in uri
    if not force_direct and not aura:
        return uri
    if uri.startswith("neo4j+s://"):
        return "bolt+s://" + uri[len("neo4j+s://") :]
    if uri.startswith("neo4j+ssc://"):
        return "bolt+ssc://" + uri[len("neo4j+ssc://") :]
    if uri.startswith("neo4j://"):
        return "bolt://" + uri[len("neo4j://") :]
    return uri


class Settings:
    # LLM配置（Qwen Turbo via DashScope）
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # DeepSeek（OpenAI兼容接口）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    # `/agent` 多智能体：请求未带 model_provider 时的默认（设为 qwen 可恢复旧行为）
    _raw_agent_llm_provider = os.getenv("AGENT_DEFAULT_LLM_PROVIDER", "deepseek").strip().lower()
    AGENT_DEFAULT_LLM_PROVIDER: str = (
        _raw_agent_llm_provider if _raw_agent_llm_provider in ("qwen", "deepseek") else "deepseek"
    )

    # Session storage (SQLite)
    CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", str((_ROOT / "python-service" / "app" / "data" / "chat_sessions.db").resolve()))

    # Milvus向量数据库配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "medical_knowledge")
    # Fail fast when Milvus is unavailable; pymilvus may otherwise retry for minutes.
    MILVUS_SEARCH_TIMEOUT_S = float(os.getenv("MILVUS_SEARCH_TIMEOUT_S", "2.5"))

    # ElasticSearch配置
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", "9200"))
    ES_INDEX = os.getenv("ES_INDEX", "medical_knowledge")

    # Neo4j配置（Aura/单机默认库名为 neo4j；自定义库须先在服务器上 CREATE DATABASE）
    NEO4J_URI = _effective_neo4j_uri(os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # Redis配置
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # 嵌入模型配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh")
    EMBEDDING_DIMENSION = 1024  # bge-large-zh 输出维度

    # 重排序模型配置
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    RERANKER_SCORE_THRESHOLD = 0.28  # 绝对分数阈值
    RERANKER_RELATIVE_THRESHOLD = 0.5  # 相对分数阈值（与最高分差50%）

    # RAG检索配置
    VECTOR_TOP_K = 6  # 向量检索返回数量（降低默认值以减少 rerank 开销与端到端延迟）
    ES_TOP_K = 6  # ES检索返回数量（降低默认值以减少 rerank 开销与端到端延迟）
    RERANK_TOP_K = 5  # 重排序后保留数量
    MMR_LAMBDA = 0.7  # MMR多样性参数（0.7相关 + 0.3多样）
    MMR_FETCH_K = 10  # MMR初始候选集大小（降低默认值以减少 rerank 开销与端到端延迟）

    # RAG rerank optimizations
    # - prefilter candidates before running the expensive cross-encoder reranker
    # - cache rerank results for a short TTL to dedupe repeat queries
    RAG_RERANK_CANDIDATES_MAX = int(os.getenv("RAG_RERANK_CANDIDATES_MAX", "16"))
    RAG_RERANK_CACHE_TTL_S = float(os.getenv("RAG_RERANK_CACHE_TTL_S", "120"))
    RAG_REWRITE_CACHE_TTL_S = float(os.getenv("RAG_REWRITE_CACHE_TTL_S", "120"))

    # Agent `full` 管线：单步 LLM 墙钟上限（超时返回空 JSON，下游走 fallback，避免整请求卡死）
    AGENT_FULL_LLM_STEP_TIMEOUT_S = float(os.getenv("AGENT_FULL_LLM_STEP_TIMEOUT_S", "150"))

    # Agent model overrides (optional): allow per-agent provider/model for "model layering".
    # provider: "qwen" | "deepseek" | "" (empty means inherit request/provider default)
    # model: OpenAI-compatible model id (string); empty means inherit request/model_name default
    AGENT_PROVIDER_VALIDATOR = os.getenv("AGENT_PROVIDER_VALIDATOR", "")
    AGENT_MODEL_VALIDATOR = os.getenv("AGENT_MODEL_VALIDATOR", "")
    AGENT_PROVIDER_EXTRACTOR = os.getenv("AGENT_PROVIDER_EXTRACTOR", "")
    AGENT_MODEL_EXTRACTOR = os.getenv("AGENT_MODEL_EXTRACTOR", "")
    AGENT_PROVIDER_ANALYST = os.getenv("AGENT_PROVIDER_ANALYST", "")
    AGENT_MODEL_ANALYST = os.getenv("AGENT_MODEL_ANALYST", "")
    AGENT_PROVIDER_TRIAGE_DEPT = os.getenv("AGENT_PROVIDER_TRIAGE_DEPT", "")
    AGENT_MODEL_TRIAGE_DEPT = os.getenv("AGENT_MODEL_TRIAGE_DEPT", "")
    AGENT_PROVIDER_PLANNER = os.getenv("AGENT_PROVIDER_PLANNER", "")
    AGENT_MODEL_PLANNER = os.getenv("AGENT_MODEL_PLANNER", "")
    AGENT_PROVIDER_COORDINATOR = os.getenv("AGENT_PROVIDER_COORDINATOR", "")
    AGENT_MODEL_COORDINATOR = os.getenv("AGENT_MODEL_COORDINATOR", "")

    # Cache prepared record outputs (validator/extractor) to skip repeated work.
    AGENT_PREPARE_CACHE_TTL_S = float(os.getenv("AGENT_PREPARE_CACHE_TTL_S", "3600"))

    # Agent evidence retrieval wall-clock budgets
    # - full: allow more time, but keep bounded to avoid huge tail latency
    # - fast: keep small to preserve responsiveness
    AGENT_EVIDENCE_TIMEOUT_FULL_S = float(os.getenv("AGENT_EVIDENCE_TIMEOUT_FULL_S", "25"))
    AGENT_EVIDENCE_TIMEOUT_FAST_S = float(os.getenv("AGENT_EVIDENCE_TIMEOUT_FAST_S", "10"))

    # Limit long medical record text in parallel sub-prompts to reduce latency/cost.
    AGENT_PARALLEL_PROMPT_MAX_CHARS = int(os.getenv("AGENT_PARALLEL_PROMPT_MAX_CHARS", "1200"))
    # Full pipeline: cap raw medical record text fed into early validator/extractor prompts.
    # Keep high enough for fidelity, but prevents pathological long prompts from dominating latency.
    AGENT_FULL_RECORD_MAX_CHARS = int(os.getenv("AGENT_FULL_RECORD_MAX_CHARS", "1800"))

    # Token管理
    MAX_CONTEXT_TOKENS = 3000  # 上下文最大Token数
    # 生成最大 Token 数（越大越不易截断，但延迟/费用更高；不同 provider 也可能有自身上限）
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))
    MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "800"))  # 历史对话窗口最大Token数（估算）

    # 文档切片配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


settings = Settings()
