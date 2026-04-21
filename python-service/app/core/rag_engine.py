"""LocalDocQA - 系统核心控制器，协调各组件完成完整RAG流程

流程：
1. 问题改写（多轮对话时）
2. 混合检索（向量MMR + ES关键词）
3. 重排序（CrossEncoder + 分数过滤 + 容错）
4. 上下文构建（Token管理）
5. 回答生成（Qwen Turbo流式）
6. 结果处理
"""
import os
import json
import hashlib
import time
import tiktoken
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime, timezone
from app.config import settings
from app.core.embeddings import BGEEmbeddings
from app.core.vector_store import VectorStoreMilvusClient
from app.core.es_store import StoreElasticSearchClient
from app.core.reranker import BGEReranker
from app.core.llm_client import OpenAILLM
from app.core.rewrite_chain import RewriteQuestionChain
from app.core.retriever import ParentRetriever
from app.core.graph_store import GraphStoreNeo4jClient
from app.core.graph_querier import GraphQuerier
from app.core.intent_classifier_v2 import HybridIntentClassifier
from app.core.intent_router import IntentRouter
from app.core.memory_manager import MemoryManager
from app.core.context_resolver import ContextResolver
from app.core.session_store import SessionStore
from app.core.context_builder import ContextBuilder
from app.core.telemetry import Telemetry
from app.core.ls_timing import now_utc, perf_ms_since, span_times
from app.core.tools.base import ToolContext, ToolError
from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolRegistry
from app.core.tools.impl import GraphQueryTool, HybridRetrieveTool, RerankTool, RewriteQuestionTool
from app.core.run_cancel import is_cancelled as run_cancelled

from langsmith import RunTree
from langsmith.run_helpers import get_current_run_tree, tracing_context


class LocalDocQA:
    """系统核心控制器，协调各组件完成RAG流程"""

    def __init__(self):
        self.initialized = False
        # 各组件
        self.embeddings: Optional[BGEEmbeddings] = None
        self.vector_store: Optional[VectorStoreMilvusClient] = None
        self.es_store: Optional[StoreElasticSearchClient] = None
        self.reranker: Optional[BGEReranker] = None
        self.llm: Optional[OpenAILLM] = None
        self.rewrite_chain: Optional[RewriteQuestionChain] = None
        self.retriever: Optional[ParentRetriever] = None
        self.graph_store: Optional[GraphStoreNeo4jClient] = None
        self.graph_querier: Optional[GraphQuerier] = None
        self.intent_classifier: Optional[HybridIntentClassifier] = None
        self.intent_router: Optional[IntentRouter] = None
        self.tools: Optional[ToolExecutor] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.context_resolver: Optional[ContextResolver] = None
        self.session_store: Optional[SessionStore] = None
        # Token编码器
        self.tokenizer = None

    async def initialize(self):
        """初始化所有组件"""
        print("=" * 50)
        print("正在初始化 LocalDocQA RAG引擎...")
        print("=" * 50)

        # 1. 初始化嵌入模型
        self.embeddings = BGEEmbeddings()
        self.embeddings.load()

        # 2. 初始化Milvus向量库
        self.vector_store = VectorStoreMilvusClient()
        try:
            self.vector_store.connect()
            self.vector_store.create_collection(self.embeddings.dimension)
        except Exception as e:
            print(f"Milvus连接失败（将在检索时重试）: {e}")

        # 3. 初始化ElasticSearch
        self.es_store = StoreElasticSearchClient()
        try:
            self.es_store.connect()
            self.es_store.create_index()
        except Exception as e:
            print(f"ES连接失败（将在检索时重试）: {e}")

        # 4. 初始化重排序模型
        self.reranker = BGEReranker()
        try:
            self.reranker.load()
        except Exception as e:
            print(f"重排序模型加载失败（将使用容错模式）: {e}")

        # 5. 初始化LLM客户端（默认 provider/model，支持每次请求覆盖）
        self.llm = OpenAILLM.from_provider(provider="qwen", model_name=settings.LLM_MODEL)

        # 6. 初始化问题改写链
        self.rewrite_chain = RewriteQuestionChain(self.llm)

        # 7. 初始化Neo4j图谱（可选）
        try:
            self.graph_store = GraphStoreNeo4jClient()
            self.graph_store.connect()
            self.graph_querier = GraphQuerier(self.graph_store)
            print("Neo4j图谱已连接")
        except Exception as e:
            print(f"Neo4j连接失败（将跳过图谱检索）: {e}")
            self.graph_querier = None

        # 8. 初始化统一检索器
        self.retriever = ParentRetriever(
            vector_store=self.vector_store,
            es_store=self.es_store,
            embeddings=self.embeddings,
        )

        # 9. 初始化意图识别器和路由器
        self.intent_classifier = HybridIntentClassifier(llm=self.llm)
        # Tool registry/executor: unify validation/error codes/trace spans for retrieval sub-steps.
        reg = ToolRegistry()
        reg.register(HybridRetrieveTool(self.retriever))
        reg.register(RerankTool(self.reranker))
        reg.register(GraphQueryTool(self.graph_querier))
        reg.register(RewriteQuestionTool(self.rewrite_chain))
        self.tools = ToolExecutor(registry=reg)
        self.intent_router = IntentRouter(
            graph_querier=self.graph_querier,
            retriever=self.retriever,
            reranker=self.reranker,
            llm=self.llm,
            tool_executor=self.tools,
        )

        # 10. 初始化记忆管理器
        try:
            self.memory_manager = MemoryManager()
            print("记忆管理器已初始化")
        except Exception as e:
            print(f"记忆管理器初始化失败: {e}")
            self.memory_manager = None

        # 10.5 初始化会话存储（SQLite）
        try:
            from app.config import settings as _settings
            self.session_store = SessionStore(_settings.CHAT_DB_PATH)
        except Exception as e:
            print(f"会话存储初始化失败: {e}")
            self.session_store = None

        # 11. 初始化上下文消解器
        self.context_resolver = ContextResolver(llm=self.llm)
        print("上下文消解器已初始化")

        # 12. 初始化Token编码器
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        self.initialized = True
        print("=" * 50)
        print("LocalDocQA RAG引擎初始化完成!")
        print("=" * 50)

    def _count_tokens(self, text: str) -> int:
        """计算文本的Token数量"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # 粗略估算：中文约1.5字符/token
        return int(len(text) * 0.7)

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """按 token 数截断文本（tiktoken 精确控制）"""
        if max_tokens <= 0:
            return ""

        # 容错：如果 tokenizer 初始化失败/为空，就用字符长度估算截断
        # （避免把上下文截成空，导致 prompt 丢失参考资料）
        if not self.tokenizer:
            est_tokens = max(1, int(len(text) * 0.7))  # 与 _count_tokens 的估算保持一致
            if est_tokens <= max_tokens:
                return text
            keep_chars = int(len(text) * (max_tokens / est_tokens))
            keep_chars = max(0, keep_chars)
            return text[:keep_chars] + "..."

        token_ids = self.tokenizer.encode(text)
        if len(token_ids) <= max_tokens:
            return text

        suffix = "..."
        suffix_tokens = len(self.tokenizer.encode(suffix))
        keep = max(0, max_tokens - suffix_tokens)
        truncated_ids = token_ids[:keep]
        return self.tokenizer.decode(truncated_ids) + suffix

    def _build_context(self, documents: List[Dict]) -> str:
        """智能构建上下文，管理Token限制"""
        context_parts = []
        total_tokens = 0
        max_tokens = settings.MAX_CONTEXT_TOKENS

        for i, doc in enumerate(documents):
            doc_text = f"[参考{i + 1}] {doc.get('title', '未知来源')}\n{doc['text']}"
            doc_tokens = self._count_tokens(doc_text)
            # 如果加上当前文档的Token数量超过最大Token限制，则截断当前文档
            if total_tokens + doc_tokens > max_tokens:
                # 截断当前文档以适应Token限制
                remaining = max_tokens - total_tokens
                # 如果剩余Token数量大于50，则截断当前文档
                if remaining > 50:  # 至少保留50个token
                    # 精确按 token 截断（比按字符更准确）
                    context_parts.append(self._truncate_by_tokens(doc_text, remaining))
                break

            context_parts.append(doc_text)
            total_tokens += doc_tokens

        return "\n\n".join(context_parts)

    @staticmethod
    def _normalize_page(value) -> Optional[int]:
        """仅返回有效页码（>=1）；0/空/非法值统一视为无页码。"""
        if value is None:
            return None
        try:
            p = int(value)
        except (TypeError, ValueError):
            return None
        return p if p >= 1 else None

    async def _rag_stream_impl(
        self,
        *,
        question: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, str]]],
        use_graph: bool,
        user_id: Optional[str],
        model_provider: Optional[str],
        model_name: Optional[str],
        emit_thinking: bool,
        emit_intent: bool,
        persist: bool,
        cancel_run_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        """共享的 RAG 内部实现：用事件流表达完整流程。

        - `query_stream()` 直接转发这些事件到 SSE。
        - `query()` 通过消费 token 事件拼接成最终 answer。
        """
        tel = Telemetry()
        tel.start("rag_total")

        parent = get_current_run_tree()
        run_t0 = time.perf_counter()
        if parent is not None:
            run = parent.create_child(
                name="python_rag",
                run_type="chain",
                inputs={
                    "question": question,
                    "session_id": session_id,
                    "user_id": user_id,
                    "use_graph": use_graph,
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "stream": True,
                },
            )
        else:
            run = RunTree(
                name="python_rag",
                run_type="chain",
                inputs={
                    "question": question,
                    "session_id": session_id,
                    "user_id": user_id,
                    "use_graph": use_graph,
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "stream": True,
                },
                project_name=os.getenv("LANGSMITH_PROJECT"),
            )
        run.post()

        thinking_steps: List[str] = []

        def _thinking(content: str) -> Dict:
            thinking_steps.append(content)
            return {"type": "thinking", "content": content}

        full_answer = ""
        sources: List[Dict] = []
        first_token_seen_at: Optional[float] = None
        t_request_start = time.perf_counter()

        with tracing_context(parent=run):
            try:
                if not self.initialized:
                    if emit_thinking:
                        yield _thinking("⏳ 正在初始化检索与模型组件，请稍候...")
                    _init_start, init_t0 = span_times()
                    init_span = run.create_child(
                        name="initialize",
                        run_type="tool",
                        inputs={},
                    )
                    init_span.post()
                    await self.initialize()
                    init_span.end(
                        outputs={"initialized": True, "duration_ms": perf_ms_since(init_t0)},
                        metadata={"duration_ms": perf_ms_since(init_t0)},
                    )
                    init_span.patch()
                    if emit_thinking:
                        yield _thinking("✅ 初始化完成，开始处理请求...")

                # 记忆/会话持久化：仅在流式链路（或显式需要）时启用，避免非流式副作用
                if persist:
                    if self.memory_manager:
                        if not chat_history and user_id:
                            try:
                                # Keep 3 turns (user+assistant)*3 = 6 messages for better follow-up resolution.
                                history = self.memory_manager.short_term.get_history(session_id, limit=3)
                                chat_history = [{"role": h["role"], "content": h["content"]} for h in history[-6:]]
                            except Exception as e:
                                print(f"获取短期记忆失败: {e}")
                                chat_history = []
                        try:
                            self.memory_manager.short_term.add_message(session_id, "user", question)
                        except Exception as e:
                            print(f"保存短期记忆失败（Redis可能未启动）: {e}")

                    if (not chat_history) and user_id and self.session_store:
                        try:
                            rows = self.session_store.list_messages(
                                session_id=session_id,
                                user_id=user_id,
                                mode="rag",
                                limit=50,
                            )
                            chat_history = [{"role": m.role, "content": m.content} for m in rows]
                        except Exception as e:
                            print(f"从会话存储恢复历史失败: {e}")

                    if user_id and self.session_store:
                        try:
                            import uuid

                            self.session_store.add_message(
                                message_id=str(uuid.uuid4()),
                                session_id=session_id,
                                user_id=user_id,
                                mode="rag",
                                role="user",
                                content=question,
                            )
                        except Exception as e:
                            print(f"写入会话消息失败: {e}")

                # 步骤0：上下文消解
                original_question = question
                resolved_question = question
                if chat_history and len(chat_history) >= 2 and self.context_resolver:
                    if emit_thinking:
                        yield _thinking("🔗 正在消解上下文指代...")
                    _cr_start, cr_t0 = span_times()
                    cr_span = run.create_child(
                        name="context_resolve",
                        run_type="tool",
                        inputs={"question": question, "chat_history": chat_history},
                    )
                    cr_span.post()
                    resolved_question, _extracted_entity = await self.context_resolver.resolve_query(question, chat_history)
                    cr_span.end(
                        outputs={"resolved_question": resolved_question, "duration_ms": perf_ms_since(cr_t0)},
                        metadata={"duration_ms": perf_ms_since(cr_t0)},
                    )
                    cr_span.patch()
                    if resolved_question != question:
                        if emit_thinking:
                            yield _thinking(f"✓ 消解后: {resolved_question}")
                        question = resolved_question

                # 步骤1：意图识别
                if emit_thinking:
                    yield _thinking("🤔 正在分析问题意图...")
                tel.start("intent_classify")
                _intent_start, intent_t0 = span_times()
                intent_span = run.create_child(
                    name="intent_classify",
                    run_type="tool",
                    inputs={"question": question},
                )
                intent_span.post()
                intent_result = await self.intent_classifier.classify(question, cancel_run_id=cancel_run_id)
                # Heuristic: follow-up questions like "那应该怎么办" are often medical in context.
                # If the classifier returns out_of_scope but recent history looks medical,
                # override to general_medical to avoid false refusal.
                try:
                    q_norm = (question or "").strip()
                    followup_phrases = ("那应该怎么办", "那怎么办", "怎么办", "然后呢", "接下来呢", "怎么处理", "怎么做", "该怎么做", "该咋办", "要怎么做")
                    is_followup = (len(q_norm) <= 12) and any(p in q_norm for p in followup_phrases)
                    if is_followup and str(intent_result.get("intent") or "") == "out_of_scope":
                        hist_text = "\n".join([str(m.get("content") or "") for m in (chat_history or [])][-8:])
                        medical_markers = ("科", "挂号", "医院", "就诊", "症状", "疼", "痛", "发烧", "咳", "药", "用药", "检查", "诊断", "治疗", "感染")
                        if any(k in hist_text for k in medical_markers):
                            intent_result["intent"] = "general_medical"
                            intent_result["use_graph"] = False
                            intent_result["method"] = "heuristic_followup"
                            try:
                                c = float(intent_result.get("confidence", 0.0))
                            except Exception:
                                c = 0.0
                            intent_result["confidence"] = max(0.55, c)
                except Exception:
                    pass
                intent_span.end(
                    outputs={**intent_result, "duration_ms": perf_ms_since(intent_t0)},
                    metadata={"duration_ms": perf_ms_since(intent_t0)},
                )
                intent_span.patch()
                intent_evt = tel.end(
                    "intent_classify",
                    intent=intent_result.get("intent"),
                    confidence=float(intent_result.get("confidence", 0.0)),
                )
                if emit_thinking and intent_evt:
                    yield _thinking(f"⏱️ intent_classify: {intent_evt['duration_ms']}ms")
                if not use_graph:
                    intent_result["use_graph"] = False

                intent_desc = {
                    "greeting": "问候",
                    "thanks": "感谢",
                    "disease_drug": "疾病用药查询",
                    "disease_symptom": "疾病症状查询",
                    "drug_contraindication": "药品禁忌查询",
                    "disease_department": "疾病科室查询",
                    "disease_food": "疾病饮食查询",
                    "symptom_disease": "症状反查疾病",
                    "general_medical": "一般医疗问答",
                    "out_of_scope": "非医疗问题",
                }.get(intent_result["intent"], "未知")

                if emit_thinking:
                    yield {
                        "type": "thinking",
                        "content": f"✓ 识别意图: {intent_desc} (置信度: {intent_result['confidence']:.0%})",
                    }
                thinking_steps.append(f"✓ 识别意图: {intent_desc} (置信度: {intent_result['confidence']:.0%})")

                if cancel_run_id and run_cancelled(cancel_run_id):
                    run.end(
                        outputs={"cancelled": True, "duration_ms": perf_ms_since(run_t0)},
                        metadata={"cancelled": True},
                    )
                    run.patch()
                    if emit_thinking:
                        yield _thinking("— 已取消（服务器已停止后续处理）")
                    yield {"type": "done", "content": ""}
                    return

                # 步骤1.5：问题改写
                if intent_result["intent"] not in ["greeting", "thanks", "out_of_scope"]:
                    if emit_thinking:
                        yield _thinking("🔄 正在优化问题表述...")
                    tel.start("rewrite")
                    try:
                        hist_sig = ""
                        try:
                            tail = (chat_history or [])[-6:]
                            hist_sig = hashlib.sha256(
                                json.dumps(tail, ensure_ascii=False, sort_keys=True).encode("utf-8")
                            ).hexdigest()[:12]
                        except Exception:
                            hist_sig = ""
                        rewritten_question = await self.tools.run(
                            "rewrite_question",
                            args={"question": question, "chat_history": chat_history},
                            ctx=ToolContext(
                                user_id=str(user_id or ""),
                                session_id=str(session_id or ""),
                                mode="rag",
                            ),
                            trace_inputs={"question": question, "chat_history_len": len(chat_history or [])},
                            # Do NOT key by session_id; users can ask the same question in a new session.
                            idempotency_key=f"rewrite:{str(user_id or '')}:{hist_sig}:{question}",
                            idempotency_ttl_s=float(getattr(settings, "RAG_REWRITE_CACHE_TTL_S", 120.0)),
                        )
                    except ToolError as e:
                        # Keep compatibility: error content remains string; add code/retriable for clients that read it.
                        yield {"type": "error", "content": e.message, **e.to_event_fields()}
                        rewritten_question = question
                    rw_evt = tel.end("rewrite")
                    if emit_thinking and rw_evt:
                        yield _thinking(f"⏱️ rewrite: {rw_evt['duration_ms']}ms")
                    if rewritten_question != question and emit_thinking:
                        yield _thinking(f"✓ 问题改写: {rewritten_question}")
                else:
                    rewritten_question = question

                intent_result["raw_question"] = original_question
                intent_result["resolved_question"] = resolved_question
                intent_result["retrieval_query"] = rewritten_question
                if emit_intent:
                    yield {"type": "intent", "content": intent_result}

                # 步骤2-3：检索+重排序
                if emit_thinking:
                    if use_graph and intent_result.get("use_graph"):
                        yield _thinking("🔍 正在查询知识图谱...")
                    else:
                        yield _thinking("🔍 正在检索相关文档...")

                if cancel_run_id and run_cancelled(cancel_run_id):
                    run.end(
                        outputs={"cancelled": True, "duration_ms": perf_ms_since(run_t0)},
                        metadata={"cancelled": True},
                    )
                    run.patch()
                    if emit_thinking:
                        yield _thinking("— 已取消（服务器已停止检索）")
                    yield {"type": "done", "content": ""}
                    return

                _route_start, route_t0 = span_times()
                try:
                    reranked, _route_strategy = await self.intent_router.route(
                        intent_result,
                        rewritten_question,
                        graph_enabled=use_graph,
                    )
                except ToolError as e:
                    yield {"type": "error", "content": e.message, **e.to_event_fields()}
                    reranked, _route_strategy = [], "tool_error"

                if emit_thinking:
                    if reranked:
                        graph_count = sum(1 for d in reranked if d.get("retrieval_source") == "graph")
                        vector_count = len(reranked) - graph_count
                        yield _thinking(f"✓ 检索完成: 图谱{graph_count}条, 向量{vector_count}条")
                        yield _thinking("⚡ 正在重排序优化结果...")
                    else:
                        yield _thinking("⚠️ 未找到相关文档")

                # 步骤4：上下文构建
                builder = ContextBuilder()
                tel.start("build_context")
                _ctx_start, ctx_t0 = span_times()
                ctx_span = run.create_child(
                    name="build_context",
                    run_type="tool",
                    inputs={"docs": reranked or [], "max_tokens": settings.MAX_CONTEXT_TOKENS},
                )
                ctx_span.post()
                context, ctx_stats = builder.build_retrieval_context(
                    reranked or [],
                    max_tokens=settings.MAX_CONTEXT_TOKENS,
                )
                ctx_span.end(
                    outputs={"context": context, "stats": ctx_stats, "duration_ms": perf_ms_since(ctx_t0)},
                    metadata={"duration_ms": perf_ms_since(ctx_t0)},
                )
                ctx_span.patch()
                ctx_evt = tel.end("build_context", **ctx_stats)
                if emit_thinking and ctx_evt:
                    yield _thinking(f"⏱️ build_context: {ctx_evt['duration_ms']}ms")
                if emit_thinking and context:
                    yield _thinking(f"📝 构建上下文: {len(context)}字符")
                if emit_thinking:
                    yield _thinking(
                        f"📦 上下文预算: {ctx_stats.get('context_tokens', 0)} tokens, 文档{ctx_stats.get('context_docs_included', 0)}条"
                    )
                    yield _thinking("💡 正在生成回答...\n")

                # 步骤5：流式生成回答
                llm = OpenAILLM.from_provider(provider=model_provider, model_name=model_name)
                messages = llm.build_rag_messages(question, context, chat_history)
                # Some "reasoner" models may stream only reasoning fields and produce empty visible content.
                # For RAG user-visible streaming, prefer a chat model that reliably streams `content`.
                stream_llm = llm
                try:
                    if (model_provider or "").lower() == "deepseek" and (
                        "r1" in (llm.model or "").lower()
                    ):
                        stream_llm = OpenAILLM.from_provider(provider="deepseek", model_name="deepseek-chat")
                        if emit_thinking:
                            yield _thinking("⚠️ 当前模型流式可能不返回正文，已自动切换为 deepseek-chat 进行流式输出")
                except Exception:
                    stream_llm = llm
                first_token = True
                tel.start("llm_stream")
                _llm_start, llm_t0 = span_times()
                llm_span = run.create_child(
                    name=f"llm_stream:{stream_llm.model}",
                    run_type="llm",
                    inputs={
                        "model": stream_llm.model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": settings.MAX_OUTPUT_TOKENS,
                    },
                )
                llm_span.post()
                try:
                    async for token in stream_llm.generate_stream(messages, cancel_run_id=cancel_run_id):
                        if cancel_run_id and run_cancelled(cancel_run_id):
                            llm_span.end(
                                outputs={
                                    "cancelled": True,
                                    "output_chars": len(full_answer),
                                    "duration_ms": perf_ms_since(llm_t0),
                                },
                                metadata={"cancelled": True},
                            )
                            llm_span.patch()
                            run.end(
                                outputs={
                                    "answer": full_answer,
                                    "cancelled": True,
                                    "duration_ms": perf_ms_since(run_t0),
                                },
                                metadata={"cancelled": True},
                            )
                            run.patch()
                            if emit_thinking:
                                yield _thinking("— 已取消（服务器已停止后续生成）")
                            yield {"type": "done", "content": ""}
                            return
                        if first_token:
                            first_token = False
                            first_token_seen_at = time.perf_counter()
                            if emit_thinking:
                                yield _thinking("⏱️ 首 token 已返回")
                        full_answer += token
                        yield {"type": "token", "content": token}
                except asyncio.CancelledError:
                    # If llm_client observed a cooperative cancel and raised, treat it as a clean stop.
                    if cancel_run_id and run_cancelled(cancel_run_id):
                        llm_span.end(
                            outputs={
                                "cancelled": True,
                                "output_chars": len(full_answer),
                                "duration_ms": perf_ms_since(llm_t0),
                            },
                            metadata={"cancelled": True},
                        )
                        llm_span.patch()
                        run.end(
                            outputs={
                                "answer": full_answer,
                                "cancelled": True,
                                "duration_ms": perf_ms_since(run_t0),
                            },
                            metadata={"cancelled": True},
                        )
                        run.patch()
                        if emit_thinking:
                            yield _thinking("— 已取消（服务器已停止后续生成）")
                        yield {"type": "done", "content": ""}
                        return
                    raise

                # Some providers/models (e.g. DeepSeek reasoner) may stream only reasoning fields and leave
                # user-visible content empty. If we ended up with an empty answer, fall back to a single
                # non-stream completion to get the final answer content.
                if not full_answer.strip():
                    try:
                        # If the selected model is a "reasoner" that doesn't return user-visible content,
                        # fall back to a chat model to produce the final answer text (without reasoning).
                        fallback_llm = llm
                        try:
                            m = (llm.model or "")
                            if (model_provider or "").lower() == "deepseek" and ("r1" in m.lower()):
                                fallback_llm = OpenAILLM.from_provider(provider="deepseek", model_name="deepseek-chat")
                        except Exception:
                            fallback_llm = llm

                        fallback_text = await fallback_llm.generate(
                            messages,
                            temperature=0.7,
                            max_tokens=settings.MAX_OUTPUT_TOKENS,
                            cancel_run_id=cancel_run_id,
                        )
                        full_answer = (fallback_text or "").strip()
                        if full_answer:
                            # Emit as one token event for SSE clients; non-stream callers will still
                            # build the final answer from these token events.
                            yield {"type": "token", "content": full_answer}
                    except Exception:
                        # If fallback fails, keep empty answer; caller will handle downstream.
                        pass
                # Final guardrail: never leave SSE clients with an empty visible answer.
                if not full_answer.strip():
                    hint = (
                        "（当前模型可能只返回推理内容而未返回可见正文，或回退生成失败。\n"
                        "可尝试切换为 `deepseek-chat` 或 `qwen3.5-flash` 再试。）"
                    )
                    full_answer = hint
                    yield {"type": "token", "content": full_answer}
                llm_span.end(
                    outputs={
                        "answer": full_answer,
                        "output_chars": len(full_answer),
                        "time_to_first_token_ms": (
                            int((first_token_seen_at - t_request_start) * 1000) if first_token_seen_at else None
                        ),
                        "duration_ms": perf_ms_since(llm_t0),
                    }
                    ,
                    metadata={"duration_ms": perf_ms_since(llm_t0)},
                )
                llm_span.patch()
                llm_evt = tel.end("llm_stream", output_chars=len(full_answer))
                if emit_thinking and llm_evt:
                    yield _thinking(f"⏱️ llm_stream: {llm_evt['duration_ms']}ms")

                # 步骤6：来源信息
                sources = [
                    {
                        "title": doc.get("title", "未知来源"),
                        "content": doc.get("text", "")[:300],
                        "page": self._normalize_page(doc.get("page")),
                        "retrieval_source": doc.get("retrieval_source", "unknown"),
                    }
                    for doc in (reranked[:5] if reranked else [])
                ]

                if persist and self.memory_manager:
                    try:
                        self.memory_manager.short_term.add_message(session_id, "assistant", full_answer)
                    except Exception:
                        pass
                    if user_id:
                        try:
                            self.memory_manager.long_term.add_consultation(
                                user_id, question, full_answer, intent_result["intent"]
                            )
                        except Exception as e:
                            print(f"保存咨询历史失败: {e}")

                if persist and user_id and self.session_store:
                    try:
                        import uuid

                        self.session_store.add_message(
                            message_id=str(uuid.uuid4()),
                            session_id=session_id,
                            user_id=user_id,
                            mode="rag",
                            role="assistant",
                            content=full_answer,
                            sources=sources,
                            thinking_steps=thinking_steps if thinking_steps else None,
                        )
                    except Exception as e:
                        print(f"写入会话消息失败: {e}")

                yield {"type": "sources", "content": sources}
                total_evt = tel.end("rag_total")
                if emit_thinking and total_evt:
                    yield _thinking(f"⏱️ total: {total_evt['duration_ms']}ms")

                run.end(
                    outputs={
                        "answer": full_answer,
                        "sources": sources,
                        "output_chars": len(full_answer),
                        "sources_count": len(sources),
                        "duration_ms": perf_ms_since(run_t0),
                    },
                    metadata={"duration_ms": perf_ms_since(run_t0)},
                )
                run.patch()
                yield {"type": "done", "content": ""}
            except Exception as e:
                run.end(
                    error=str(e),
                    end_time=now_utc(),
                    metadata={"duration_ms": perf_ms_since(run_t0)},
                )
                run.post()
                raise

    async def query(
        self,
        question: str,
        session_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_graph: bool = True,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        """
        完整RAG查询流程（非流式）

        Returns:
            {"answer": str, "sources": List[Dict]}
        """
        # 通过消费共享的事件流来构建非流式返回，确保两条链路行为一致
        answer_parts: List[str] = []
        sources: List[Dict] = []
        async for evt in self._rag_stream_impl(
            question=question,
            session_id=session_id,
            chat_history=chat_history,
            use_graph=use_graph,
            user_id=user_id,
            model_provider=model_provider,
            model_name=model_name,
            emit_thinking=False,
            emit_intent=False,
            persist=False,
            cancel_run_id=None,
        ):
            if evt.get("type") == "token":
                answer_parts.append(evt.get("content") or "")
            elif evt.get("type") == "sources":
                sources = evt.get("content") or []

        return {"answer": "".join(answer_parts), "sources": sources}

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_graph: bool = True,
        user_id: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        cancel_run_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        完整RAG查询流程（流式输出，集成意图路由和记忆）

        Yields:
            {"type": "token", "content": str} - 流式token
            {"type": "sources", "content": List[Dict]} - 来源信息
            {"type": "intent", "content": Dict} - 意图识别结果
            {"type": "thinking", "content": str} - 思考过程
            {"type": "done", "content": ""} - 完成标记
        """
        async for evt in self._rag_stream_impl(
            question=question,
            session_id=session_id,
            chat_history=chat_history,
            use_graph=use_graph,
            user_id=user_id,
            model_provider=model_provider,
            model_name=model_name,
            emit_thinking=True,
            emit_intent=True,
            persist=True,
            cancel_run_id=cancel_run_id,
        ):
            yield evt
