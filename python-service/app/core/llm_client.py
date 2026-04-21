"""LLM客户端 - OpenAI兼容接口（Qwen/DashScope, DeepSeek 等）"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
from datetime import datetime, timezone
from openai import AsyncOpenAI
from app.config import settings
from app.core.context_builder import ContextBuilder
from app.core.ls_timing import now_utc, perf_ms_since, span_times
from app.core.run_cancel import is_cancelled as run_cancelled

from langsmith.run_helpers import get_current_run_tree

_LANGSMITH_BUILD_ID = "ls-timing-fix-2026-04-14"
_LANGSMITH_SPAN_SUFFIX = f"@{_LANGSMITH_BUILD_ID}"


def _is_retryable_llm_http_error(exc: BaseException) -> bool:
    """429 / 5xx from OpenAI-compatible providers."""
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError):
            c = int(exc.status_code)
            return c == 429 or (500 <= c < 600)
    except Exception:
        pass
    status = getattr(exc, "status_code", None)
    if status is None:
        return False
    try:
        c = int(status)
        return c == 429 or (500 <= c < 600)
    except (TypeError, ValueError):
        return False


async def _llm_http_backoff(attempt: int) -> None:
    base_s = float(getattr(settings, "LLM_HTTP_RETRY_BASE_MS", 400)) / 1000.0
    delay = min(30.0, base_s * (2**attempt) + random.uniform(0, base_s))
    await asyncio.sleep(delay)

def _cancel_requested(cancel_run_id: Optional[str]) -> bool:
    return bool(cancel_run_id) and run_cancelled(cancel_run_id)


class LLMRunCancelled(asyncio.CancelledError):
    """Raised when a cooperative cancel flag is observed inside llm_client."""


class OpenAILLM:
    """LLM客户端，与大语言模型交互，支持流式生成回答"""

    def __init__(self, *, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    @classmethod
    def from_provider(
        cls,
        *,
        provider: Optional[Literal["qwen", "deepseek"]] = None,
        model_name: Optional[str] = None,
    ) -> "OpenAILLM":
        """Create an LLM client based on selected provider/model.

        - qwen: uses DASHSCOPE_API_KEY + LLM_API_BASE
        - deepseek: uses DEEPSEEK_API_KEY + DEEPSEEK_API_BASE
        """
        def _normalize_model_id(p: str, m: Optional[str]) -> Optional[str]:
            mm = (m or "").strip()
            if not mm:
                return m
            if p == "deepseek":
                # DeepSeek official OpenAI-compatible API uses stable ids like `deepseek-chat`.
                # Some UIs/configs may pass non-stable names. Map to a supported id.
                if mm.lower() in {"deepseek-chat"}:
                    return "deepseek-chat"
            return m

        p = provider or "qwen"
        if p == "deepseek":
            api_key = settings.DEEPSEEK_API_KEY or settings.DASHSCOPE_API_KEY
            base_url = settings.DEEPSEEK_API_BASE
            model = _normalize_model_id(p, model_name) or "deepseek-chat"
        else:
            api_key = settings.DASHSCOPE_API_KEY
            base_url = settings.LLM_API_BASE
            model = model_name or settings.LLM_MODEL
        return cls(api_key=api_key, base_url=base_url, model=model)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        model: Optional[str] = None,
        cancel_run_id: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """非流式生成回答"""
        m = model or self.model
        max_out = max_tokens or settings.MAX_OUTPUT_TOKENS
        parent = get_current_run_tree()
        span = None
        if parent is not None:
            span = parent.create_child(
                name=f"openai_chat_completions:{m}{_LANGSMITH_SPAN_SUFFIX}",
                run_type="llm",
                inputs={
                    "model": m,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_out,
                },
            )
            span.post()
            t0_span = time.perf_counter()

        t0 = time.perf_counter()
        max_retries = int(getattr(settings, "LLM_HTTP_MAX_RETRIES", 3))
        attempt_used = 0
        response = None
        last_exc: Optional[BaseException] = None
        try:
            for attempt in range(max_retries + 1):
                attempt_used = attempt
                if _cancel_requested(cancel_run_id):
                    raise LLMRunCancelled("cancelled")
                try:
                    response = await self.client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_out,
                        response_format=response_format,
                    )
                    break
                except Exception as e:
                    last_exc = e
                    if not _is_retryable_llm_http_error(e) or attempt >= max_retries:
                        if span is not None:
                            span.end(
                                error=str(e),
                                metadata={
                                    "duration_ms": perf_ms_since(t0_span),
                                    "build_id": _LANGSMITH_BUILD_ID,
                                    "llm_http_retries": attempt,
                                },
                            )
                            span.patch()
                        raise
                    if _cancel_requested(cancel_run_id):
                        raise LLMRunCancelled("cancelled")
                    await _llm_http_backoff(attempt)
            if response is None:
                raise last_exc if last_exc else RuntimeError("LLM chat.completions failed")
            text = response.choices[0].message.content or ""
            if span is not None:
                span.end(
                    outputs={
                        "text": text,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "duration_ms": perf_ms_since(t0_span),
                        "build_id": _LANGSMITH_BUILD_ID,
                        "llm_http_retries": attempt_used,
                    },
                    metadata={"duration_ms": perf_ms_since(t0_span), "build_id": _LANGSMITH_BUILD_ID},
                )
                span.patch()
            return text
        except LLMRunCancelled as e:
            if span is not None:
                span.end(
                    error="cancelled",
                    metadata={"cancelled": True, "duration_ms": perf_ms_since(t0_span), "build_id": _LANGSMITH_BUILD_ID},
                )
                span.patch()
            raise e
        except asyncio.CancelledError as e:
            # Task was cancelled (e.g. upstream timeout / gather cancellation). Treat as cancelled for tracing.
            if span is not None:
                span.end(
                    error="cancelled",
                    metadata={
                        "cancelled": True,
                        "duration_ms": perf_ms_since(t0_span),
                        "build_id": _LANGSMITH_BUILD_ID,
                        "llm_http_retries": attempt_used,
                    },
                )
                span.patch()
            raise e
        except Exception:
            raise

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        model: Optional[str] = None,
        include_reasoning: bool = False,
        cancel_run_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成回答，逐token返回"""
        m = model or self.model
        max_out = max_tokens or settings.MAX_OUTPUT_TOKENS
        parent = get_current_run_tree()
        span = None
        if parent is not None:
            span = parent.create_child(
                name=f"openai_chat_completions_stream:{m}{_LANGSMITH_SPAN_SUFFIX}",
                run_type="llm",
                inputs={
                    "model": m,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_out,
                    "stream": True,
                },
            )
            span.post()
            t0_span = time.perf_counter()

        t0 = time.perf_counter()
        out = ""
        first_token_at: Optional[float] = None
        max_retries = int(getattr(settings, "LLM_HTTP_MAX_RETRIES", 3))
        attempt_used = 0
        response = None
        last_exc: Optional[BaseException] = None
        try:
            for attempt in range(max_retries + 1):
                attempt_used = attempt
                if _cancel_requested(cancel_run_id):
                    raise LLMRunCancelled("cancelled")
                try:
                    response = await self.client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_out,
                        stream=True,
                    )
                    break
                except Exception as e:
                    last_exc = e
                    if not _is_retryable_llm_http_error(e) or attempt >= max_retries:
                        if span is not None:
                            span.end(
                                error=str(e),
                                metadata={
                                    "duration_ms": perf_ms_since(t0_span),
                                    "build_id": _LANGSMITH_BUILD_ID,
                                    "llm_http_retries": attempt,
                                },
                            )
                            span.patch()
                        raise
                    if _cancel_requested(cancel_run_id):
                        raise LLMRunCancelled("cancelled")
                    await _llm_http_backoff(attempt)
            if response is None:
                raise last_exc if last_exc else RuntimeError("LLM stream create failed")
            # Some providers block waiting for the next SSE chunk; poll cancellation periodically
            # so stop feels immediate even when no new tokens arrive.
            aiter = response.__aiter__() if hasattr(response, "__aiter__") else response
            while True:
                if _cancel_requested(cancel_run_id):
                    try:
                        aclose = getattr(response, "aclose", None)
                        if callable(aclose):
                            await aclose()
                        else:
                            close = getattr(response, "close", None)
                            if callable(close):
                                close()
                    except Exception:
                        pass
                    raise LLMRunCancelled("cancelled")
                try:
                    chunk = await asyncio.wait_for(aiter.__anext__(), timeout=0.5)  # type: ignore[attr-defined]
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    break
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                if not delta:
                    continue

                # OpenAI-compatible providers may stream:
                # - delta.content (final answer text)
                # - delta.reasoning_content (internal reasoning; should NOT be mixed into user-visible answer by default)
                token_parts: List[str] = []
                v = getattr(delta, "content", None)
                if isinstance(v, str) and v:
                    token_parts.append(v)
                if include_reasoning:
                    rv = getattr(delta, "reasoning_content", None)
                    if isinstance(rv, str) and rv:
                        token_parts.append(rv)
                if not token_parts:
                    continue

                if first_token_at is None:
                    first_token_at = time.perf_counter()
                token = "".join(token_parts)
                out += token
                yield token
            if span is not None:
                span.end(
                    outputs={
                        "text": out,
                        "output_chars": len(out),
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "time_to_first_token_ms": (
                            int((first_token_at - t0) * 1000) if first_token_at is not None else None
                        ),
                        "duration_ms": perf_ms_since(t0_span),
                        "build_id": _LANGSMITH_BUILD_ID,
                        "llm_http_retries": attempt_used,
                    },
                    metadata={"duration_ms": perf_ms_since(t0_span), "build_id": _LANGSMITH_BUILD_ID},
                )
                span.patch()
        except LLMRunCancelled as e:
            if span is not None:
                span.end(
                    error="cancelled",
                    metadata={
                        "cancelled": True,
                        "duration_ms": perf_ms_since(t0_span),
                        "build_id": _LANGSMITH_BUILD_ID,
                        "llm_http_retries": attempt_used,
                    },
                )
                span.patch()
            raise e
        except asyncio.CancelledError as e:
            # If we are cancelled while awaiting/iterating, best-effort close the upstream stream and end span.
            try:
                if response is not None:
                    aclose = getattr(response, "aclose", None)
                    if callable(aclose):
                        await aclose()
                    else:
                        close = getattr(response, "close", None)
                        if callable(close):
                            close()
            except Exception:
                pass
            if span is not None:
                span.end(
                    error="cancelled",
                    metadata={
                        "cancelled": True,
                        "duration_ms": perf_ms_since(t0_span),
                        "build_id": _LANGSMITH_BUILD_ID,
                        "llm_http_retries": attempt_used,
                    },
                )
                span.patch()
            raise e
        except Exception as e:
            if span is not None:
                span.end(
                    error=str(e),
                    metadata={"duration_ms": perf_ms_since(t0_span), "build_id": _LANGSMITH_BUILD_ID},
                )
                span.patch()
            raise

    def build_rag_messages(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建RAG问答的消息列表"""
        system_rules = """你是一个专业的医疗AI助手。请根据提供的参考资料回答用户的医疗健康问题。

要求：
1. 回答必须基于提供的参考资料，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 使用专业但易懂的语言
4. 在适当位置标注参考来源
5. 在回答末尾添加免责声明：本回答仅供参考，不能替代专业医生的诊断和建议
6. 实体一致性：只讨论用户问题或对话历史中明确出现的疾病/症状/药物/检查等实体，严禁凭空引入新的疾病名称（例如用户只提到“头疼、糖尿病”，就不要额外加入其他疾病）
7. 如果用户问题涉及“这两个/这些”但无法从对话历史确定对应哪些实体，请先反问澄清，不要自行猜测补全
8. 禁止“捏造用户画像”：不得凭空补全年龄/性别/职业/既往史/检查结果/治疗方案等个人信息。参考资料可能包含其它病例或人物信息，即使资料里出现“患者xx岁/男/透析/高血压”等，也不能当作用户信息；只能把它当作通用医学知识总结。
9. 当用户信息不足以给出具体用药/处置建议时，先给出安全的通用建议，并列出需要补充的关键问题（例如：年龄、是否妊娠、基础病、当前用药、过敏史、是否伴随红旗征等）。
10. 对于“挂什么科/看什么科/就诊科室/去哪个科”等问题：只回答**科室选择与就医路径**（例如首选科室、何时急诊、需要准备的信息/检查），不要展开病因鉴别或罗列具体诊断名称；更不能把参考资料里的其他病例诊断当作用户情况。
"""
        builder = ContextBuilder()
        messages, _stats = builder.build_rag_messages(
            system_rules=system_rules,
            question=question,
            retrieval_context=context,
            chat_history=chat_history,
            max_history_tokens=getattr(settings, "MAX_HISTORY_TOKENS", 800),
        )
        return messages
