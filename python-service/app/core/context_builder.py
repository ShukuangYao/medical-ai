from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken


class TokenCounter:
    """Small utility to count/truncate tokens with a safe fallback.

    Note: we intentionally use cl100k_base as an approximation across OpenAI-compatible providers.
    """

    def __init__(self) -> None:
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc:
            return len(self._enc.encode(text))
        # Rough estimate: ~0.7 token/char for CJK-heavy text (same as existing rag_engine heuristic)
        return max(1, int(len(text) * 0.7))

    def truncate(self, text: str, max_tokens: int, *, suffix: str = "...") -> str:
        if max_tokens <= 0:
            return ""
        if not text:
            return ""
        if self._enc is None:
            est_tokens = self.count(text)
            if est_tokens <= max_tokens:
                return text
            keep_chars = int(len(text) * (max_tokens / max(1, est_tokens)))
            keep_chars = max(0, keep_chars)
            return text[:keep_chars] + suffix

        token_ids = self._enc.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        suffix_tokens = len(self._enc.encode(suffix))
        keep = max(0, max_tokens - suffix_tokens)
        return self._enc.decode(token_ids[:keep]) + suffix


@dataclass(frozen=True)
class RetrievalDoc:
    title: str
    text: str
    page: Optional[int] = None
    retrieval_source: str = "unknown"


class ContextBuilder:
    def __init__(self, *, counter: Optional[TokenCounter] = None):
        self.counter = counter or TokenCounter()

    def build_history_window(
        self,
        chat_history: Optional[Sequence[Dict[str, str]]],
        *,
        max_tokens: int,
        min_messages: int = 0,
        keep_first_user: bool = False,
    ) -> List[Dict[str, str]]:
        """Take the most recent messages within token budget.

        - Preserves order.
        - Prefers keeping more recent messages.
        """
        if not chat_history:
            return []

        first_user: Optional[Dict[str, str]] = None
        if keep_first_user:
            for m in chat_history:
                if m.get("role") == "user" and m.get("content"):
                    first_user = {"role": "user", "content": str(m.get("content") or "")}
                    break

        kept_rev: List[Dict[str, str]] = []
        total = 0
        for msg in reversed(list(chat_history)):
            role = str(msg.get("role", "") or "")
            content = str(msg.get("content", "") or "")
            if not role or not content:
                continue
            cost = self.counter.count(content) + 6  # tiny overhead per message
            if kept_rev and total + cost > max_tokens:
                break
            if not kept_rev and total + cost > max_tokens and min_messages <= 0:
                break
            kept_rev.append({"role": role, "content": content})
            total += cost

        kept = list(reversed(kept_rev))
        if first_user:
            # Pin the first user message for "initial question" queries.
            if not kept or kept[0].get("content") != first_user.get("content"):
                kept = [first_user] + [m for m in kept if m.get("content") != first_user.get("content")]

            # If the token budget is exceeded, drop from the oldest part after the pinned first user message
            # while keeping the most recent tail.
            while len(kept) > 1:
                token_est = sum(self.counter.count(m.get("content", "")) + 6 for m in kept)
                if token_est <= max_tokens:
                    break
                kept.pop(1)

        if min_messages and len(kept) < min_messages:
            # Ensure at least N last messages (may exceed budget slightly, but bounded).
            tail = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in chat_history[-min_messages:]
                if m.get("role") and m.get("content")
            ]
            return tail
        return kept

    def build_retrieval_context(
        self,
        docs: Sequence[Dict[str, Any]],
        *,
        max_tokens: int,
        min_remaining_tokens_for_partial: int = 50,
    ) -> Tuple[str, Dict[str, Any]]:
        """Format retrieval docs into a compact context string with token budget.

        Returns (context_text, stats)
        """
        parts: List[str] = []
        used = 0
        included_docs = 0
        truncated_docs = 0

        for i, raw in enumerate(docs):
            title = str(raw.get("title") or "未知来源")
            text = str(raw.get("text") or "")
            if not text:
                continue
            header = f"[参考{i + 1}] {title}"
            doc_text = f"{header}\n{text}"
            doc_tokens = self.counter.count(doc_text)

            if used + doc_tokens > max_tokens:
                remaining = max_tokens - used
                if remaining > min_remaining_tokens_for_partial:
                    parts.append(self.counter.truncate(doc_text, remaining))
                    included_docs += 1
                    truncated_docs += 1
                    used = max_tokens
                break

            parts.append(doc_text)
            used += doc_tokens
            included_docs += 1

        context = "\n\n".join(parts)
        stats = {
            "context_tokens": used,
            "context_docs_included": included_docs,
            "context_docs_truncated": truncated_docs,
        }
        return context, stats

    def build_rag_messages(
        self,
        *,
        system_rules: str,
        question: str,
        retrieval_context: str,
        chat_history: Optional[Sequence[Dict[str, str]]],
        max_history_tokens: int,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Build messages with basic position engineering.

        Strategy:
        - System rules first (static).
        - History window next.
        - Retrieval context as a system message placed right before the user question.
        - User question last.
        """
        q = (question or "").strip()
        initial_keywords = ("最初", "最早", "一开始", "第一个", "第一问", "起初")
        keep_first_user = any(k in q for k in initial_keywords)
        history_msgs = self.build_history_window(
            chat_history,
            max_tokens=max_history_tokens,
            min_messages=0,
            keep_first_user=keep_first_user,
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_rules.strip()}]
        messages.extend(history_msgs)

        retrieval_context = (retrieval_context or "").strip()
        if retrieval_context:
            messages.append({"role": "system", "content": f"参考资料：\n{retrieval_context}"})

        messages.append({"role": "user", "content": question})

        stats = {
            "history_tokens_est": sum(self.counter.count(m.get("content", "")) for m in history_msgs),
            "history_messages": len(history_msgs),
            "has_retrieval": bool(retrieval_context),
            "kept_first_user": keep_first_user,
        }
        return messages, stats

