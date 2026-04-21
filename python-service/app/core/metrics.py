from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore

    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["service", "method", "path", "status_code"],
    )
    http_request_duration_ms = Histogram(
        "http_request_duration_ms",
        "HTTP request duration in milliseconds",
        ["service", "method", "path", "status_code"],
        buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 15000, 60000),
    )
    cancels_total = Counter(
        "cancels_total",
        "Total cancellations requested/observed",
        ["service", "source"],  # source: api_cancel|client_abort|llm_cancel
    )
    llm_http_retries_total = Counter(
        "llm_http_retries_total",
        "Total LLM HTTP retries",
        ["service", "provider"],
    )
    llm_failures_total = Counter(
        "llm_failures_total",
        "Total LLM failures (non-cancel)",
        ["service", "provider"],
    )
    tool_errors_total = Counter(
        "tool_errors_total",
        "Total tool execution errors",
        ["service", "tool", "code", "retriable"],
    )

    def render_latest() -> bytes:
        return generate_latest()

    def content_type() -> str:
        return CONTENT_TYPE_LATEST

except Exception:  # pragma: no cover
    http_requests_total = None
    http_request_duration_ms = None
    cancels_total = None
    llm_http_retries_total = None
    llm_failures_total = None
    tool_errors_total = None

    def render_latest() -> bytes:
        return b""

    def content_type() -> str:
        return "text/plain; charset=utf-8"


def inc_cancel(source: str) -> None:
    if cancels_total is not None:
        cancels_total.labels(service="python", source=source).inc()


def inc_llm_retry(provider: Optional[str]) -> None:
    if llm_http_retries_total is not None:
        llm_http_retries_total.labels(service="python", provider=(provider or "unknown")).inc()


def inc_llm_failure(provider: Optional[str]) -> None:
    if llm_failures_total is not None:
        llm_failures_total.labels(service="python", provider=(provider or "unknown")).inc()


def inc_tool_error(*, tool: str, code: str, retriable: bool) -> None:
    if tool_errors_total is not None:
        tool_errors_total.labels(service="python", tool=str(tool or "unknown"), code=str(code or "UNKNOWN"), retriable=str(bool(retriable))).inc()

