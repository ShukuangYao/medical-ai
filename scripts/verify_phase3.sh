#!/usr/bin/env bash
set -eo pipefail

NODE_BASE="${NODE_BASE:-http://localhost:3001}"
PY_BASE="${PY_BASE:-http://localhost:8000}"

run_id="verify-$(date +%s)"

echo "NODE_BASE=$NODE_BASE"
echo "PY_BASE=$PY_BASE"
echo "RUN_ID=$run_id"
echo

echo "== metrics(before) =="
curl -sS "$NODE_BASE/metrics" | grep -E "cancels_total|http_requests_total|http_request_duration_ms" || true
curl -sS "$PY_BASE/metrics" | grep -E "cancels_total|llm_http_retries_total|llm_failures_total|tool_errors_total|http_requests_total|http_request_duration_ms" || true
echo

echo "== start SSE stream (RAG) =="
curl -sS -N \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: $run_id" \
  -H "X-Run-Id: $run_id" \
  -X POST "$NODE_BASE/api/chat/stream" \
  -d "{\"mode\":\"rag\",\"message\":\"请用较长的回答解释高血压的常见原因、危险信号和就医建议。\"}" \
  >"/tmp/sse_${run_id}.log" 2>"/tmp/sse_${run_id}.err" &
sse_pid=$!

sleep 1

echo "== cancel =="
curl -sS -X POST "$NODE_BASE/api/cancel" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"$run_id\"}" || true
echo

echo "== wait stream (max 10s) =="
sleep_s=0
while kill -0 "$sse_pid" 2>/dev/null; do
  sleep 0.5
  sleep_s=$((sleep_s + 1))
  if [ "$sleep_s" -ge 20 ]; then
    echo "stream still running after 10s; continuing"
    break
  fi
done
echo

echo "== metrics(after) =="
curl -sS "$NODE_BASE/metrics" | grep -E "cancels_total" || true
curl -sS "$PY_BASE/metrics" | grep -E "cancels_total|llm_http_retries_total|llm_failures_total|tool_errors_total" || true
echo

echo "== SSE tail =="
tail -n 10 "/tmp/sse_${run_id}.log" || true
echo
echo "saved: /tmp/sse_${run_id}.log /tmp/sse_${run_id}.err"

