from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import pytest


BASE_URL = os.getenv("MEDICAL_AI_BASE_URL", "http://localhost:3001")

# Wide timeouts: model + retrieval can be slow.
NON_STREAM_TIMEOUT_S = float(os.getenv("MEDICAL_AI_NON_STREAM_TIMEOUT_S", "120"))
STREAM_TIMEOUT_S = float(os.getenv("MEDICAL_AI_STREAM_TIMEOUT_S", "180"))


def _now_ms() -> int:
    return int(time.time() * 1000)

def _group_id(case: "Case") -> str:
    # Use fixed-width prefixes so -q output is easy to scan.
    return f"[{case.mode.upper()}|{'S' if case.stream else 'NS'}]"


def _build_user_id() -> str:
    # Keep stable across a single test run, but unique per run.
    return f"pytest_user_{_now_ms()}"


@dataclass(frozen=True)
class Case:
    id: str
    mode: str  # rag | agent
    stream: bool
    message: str


def _cases() -> List[Case]:
    """Generate 50 smoke cases.

    - 25 rag, 25 agent
    - half stream, half non-stream (roughly)
    """
    rag_msgs = [
        "头疼怎么办？",
        "高烧怎么降温？",
        "咳嗽两天了要注意什么？",
        "腹泻一天怎么办？",
        "过敏起疹子该怎么处理？",
        "喉咙痛吃什么缓解？",
        "胃痛可能是什么原因？",
        "失眠怎么改善？",
        "血压高要注意什么？",
        "血糖高饮食怎么控制？",
        "孩子发烧需要去医院吗？",
        "胸闷气短可能是什么？",
        "心跳很快要紧吗？",
        "流鼻涕打喷嚏是感冒还是过敏？",
        "扭伤了怎么处理？",
        "牙疼怎么办？",
        "皮肤瘙痒可能是什么原因？",
        "便秘怎么缓解？",
        "耳鸣需要看什么科？",
        "眼睛红痛怎么办？",
        "月经推迟可能原因？",
        "孕期感冒怎么办？",
        "吃了药恶心怎么办？",
        "挂什么科？（结合上文）",
        "谢谢",
    ]
    agent_msgs = [
        "主诉：发热39℃ 2天，伴咳嗽、乏力。既往无明确慢病。现有：无胸痛，无呼吸困难。请分析紧急程度与就诊科室。",
        "病历：头痛3天，伴恶心，畏光。无外伤史。请给出鉴别诊断候选与红旗征。",
        "症状：腹痛右下腹，发热，恶心。请分析可能情况与下一步检查。",
        "病历：胸闷气短，活动后加重，夜间不能平卧。请评估紧急程度并建议就医科室。",
        "病历：皮疹瘙痒，接触海鲜后出现，伴轻微喉咙紧。请评估风险与处理建议。",
        "病历：眩晕，伴耳鸣，恶心，站立不稳。请分析就诊科室与检查建议。",
        "病历：咳嗽两周，痰黄，偶有低热。请分析可能原因与下一步检查。",
        "病历：突然剧烈头痛（爆炸样），伴呕吐。请评估是否急诊。",
        "病历：发热伴皮肤出血点。请给出紧急程度与红旗征。",
        "病历：尿频尿痛2天，无发热。请分析科室与建议检查。",
        "病历：高血压史，现头晕头痛，血压180/110。请评估紧急程度。",
        "病历：糖尿病史，足部伤口不愈合。请给出风险与下一步建议。",
        "病历：上腹痛，黑便。请评估风险与就医建议。",
        "病历：持续咽痛，吞咽困难，流口水。请评估紧急程度。",
        "病历：颈部僵硬，发热，头痛。请评估风险。",
        "病历：持续胸痛30分钟，出汗。请评估是否急诊。",
        "病历：儿童发热、精神差、反复呕吐。请评估风险。",
        "病历：孕期腹痛伴阴道出血。请评估紧急程度。",
        "病历：手指割伤后红肿热痛，发热。请评估风险与检查。",
        "病历：腰痛伴下肢麻木无力。请给出红旗征。",
        "病历：视力突然下降。请评估是否急诊。",
        "病历：持续高热不退伴意识模糊。请评估风险。",
        "病历：皮肤黄染、尿黄。请建议检查与科室。",
        "挂什么科？（结合上文病历）",
        "谢谢",
    ]

    out: List[Case] = []
    # Rag: 13 stream + 12 non-stream
    for i, msg in enumerate(rag_msgs):
        out.append(Case(id=f"rag_s_{i}", mode="rag", stream=(i % 2 == 0), message=msg))
    # Agent: 12 stream + 13 non-stream
    for i, msg in enumerate(agent_msgs):
        out.append(Case(id=f"agent_s_{i}", mode="agent", stream=(i % 2 == 1), message=msg))

    assert len(out) == 50
    return out


def _parse_sse_lines(lines: Iterable[str]) -> Iterable[Dict[str, Any]]:
    """Parse SSE text stream, yielding decoded JSON objects from `data:` lines."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        raw = line[len("data:") :].strip()
        try:
            yield json.loads(raw)
        except Exception:
            continue


def _assert_agent_report_shape(report: Dict[str, Any]) -> None:
    # Key sections expected by frontend cards
    for key in ["triage", "department", "next_steps", "summary"]:
        assert key in report, f"missing report.{key}"


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_api_smoke_50_cases(case: Case):
    # Lightweight progress marker visible even under -q
    print(f"{_group_id(case)} {case.id}", flush=True)
    user_id = _build_user_id()
    session_id = f"{case.mode}_pytest_{_now_ms()}_{case.id}"

    if case.stream:
        url = f"{BASE_URL}/api/chat/stream"
        payload = {
            "message": case.message,
            "mode": case.mode,
            "sessionId": session_id,
            "userId": user_id,
            "useGraph": True,
            "chat_history": [],
            # keep model defaults (server-side)
        }
        t0 = time.time()
        seen = {"thinking": 0, "token": 0, "sources": 0, "result": 0, "done": 0, "error": 0, "session": 0}
        last_result: Optional[Dict[str, Any]] = None

        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_S) as client:
            async with client.stream("POST", url, json=payload) as resp:
                assert resp.status_code == 200
                ct = resp.headers.get("content-type", "")
                assert "text/event-stream" in ct

                async for line in resp.aiter_lines():
                    for evt in _parse_sse_lines([line]):
                        t = str(evt.get("type") or "")
                        if t in seen:
                            seen[t] += 1
                        if t == "result" and isinstance(evt.get("content"), dict):
                            last_result = evt["content"]
                        if t == "done":
                            break
                    if seen["done"] > 0 or seen["error"] > 0:
                        break
                    # hard stop on timeout budget even if server hangs
                    if time.time() - t0 > STREAM_TIMEOUT_S:
                        break

        assert seen["thinking"] >= 1, f"no thinking events: {seen}"
        assert seen["done"] >= 1, f"no done event: {seen}"
        assert seen["error"] == 0, f"error event seen: {seen}"

        if case.mode == "rag":
            # Either tokens arrive, or we still expect sources event.
            assert seen["sources"] >= 1, f"no sources event for rag: {seen}"
            assert seen["token"] >= 1, f"no token events for rag: {seen}"
        else:
            assert seen["result"] >= 1 and last_result is not None, f"no result event for agent: {seen}"
            _assert_agent_report_shape(last_result)
    else:
        url = f"{BASE_URL}/api/chat"
        # Node backend accepts multipart or JSON; use JSON for simplicity.
        payload = {
            "message": case.message,
            "mode": case.mode,
            "sessionId": session_id,
            "userId": user_id,
            "useGraph": True,
        }
        async with httpx.AsyncClient(timeout=NON_STREAM_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("sessionId"), str) and data["sessionId"]
        assert isinstance(data.get("answer"), str) and data["answer"].strip()

        if case.mode == "rag":
            if data.get("sources") is not None:
                assert isinstance(data["sources"], list)
        else:
            # Some implementations may only return `answer`; if report exists, validate.
            if isinstance(data.get("report"), dict):
                _assert_agent_report_shape(data["report"])

