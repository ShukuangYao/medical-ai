from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.models.response import AgentReport


_DOSE_PATTERNS: List[re.Pattern[str]] = [
    # Common dosage-like tokens (keep conservative; we only sanitize summary text)
    re.compile(r"\b\d+(\.\d+)?\s*(mg|g|mcg|μg|ml|mL|IU|单位)\b", re.IGNORECASE),
    re.compile(r"\b(qd|bid|tid|qid|q\d+h|qhs|prn)\b", re.IGNORECASE),
    re.compile(r"\b(每天|每日|一日|每\d+小时|每晚|睡前)\b"),
]

_DIAGNOSIS_CLAIM_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(确诊|诊断为|考虑为?是|就是|一定是|肯定是|明确是)"),
    re.compile(r"(患有|得了|已是)"),
]


def _safe_str(v: Any) -> str:
    return str(v) if v is not None else ""


def _ensure_disclaimer(summary: str) -> str:
    s = (summary or "").strip()
    if not s:
        return "已完成病历分析（结果仅供参考，不能替代专业医生的诊断与建议）。"
    if "不能替代" in s and "医生" in s:
        return s
    # Keep it short and consistent with existing wording in orchestrator
    return s.rstrip("。") + "。（结果仅供参考，不能替代专业医生的诊断与建议。）"


def _sanitize_summary_no_dosage(summary: str) -> Tuple[str, List[str]]:
    """Best-effort: remove explicit dosage/frequency patterns from summary."""
    s = (summary or "").strip()
    if not s:
        return s, []
    hit = False
    for p in _DOSE_PATTERNS:
        if p.search(s):
            hit = True
            break
    if not hit:
        return s, []
    # Do not attempt complicated rewriting; just replace with safer generic text.
    return (
        "为安全起见，我不能在此给出具体处方或剂量/频次建议；请结合线下医生评估与检查结果遵医嘱处理。"
        + ("（结果仅供参考，不能替代专业医生的诊断与建议。）" if "不能替代" not in s else "")
    ), ["SUMMARY_CONTAINS_DOSAGE_OR_FREQUENCY"]


def _soften_diagnosis_claim(text: str) -> Tuple[str, bool]:
    """Downgrade overly-certain diagnosis claims to cautious wording."""
    s = (text or "").strip()
    if not s:
        return s, False
    hit = any(p.search(s) for p in _DIAGNOSIS_CLAIM_PATTERNS)
    if not hit:
        return s, False
    s2 = s
    for p in _DIAGNOSIS_CLAIM_PATTERNS:
        s2 = p.sub("可能", s2)
    if "可能" not in s2:
        s2 = "可能存在相关问题，需结合检查进一步评估"
    return s2, True


def _clean_list_str(v: Any, *, max_items: int = 20) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        s = str(x).strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _clean_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _clean_list_dict(v: Any, *, max_items: int = 50) -> List[Dict[str, Any]]:
    if not isinstance(v, list):
        return []
    out: List[Dict[str, Any]] = []
    for x in v:
        if isinstance(x, dict):
            out.append(x)
        if len(out) >= max_items:
            break
    return out


def _sanitize_list_no_dosage(items: List[str]) -> Tuple[List[str], bool]:
    """Remove/replace items that look like prescriptions or explicit dosing."""
    changed = False
    out: List[str] = []
    for it in items:
        bad = any(p.search(it) for p in _DOSE_PATTERNS)
        if bad:
            changed = True
            continue
        out.append(it)
    return out, changed


def _ensure_next_steps_minimum(next_steps: Dict[str, Any], *, severity_level: str) -> Dict[str, Any]:
    ns = next_steps if isinstance(next_steps, dict) else {}

    immediate = _clean_list_str(ns.get("immediate_actions"))
    tests = _clean_list_str(ns.get("recommended_tests"))
    when = _clean_list_str(ns.get("when_to_seek_care"))

    if not immediate:
        immediate = [
            "补充休息与补液，避免剧烈运动与刺激性食物",
            "监测体温/症状变化（如加重及时就医）",
        ]
        if severity_level in {"urgent", "emergency"}:
            immediate.insert(0, "尽快前往就近医院/急诊评估（避免自行用药拖延）")
    if len(immediate) == 1:
        immediate.append("如出现明显加重或红旗征，及时就医/急诊")

    if not tests:
        tests = ["血常规（如未做）", "C反应蛋白/炎症指标（如医生认为需要）"]
        if severity_level in {"urgent", "emergency"}:
            tests.append("血氧饱和度监测/必要时胸部影像检查（由医生决定）")
    if len(tests) == 1:
        tests.append("必要时复查/进一步检查（由医生根据查体决定）")

    if not when:
        when = [
            "出现呼吸困难/胸痛/意识改变/持续高热不退等红旗征 → 立即急诊",
            "症状持续不缓解或逐渐加重 → 24–48小时内就医",
        ]
    if len(when) == 1:
        when.append("对症处理后仍不改善或出现新症状 → 尽快就医")

    ns["immediate_actions"] = immediate
    ns["recommended_tests"] = tests
    ns["when_to_seek_care"] = when
    return ns


def _ensure_treatment_safety_minimum(treatment_safety: Dict[str, Any]) -> Dict[str, Any]:
    ts = treatment_safety if isinstance(treatment_safety, dict) else {}
    meds = _clean_list_str(ts.get("medication_considerations"))
    contraindications = _clean_list_str(ts.get("contraindications"))
    cautions = _clean_list_str(ts.get("cautions"))

    meds2, meds_changed = _sanitize_list_no_dosage(meds)
    if meds_changed:
        # Replace with a generic safe note if we removed too much
        if not meds2:
            meds2 = ["不建议在此处自行加减处方药；如需用药请在线下医生评估后遵医嘱"]
    if not meds2:
        meds2 = ["避免自行使用处方药；如需用药请咨询医生/药师并遵医嘱"]
    if not contraindications:
        contraindications = ["对已知过敏药物/成分避免使用；不确定时先咨询医生/药师"]
    if not cautions:
        cautions = ["如出现症状加重或红旗征，请及时就医/急诊评估"]

    ts["medication_considerations"] = meds2
    ts["contraindications"] = contraindications
    ts["cautions"] = cautions
    return ts


def _sanitize_possible_conditions(symptom_analysis: Dict[str, Any]) -> Dict[str, Any]:
    sa = symptom_analysis if isinstance(symptom_analysis, dict) else {}
    pc = _clean_list_str(sa.get("possible_conditions"), max_items=20)
    out: List[str] = []
    for item in pc:
        s2, _changed = _soften_diagnosis_claim(item)
        s2 = s2.strip("。；;，, ")
        if not s2:
            continue
        if len(s2) > 80:
            s2 = s2[:80]
        out.append(s2)
    sa["possible_conditions"] = out
    if not isinstance(sa.get("key_findings"), list):
        sa["key_findings"] = []
    return sa


def _sanitize_structured_case(structured_case: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize structured_case to a stable shape; returns (normalized, tags)."""
    tags: List[str] = []
    sc = structured_case if isinstance(structured_case, dict) else {}

    if "chief_complaint" in sc and sc["chief_complaint"] is not None and not isinstance(sc["chief_complaint"], str):
        sc["chief_complaint"] = _safe_str(sc.get("chief_complaint"))
        tags.append("STRUCTURED_CASE_CHIEF_COMPLAINT_COERCED")
    if "duration" in sc and sc["duration"] is not None and not isinstance(sc["duration"], str):
        sc["duration"] = _safe_str(sc.get("duration"))
        tags.append("STRUCTURED_CASE_DURATION_COERCED")

    symptoms = _clean_list_str(sc.get("symptoms"), max_items=30)
    if not symptoms and sc.get("symptoms") not in (None, [], ""):
        tags.append("STRUCTURED_CASE_SYMPTOMS_NORMALIZED")
    sc["symptoms"] = symptoms

    sc["vitals"] = _clean_dict(sc.get("vitals"))
    sc["history"] = _clean_dict(sc.get("history"))
    sc["medications"] = _clean_list_str(sc.get("medications"), max_items=30)
    sc["allergies"] = _clean_list_str(sc.get("allergies"), max_items=30)

    tests_in = sc.get("tests")
    tests = _clean_list_dict(tests_in, max_items=50)
    norm_tests: List[Dict[str, Any]] = []
    for t in tests:
        name = _safe_str(t.get("name") or "").strip()
        value = _safe_str(t.get("value") or "").strip()
        unit = _safe_str(t.get("unit") or "").strip()
        note = _safe_str(t.get("note") or "").strip()
        if not (name or value or unit or note):
            continue
        norm_tests.append({"name": name, "value": value, "unit": unit, "note": note})
    if tests_in is not None and tests_in != norm_tests:
        tags.append("STRUCTURED_CASE_TESTS_NORMALIZED")
    sc["tests"] = norm_tests
    return sc, tags


def normalize_and_validate_agent_report(
    report: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[str]]:
    """Normalize + validate AgentReport.

    Returns:
      (validated_report_dict, error_detail_or_none)
    """
    r: Dict[str, Any] = report if isinstance(report, dict) else {}
    # Minimal normalization for required nested shapes
    if not isinstance(r.get("validated_record"), dict):
        r["validated_record"] = {}
    if not isinstance(r.get("intent"), dict):
        r["intent"] = {}
    if not isinstance(r.get("structured_case"), dict):
        r["structured_case"] = {}
    if not isinstance(r.get("symptom_analysis"), dict):
        r["symptom_analysis"] = {}
    if not isinstance(r.get("triage"), dict):
        r["triage"] = {}
    if not isinstance(r.get("department"), dict):
        r["department"] = {}
    if not isinstance(r.get("next_steps"), dict):
        r["next_steps"] = {}
    if not isinstance(r.get("treatment_safety"), dict):
        r["treatment_safety"] = {}
    if not isinstance(r.get("trace"), list):
        r["trace"] = []

    tags: List[str] = []

    # Normalize triage severity_level
    triage = r.get("triage") or {}
    if isinstance(triage, dict):
        level = _safe_str(triage.get("severity_level") or "").strip() or "routine"
        if level not in {"emergency", "urgent", "routine"}:
            level = "routine"
        triage["severity_level"] = level
        triage["why"] = _safe_str(triage.get("why") or "")
        if not isinstance(triage.get("red_flags"), list):
            triage["red_flags"] = []
        r["triage"] = triage

    level2 = (r.get("triage") or {}).get("severity_level") if isinstance(r.get("triage"), dict) else "routine"
    severity_level = str(level2) if level2 in {"emergency", "urgent", "routine"} else "routine"

    # Ensure department has minimally usable content for UI
    dept = r.get("department") or {}
    if isinstance(dept, dict):
        rec = _clean_list_str(dept.get("recommended"), max_items=8)
        alt = _clean_list_str(dept.get("alternatives"), max_items=8)
        reason = _safe_str(dept.get("reason") or "")
        if not rec:
            rec = ["内科/全科门诊（先评估，再按症状分科）"]
            if severity_level in {"urgent", "emergency"}:
                rec = ["急诊科（优先评估）"]
        dept["recommended"] = rec
        dept["alternatives"] = alt
        dept["reason"] = reason
        r["department"] = dept

    # Ensure next_steps and treatment_safety are safe + non-empty
    r["next_steps"] = _ensure_next_steps_minimum(r.get("next_steps") if isinstance(r.get("next_steps"), dict) else {}, severity_level=severity_level)
    r["treatment_safety"] = _ensure_treatment_safety_minimum(
        r.get("treatment_safety") if isinstance(r.get("treatment_safety"), dict) else {}
    )

    # Symptom analysis: keep "possible_conditions" non-diagnostic and list-like
    r["symptom_analysis"] = _sanitize_possible_conditions(
        r.get("symptom_analysis") if isinstance(r.get("symptom_analysis"), dict) else {}
    )

    # Structured case: normalize type drifts for stable UI rendering
    sc_norm, sc_tags = _sanitize_structured_case(
        r.get("structured_case") if isinstance(r.get("structured_case"), dict) else {}
    )
    r["structured_case"] = sc_norm
    tags.extend(sc_tags)

    # Summary executable rules (minimal set)
    summary = _safe_str(r.get("summary") or "")
    summary2, _tags = _sanitize_summary_no_dosage(summary)
    summary3, _changed = _soften_diagnosis_claim(summary2)
    r["summary"] = _ensure_disclaimer(summary3)

    try:
        parsed = AgentReport.parse_obj(r)
        return parsed.dict(), None, tags
    except ValidationError as e:
        detail: Dict[str, Any] = {"errors": e.errors()}
        if tags:
            detail["tags"] = tags
        return r, detail, tags

