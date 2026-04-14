from pydantic import BaseModel
from typing import List, Optional, Literal, Any, Dict

class Source(BaseModel):
    title: str
    content: str
    page: Optional[int] = None

class TraceItem(BaseModel):
    agent: str
    message: str
    timestamp: Optional[str] = None


class ValidatedRecord(BaseModel):
    normalized_text: str
    missing_fields: List[str] = []
    contradictions: List[str] = []
    corrections: List[str] = []


class IntentResult(BaseModel):
    raw_question: str
    resolved_question: str
    intent_type: str
    confidence: float


class StructuredCase(BaseModel):
    chief_complaint: Optional[str] = None
    symptoms: List[str] = []
    duration: Optional[str] = None
    vitals: Dict[str, Any] = {}
    history: Dict[str, Any] = {}
    medications: List[str] = []
    allergies: List[str] = []
    tests: List[Dict[str, Any]] = []


class SymptomAnalysis(BaseModel):
    key_findings: List[str] = []
    possible_conditions: List[str] = []


class TriageResult(BaseModel):
    severity_level: Literal["emergency", "urgent", "routine"]
    red_flags: List[str] = []
    why: str


class DepartmentRecommendation(BaseModel):
    recommended: List[str] = []
    alternatives: List[str] = []
    reason: str


class NextSteps(BaseModel):
    immediate_actions: List[str] = []
    recommended_tests: List[str] = []
    when_to_seek_care: List[str] = []


class TreatmentSafety(BaseModel):
    medication_considerations: List[str] = []
    contraindications: List[str] = []
    cautions: List[str] = []


class AgentReport(BaseModel):
    validated_record: ValidatedRecord
    intent: IntentResult
    structured_case: StructuredCase
    symptom_analysis: SymptomAnalysis
    triage: TriageResult
    department: DepartmentRecommendation
    next_steps: NextSteps
    treatment_safety: TreatmentSafety
    summary: str
    trace: List[TraceItem] = []

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None
    trace: Optional[List[TraceItem]] = None
    report: Optional[AgentReport] = None
