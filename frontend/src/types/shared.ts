export type ChatMode = 'rag' | 'agent'

export type ModelProvider = 'qwen' | 'deepseek'
export type QwenModel = 'qwen3.5-flash' | 'qwen3.5-plus'
export type DeepseekModel = 'deepseek-chat'
export type ModelName = QwenModel | DeepseekModel
export type AgentPipeline = 'fast' | 'full'

export type TriageSeverity = 'emergency' | 'urgent' | 'routine'

export interface AgentReport {
  validated_record: {
    normalized_text: string
    missing_fields: string[]
    contradictions: string[]
    corrections: string[]
  }
  intent: {
    raw_question: string
    resolved_question: string
    intent_type: string
    confidence: number
  }
  structured_case: {
    chief_complaint?: string | null
    symptoms: string[]
    duration?: string | null
    vitals: Record<string, unknown>
    history: Record<string, unknown>
    medications: string[]
    allergies: string[]
    tests: Array<Record<string, unknown>>
  }
  symptom_analysis: {
    key_findings: string[]
    possible_conditions: string[]
  }
  triage: {
    severity_level: TriageSeverity
    red_flags: string[]
    why: string
  }
  department: {
    recommended: string[]
    alternatives: string[]
    reason: string
  }
  next_steps: {
    immediate_actions: string[]
    recommended_tests: string[]
    when_to_seek_care: string[]
  }
  treatment_safety: {
    medication_considerations: string[]
    contraindications: string[]
    cautions: string[]
  }
  summary: string
  trace: TraceItem[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** Correlates to Node/Python `run_id` (needed for cancel/feedback). */
  runId?: string
  sources?: Source[]
  trace?: TraceItem[]
  report?: AgentReport
  thinkingSteps?: string[]
  thinkingExpanded?: boolean
  streaming?: boolean    // 流式输出进行中（用于展示层避免裁剪“半句话”）
  /** Stream finished (SSE done) AND local typewriter queue flushed to DOM */
  typewriterDone?: boolean
  pinned?: boolean       // 置顶，始终包含在上下文中
  isSummary?: boolean    // 历史摘要消息
}

export interface Source {
  title: string
  content: string
  page?: number
  retrieval_source?: 'graph' | 'vector' | 'elasticsearch' | 'unknown' | string
}

export interface TraceItem {
  agent: string
  message: string
  timestamp?: string
}

export interface ChatRequest {
  message: string
  mode: ChatMode
  sessionId?: string
  userId?: string
  file?: File | null
  useGraph?: boolean
  chatHistory?: Array<{ role: 'user' | 'assistant'; content: string }>
  modelProvider?: ModelProvider
  modelName?: ModelName
  /** Agent tab only: fast merges steps; full restores legacy multi-step agents (slower, richer). */
  agentPipeline?: AgentPipeline
}

export interface ChatResponse {
  sessionId: string
  answer: string
  sources?: Source[]
  trace?: TraceItem[]
  report?: AgentReport
}
