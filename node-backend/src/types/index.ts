export type ChatMode = 'rag' | 'agent'

export interface Source {
  title: string
  content: string
  page?: number
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
}

export interface ChatResponse {
  sessionId: string
  answer: string
  sources?: Source[]
  trace?: TraceItem[]
  /**
   * Agent mode may return a structured report for rich UI rendering.
   * RAG mode usually leaves this empty.
   */
  report?: unknown
}

export interface PythonServiceResponse {
  answer: string
  sources?: Source[]
  trace?: TraceItem[]
  report?: unknown
}
