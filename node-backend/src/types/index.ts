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
}

export interface PythonServiceResponse {
  answer: string
  sources?: Source[]
  trace?: TraceItem[]
}
