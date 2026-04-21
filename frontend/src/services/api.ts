import axios from 'axios'
import type { AgentReport, ChatRequest, ChatResponse, Source } from '../types/shared'
import { createSSEConnection } from './sseClient'

const api = axios.create({
  baseURL: '/api',
  // Non-stream: agent `full` runs many LLM steps; 10m default (streaming uses fetch/SSE separately).
  timeout: 600000,
})

export interface StreamCallbacks {
  onToken: (token: string) => void
  onSources?: (sources: Source[]) => void
  onThinking?: (thinking: string) => void
  onIntent?: (intent: unknown) => void
  onAgentStep?: (step: unknown) => void
  onResult?: (report: AgentReport) => void
  onSession?: (sessionId: string) => void
  /** Same id as X-Run-Id; call POST /api/cancel before aborting the stream for server-side stop. */
  onRunId?: (runId: string) => void
  onDone: () => void
  onError: (error: string) => void
  onRetry?: (attempt: number, delayMs: number) => void
}

export const chatAPI = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const formData = new FormData()
    formData.append('message', request.message)
    formData.append('mode', request.mode)
    if (request.sessionId) formData.append('sessionId', request.sessionId)
    if (request.userId) formData.append('userId', request.userId)
    formData.append('useGraph', String(request.useGraph ?? true))
    // Agent mode does NOT allow manual model selection; models are configured per-agent on the server.
    if (request.mode === 'rag') {
      if (request.modelProvider) formData.append('modelProvider', request.modelProvider)
      if (request.modelName) formData.append('modelName', request.modelName)
    }
    if (request.agentPipeline) formData.append('agentPipeline', request.agentPipeline)
    if (request.file) formData.append('file', request.file)

    const response = await api.post<ChatResponse>('/chat', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  /**
   * SSE 流式发送消息
   * 返回 AbortController，调用 .abort() 可中断生成
   */
  sendMessageStream: (request: ChatRequest, callbacks: StreamCallbacks): AbortController => {
    return createSSEConnection({
      url: '/api/chat/stream',
      body: {
        message: request.message,
        mode: request.mode,
        sessionId: request.sessionId,
        userId: request.userId,
        useGraph: request.useGraph ?? true,
        chat_history: request.chatHistory ?? [],
        ...(request.mode === 'rag' ? { modelProvider: request.modelProvider, modelName: request.modelName } : {}),
        agentPipeline: request.agentPipeline,
      },
      callbacks: {
        onToken: callbacks.onToken,
        onRunId: callbacks.onRunId,
        onDone: callbacks.onDone,
        onError: callbacks.onError,
        onRetry: callbacks.onRetry,
        onEvent: (type, data) => {
          switch (type) {
            case 'thinking':
              callbacks.onThinking?.(data as string)
              break
            case 'intent':
              callbacks.onIntent?.(data)
              break
            case 'agent_step':
              callbacks.onAgentStep?.(data)
              break
            case 'sources':
              callbacks.onSources?.(data as Source[])
              break
            case 'result':
              callbacks.onResult?.(data as AgentReport)
              break
            case 'session':
              callbacks.onSession?.(data as string)
              break
          }
        },
      },
    })
  },

  healthCheck: async (): Promise<{ status: string }> => {
    const response = await api.get('/health')
    return response.data
  },
}

export default api
