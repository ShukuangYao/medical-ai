import axios from 'axios'
import type { AgentReport, ChatRequest, ChatResponse, Source } from '../types/shared'
import { createSSEConnection } from './sseClient'

const api = axios.create({
  baseURL: '/api',
  // Non-stream requests can be slow (model + retrieval). Streaming uses fetch/SSE, not this timeout.
  timeout: 120000,
})

export interface StreamCallbacks {
  onToken: (token: string) => void
  onSources?: (sources: Source[]) => void
  onThinking?: (thinking: string) => void
  onIntent?: (intent: unknown) => void
  onAgentStep?: (step: unknown) => void
  onResult?: (report: AgentReport) => void
  onSession?: (sessionId: string) => void
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
    if (request.modelProvider) formData.append('modelProvider', request.modelProvider)
    if (request.modelName) formData.append('modelName', request.modelName)
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
        modelProvider: request.modelProvider,
        modelName: request.modelName,
      },
      callbacks: {
        onToken: callbacks.onToken,
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
