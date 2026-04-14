import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'
import type { ChatResponse } from '../types/index.js'
import { v4 as uuidv4 } from 'uuid'

export default async function chatRoutes(fastify: FastifyInstance) {
  // 非流式聊天接口
  fastify.post('/chat', async (request: any, reply: any) => {
    try {
      const data = await request.file()

      let message = ''
      let mode = 'rag'
      let sessionId = ''
      let modelProvider = ''
      let modelName = ''
      let userId = ''

      if (data) {
        const fields = data.fields
        message = (fields.message as any)?.value || ''
        mode = (fields.mode as any)?.value || 'rag'
        sessionId = (fields.sessionId as any)?.value || ''
        modelProvider = (fields.modelProvider as any)?.value || ''
        modelName = (fields.modelName as any)?.value || ''
        userId = (fields.userId as any)?.value || ''
      }

      const sid = sessionId || `${mode}_${uuidv4()}`

      const pythonEndpoint = mode === 'agent'
        ? `${config.pythonServiceUrl}/api/agent`
        : `${config.pythonServiceUrl}/api/rag`

      const response = await fetch(pythonEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sid,
          user_id: userId || undefined,
          message,
          question: message,
          model_provider: modelProvider || undefined,
          model_name: modelName || undefined,
        }),
      })

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`)
      }

      const result = await response.json() as any

      const chatResponse: ChatResponse = {
        sessionId: sid,
        answer: result.answer,
        sources: result.sources,
        trace: result.trace,
      }

      return chatResponse
    } catch (error) {
      fastify.log.error(error)
      reply.status(500).send({ error: 'Internal server error' })
    }
  })

  // SSE流式聊天接口
  fastify.post('/chat/stream', async (request: any, reply: any) => {
    try {
      const body = request.body as any
      const message = body?.message || body?.question || ''
      const mode = body?.mode || 'rag'
      const sessionId = body?.sessionId || `${mode}_${uuidv4()}`
      const chatHistory = body?.chat_history || null
      const useGraph = body?.useGraph
      const modelProvider = body?.modelProvider
      const modelName = body?.modelName
      const userId = body?.userId

      if (!message) {
        return reply.send({ error: 'message is required' })
      }

      const pythonEndpoint = mode === 'agent'
        ? `${config.pythonServiceUrl}/api/agent/stream`
        : `${config.pythonServiceUrl}/api/rag/stream`

      const response = await fetch(pythonEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
          message,
          question: message,
          chat_history: chatHistory,
          use_graph: useGraph,
          model_provider: modelProvider,
          model_name: modelName,
        }),
      })

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`)
      }

      // 设置SSE响应头
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*',
      })

      // 先发送sessionId
      reply.raw.write(`data: ${JSON.stringify({ type: 'session', content: sessionId })}\n\n`)

      // 转发Python服务的SSE流
      const reader = response.body?.getReader()
      if (reader) {
        const decoder = new TextDecoder()
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value, { stream: true })
          reply.raw.write(chunk)
        }
      }

      reply.raw.end()
    } catch (error) {
      fastify.log.error(error)
      if (!reply.raw.headersSent) {
        reply.status(500).send({ error: 'Internal server error' })
      } else {
        reply.raw.write(`data: ${JSON.stringify({ type: 'error', content: 'Stream error' })}\n\n`)
        reply.raw.end()
      }
    }
  })
}
