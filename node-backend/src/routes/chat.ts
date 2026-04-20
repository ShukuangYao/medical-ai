import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'
import { postPythonBuffer, postPythonStream } from '../pythonUpstream.js'
import type { ChatResponse } from '../types/index.js'
import { v4 as uuidv4 } from 'uuid'
import { getCurrentRunTree, traceable } from 'langsmith/traceable'

/** LangSmith only: avoid logging huge `report` / long strings (does not change HTTP response). */
function sanitizeNodeChatTraceOutputs(outputs: Readonly<ChatResponse>): Record<string, unknown> {
  const answer = outputs.answer ?? ''
  const sources = outputs.sources ?? []
  const trace = outputs.trace ?? []
  const report = outputs.report
  let reportKeys: string[] = []
  let reportSummaryLen = 0
  if (report && typeof report === 'object' && !Array.isArray(report)) {
    reportKeys = Object.keys(report as object).slice(0, 40)
    const s = (report as { summary?: unknown }).summary
    if (typeof s === 'string') reportSummaryLen = s.length
  }
  return {
    sessionId: outputs.sessionId,
    answer_len: answer.length,
    answer_preview: answer.slice(0, 400),
    sources_count: sources.length,
    trace_count: trace.length,
    report_present: report != null,
    report_keys_sample: reportKeys,
    report_summary_len: reportSummaryLen,
  }
}

export default async function chatRoutes(fastify: FastifyInstance) {
  const DEFAULT_USER_ID = 'anonymous'
  // 非流式聊天接口
  fastify.post('/chat', async (request: any, reply: any) => {
    const bodyForMeta = (request.body as any) || {}
    const handler = traceable(
      async () => {
      let message = ''
      let mode = 'rag'
      let sessionId = ''
      let modelProvider = ''
      let modelName = ''
      let userId = ''
      let agentPipeline = ''

      // Parse multipart fields reliably (even when there's no file).
      if (typeof request.isMultipart === 'function' && request.isMultipart()) {
        const parts = request.parts()
        for await (const part of parts) {
          if (part.type === 'file') {
            // File upload is currently not used by the python service in this endpoint.
            // Consume stream to avoid hanging the request.
            try {
              await part.toBuffer()
            } catch {
              // ignore
            }
            continue
          }
          const fieldname = String(part.fieldname || '')
          const value = String(part.value ?? '')
          if (fieldname === 'message') message = value
          else if (fieldname === 'mode') mode = value || 'rag'
          else if (fieldname === 'sessionId') sessionId = value
          else if (fieldname === 'modelProvider') modelProvider = value
          else if (fieldname === 'modelName') modelName = value
          else if (fieldname === 'userId') userId = value
          else if (fieldname === 'agentPipeline') agentPipeline = value
        }
      } else {
        const body = (request.body as any) || {}
        message = body.message || body.question || ''
        mode = body.mode || 'rag'
        sessionId = body.sessionId || ''
        modelProvider = body.modelProvider || ''
        modelName = body.modelName || ''
        userId = body.userId || ''
        agentPipeline = body.agentPipeline || ''
      }

      if (!message) {
        return { sessionId: sessionId || `${mode}_${uuidv4()}`, answer: '请输入您的健康问题', sources: [], trace: [] }
      }

      const effectiveUserId = (userId || '').trim() || DEFAULT_USER_ID
      const sid = sessionId || `${mode}_${uuidv4()}`

      const pythonEndpoint = mode === 'agent'
        ? `${config.pythonServiceUrl}/api/agent`
        : `${config.pythonServiceUrl}/api/rag`

      const lsHeaders: Record<string, string> = {}
      const runTree = getCurrentRunTree() as any
      if (runTree && typeof runTree.toHeaders === 'function') Object.assign(lsHeaders, runTree.toHeaders())

      const ap = (agentPipeline || '').trim().toLowerCase()
      const agent_pipeline = mode === 'agent' && (ap === 'full' || ap === 'fast') ? ap : undefined

      const { status: pyStatus, body: pyBody } = await postPythonBuffer(
        pythonEndpoint,
        {
          session_id: sid,
          user_id: effectiveUserId,
          message,
          question: message,
          model_provider: modelProvider || undefined,
          model_name: modelName || undefined,
          ...(agent_pipeline ? { agent_pipeline } : {}),
        },
        lsHeaders,
      )

      if (pyStatus < 200 || pyStatus >= 300) {
        throw new Error(`Python service error: HTTP ${pyStatus} ${pyBody.toString('utf8').slice(0, 400)}`)
      }

      const result = JSON.parse(pyBody.toString('utf8') || '{}') as any

      const chatResponse: ChatResponse = {
        sessionId: sid,
        answer: result.answer,
        sources: result.sources,
        trace: result.trace,
        report: result.report,
      }

      return chatResponse
      },
      {
        name: 'node_chat',
        run_type: 'chain',
        tags: ['service:node', 'endpoint:/api/chat'],
        metadata: {
          mode: bodyForMeta?.mode || 'rag',
          stream: false,
          session_id: bodyForMeta?.sessionId || null,
          user_id: bodyForMeta?.userId || null,
        },
        processOutputs: (outputs) => sanitizeNodeChatTraceOutputs(outputs as Readonly<ChatResponse>),
      },
    )

    try {
      return await handler()
    } catch (error) {
      fastify.log.error(error)
      const detail = error instanceof Error ? error.message : String(error)
      reply.status(500).send({
        error: 'chat_failed',
        detail: detail.slice(0, 4000),
      })
    }
  })

  // SSE流式聊天接口
  fastify.post('/chat/stream', async (request: any, reply: any) => {
    const handler = traceable(
      async () => {
      const body = request.body as any
      const message = body?.message || body?.question || ''
      const mode = body?.mode || 'rag'
      const sessionId = body?.sessionId || `${mode}_${uuidv4()}`
      const chatHistory = body?.chat_history || null
      const useGraph = body?.useGraph
      const modelProvider = body?.modelProvider
      const modelName = body?.modelName
      const userId = (body?.userId || '').trim() || DEFAULT_USER_ID
      const agentPipeline = String(body?.agentPipeline || '').trim().toLowerCase()

      if (!message) {
        return reply.send({ error: 'message is required' })
      }

      const pythonEndpoint = mode === 'agent'
        ? `${config.pythonServiceUrl}/api/agent/stream`
        : `${config.pythonServiceUrl}/api/rag/stream`

      const lsHeaders: Record<string, string> = {}
      const runTree = getCurrentRunTree() as any
      if (runTree && typeof runTree.toHeaders === 'function') Object.assign(lsHeaders, runTree.toHeaders())

      const agent_pipeline =
        mode === 'agent' && (agentPipeline === 'full' || agentPipeline === 'fast') ? agentPipeline : undefined

      const { status: pyStatus, incoming } = await postPythonStream(
        pythonEndpoint,
        {
          session_id: sessionId,
          user_id: userId,
          message,
          question: message,
          chat_history: chatHistory,
          use_graph: useGraph,
          model_provider: modelProvider,
          model_name: modelName,
          ...(agent_pipeline ? { agent_pipeline } : {}),
        },
        lsHeaders,
      )

      if (pyStatus < 200 || pyStatus >= 300) {
        incoming.resume()
        throw new Error(`Python service error: HTTP ${pyStatus}`)
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

      await new Promise<void>((resolve, reject) => {
        incoming.on('data', (chunk: Buffer) => {
          reply.raw.write(chunk)
        })
        incoming.on('end', () => resolve())
        incoming.on('error', reject)
      })

      reply.raw.end()
      // IMPORTANT: this route streams the response; do not return a JSON body,
      // otherwise Fastify will attempt to send a second response.
      return null
      },
      {
        name: 'node_chat_stream',
        run_type: 'chain',
        tags: ['service:node', 'endpoint:/api/chat/stream'],
        metadata: {
          mode: (request.body as any)?.mode || 'rag',
          stream: true,
          session_id: (request.body as any)?.sessionId || null,
          user_id: (request.body as any)?.userId || null,
        },
      },
    )

    try {
      await handler()
      return reply
    } catch (error) {
      fastify.log.error(error)
      if (!reply.raw.headersSent) {
        const detail = error instanceof Error ? error.message : String(error)
        reply.status(500).send({
          error: 'chat_stream_failed',
          detail: detail.slice(0, 4000),
        })
      } else {
        reply.raw.write(`data: ${JSON.stringify({ type: 'error', content: 'Stream error' })}\n\n`)
        reply.raw.end()
      }
    }
  })
}
