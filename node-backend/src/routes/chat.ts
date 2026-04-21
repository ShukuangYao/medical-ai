import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'
import { postPythonBuffer, postPythonStream } from '../pythonUpstream.js'
import type { ChatResponse } from '../types/index.js'
import { v4 as uuidv4 } from 'uuid'
import { getCurrentRunTree, traceable } from 'langsmith/traceable'
import { consumeCancelled } from '../cancelRegistry.js'

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

  const buildLangsmithRunName = (args: {
    endpoint: string
    mode: string
    stream: boolean
    modelProvider?: string
    modelName?: string
    agentPipeline?: string
  }) => {
    const mode = (args.mode || 'rag').trim() || 'rag'
    const stream = args.stream ? 'stream' : 'buffer'
    const mp = (args.modelProvider || '').trim()
    const mn = (args.modelName || '').trim()
    const ap = (args.agentPipeline || '').trim()
    const modelPart =
      mode === 'rag' && (mp || mn)
        ? `${mp ? mp : 'model'}:${mn ? mn : 'default'}`
        : mode === 'agent' && ap
          ? `agent:${ap}`
          : mode === 'agent'
            ? 'agent'
            : 'rag'
    // Keep it readable in LangSmith tables, but still grep-friendly.
    return `node_chat:${args.endpoint}:${mode}:${stream}:${modelPart}`
  }

  // 非流式聊天接口
  fastify.post('/chat', async (request: any, reply: any) => {
    const bodyForMeta = (request.body as any) || {}
    const requestId: string = String(request.headers?.['x-request-id'] || '') || uuidv4()
    const runId: string = String(request.headers?.['x-run-id'] || '') || uuidv4()
    const handler = traceable(
      async () => {
      try {
      // Surface IDs to the browser for debugging/correlation (request headers are client-owned).
      // Note: these are response headers for the browser -> node hop.
      try {
        reply.header('X-Request-ID', requestId)
        reply.header('X-Run-Id', runId)
      } catch {
        // ignore header-setting failures
      }
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

      const requestHeaders: Record<string, string> = {
        'X-Request-ID': requestId,
        'X-Run-Id': runId,
      }

      const ap = (agentPipeline || '').trim().toLowerCase()
      const agent_pipeline = mode === 'agent' && (ap === 'full' || ap === 'fast') ? ap : undefined

      // Agent mode does NOT allow client-side model selection: server uses per-agent config/env.
      const effectiveModelProvider = mode === 'rag' ? (modelProvider || undefined) : undefined
      const effectiveModelName = mode === 'rag' ? (modelName || undefined) : undefined

      const { status: pyStatus, body: pyBody } = await postPythonBuffer(
        pythonEndpoint,
        {
          session_id: sid,
          user_id: effectiveUserId,
          message,
          question: message,
          model_provider: effectiveModelProvider,
          model_name: effectiveModelName,
          ...(agent_pipeline ? { agent_pipeline } : {}),
        },
        { ...lsHeaders, ...requestHeaders },
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
      } finally {
        const cancelled = consumeCancelled(runId)
        if (cancelled) {
          try {
            const rt = getCurrentRunTree() as any
            if (rt) rt.metadata = { cancelled: true }
          } catch {
            // ignore metadata update errors
          }
        }
      }
      },
      {
        name: buildLangsmithRunName({
          endpoint: '/api/chat',
          mode: String(bodyForMeta?.mode || 'rag'),
          stream: false,
          modelProvider: String(bodyForMeta?.modelProvider || ''),
          modelName: String(bodyForMeta?.modelName || ''),
          agentPipeline: String(bodyForMeta?.agentPipeline || ''),
        }),
        run_type: 'chain',
        tags: [
          'service:node',
          'endpoint:/api/chat',
          `mode:${String(bodyForMeta?.mode || 'rag')}`,
          `stream:false`,
          ...(String(bodyForMeta?.modelProvider || '').trim()
            ? [`model_provider:${String(bodyForMeta.modelProvider).trim()}`]
            : []),
          ...(String(bodyForMeta?.modelName || '').trim()
            ? [`model_name:${String(bodyForMeta.modelName).trim()}`]
            : []),
          ...(String(bodyForMeta?.agentPipeline || '').trim()
            ? [`agent_pipeline:${String(bodyForMeta.agentPipeline).trim().toLowerCase()}`]
            : []),
        ],
        metadata: {
          mode: bodyForMeta?.mode || 'rag',
          stream: false,
          session_id: bodyForMeta?.sessionId || null,
          user_id: bodyForMeta?.userId || null,
          request_id: requestId,
          run_id: runId,
          model_provider: String(bodyForMeta?.modelProvider || '').trim() || null,
          model_name: String(bodyForMeta?.modelName || '').trim() || null,
          agent_pipeline: String(bodyForMeta?.agentPipeline || '').trim().toLowerCase() || null,
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
    const requestId: string = String(request.headers?.['x-request-id'] || '') || uuidv4()
    const runId: string = String(request.headers?.['x-run-id'] || '') || uuidv4()
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

      const requestHeaders: Record<string, string> = {
        'X-Request-ID': requestId,
        'X-Run-Id': runId,
      }

      const agent_pipeline =
        mode === 'agent' && (agentPipeline === 'full' || agentPipeline === 'fast') ? agentPipeline : undefined

      // Agent mode does NOT allow client-side model selection: server uses per-agent config/env.
      const effectiveModelProvider = mode === 'rag' ? modelProvider : undefined
      const effectiveModelName = mode === 'rag' ? modelName : undefined

      const { status: pyStatus, incoming } = await postPythonStream(
        pythonEndpoint,
        {
          session_id: sessionId,
          user_id: userId,
          message,
          question: message,
          chat_history: chatHistory,
          use_graph: useGraph,
          model_provider: effectiveModelProvider,
          model_name: effectiveModelName,
          ...(agent_pipeline ? { agent_pipeline } : {}),
        },
        { ...lsHeaders, ...requestHeaders },
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
        'X-Request-ID': requestId,
        'X-Run-Id': runId,
        'Access-Control-Allow-Origin': '*',
      })

      // 先发送sessionId
      reply.raw.write(`data: ${JSON.stringify({ type: 'session', content: sessionId, request_id: requestId, run_id: runId })}\n\n`)

      // If the browser aborts the SSE connection without calling /api/cancel,
      // best-effort stop piping Python upstream and mark the node run as cancelled.
      let clientAborted = false
      const onClose = () => {
        clientAborted = true
        try {
          incoming.destroy(new Error('client_disconnected'))
        } catch {
          // ignore
        }
      }
      reply.raw.once('close', onClose)

      await new Promise<void>((resolve, reject) => {
        incoming.on('data', (chunk: Buffer) => {
          reply.raw.write(chunk)
        })
        incoming.on('end', () => resolve())
        incoming.on('error', reject)
      })

      reply.raw.end()
      const cancelled = clientAborted || consumeCancelled(runId)
      try {
        const rt = getCurrentRunTree() as any
        if (cancelled && rt) rt.metadata = { cancelled: true }
      } catch {
        // ignore metadata update errors
      }
      if (cancelled) {
        // Mark the node root run as cancelled (shows up in LangSmith "Error" column).
        throw new Error(clientAborted ? 'client_disconnected' : 'cancelled')
      }
      // IMPORTANT: this route streams the response; do not return a JSON body,
      // otherwise Fastify will attempt to send a second response.
      return { cancelled }
      },
      {
        name: buildLangsmithRunName({
          endpoint: '/api/chat/stream',
          mode: String((request.body as any)?.mode || 'rag'),
          stream: true,
          modelProvider: String((request.body as any)?.modelProvider || ''),
          modelName: String((request.body as any)?.modelName || ''),
          agentPipeline: String((request.body as any)?.agentPipeline || ''),
        }),
        run_type: 'chain',
        tags: [
          'service:node',
          'endpoint:/api/chat/stream',
          `mode:${String((request.body as any)?.mode || 'rag')}`,
          `stream:true`,
          ...(((request.body as any)?.modelProvider || '').toString().trim()
            ? [`model_provider:${String((request.body as any).modelProvider).trim()}`]
            : []),
          ...(((request.body as any)?.modelName || '').toString().trim()
            ? [`model_name:${String((request.body as any).modelName).trim()}`]
            : []),
          ...(((request.body as any)?.agentPipeline || '').toString().trim()
            ? [`agent_pipeline:${String((request.body as any).agentPipeline).trim().toLowerCase()}`]
            : []),
        ],
        metadata: {
          mode: (request.body as any)?.mode || 'rag',
          stream: true,
          session_id: (request.body as any)?.sessionId || null,
          user_id: (request.body as any)?.userId || null,
          request_id: requestId,
          run_id: runId,
          model_provider: String((request.body as any)?.modelProvider || '').trim() || null,
          model_name: String((request.body as any)?.modelName || '').trim() || null,
          agent_pipeline: String((request.body as any)?.agentPipeline || '').trim().toLowerCase() || null,
        },
      },
    )

    try {
      await handler()
      return reply
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error)
      if (msg === 'cancelled' || msg === 'client_disconnected') {
        // Expected: user clicked Stop or browser aborted the SSE connection.
        // Traceable run has been marked cancelled; do not write an SSE error event.
        try {
          if (!reply.raw.writableEnded) reply.raw.end()
        } catch {
          // ignore
        }
        return reply
      }
      fastify.log.error(error)
      if (!reply.raw.headersSent) {
        const detail = msg
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
