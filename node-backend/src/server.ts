import dotenv from 'dotenv'
dotenv.config()

// Hard default: keep all traces in a single LangSmith project.
// Avoid accidental drift to "medicine-ai" from a stale shell env.
if (!process.env.LANGSMITH_PROJECT || process.env.LANGSMITH_PROJECT === 'medicine-ai') {
  process.env.LANGSMITH_PROJECT = 'medical-ai'
}
if (!process.env.LANGSMITH_TRACING_V2) process.env.LANGSMITH_TRACING_V2 = 'true'

import { randomUUID } from 'node:crypto'
import Fastify from 'fastify'
import cors from '@fastify/cors'
import multipart from '@fastify/multipart'
import { config } from './config.js'
import { cancelsTotal, httpRequestDurationMs, httpRequestsTotal, metricsText } from './metrics.js'
import chatRoutes from './routes/chat.js'
import cancelRoutes from './routes/cancel.js'
import configRoutes from './routes/config.js'
import healthRoutes from './routes/health.js'
import sessionRoutes from './routes/sessions.js'

const fastify = Fastify({
  logger: true,
  // Phase 3: structured correlation id for logs (Fastify uses this as `req.id`)
  genReqId: (req) => (String(req.headers['x-request-id'] || '').trim() || randomUUID()),
})

// Phase 3: request_id/run_id middleware + structured logs
fastify.addHook('onRequest', async (req, reply) => {
  const requestId = String(req.headers['x-request-id'] || '').trim() || String((req as any).id || '') || randomUUID()
  const runId = String(req.headers['x-run-id'] || '').trim() || randomUUID()
  ;(req as any).requestId = requestId
  ;(req as any).runId = runId
  try {
    reply.header('X-Request-ID', requestId)
    reply.header('X-Run-Id', runId)
  } catch {
    // ignore
  }
  // Bind to logger so every log line carries these fields
  ;(req as any).log = (req as any).log.child({ request_id: requestId, run_id: runId })
})

fastify.addHook('onResponse', async (req, reply) => {
  try {
    const requestId = (req as any).requestId
    const runId = (req as any).runId
    const route = String((req as any).routerPath || req.url || '')
    const status = String(reply.statusCode || 0)
    httpRequestsTotal.inc({ service: 'node', method: req.method, route, status_code: status })
    httpRequestDurationMs.observe({ service: 'node', method: req.method, route, status_code: status }, reply.elapsedTime)
    ;(req as any).log.info(
      {
        request_id: requestId,
        run_id: runId,
        method: req.method,
        path: req.url,
        status_code: reply.statusCode,
        duration_ms: reply.elapsedTime,
      },
      'http_request',
    )
  } catch {
    // ignore
  }
})

// Phase 3: Prometheus metrics endpoint
fastify.get('/metrics', async (_req, reply) => {
  const text = await metricsText()
  reply.header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
  return reply.send(text)
})

// 注册插件
await fastify.register(cors, {
  origin: true,
})

await fastify.register(multipart)

// 注册路由
await fastify.register(chatRoutes, { prefix: '/api' })
await fastify.register(cancelRoutes, { prefix: '/api' })
await fastify.register(sessionRoutes, { prefix: '/api' })
await fastify.register(configRoutes, { prefix: '/api' })
await fastify.register(healthRoutes)

// 启动服务器
const start = async () => {
  try {
    fastify.log.info(
      { LANGSMITH_PROJECT: process.env.LANGSMITH_PROJECT, LANGSMITH_TRACING_V2: process.env.LANGSMITH_TRACING_V2 },
      'LangSmith env',
    )
    await fastify.listen({ port: config.port, host: '0.0.0.0' })
    console.log(`Server listening on port ${config.port}`)
  } catch (err) {
    fastify.log.error(err)
    process.exit(1)
  }
}

start()
