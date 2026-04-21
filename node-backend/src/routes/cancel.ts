import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'
import { markCancelled } from '../cancelRegistry.js'
import { postPythonBuffer } from '../pythonUpstream.js'
import { cancelsTotal } from '../metrics.js'

/** Forward cooperative cancel to Python (same run_id as X-Run-Id on /api/chat/stream). */
export default async function cancelRoutes(fastify: FastifyInstance) {
  fastify.post('/cancel', async (request, reply) => {
    const body = (request.body as { run_id?: string; runId?: string }) || {}
    const runId = String(body.run_id ?? body.runId ?? '').trim()
    if (!runId) {
      return reply.status(400).send({ ok: false, reason: 'missing_run_id' })
    }
    markCancelled(runId)
    cancelsTotal.inc({ service: 'node', source: 'api_cancel' })
    const { status, body: pyBody } = await postPythonBuffer(`${config.pythonServiceUrl}/api/cancel`, {
      run_id: runId,
    })
    if (status < 200 || status >= 300) {
      return reply.status(502).send({
        ok: false,
        reason: 'python_cancel_failed',
        status,
        detail: pyBody.toString('utf8').slice(0, 500),
      })
    }
    try {
      const parsed = JSON.parse(pyBody.toString('utf8') || '{}') as Record<string, unknown>
      return reply.send(parsed)
    } catch {
      return reply.send({ ok: true, run_id: runId })
    }
  })
}
