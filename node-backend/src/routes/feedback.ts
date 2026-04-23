import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'
import { postPythonBuffer } from '../pythonUpstream.js'
import { getLangsmithRunId } from '../runRegistry.js'
import { Client as LangSmithClient } from 'langsmith'

export default async function feedbackRoutes(fastify: FastifyInstance) {
  fastify.post('/feedback', async (request, reply) => {
    const body = (request.body as Record<string, unknown>) || {}

    // Best-effort: write feedback to LangSmith using the LangSmith run id,
    // which is different from our correlation run_id (X-Run-Id).
    try {
      const correlationRunId = String(body.run_id ?? body.runId ?? '').trim()
      const lsRunId = correlationRunId ? getLangsmithRunId(correlationRunId) : null
      if (lsRunId) {
        const rating = Number(body.rating ?? 0)
        const score = rating > 0 ? 1 : rating < 0 ? 0 : null
        const comment = typeof body.comment === 'string' ? body.comment : ''
        const corrected = typeof body.corrected_answer === 'string' ? body.corrected_answer : ''
        const extra = {
          correlation_run_id: correlationRunId,
          session_id: body.session_id ?? null,
          message_id: body.message_id ?? null,
          mode: body.mode ?? null,
          user_id: body.user_id ?? null,
        } as Record<string, unknown>
        const client = new LangSmithClient()
        await (client as any).createFeedback(lsRunId, 'user_feedback', {
          score,
          // Put all user-visible fields into `value` so the UI shows them together.
          value: {
            rating,
            comment: comment ? comment.slice(0, 2000) : '',
            corrected_answer: corrected ? corrected.slice(0, 8000) : '',
          },
          comment: comment ? comment.slice(0, 2000) : undefined,
          correction: corrected.trim() ? { corrected_answer: corrected.slice(0, 8000) } : undefined,
          extra,
        } as any)
      }
    } catch {
      // ignore
    }

    const { status, body: pyBody } = await postPythonBuffer(`${config.pythonServiceUrl}/api/feedback`, body)
    if (status < 200 || status >= 300) {
      return reply.status(502).send({
        ok: false,
        reason: 'python_feedback_failed',
        status,
        detail: pyBody.toString('utf8').slice(0, 500),
      })
    }
    try {
      const parsed = JSON.parse(pyBody.toString('utf8') || '{}') as Record<string, unknown>
      return reply.send(parsed)
    } catch {
      return reply.send({ ok: true })
    }
  })
}

