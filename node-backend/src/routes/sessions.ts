import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'

export default async function sessionRoutes(fastify: FastifyInstance) {
  const DEFAULT_USER_ID = 'anonymous'
  fastify.post('/sessions', async (request: any, reply: any) => {
    try {
      const body = request.body as any
      const userId = (body?.userId || '').trim() || DEFAULT_USER_ID
      const mode = body?.mode || 'rag'
      const sessionId = body?.sessionId
      const title = body?.title

      const resp = await fetch(`${config.pythonServiceUrl}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, mode, sessionId, title }),
      })
      if (!resp.ok) throw new Error(`Python service error: ${resp.statusText}`)
      const data = await resp.json()
      return data
    } catch (e) {
      fastify.log.error(e)
      reply.status(500).send({ error: 'Internal server error' })
    }
  })

  fastify.get('/sessions', async (request: any, reply: any) => {
    try {
      const { userId, mode, includeArchived } = request.query as any
      const url = new URL(`${config.pythonServiceUrl}/api/sessions`)
      url.searchParams.set('userId', (userId || '').trim() || DEFAULT_USER_ID)
      if (mode) url.searchParams.set('mode', mode)
      if (includeArchived !== undefined) url.searchParams.set('includeArchived', String(includeArchived))

      const resp = await fetch(url.toString())
      if (!resp.ok) throw new Error(`Python service error: ${resp.statusText}`)
      const data = await resp.json()
      return data
    } catch (e) {
      fastify.log.error(e)
      reply.status(500).send({ error: 'Internal server error' })
    }
  })

  fastify.get('/sessions/:sessionId/messages', async (request: any, reply: any) => {
    try {
      const { sessionId } = request.params as any
      const { userId, mode, limit } = request.query as any
      const url = new URL(`${config.pythonServiceUrl}/api/sessions/${sessionId}/messages`)
      url.searchParams.set('userId', (userId || '').trim() || DEFAULT_USER_ID)
      if (mode) url.searchParams.set('mode', mode)
      if (limit) url.searchParams.set('limit', String(limit))

      const resp = await fetch(url.toString())
      if (!resp.ok) throw new Error(`Python service error: ${resp.statusText}`)
      const data = await resp.json()
      return data
    } catch (e) {
      fastify.log.error(e)
      reply.status(500).send({ error: 'Internal server error' })
    }
  })

  fastify.post('/sessions/:sessionId/rename', async (request: any, reply: any) => {
    try {
      const { sessionId } = request.params as { sessionId: string }
      const body = request.body as { userId?: string; mode?: string; title?: string }
      const resp = await fetch(`${config.pythonServiceUrl}/api/sessions/${sessionId}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: (body?.userId || '').trim() || DEFAULT_USER_ID, mode: body?.mode, title: body?.title }),
      })
      if (!resp.ok) throw new Error(`Python service error: ${resp.statusText}`)
      return await resp.json()
    } catch (e) {
      fastify.log.error(e)
      reply.status(500).send({ error: 'Internal server error' })
    }
  })

  fastify.post('/sessions/:sessionId/archive', async (request: any, reply: any) => {
    try {
      const { sessionId } = request.params as { sessionId: string }
      const body = (request.body as { userId?: string; mode?: string; archived?: boolean }) || {}
      const resp = await fetch(`${config.pythonServiceUrl}/api/sessions/${sessionId}/archive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: (body.userId || '').trim() || DEFAULT_USER_ID, mode: body.mode, archived: body.archived !== false }),
      })
      if (!resp.ok) throw new Error(`Python service error: ${resp.statusText}`)
      return await resp.json()
    } catch (e) {
      fastify.log.error(e)
      reply.status(500).send({ error: 'Internal server error' })
    }
  })
}

