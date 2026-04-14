import type { FastifyInstance } from 'fastify'
import { config } from '../config.js'

export default async function configRoutes(fastify: FastifyInstance) {
  fastify.get('/config', async () => {
    return {
      pythonServiceUrl: config.pythonServiceUrl,
      nodeEnv: config.nodeEnv,
      useGraphDefault: true,
      timestamp: new Date().toISOString(),
    }
  })
}
