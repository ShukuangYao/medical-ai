import dotenv from 'dotenv'
dotenv.config()

// Hard default: keep all traces in a single LangSmith project.
// Avoid accidental drift to "medicine-ai" from a stale shell env.
if (!process.env.LANGSMITH_PROJECT || process.env.LANGSMITH_PROJECT === 'medicine-ai') {
  process.env.LANGSMITH_PROJECT = 'medical-ai'
}
if (!process.env.LANGSMITH_TRACING_V2) process.env.LANGSMITH_TRACING_V2 = 'true'

import Fastify from 'fastify'
import cors from '@fastify/cors'
import multipart from '@fastify/multipart'
import { config } from './config.js'
import chatRoutes from './routes/chat.js'
import configRoutes from './routes/config.js'
import healthRoutes from './routes/health.js'
import sessionRoutes from './routes/sessions.js'

const fastify = Fastify({
  logger: true,
})

// 注册插件
await fastify.register(cors, {
  origin: true,
})

await fastify.register(multipart)

// 注册路由
await fastify.register(chatRoutes, { prefix: '/api' })
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
