import dotenv from 'dotenv'

dotenv.config()

export const config = {
  port: parseInt(process.env.PORT || '3001', 10),
  pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:8000',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
  nodeEnv: process.env.NODE_ENV || 'development',
}
