import client from 'prom-client'

// Default Node.js process metrics (event loop, memory, GC, etc.)
client.collectDefaultMetrics()

export const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['service', 'method', 'route', 'status_code'] as const,
})

export const httpRequestDurationMs = new client.Histogram({
  name: 'http_request_duration_ms',
  help: 'HTTP request duration in milliseconds',
  labelNames: ['service', 'method', 'route', 'status_code'] as const,
  buckets: [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 15000, 60000],
})

export const cancelsTotal = new client.Counter({
  name: 'cancels_total',
  help: 'Total cancellations requested/observed',
  labelNames: ['service', 'source'] as const, // source: api_cancel|client_abort
})

export function metricsText(): Promise<string> {
  return client.register.metrics()
}

