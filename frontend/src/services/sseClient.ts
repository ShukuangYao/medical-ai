/**
 * SSE 客户端
 * - AbortController 中断生成
 * - 指数退避断线重连（最多 maxRetries 次）
 * - Last-Event-ID 续传
 * - Token 级别异常隔离（单条解析失败不中断流）
 */

export interface SSECallbacks {
  onToken: (token: string) => void
  onEvent: (type: string, data: unknown) => void
  onDone: () => void
  onError: (err: string) => void
  onRetry?: (attempt: number, delayMs: number) => void
  /** Present on envelopes from Node/Python (e.g. session line); used for POST /api/cancel. */
  onRunId?: (runId: string) => void
}

export interface SSEOptions {
  url: string
  body: Record<string, unknown>
  callbacks: SSECallbacks
  maxRetries?: number
}

/**
 * 建立 SSE 连接，返回 AbortController（调用 .abort() 可中断）
 */
export function createSSEConnection(options: SSEOptions): AbortController {
  const controller = new AbortController()
  const maxRetries = options.maxRetries ?? 3

  async function connect(retryCount: number, lastEventId: string | null) {
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (lastEventId) headers['Last-Event-ID'] = lastEventId

      const response = await fetch(options.url, {
        method: 'POST',
        headers,
        body: JSON.stringify(options.body),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('无法获取响应流')

      const decoder = new TextDecoder()
      let buffer = ''
      let currentEventId: string | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (line.startsWith('id: ')) {
            currentEventId = line.slice(4).trim()
          } else if (line.startsWith('data: ')) {
            // Token 级别异常隔离：单条解析失败不影响后续
            try {
              const data = JSON.parse(line.slice(6)) as Record<string, unknown>
              const rid = typeof data.run_id === 'string' ? data.run_id.trim() : ''
              if (rid) options.callbacks.onRunId?.(rid)
              if (data.type === 'token') {
                options.callbacks.onToken(String(data.content ?? ''))
              } else if (data.type === 'done') {
                options.callbacks.onDone()
                return
              } else if (data.type === 'error') {
                options.callbacks.onError(String(data.content ?? '未知错误'))
                return
              } else {
                options.callbacks.onEvent(String(data.type), data.content)
              }
            } catch {
              // 忽略单条 JSON 解析错误，继续处理后续 token
            }
          }
        }

        if (currentEventId) lastEventId = currentEventId
      }

      options.callbacks.onDone()
    } catch (err) {
      // 用户主动中断，不触发错误回调
      if (controller.signal.aborted) return

      if (retryCount < maxRetries) {
        const delayMs = Math.pow(2, retryCount) * 1000 // 1s, 2s, 4s
        options.callbacks.onRetry?.(retryCount + 1, delayMs)
        await sleep(delayMs)
        if (!controller.signal.aborted) {
          connect(retryCount + 1, lastEventId)
        }
      } else {
        const msg = err instanceof Error ? err.message : '连接失败'
        options.callbacks.onError(`重连 ${maxRetries} 次后仍失败: ${msg}`)
      }
    }
  }

  connect(0, null)
  return controller
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
