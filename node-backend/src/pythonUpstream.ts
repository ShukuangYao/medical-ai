import http from 'node:http'
import https from 'node:https'
import type { IncomingMessage } from 'node:http'
import { URL } from 'node:url'

function pick(url: URL): typeof http | typeof https {
  return url.protocol === 'https:' ? https : http
}

function portFor(url: URL): number {
  if (url.port) return Number(url.port)
  return url.protocol === 'https:' ? 443 : 80
}

/**
 * POST JSON to Python without global `fetch` (Undici) headers/body timeouts that
 * break long agent runs before the response head is sent.
 */
export async function postPythonBuffer(
  urlStr: string,
  body: unknown,
  extraHeaders: Record<string, string> = {},
): Promise<{ status: number; body: Buffer }> {
  const url = new URL(urlStr)
  const lib = pick(url)
  const payload = JSON.stringify(body)
  return await new Promise((resolve, reject) => {
    const req = lib.request(
      {
        hostname: url.hostname,
        port: portFor(url),
        path: `${url.pathname}${url.search}`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
          ...extraHeaders,
        },
      },
      (res) => {
        const chunks: Buffer[] = []
        res.on('data', (c: Buffer) => chunks.push(c))
        res.on('end', () => resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks) }))
        res.on('error', reject)
      },
    )
    req.setTimeout(0)
    req.on('error', reject)
    req.write(payload)
    req.end()
  })
}

export async function postPythonStream(
  urlStr: string,
  body: unknown,
  extraHeaders: Record<string, string> = {},
): Promise<{ status: number; incoming: IncomingMessage }> {
  const url = new URL(urlStr)
  const lib = pick(url)
  const payload = JSON.stringify(body)
  return await new Promise((resolve, reject) => {
    const req = lib.request(
      {
        hostname: url.hostname,
        port: portFor(url),
        path: `${url.pathname}${url.search}`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
          ...extraHeaders,
        },
      },
      (incoming) => {
        resolve({ status: incoming.statusCode ?? 0, incoming })
      },
    )
    req.setTimeout(0)
    req.on('error', reject)
    req.write(payload)
    req.end()
  })
}
