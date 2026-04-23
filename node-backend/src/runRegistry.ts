const TTL_MS = 60 * 60 * 1000

// correlation run_id (X-Run-Id) -> { langsmith_run_id, expiry }
const runMap = new Map<string, { langsmithRunId: string; exp: number }>()

function cleanup(now: number) {
  let n = 0
  for (const [rid, v] of runMap) {
    if (v.exp <= now) runMap.delete(rid)
    if (++n >= 200) break
  }
}

export function markLangsmithRunId(correlationRunId: string, langsmithRunId: string) {
  const rid = String(correlationRunId || '').trim()
  const lid = String(langsmithRunId || '').trim()
  if (!rid || !lid) return
  const now = Date.now()
  cleanup(now)
  runMap.set(rid, { langsmithRunId: lid, exp: now + TTL_MS })
}

export function getLangsmithRunId(correlationRunId: string): string | null {
  const rid = String(correlationRunId || '').trim()
  if (!rid) return null
  const now = Date.now()
  cleanup(now)
  const v = runMap.get(rid)
  if (!v) return null
  if (v.exp <= now) {
    runMap.delete(rid)
    return null
  }
  return v.langsmithRunId
}

