const TTL_MS = 10 * 60 * 1000

// run_id -> expiryEpochMs
const cancelled = new Map<string, number>()

function cleanup(now: number) {
  // Bound work: clean at most 200 entries per call
  let n = 0
  for (const [rid, exp] of cancelled) {
    if (exp <= now) cancelled.delete(rid)
    if (++n >= 200) break
  }
}

export function markCancelled(runId: string) {
  const rid = String(runId || '').trim()
  if (!rid) return
  const now = Date.now()
  cleanup(now)
  cancelled.set(rid, now + TTL_MS)
}

/** Consume the cancellation mark for this run_id (best-effort). */
export function consumeCancelled(runId: string): boolean {
  const rid = String(runId || '').trim()
  if (!rid) return false
  const now = Date.now()
  cleanup(now)
  const exp = cancelled.get(rid)
  if (exp == null) return false
  if (exp <= now) {
    cancelled.delete(rid)
    return false
  }
  cancelled.delete(rid)
  return true
}

