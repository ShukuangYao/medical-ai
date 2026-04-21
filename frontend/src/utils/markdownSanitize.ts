/**
 * 展示前轻量清理：去掉行尾半截 `**`、仅含星号的列表行、以及无法配对的末尾 `**`（未闭合加粗）。
 * 不尝试完整 Markdown AST，避免误伤合法内容。
 */
export function sanitizeDisplayText(
  raw: string,
  opts?: { trimIncompleteTail?: boolean; maxRef?: number; remapRefs?: boolean }
): string {
  let s = raw ?? ''

  // Remove zero-width chars that often appear in streamed/model outputs and break regex matching.
  // - ZWSP/ZWNJ/ZWJ/BOM/word-joiner
  s = s.replace(/[\u200B-\u200D\u2060\uFEFF]/g, '')

  // 去掉仅含 ** 的行（含无序/有序列表前缀）
  s = s.replace(/^\s*[-*+]\s*\*\*\s*$/gm, '')
  s = s.replace(/^\s*\d+[.)]\s*\*\*\s*$/gm, '')
  s = s.replace(/^\s*\*\*\s*$/gm, '')

  // NOTE: do NOT blindly strip trailing `**`.
  // Legit markdown often ends a line with `**` (e.g. `**标题：**`).
  // We handle truly unbalanced bold markers via the odd-pairs fix below.

  // 若 `**` 出现奇数次，去掉最后一次（视为未闭合加粗）
  let pairs = (s.match(/\*\*/g) ?? []).length
  if (pairs % 2 === 1) {
    const last = s.lastIndexOf('**')
    if (last !== -1) {
      s = s.slice(0, last) + s.slice(last + 2)
    }
  }

  // Avoid a second "strip trailing **" pass; it can break valid markdown.

  // 与 MessageList 原逻辑一致：孤立编号行、过多空行
  s = s
    // - isolated numbering lines (common when models output numbering on its own line)
    //   covers ASCII + fullwidth digits, and common punctuations: `. ) 、 ） ． 。`
    .replace(/^\s*[\p{Number}]+\s*(?:[.)]|[．。]|、|）)?\s*$/gmu, '')
    .replace(/\n{3,}/g, '\n\n')
    .trimEnd()

  // Drop invalid reference markers like `[参考6]` when we only have N sources.
  // We keep the marker format but remove out-of-range ones to avoid confusing UX.
  const maxRef = typeof opts?.maxRef === 'number' ? Number(opts.maxRef) : null
  if (maxRef != null && Number.isFinite(maxRef) && maxRef >= 0) {
    const stripOutOfRange = (m: string, nRaw: string) => {
      const n = Number(nRaw)
      if (!Number.isFinite(n) || n < 1) return ''
      return n > maxRef ? '' : m
    }
    // [参考1] / ［参考1］ / （参考1） / (参考1)
    s = s.replace(/\[\s*参考\s*([0-9]+)\s*\]/g, stripOutOfRange)
    s = s.replace(/［\s*参考\s*([0-9]+)\s*］/g, stripOutOfRange)
    s = s.replace(/（\s*参考\s*([0-9]+)\s*）/g, stripOutOfRange)
    s = s.replace(/\(\s*参考\s*([0-9]+)\s*\)/g, stripOutOfRange)
  }

  // Remap reference indices to be continuous (based on first appearance order).
  // Example: if text contains [参考3] ... [参考5] with maxRef>=5, remap to [参考1] ... [参考2].
  // This improves UX when the model "skips" numbers.
  if (opts?.remapRefs && maxRef != null && Number.isFinite(maxRef) && maxRef >= 1) {
    const seen: number[] = []
    const addSeen = (n: number) => {
      if (!Number.isFinite(n) || n < 1 || n > maxRef) return
      if (!seen.includes(n)) seen.push(n)
    }
    const scan = (re: RegExp) => {
      re.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = re.exec(s)) !== null) {
        addSeen(Number(m[1]))
      }
    }
    scan(/\[\s*参考\s*([0-9]+)\s*\]/g)
    scan(/［\s*参考\s*([0-9]+)\s*］/g)
    scan(/（\s*参考\s*([0-9]+)\s*）/g)
    scan(/\(\s*参考\s*([0-9]+)\s*\)/g)

    const map = new Map<number, number>()
    seen.forEach((oldN, i) => map.set(oldN, i + 1))
    const remap = (_m: string, nRaw: string, open: string, close: string) => {
      const oldN = Number(nRaw)
      const newN = map.get(oldN)
      if (!newN) return '' // out-of-range or removed
      return `${open}参考${newN}${close}`
    }
    s = s.replace(/\[\s*参考\s*([0-9]+)\s*\]/g, (_m, n) => remap(_m, n, '[', ']'))
    s = s.replace(/［\s*参考\s*([0-9]+)\s*］/g, (_m, n) => remap(_m, n, '［', '］'))
    s = s.replace(/（\s*参考\s*([0-9]+)\s*）/g, (_m, n) => remap(_m, n, '（', '）'))
    s = s.replace(/\(\s*参考\s*([0-9]+)\s*\)/g, (_m, n) => remap(_m, n, '(', ')'))
  }

  // Normalize reference markers to a format that won't be swallowed by Markdown renderers.
  // Use full-width corner brackets: 【参考N】
  // This also makes the marker visually consistent even if the model outputs mixed brackets/spaces.
  // First, aggressively canonicalize any bracketed variants (including nested / half-closed ones)
  // into a SINGLE pair: 【参考N】.
  //
  // Examples that should all become 【参考4】:
  // - [参考4] / ［参考4］ / (参考4) / （参考4）
  // - [[参考4]] / 【【参考4】】 / 【[参考4] / [参考4】 / 【参考4]
  // Pass 1: safe-ish normalization when a non-letter/number boundary exists (avoids matching inside English words).
  s = s.replace(
    /(^|[^\p{Letter}\p{Number}])(?:[【\[\(（［]\s*){1,4}参考\s*([0-9]+)\s*(?:[】\]\)）］]\s*){0,4}/gmu,
    (_m, prefix, n) => `${prefix}【参考${n}】`
  )
  // Pass 2: CJK-heavy outputs often attach markers right after a Chinese character (which counts as \p{Letter}),
  // e.g. "有关。【【参考3】】" or "有关[参考3]" with no space. Normalize those too.
  s = s.replace(
    /(?:[【\[\(（［]\s*){1,4}参考\s*([0-9]+)\s*(?:[】\]\)）］]\s*){0,4}/gmu,
    (_m, n) => `【参考${n}】`
  )
  // If brackets were already dropped (e.g. rendered as plain "参考 4"), normalize that too.
  s = s.replace(
    /(^|[^\p{Letter}\p{Number}])参考\s*([0-9]+)(?=[^\p{Letter}\p{Number}]|$)/gmu,
    (_m, prefix, n) => `${prefix}【参考${n}】`
  )
  // Finally, collapse any accidental double-wrapping produced earlier in the pipeline.
  s = s.replace(/【\s*【\s*参考\s*([0-9]+)\s*】\s*】/g, '【参考$1】')

  // If the tail looks like it was truncated mid-sentence (common when hitting max_tokens),
  // optionally hide the incomplete last fragment for a cleaner UI.
  // IMPORTANT: do NOT enable this during streaming; it would hide the in-progress sentence.
  const trimIncompleteTail = opts?.trimIncompleteTail !== false
  if (trimIncompleteTail) {
    // Also drop a dangling trailing numbering-only line (even if it's the last line).
    s = s.replace(/\n\s*[\p{Number}]+\s*(?:[.)]|[．。]|、|）)?\s*$/gu, '')
    const terminators = ['。', '！', '？', '.', '!', '?', '…']
    const trimmed = s.trimEnd()
    const endsWithTerminator = terminators.some((t) => trimmed.endsWith(t))
    if (!endsWithTerminator) {
      const lastIdx = Math.max(
        trimmed.lastIndexOf('。'),
        trimmed.lastIndexOf('！'),
        trimmed.lastIndexOf('？'),
        trimmed.lastIndexOf('.'),
        trimmed.lastIndexOf('!'),
        trimmed.lastIndexOf('?'),
        trimmed.lastIndexOf('…'),
      )
      // only trim when we likely have a substantial incomplete tail
      const tailLen = lastIdx >= 0 ? trimmed.length - (lastIdx + 1) : 0
      if (lastIdx >= 0 && tailLen >= 12) {
        s = trimmed.slice(0, lastIdx + 1)
      }
    }
  }

  return s
}
