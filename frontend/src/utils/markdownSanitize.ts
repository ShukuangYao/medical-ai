/**
 * 展示前轻量清理：去掉行尾半截 `**`、仅含星号的列表行、以及无法配对的末尾 `**`（未闭合加粗）。
 * 不尝试完整 Markdown AST，避免误伤合法内容。
 */
export function sanitizeDisplayText(raw: string): string {
  let s = raw ?? ''

  // 去掉仅含 ** 的行（含无序/有序列表前缀）
  s = s.replace(/^\s*[-*+]\s*\*\*\s*$/gm, '')
  s = s.replace(/^\s*\d+[.)]\s*\*\*\s*$/gm, '')
  s = s.replace(/^\s*\*\*\s*$/gm, '')

  // 行尾悬空的 **（列表项内常见：`- xxx -> 可考虑：**`）
  s = s.replace(/\*\*\s*$/gm, '')

  // 全文末尾空白后的 **
  s = s.replace(/\*\*\s*$/g, '')

  // 若 `**` 出现奇数次，去掉最后一次（视为未闭合加粗）
  let pairs = (s.match(/\*\*/g) ?? []).length
  if (pairs % 2 === 1) {
    const last = s.lastIndexOf('**')
    if (last !== -1) {
      s = s.slice(0, last) + s.slice(last + 2)
    }
  }

  // 再扫一遍行尾 **（奇数修复后可能仍留半行）
  s = s.replace(/\*\*\s*$/gm, '')
  s = s.replace(/\*\*\s*$/g, '')

  // 与 MessageList 原逻辑一致：孤立编号行、过多空行
  s = s
    .replace(/^\s*\d+\.?\s*$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trimEnd()

  return s
}
