/**
 * 前端上下文管理器
 *
 * 策略：滑动窗口 + 重要信息置顶 + 历史对话摘要
 *
 * 优先级（高→低）：
 *   1. 历史摘要（system 消息，始终置顶）
 *   2. pinned 消息（用户手动标记或自动标记的重要消息）
 *   3. 滑动窗口内的普通消息（最近 windowSize 轮）
 *
 * Token 预算：
 *   总预算 maxTokens，先扣除摘要和 pinned，剩余分配给窗口消息。
 *   窗口消息从最新往旧填充，超出预算则截断。
 */

import type { Message } from '../types/shared'

export interface ContextConfig {
  maxTokens: number       // 发给后端的历史 token 上限，默认 4000
  windowSize: number      // 滑动窗口轮数（1轮 = 1问1答），默认 10
  summaryThreshold: number // 用户消息超过此数量时触发摘要，默认 15
}

export type ContextRole = 'user' | 'assistant' | 'system'

export interface ContextMessage {
  role: ContextRole
  content: string
}

const DEFAULT_CONFIG: ContextConfig = {
  maxTokens: 4000,
  windowSize: 10,
  summaryThreshold: 15,
}

/**
 * 粗略估算 token 数
 * 中文约 1.5 字/token，英文约 4 字/token，取折中 length/2
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 2)
}

/**
 * 构建发送给后端的上下文消息列表
 */
export function buildContextMessages(
  messages: Message[],
  summary: string | null,
  config: Partial<ContextConfig> = {}
): ContextMessage[] {
  const cfg: ContextConfig = { ...DEFAULT_CONFIG, ...config }

  // 只处理真实对话消息（排除摘要占位消息）
  const chatMsgs = messages.filter(
    (m) => !m.isSummary && (m.role === 'user' || m.role === 'assistant')
  )

  const pinned = chatMsgs.filter((m) => m.pinned)
  const normal = chatMsgs.filter((m) => !m.pinned)

  // 滑动窗口：取最近 windowSize 轮（每轮 = user + assistant，共 2 条）
  const windowMsgs = normal.slice(-(cfg.windowSize * 2))

  const result: ContextMessage[] = []
  let usedTokens = 0

  // 1. 历史摘要（system 消息）
  if (summary) {
    const t = estimateTokens(summary)
    result.push({ role: 'system', content: `【历史对话摘要】\n${summary}` })
    usedTokens += t
  }

  // 2. pinned 消息（按原始顺序，超预算则跳过）
  for (const msg of pinned) {
    const t = estimateTokens(msg.content)
    if (usedTokens + t <= cfg.maxTokens) {
      result.push({ role: msg.role, content: msg.content })
      usedTokens += t
    }
  }

  // 3. 滑动窗口消息（从最新往旧填充，保证最新消息优先）
  const remaining = cfg.maxTokens - usedTokens
  let windowUsed = 0
  const windowResult: ContextMessage[] = []

  for (let i = windowMsgs.length - 1; i >= 0; i--) {
    const msg = windowMsgs[i]
    const t = estimateTokens(msg.content)
    if (windowUsed + t > remaining) break
    windowResult.unshift({ role: msg.role, content: msg.content })
    windowUsed += t
  }

  return [...result, ...windowResult]
}

/**
 * 判断是否需要触发摘要压缩
 */
export function shouldSummarize(
  messages: Message[],
  threshold = DEFAULT_CONFIG.summaryThreshold
): boolean {
  const userCount = messages.filter((m) => !m.isSummary && m.role === 'user').length
  return userCount > threshold
}

/**
 * 返回需要被摘要的消息（窗口之外的旧消息）
 * keepLast：保留最近几条不参与摘要
 */
export function getMessagesToSummarize(
  messages: Message[],
  keepLast = DEFAULT_CONFIG.windowSize * 2
): Message[] {
  const chatMsgs = messages.filter(
    (m) => !m.isSummary && !m.pinned && (m.role === 'user' || m.role === 'assistant')
  )
  return chatMsgs.slice(0, Math.max(0, chatMsgs.length - keepLast))
}

/**
 * 客户端简单摘要（模板式，不调用 LLM）
 * 生产环境可替换为调用后端摘要接口
 */
export function buildSimpleSummary(messages: Message[]): string {
  const pairs: string[] = []
  for (let i = 0; i < messages.length - 1; i += 2) {
    const user = messages[i]
    const assistant = messages[i + 1]
    if (user?.role === 'user' && assistant?.role === 'assistant') {
      const q = user.content.slice(0, 60)
      const a = assistant.content.slice(0, 80)
      pairs.push(`用户询问"${q}"，助手回答"${a}"`)
    }
  }
  return pairs.join('；') || ''
}
