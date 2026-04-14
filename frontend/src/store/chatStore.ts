import { create } from 'zustand'
import type { Message, ChatMode, ModelName, ModelProvider } from '../types/shared'
import { sessionsAPI, type SessionListItem } from '../services/sessionsApi'
import {
  buildContextMessages,
  shouldSummarize,
  getMessagesToSummarize,
  buildSimpleSummary,
} from '../utils/contextManager'

let _idCounter = 0
function genId(): string {
  return `msg_${Date.now()}_${++_idCounter}`
}

/** 同一 mode 上并发的 hydrate（如 React Strict Mode 双跑 useEffect）合并为一次，避免空列表时创建多个默认会话 */
const _hydrateInFlight: Partial<Record<ChatMode, Promise<void>>> = {}

export interface PerModeState {
  sessionId: string | null
  messages: Message[]
  loading: boolean
  messageSummary: string | null
}

const initMode = (): PerModeState => ({
  sessionId: null,
  messages: [],
  loading: false,
  messageSummary: null,
})

interface ChatState {
  mode: ChatMode
  perMode: Record<ChatMode, PerModeState>

  /** 匿名用户ID（localStorage 持久化） */
  userId: string

  /** 会话列表（rag/agent 各一套） */
  sessions: Record<ChatMode, SessionListItem[]>

  /** 当前激活会话（rag/agent 各自独立） */
  activeSessionIdByMode: Record<ChatMode, string | null>

  /** 模型选择（全局） */
  modelProvider: ModelProvider
  modelName: ModelName
  setModelProvider: (provider: ModelProvider) => void
  setModelName: (model: ModelName) => void

  hydrateSessions: (mode: ChatMode) => Promise<void>
  createNewSession: (mode: ChatMode) => Promise<string>
  setActiveSession: (mode: ChatMode, sessionId: string) => Promise<void>
  renameSession: (mode: ChatMode, sessionId: string, title: string) => Promise<void>
  archiveSession: (mode: ChatMode, sessionId: string) => Promise<void>

  /** 每个 session 的消息缓存（避免切换会话时 SSE 写串） */
  messagesBySession: Record<ChatMode, Record<string, Message[]>>
  addMessageForSession: (mode: ChatMode, sessionId: string, message: Omit<Message, 'id'> & { id?: string }) => string
  updateMessageByIdForSession: (mode: ChatMode, sessionId: string, id: string, updater: (msg: Message) => Message) => void

  /** 切换 tab，不清空历史 */
  setMode: (mode: ChatMode) => void
  setSessionId: (id: string) => void
  /** 写入指定 tab 的 session，用于异步回调晚于切 tab 的场景 */
  setSessionIdForMode: (mode: ChatMode, id: string) => void
  addMessage: (message: Omit<Message, 'id'> & { id?: string }, forMode?: ChatMode) => string
  /** 更新当前 mode 最后一条消息 */
  updateLastMessage: (updater: (msg: Message) => Message) => void
  /** 按 ID 更新消息（跨 mode 搜索，支持切 tab 后流式回调仍写入正确 tab） */
  updateMessageById: (id: string, updater: (msg: Message) => Message) => void
  setLoading: (loading: boolean) => void
  /** 显式指定 mode 的 loading，用于流式回调跨 tab 场景 */
  setLoadingForMode: (mode: ChatMode, loading: boolean) => void
  clearMessages: () => void
  togglePin: (messageId: string) => void
  setSummary: (summary: string) => void
  checkAndSummarize: () => void
  checkAndSummarizeForMode: (mode: ChatMode) => void
  getContextMessages: () => Array<{ role: string; content: string }>
  getContextMessagesForMode: (mode: ChatMode) => Array<{ role: string; content: string }>
}

export const useChatStore = create<ChatState>((set, get) => ({
  mode: 'rag',
  perMode: { rag: initMode(), agent: initMode() },

  userId: (() => {
    const key = 'medicalai.userId'
    const existed = localStorage.getItem(key)
    if (existed) return existed
    const gen = (globalThis.crypto?.randomUUID?.() ?? `u_${Date.now()}_${Math.random().toString(16).slice(2)}`)
    localStorage.setItem(key, gen)
    return gen
  })(),

  sessions: { rag: [], agent: [] },
  activeSessionIdByMode: {
    rag: localStorage.getItem('medicalai.activeSessionId.rag'),
    agent: localStorage.getItem('medicalai.activeSessionId.agent'),
  },

  messagesBySession: { rag: {}, agent: {} },

  modelProvider: (localStorage.getItem('medicalai.modelProvider') as ModelProvider) || 'qwen',
  modelName: (localStorage.getItem('medicalai.modelName') as ModelName) || 'qwen-turbo',

  setModelProvider: (provider) => {
    set({ modelProvider: provider })
    localStorage.setItem('medicalai.modelProvider', provider)
    // provider 变更时，若当前 model 不属于该 provider，则切到默认
    const { modelName } = get()
    const nextDefault: ModelName = provider === 'qwen' ? 'qwen-turbo' : 'deepseek-chat'
    const qwenModels = new Set<ModelName>(['qwen-turbo', 'qwen-plus', 'qwen-max'])
    const deepseekModels = new Set<ModelName>(['deepseek-reasoner', 'deepseek-chat'])
    const ok = provider === 'qwen' ? qwenModels.has(modelName) : deepseekModels.has(modelName)
    if (!ok) get().setModelName(nextDefault)
  },

  setModelName: (model) => {
    set({ modelName: model })
    localStorage.setItem('medicalai.modelName', model)
  },

  hydrateSessions: async (mode) => {
    const inflight = _hydrateInFlight[mode]
    if (inflight) {
      await inflight
      return
    }
    const run = (async () => {
      const { userId } = get()
      const list = await sessionsAPI.listSessions({ userId, mode })
      set((s) => ({ sessions: { ...s.sessions, [mode]: list } }))

      const active = get().activeSessionIdByMode[mode]
      if (active && list.some((x) => x.sessionId === active)) {
        await get().setActiveSession(mode, active)
        return
      }
      if (list.length > 0) {
        await get().setActiveSession(mode, list[0].sessionId)
        return
      }
      await get().createNewSession(mode)
    })().finally(() => {
      if (_hydrateInFlight[mode] === run) delete _hydrateInFlight[mode]
    })
    _hydrateInFlight[mode] = run
    await run
  },

  createNewSession: async (mode) => {
    const { userId } = get()
    const raw = globalThis.crypto?.randomUUID?.() ?? `s_${Date.now()}_${Math.random().toString(16).slice(2)}`
    const sid = `${mode}_${raw}`
    await sessionsAPI.createSession({ userId, mode, sessionId: sid, title: '新会话' })
    const list = await sessionsAPI.listSessions({ userId, mode })
    set((s) => ({ sessions: { ...s.sessions, [mode]: list } }))
    await get().setActiveSession(mode, sid)
    return sid
  },

  renameSession: async (mode, sessionId, title) => {
    const { userId } = get()
    await sessionsAPI.renameSession({ userId, sessionId, mode, title })
    set((s) => ({
      sessions: {
        ...s.sessions,
        [mode]: s.sessions[mode].map((x) => (x.sessionId === sessionId ? { ...x, title } : x)),
      },
    }))
  },

  archiveSession: async (mode, sessionId) => {
    const { userId } = get()
    await sessionsAPI.archiveSession({ userId, sessionId, mode })
    set((s) => {
      const { [sessionId]: _removed, ...restBySession } = s.messagesBySession[mode]
      return { messagesBySession: { ...s.messagesBySession, [mode]: restBySession } }
    })
    await get().hydrateSessions(mode)
  },

  setActiveSession: async (mode, sessionId) => {
    const { userId } = get()
    const cached = get().messagesBySession[mode][sessionId]
    const msgs = cached ?? (await sessionsAPI.getMessages({ userId, sessionId, mode }))
    set((s) => ({
      activeSessionIdByMode: { ...s.activeSessionIdByMode, [mode]: sessionId },
      messagesBySession: {
        ...s.messagesBySession,
        [mode]: { ...s.messagesBySession[mode], [sessionId]: msgs.map((m) => ({ ...m, thinkingExpanded: false })) },
      },
      perMode: {
        ...s.perMode,
        [mode]: {
          ...s.perMode[mode],
          sessionId,
          messages: msgs.map((m) => ({ ...m, thinkingExpanded: false })),
        },
      },
    }))
    localStorage.setItem(`medicalai.activeSessionId.${mode}`, sessionId)
  },

  addMessageForSession: (mode, sessionId, message) => {
    const id = message.id ?? genId()
    const full = { ...message, id } as Message
    set((s) => {
      const prev = s.messagesBySession[mode][sessionId] ?? []
      const nextMsgs = [...prev, full]
      const nextBySession = { ...s.messagesBySession[mode], [sessionId]: nextMsgs }
      const isActive = s.activeSessionIdByMode[mode] === sessionId
      return {
        messagesBySession: { ...s.messagesBySession, [mode]: nextBySession },
        perMode: isActive ? { ...s.perMode, [mode]: { ...s.perMode[mode], messages: nextMsgs } } : s.perMode,
      }
    })
    return id
  },

  updateMessageByIdForSession: (mode, sessionId, id, updater) => {
    set((s) => {
      const prev = s.messagesBySession[mode][sessionId]
      if (!prev) return s
      const idx = prev.findIndex((m) => m.id === id)
      if (idx === -1) return s
      const nextMsgs = [...prev]
      nextMsgs[idx] = updater(nextMsgs[idx])
      const nextBySession = { ...s.messagesBySession[mode], [sessionId]: nextMsgs }
      const isActive = s.activeSessionIdByMode[mode] === sessionId
      return {
        messagesBySession: { ...s.messagesBySession, [mode]: nextBySession },
        perMode: isActive ? { ...s.perMode, [mode]: { ...s.perMode[mode], messages: nextMsgs } } : s.perMode,
      }
    })
  },

  setMode: (mode) => set({ mode }),

  setSessionId: (id) => get().setSessionIdForMode(get().mode, id),

  setSessionIdForMode: (mode, id) => {
    set((s) => ({ perMode: { ...s.perMode, [mode]: { ...s.perMode[mode], sessionId: id } } }))
  },

  addMessage: (message, forMode) => {
    const id = message.id ?? genId()
    const full = { ...message, id } as Message
    const modeKey = forMode ?? get().mode
    set((s) => ({
      perMode: {
        ...s.perMode,
        [modeKey]: { ...s.perMode[modeKey], messages: [...s.perMode[modeKey].messages, full] },
      },
    }))
    return id
  },

  updateLastMessage: (updater) => {
    const { mode } = get()
    set((s) => {
      const msgs = s.perMode[mode].messages
      if (!msgs.length) return s
      const updated = [...msgs]
      updated[updated.length - 1] = updater(updated[updated.length - 1])
      return { perMode: { ...s.perMode, [mode]: { ...s.perMode[mode], messages: updated } } }
    })
  },

  updateMessageById: (id, updater) => {
    set((s) => {
      const next = { ...s.perMode }
      for (const m of Object.keys(next) as ChatMode[]) {
        const idx = next[m].messages.findIndex((msg) => msg.id === id)
        if (idx !== -1) {
          const msgs = [...next[m].messages]
          msgs[idx] = updater(msgs[idx])
          next[m] = { ...next[m], messages: msgs }
          return { perMode: next }
        }
      }
      return s
    })
  },

  setLoading: (loading) => {
    const { mode } = get()
    set((s) => ({ perMode: { ...s.perMode, [mode]: { ...s.perMode[mode], loading } } }))
  },

  setLoadingForMode: (mode, loading) => {
    set((s) => ({ perMode: { ...s.perMode, [mode]: { ...s.perMode[mode], loading } } }))
  },

  clearMessages: () => {
    const { mode } = get()
    set((s) => ({ perMode: { ...s.perMode, [mode]: initMode() } }))
  },

  togglePin: (messageId) => {
    const { mode } = get()
    set((s) => ({
      perMode: {
        ...s.perMode,
        [mode]: {
          ...s.perMode[mode],
          messages: s.perMode[mode].messages.map((m) =>
            m.id === messageId ? { ...m, pinned: !m.pinned } : m
          ),
        },
      },
    }))
  },

  setSummary: (summary) => {
    const { mode } = get()
    set((s) => ({ perMode: { ...s.perMode, [mode]: { ...s.perMode[mode], messageSummary: summary } } }))
  },

  checkAndSummarize: () => get().checkAndSummarizeForMode(get().mode),

  checkAndSummarizeForMode: (mode) => {
    const { messages, messageSummary } = get().perMode[mode]
    if (!shouldSummarize(messages)) return
    const toSummarize = getMessagesToSummarize(messages)
    if (!toSummarize.length) return
    const newPart = buildSimpleSummary(toSummarize)
    const combined = messageSummary ? `${messageSummary}；${newPart}` : newPart
    set((s) => ({
      perMode: { ...s.perMode, [mode]: { ...s.perMode[mode], messageSummary: combined } },
    }))
  },

  getContextMessages: () => get().getContextMessagesForMode(get().mode),

  getContextMessagesForMode: (mode) => {
    const { perMode } = get()
    const { messages, messageSummary } = perMode[mode]
    return buildContextMessages(messages, messageSummary)
  },
}))
