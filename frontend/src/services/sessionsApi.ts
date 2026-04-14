import axios from 'axios'
import type { ChatMode, Message } from '../types/shared'

const api = axios.create({ baseURL: '/api', timeout: 30000 })

export interface SessionListItem {
  sessionId: string
  title: string
  createdAt: string
  updatedAt: string
  archived?: boolean
}

export const sessionsAPI = {
  createSession: async (params: { userId: string; mode: ChatMode; sessionId: string; title?: string }) => {
    const res = await api.post<{ sessionId: string }>('/sessions', {
      userId: params.userId,
      mode: params.mode,
      sessionId: params.sessionId,
      title: params.title,
    })
    return res.data
  },

  listSessions: async (params: { userId: string; mode: ChatMode }) => {
    const res = await api.get<{ sessions: SessionListItem[] }>('/sessions', {
      params: { userId: params.userId, mode: params.mode },
    })
    return res.data.sessions
  },

  getMessages: async (params: { userId: string; sessionId: string; mode: ChatMode }) => {
    const res = await api.get<{ messages: Array<Omit<Message, 'thinkingExpanded'> & { createdAt: string }> }>(
      `/sessions/${params.sessionId}/messages`,
      { params: { userId: params.userId, mode: params.mode } }
    )
    return res.data.messages
  },

  renameSession: async (params: { userId: string; sessionId: string; mode: ChatMode; title: string }) => {
    await api.post(`/sessions/${params.sessionId}/rename`, {
      userId: params.userId,
      mode: params.mode,
      title: params.title,
    })
  },

  archiveSession: async (params: { userId: string; sessionId: string; mode: ChatMode; archived?: boolean }) => {
    await api.post(`/sessions/${params.sessionId}/archive`, {
      userId: params.userId,
      mode: params.mode,
      archived: params.archived !== false,
    })
  },
}

