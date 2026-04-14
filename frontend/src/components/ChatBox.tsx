import { useState, useRef, useCallback, useMemo } from 'react'
import { Input, Button, Space, message, Spin, Switch, Typography, Select } from 'antd'
import { SendOutlined, ThunderboltOutlined, StopOutlined } from '@ant-design/icons'
import { useChatStore } from '../store/chatStore'
import { chatAPI } from '../services/api'
import MessageList from './MessageList'
import type { ChatMode, ModelName, ModelProvider } from '../types/shared'

const { TextArea } = Input
const { Text } = Typography

interface ChatBoxProps {
  mode: ChatMode
}

function ChatBox({ mode }: ChatBoxProps) {
  const [inputValue, setInputValue] = useState('')
  const [streamMode, setStreamMode] = useState(true)
  const [graphMode, setGraphMode] = useState(true)
  const streamContentRef = useRef('')
  const tokenQueueRef = useRef<string[]>([])
  const typewriterTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isDoneRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  /** 记录本次请求的 mode，用于跨 tab 场景下回调仍写入正确 tab */
  const requestModeRef = useRef<ChatMode>(mode)
  /** 记录本次请求的 sessionId，确保回调写入正确会话 */
  const requestSessionIdRef = useRef<string>('')
  /** 记录本次 assistant 消息 ID，用于跨 tab 安全更新 */
  const assistantMsgIdRef = useRef<string>('')

  // 读取当前 tab（mode prop）的状态
  const { messages, loading } = useChatStore((s) => s.perMode[mode])
  const { addMessageForSession, updateMessageByIdForSession, setLoadingForMode } = useChatStore()
  const { modelProvider, modelName, setModelProvider, setModelName } = useChatStore()
  const { userId } = useChatStore()

  const formatAgentStep = (step: any): string => {
    const agent = step?.agent ?? 'agent'
    const name = step?.step ?? 'step'
    const detail = step?.detail

    try {
      if (name === 'triage') {
        const lvl = detail?.severity_level ?? 'routine'
        const why = typeof detail?.why === 'string' ? detail.why : ''
        return `🤖 ${agent}: ${name}（${lvl}${why ? `，${why}` : ''}）`
      }
      if (name === 'department') {
        const rec = Array.isArray(detail?.recommended) ? detail.recommended.filter(Boolean).join('、') : ''
        return `🤖 ${agent}: ${name}${rec ? `（推荐：${rec}）` : ''}`
      }
      if (name === 'next_steps') {
        const ia = Array.isArray(detail?.immediate_actions) ? detail.immediate_actions.length : 0
        const rt = Array.isArray(detail?.recommended_tests) ? detail.recommended_tests.length : 0
        return `🤖 ${agent}: ${name}（立即措施${ia}条，建议检查${rt}条）`
      }
      if (name === 'validated_record') {
        const missing = Array.isArray(detail?.missing_fields) ? detail.missing_fields.length : 0
        const contra = Array.isArray(detail?.contradictions) ? detail.contradictions.length : 0
        return `🤖 ${agent}: ${name}（缺失字段${missing}项，矛盾${contra}项）`
      }
      if (name === 'structured_case') {
        const cc = typeof detail?.chief_complaint === 'string' ? detail.chief_complaint : ''
        return `🤖 ${agent}: ${name}${cc ? `（主诉：${cc}）` : ''}`
      }
      if (name === 'symptom_analysis') {
        const cond = Array.isArray(detail?.possible_conditions) ? detail.possible_conditions.slice(0, 3).filter(Boolean).join('、') : ''
        return `🤖 ${agent}: ${name}${cond ? `（可能：${cond}）` : ''}`
      }
      if (name === 'treatment_safety') {
        const caut = Array.isArray(detail?.cautions) ? detail.cautions.length : 0
        return `🤖 ${agent}: ${name}${caut ? `（提醒${caut}条）` : ''}`
      }

      const snippet = detail ? JSON.stringify(detail).slice(0, 160) : ''
      return `🤖 ${agent}: ${name}${snippet ? `（${snippet}${snippet.length >= 160 ? '…' : ''}）` : ''}`
    } catch {
      return `🤖 ${agent}: ${name}`
    }
  }

  const modelOptions = useMemo(() => {
    if (modelProvider === 'qwen') {
      return [
        { value: 'qwen-turbo', label: 'qwen-turbo' },
        { value: 'qwen-plus', label: 'qwen-plus' },
        { value: 'qwen-max', label: 'qwen-max' },
      ] as Array<{ value: ModelName; label: string }>
    }
    return [
      { value: 'deepseek-reasoner', label: 'deepseek-reasoner' },
      { value: 'deepseek-chat', label: 'deepseek-chat' },
    ] as Array<{ value: ModelName; label: string }>
  }, [modelProvider])

  const handleToggleThinking = (messageIndex: number, expanded: boolean) => {
    const msgs = useChatStore.getState().perMode[mode].messages
    if (messageIndex < 0 || messageIndex >= msgs.length) return
    useChatStore.getState().updateMessageById(msgs[messageIndex].id, (m) => ({ ...m, thinkingExpanded: expanded }))
  }

  // 停止打字机：flush=true 时立即将队列剩余内容写入
  const stopTypewriter = useCallback((flush = false) => {
    if (typewriterTimerRef.current) {
      clearInterval(typewriterTimerRef.current)
      typewriterTimerRef.current = null
    }
    if (flush && tokenQueueRef.current.length > 0) {
      streamContentRef.current += tokenQueueRef.current.join('')
      tokenQueueRef.current = []
      useChatStore
        .getState()
        .updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, assistantMsgIdRef.current, (msg) => ({
          ...msg,
          content: streamContentRef.current,
        }))
    }
    isDoneRef.current = false
  }, [])

  // 启动打字机：30ms/tick，每次出队 3 个字符；队列耗尽且流结束时自动终止
  const startTypewriter = useCallback(() => {
    if (typewriterTimerRef.current) return
    typewriterTimerRef.current = setInterval(() => {
      if (tokenQueueRef.current.length > 0) {
        const chars = tokenQueueRef.current.splice(0, 3).join('')
        streamContentRef.current += chars
        useChatStore
          .getState()
          .updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, assistantMsgIdRef.current, (msg) => ({
            ...msg,
            content: streamContentRef.current,
          }))
      } else if (isDoneRef.current) {
        clearInterval(typewriterTimerRef.current!)
        typewriterTimerRef.current = null
        isDoneRef.current = false
        useChatStore
          .getState()
          .updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, assistantMsgIdRef.current, (msg) => ({
            ...msg,
            thinkingExpanded: false,
          }))
        useChatStore.getState().setLoadingForMode(requestModeRef.current, false)
        useChatStore.getState().checkAndSummarizeForMode(requestModeRef.current)
      }
    }, 30)
  }, [])

  // 用户点击停止：中断 SSE + flush 打字机队列
  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    stopTypewriter(true)
    useChatStore.getState().updateMessageById(assistantMsgIdRef.current, (msg) => ({ ...msg, thinkingExpanded: false }))
    setLoadingForMode(requestModeRef.current, false)
  }, [stopTypewriter, setLoadingForMode])

  const handleSend = async () => {
    if (!inputValue.trim() || loading) return
    const userMessage = inputValue.trim()
    const m = mode
    let sid = useChatStore.getState().perMode[m].sessionId
    if (!sid) {
      try {
        await useChatStore.getState().hydrateSessions(m)
      } catch {
        // ignore; fallback to creating a new session below
      }
      sid = useChatStore.getState().perMode[m].sessionId
    }
    if (!sid) sid = await useChatStore.getState().createNewSession(m)
    setInputValue('')
    // 先取上下文（滑动窗口+pinned），再 addMessage，避免把当前问题算进历史
    const contextHistory = useChatStore.getState().getContextMessagesForMode(m)
      .filter((msg): msg is { role: 'user' | 'assistant'; content: string } =>
        msg.role === 'user' || msg.role === 'assistant'
      )
    requestModeRef.current = m
    requestSessionIdRef.current = sid
    addMessageForSession(m, sid, { role: 'user', content: userMessage })
    setLoadingForMode(m, true)

    if (streamMode) {
      streamContentRef.current = ''
      tokenQueueRef.current = []
      isDoneRef.current = false
      // 捕获 assistant 消息 ID，后续所有回调通过 ID 更新，跨 tab 安全
      assistantMsgIdRef.current = addMessageForSession(m, sid, { role: 'assistant', content: '', thinkingSteps: [], thinkingExpanded: true })
      const aid = assistantMsgIdRef.current
      // 关键：agent 流式可能不返回 token（只返回事件/最终 result），此时若不启动打字机，
      // 收尾逻辑（loading=false）不会触发，导致输入框一直灰。
      startTypewriter()
      abortControllerRef.current = chatAPI.sendMessageStream(
        {
          message: userMessage,
          mode: m,
          sessionId: sid,
          userId,
          useGraph: graphMode,
          chatHistory: contextHistory,
          modelProvider,
          modelName,
        },
        {
          onToken: (token) => { tokenQueueRef.current.push(...token.split('')); startTypewriter() },
          onThinking: (step) => {
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg, thinkingSteps: [...(msg.thinkingSteps ?? []), step], thinkingExpanded: true,
            }))
          },
          onAgentStep: (step) => {
            const s = step as any
            const text = formatAgentStep(s)
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg,
              thinkingSteps: [...(msg.thinkingSteps ?? []), text],
              thinkingExpanded: true,
            }))
          },
          onIntent: (intent) => {
            const i = intent as Record<string, string>
            const steps: string[] = []
            if (i?.raw_question) steps.push(`🧾 原始问题: ${i.raw_question}`)
            if (i?.resolved_question) steps.push(`🔗 消解问题: ${i.resolved_question}`)
            if (i?.retrieval_query) steps.push(`🔎 检索问题: ${i.retrieval_query}`)
            if (steps.length > 0) updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg, thinkingSteps: [...(msg.thinkingSteps ?? []), ...steps],
            }))
          },
          onSession: (id) => useChatStore.getState().setSessionIdForMode(requestModeRef.current, id),
          onSources: (sources) => updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({ ...msg, sources })),
          onResult: (report) => {
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg,
              report,
              trace: report?.trace,
              content: report?.summary ?? msg.content,
            }))
          },
          onDone: () => { isDoneRef.current = true; startTypewriter() },
          onError: (error) => {
            stopTypewriter(true)
            message.error(`生成错误: ${error}`)
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, assistantMsgIdRef.current, (msg) => ({ ...msg, thinkingExpanded: false }))
            setLoadingForMode(requestModeRef.current, false)
          },
          onRetry: (attempt, delayMs) =>
            message.warning(`连接断开，${delayMs / 1000}s 后重连（第 ${attempt} 次）`),
        }
      )
    } else {
      chatAPI.sendMessage({
        message: userMessage,
        mode: m,
        sessionId: sid,
        userId,
        useGraph: graphMode,
        modelProvider,
        modelName,
      })
        .then((response) => {
          const store = useChatStore.getState()
          store.setSessionIdForMode(m, response.sessionId)
          addMessageForSession(m, sid, { role: 'assistant', content: response.answer, sources: response.sources, trace: response.trace, report: response.report })
          store.checkAndSummarizeForMode(m)
        })
        .catch(() => message.error('发送失败，请重试'))
        .finally(() => setLoadingForMode(m, false))
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
      <MessageList messages={messages} mode={mode} onToggleThinking={handleToggleThinking} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
        <ThunderboltOutlined />
        <Text type="secondary" style={{ fontSize: 12 }}>流式输出</Text>
        <Switch size="small" checked={streamMode} onChange={setStreamMode} disabled={loading} />
        <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>图谱检索</Text>
        <Switch size="small" checked={graphMode} onChange={setGraphMode} disabled={mode !== 'rag' || loading} />
        <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>模型</Text>
        <Select
          size="small"
          value={modelProvider}
          style={{ width: 110 }}
          disabled={loading}
          options={[
            { value: 'qwen', label: 'Qwen' },
            { value: 'deepseek', label: 'DeepSeek' },
          ]}
          onChange={(v) => setModelProvider(v as ModelProvider)}
        />
        <Select
          size="small"
          value={modelName}
          style={{ width: 170 }}
          disabled={loading}
          options={modelOptions}
          onChange={(v) => setModelName(v as ModelName)}
        />
        {messages.length > 0 && (
          <Text type="secondary" style={{ fontSize: 11, marginLeft: 'auto' }}>
            {messages.filter((m) => m.role === 'user').length} 轮对话
          </Text>
        )}
      </div>
      <Space.Compact style={{ width: '100%', marginTop: 8 }}>
        <TextArea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={mode === 'rag' ? '请输入医疗相关问题...' : '请输入病历信息...'}
          autoSize={{ minRows: 2, maxRows: 4 }}
          onPressEnter={(e) => {
            if (!e.shiftKey) { e.preventDefault(); handleSend() }
          }}
          disabled={loading}
        />
        {loading && streamMode ? (
          <Button danger icon={<StopOutlined />} onClick={handleStop} style={{ height: 'auto' }}>
            停止
          </Button>
        ) : (
          <Button
            type="primary"
            icon={loading ? <Spin size="small" /> : <SendOutlined />}
            onClick={handleSend}
            disabled={loading || !inputValue.trim()}
            style={{ height: 'auto' }}
          >
            发送
          </Button>
        )}
      </Space.Compact>
    </div>
  )
}

export default ChatBox
