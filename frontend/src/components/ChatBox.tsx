import { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import { Input, Button, Space, message, Spin, Switch, Typography, Select, Tooltip } from 'antd'
import { SendOutlined, ThunderboltOutlined, StopOutlined } from '@ant-design/icons'
import { useChatStore } from '../store/chatStore'
import { chatAPI } from '../services/api'
import MessageList from './MessageList'
import type { AgentPipeline, ChatMode, ModelName, ModelProvider } from '../types/shared'

const { TextArea } = Input
const { Text } = Typography

interface ChatBoxProps {
  mode: ChatMode
}

function ChatBox({ mode }: ChatBoxProps) {
  const [inputValue, setInputValue] = useState('')
  const [streamMode, setStreamMode] = useState(true)
  const [graphMode, setGraphMode] = useState(true)
  /** Agent: full restores legacy multi-step agents (richer but slower). */
  const [agentDetailMode, setAgentDetailMode] = useState(false)
  const [stopping, setStopping] = useState(false)
  const streamContentRef = useRef('')
  const tokenQueueRef = useRef<string[]>([])
  const typewriterTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isDoneRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  /** Node `run_id` from first SSE envelope; POST /api/cancel before abort. */
  const currentRunIdRef = useRef<string>('')
  /** 记录本次请求的 mode，用于跨 tab 场景下回调仍写入正确 tab */
  const requestModeRef = useRef<ChatMode>(mode)
  /** 记录本次请求的 sessionId，确保回调写入正确会话 */
  const requestSessionIdRef = useRef<string>('')
  /** 记录本次 assistant 消息 ID，用于跨 tab 安全更新 */
  const assistantMsgIdRef = useRef<string>('')

  // 读取当前 tab（mode prop）的状态
  const { messages, loading, sessionId } = useChatStore((s) => s.perMode[mode])
  const { addMessageForSession, updateMessageByIdForSession, setLoadingForMode } = useChatStore()
  const { modelProvider, modelName, setModelProvider, setModelName } = useChatStore()
  const { userId } = useChatStore()

  // Persist agent "detailed analysis" toggle across page reloads (demo-friendly preference).
  useEffect(() => {
    const uid = (userId || '').trim() || 'anonymous'
    const key = `medical-ai:pref:agentDetailMode:${uid}`
    try {
      const raw = window.localStorage.getItem(key)
      if (raw === null) return
      const v = raw === '1' || raw.toLowerCase() === 'true'
      setAgentDetailMode(v)
    } catch {
      // ignore
    }
  }, [userId])

  const handleSetAgentDetailMode = useCallback(
    (v: boolean) => {
      setAgentDetailMode(v)
      const uid = (userId || '').trim() || 'anonymous'
      const key = `medical-ai:pref:agentDetailMode:${uid}`
      try {
        window.localStorage.setItem(key, v ? '1' : '0')
      } catch {
        // ignore
      }
    },
    [userId]
  )

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
        { value: 'qwen3.5-flash', label: 'qwen3.5-flash' },
        { value: 'qwen3.5-plus', label: 'qwen3.5-plus' },
      ] as Array<{ value: ModelName; label: string }>
    }
    return [
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
    if (flush) {
      if (tokenQueueRef.current.length > 0) {
        streamContentRef.current += tokenQueueRef.current.join('')
        tokenQueueRef.current = []
      }
      useChatStore
        .getState()
        .updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, assistantMsgIdRef.current, (msg) => ({
          ...msg,
          content: streamContentRef.current,
          typewriterDone: true,
        }))
    }
    isDoneRef.current = false
  }, [])

  // 启动打字机：20ms/tick，每次出队 10 个字符；队列耗尽且流结束时自动终止
  const startTypewriter = useCallback(() => {
    if (typewriterTimerRef.current) return
    typewriterTimerRef.current = setInterval(() => {
      if (tokenQueueRef.current.length > 0) {
        const chars = tokenQueueRef.current.splice(0, 10).join('')
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
            typewriterDone: true,
          }))
        useChatStore.getState().setLoadingForMode(requestModeRef.current, false)
        useChatStore.getState().checkAndSummarizeForMode(requestModeRef.current)
      }
    }, 20)
  }, [])

  // 用户点击停止：先等取消到达 Python，再断 SSE；快照 ref 避免 await 期间新请求覆盖
  const handleStop = useCallback(async () => {
    if (stopping) return
    setStopping(true)
    const rid = currentRunIdRef.current.trim()
    const ctrl = abortControllerRef.current
    const modeAtStop = requestModeRef.current
    const aid = assistantMsgIdRef.current

    try {
      if (rid) {
        try {
          await fetch('/api/cancel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: rid }),
            signal: AbortSignal.timeout(150_000),
          })
        } catch {
          // 网络错误 / 150s 超时：仍断开客户端流，避免挂死
        }
      }

      if (currentRunIdRef.current === rid) {
        currentRunIdRef.current = ''
      }
      ctrl?.abort()
      if (abortControllerRef.current === ctrl) {
        abortControllerRef.current = null
      }
      stopTypewriter(true)
      useChatStore.getState().updateMessageById(aid, (msg) => ({ ...msg, thinkingExpanded: false }))
      setLoadingForMode(modeAtStop, false)
    } finally {
      setStopping(false)
    }
  }, [stopTypewriter, setLoadingForMode, stopping])

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

    const agentPipeline: AgentPipeline | undefined = m === 'agent' ? (agentDetailMode ? 'full' : 'fast') : undefined

    if (streamMode) {
      setStopping(false)
      currentRunIdRef.current = ''
      streamContentRef.current = ''
      tokenQueueRef.current = []
      isDoneRef.current = false
      // 捕获 assistant 消息 ID，后续所有回调通过 ID 更新，跨 tab 安全
      assistantMsgIdRef.current = addMessageForSession(m, sid, {
        role: 'assistant',
        content: '',
        thinkingSteps: [],
        thinkingExpanded: true,
        streaming: true,
        typewriterDone: false,
      })
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
          ...(m === 'rag' ? { modelProvider, modelName } : {}),
          agentPipeline,
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
          onRunId: (runId) => {
            const rid = runId.trim()
            if (!rid) return
            currentRunIdRef.current = rid
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg,
              runId: rid,
            }))
          },
          onSession: (id) => {
            requestSessionIdRef.current = id
            useChatStore.getState().setSessionIdForMode(requestModeRef.current, id)
          },
          onSources: (sources) => updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({ ...msg, sources })),
          onResult: (report) => {
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg,
              report,
              trace: report?.trace,
              content: report?.summary ?? msg.content,
            }))
          },
          onDone: () => {
            currentRunIdRef.current = ''
            setStopping(false)
            isDoneRef.current = true
            const queueEmpty = tokenQueueRef.current.length === 0
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, aid, (msg) => ({
              ...msg,
              streaming: false,
              // Keep typewriterDone=false until local queue is flushed; prevents Markdown tail-trim flicker.
              typewriterDone: queueEmpty,
            }))
            if (queueEmpty) {
              // No local typewriter backlog: finish UI immediately.
              useChatStore.getState().setLoadingForMode(requestModeRef.current, false)
              useChatStore.getState().checkAndSummarizeForMode(requestModeRef.current)
            }
            startTypewriter()
          },
          onError: (error) => {
            currentRunIdRef.current = ''
            setStopping(false)
            stopTypewriter(true)
            message.error(`生成错误: ${error}`)
            updateMessageByIdForSession(requestModeRef.current, requestSessionIdRef.current, assistantMsgIdRef.current, (msg) => ({ ...msg, thinkingExpanded: false, streaming: false }))
            setLoadingForMode(requestModeRef.current, false)
          },
          onRetry: (attempt, delayMs) =>
            message.warning(`连接断开，${delayMs / 1000}s 后重连（第 ${attempt} 次）`),
        }
      )
    } else {
      // Non-stream: still create an assistant placeholder so UI never "looks empty" if request is slow/fails.
      const pendingId = addMessageForSession(m, sid, { role: 'assistant', content: '正在生成中，请稍候…', thinkingSteps: [], thinkingExpanded: false })
      chatAPI.sendMessage({
        message: userMessage,
        mode: m,
        sessionId: sid,
        userId,
        useGraph: graphMode,
        ...(m === 'rag' ? { modelProvider, modelName } : {}),
        agentPipeline,
      })
        .then((response) => {
          const store = useChatStore.getState()
          store.setSessionIdForMode(m, response.sessionId)
          const text =
            (response.answer && String(response.answer).trim()) ||
            (response.report && typeof response.report === 'object' && (response.report as { summary?: string }).summary
              ? String((response.report as { summary?: string }).summary).trim()
              : '') ||
            '（未返回摘要；若有结构化报告请展开下方卡片）'
          updateMessageByIdForSession(m, sid, pendingId, (msg) => ({
            ...msg,
            content: text,
            sources: response.sources,
            trace: response.trace,
            report: response.report,
          }))
          store.checkAndSummarizeForMode(m)
        })
        .catch((err: unknown) => {
          const ax = err as {
            message?: string
            code?: string
            response?: { status?: number; data?: { detail?: string; error?: string } }
          }
          const detail =
            typeof ax?.response?.data?.detail === 'string'
              ? ax.response.data.detail
              : typeof ax?.response?.data?.error === 'string'
                ? ax.response.data.error
                : ''
          const hint =
            detail ||
            (ax?.response?.status != null ? `HTTP ${ax.response.status}` : '') ||
            ax?.code ||
            ax?.message ||
            String(err)
          message.error(`发送失败：${hint}`)
          updateMessageByIdForSession(m, sid, pendingId, (msg) => ({
            ...msg,
            content: `请求失败（${hint}）。可试流式或稍后重试。`,
          }))
        })
        .finally(() => setLoadingForMode(m, false))
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
      <MessageList
        messages={messages}
        mode={mode}
        sessionId={sessionId ?? undefined}
        userId={userId}
        onToggleThinking={handleToggleThinking}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
        <ThunderboltOutlined />
        <Text type="secondary" style={{ fontSize: 12 }}>流式输出</Text>
        <Switch size="small" checked={streamMode} onChange={setStreamMode} disabled={loading} />
        <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>图谱检索</Text>
        <Switch size="small" checked={graphMode} onChange={setGraphMode} disabled={mode !== 'rag' || loading} />
        {mode === 'agent' ? (
          <>
            <Tooltip title={agentDetailMode ? '已开启：full 管线（更慢、更多阶段与中间产物）' : '未开启：fast 管线（更快、步骤更少）'}>
              <Text type="secondary" style={{ fontSize: 12, marginLeft: 8, cursor: 'help' }}>详细分析</Text>
            </Tooltip>
            <Tooltip title={agentDetailMode ? 'full：更详细但更慢' : 'fast：更快但更简略'}>
              <Switch size="small" checked={agentDetailMode} onChange={handleSetAgentDetailMode} disabled={loading} />
            </Tooltip>
          </>
        ) : null}
        {mode === 'rag' ? (
          <>
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
          </>
        ) : null}
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
          <Button
            danger
            icon={<StopOutlined />}
            onClick={handleStop}
            loading={stopping}
            disabled={stopping}
            style={{ height: 'auto' }}
          >
            {stopping ? '正在停止…' : '停止'}
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
