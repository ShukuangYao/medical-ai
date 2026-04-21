import { Card, Typography, Collapse, Tag, Space, Descriptions, Divider } from 'antd'
import { UserOutlined, RobotOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import { useLayoutEffect, useMemo, useRef } from 'react'
import type { Message, ChatMode } from '../types/shared'
import { sanitizeDisplayText } from '../utils/markdownSanitize'

const { Text, Paragraph } = Typography
const { Panel } = Collapse

const sourceTagStyle: Record<string, { label: string; color: string }> = {
  graph: { label: '图谱', color: 'green' },
  vector: { label: '向量', color: 'gold' },
  elasticsearch: { label: 'ES', color: 'blue' },
  unknown: { label: '未知', color: 'default' },
}

interface MessageListProps {
  messages: Message[]
  mode: ChatMode
  onToggleThinking?: (messageIndex: number, expanded: boolean) => void
}

function MessageList({ messages, mode, onToggleThinking }: MessageListProps) {
  const pageText = (page?: number) => (typeof page === 'number' && page >= 1 ? `第${page}页` : null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const shouldStickRef = useRef(true)
  const triageColor = (level?: string) => {
    if (level === 'emergency') return 'red'
    if (level === 'urgent') return 'orange'
    return 'green'
  }

  // 切换 tab 或消息变化时自动滚动到底部（若用户手动上滑阅读，则不强制抢滚动条）
  // 注意：流式输出通常只是更新最后一条消息 content，不会让 messages.length 增长，
  // 因此需要把最后一条消息的变化也纳入依赖。
  const lastMsg = messages[messages.length - 1]
  const lastMsgKey = useMemo(
    () =>
      `${lastMsg?.id ?? ''}:${(lastMsg?.content ?? '').length}:${(lastMsg?.thinkingSteps?.length ?? 0)}:${lastMsg?.report ? 1 : 0}`,
    [lastMsg?.id, lastMsg?.content, lastMsg?.thinkingSteps, lastMsg?.report]
  )

  const scrollToBottom = () => {
    const el = containerRef.current
    if (!el) return
    // Set scrollTop directly is more reliable than scrollIntoView for streaming updates.
    el.scrollTop = el.scrollHeight
  }

  const recomputeStickiness = () => {
    const el = containerRef.current
    if (!el) return
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    shouldStickRef.current = distanceToBottom < 120
  }

  useLayoutEffect(() => {
    if (!shouldStickRef.current) return
    // Ensure layout finished before reading scrollHeight.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scrollToBottom()
      })
    })
  }, [mode, messages.length, lastMsgKey])

  const parsedAgentSummary = (summaryRaw?: string): { summary: string; disclaimer?: string } => {
    const t = (summaryRaw ?? '').trim()
    if (!t) return { summary: '' }
    if (!(t.startsWith('{') && t.endsWith('}'))) return { summary: t }
    try {
      const obj = JSON.parse(t) as unknown
      if (obj && typeof obj === 'object') {
        const o = obj as Record<string, unknown>
        const s = typeof o.summary === 'string' ? o.summary.trim() : ''
        const d = typeof o.disclaimer === 'string' ? o.disclaimer.trim() : undefined
        if (s) return { summary: s, disclaimer: d }
      }
      return { summary: t }
    } catch {
      return { summary: t }
    }
  }

  const reorderSourcesByRemappedRefs = (
    text: string,
    sources: Array<{ title: string; content: string; page?: number; retrieval_source?: string }>
  ) => {
    const maxRef = sources.length
    if (!text || maxRef <= 0) return sources

    // Extract original reference numbers in first-appearance order (before remapping),
    // then reorder sources so referenced ones come first.
    const s = String(text).replace(/[\u200B-\u200D\u2060\uFEFF]/g, '')
    const refs: number[] = []
    const add = (n: number) => {
      if (!Number.isFinite(n) || n < 1 || n > maxRef) return
      if (!refs.includes(n)) refs.push(n)
    }
    const scan = (re: RegExp) => {
      re.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = re.exec(s)) !== null) add(Number(m[1]))
    }
    scan(/\[\s*参考\s*([0-9]+)\s*\]/g)
    scan(/［\s*参考\s*([0-9]+)\s*］/g)
    scan(/（\s*参考\s*([0-9]+)\s*）/g)
    scan(/\(\s*参考\s*([0-9]+)\s*\)/g)

    if (refs.length === 0) return sources
    const picked = refs.map((n) => sources[n - 1]).filter(Boolean)
    const pickedSet = new Set(refs.map((n) => n - 1))
    const rest = sources.filter((_, i) => !pickedSet.has(i))
    return [...picked, ...rest]
  }

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, overflowY: 'auto', paddingRight: 8 }}
      onScroll={() => recomputeStickiness()}
      onWheel={() => recomputeStickiness()}
      onTouchMove={() => recomputeStickiness()}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="small">
        {messages.map((msg, index) => (
          <Card
            key={index}
            size="small"
            style={{
              background: msg.role === 'user' ? '#e6f7ff' : '#f6ffed',
              borderLeft: `4px solid ${msg.role === 'user' ? '#1890ff' : '#52c41a'}`,
              marginBottom: 6,
            }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size={6}>
              <Space size={6}>
                {msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                <Text strong>{msg.role === 'user' ? '用户' : 'AI助手'}</Text>
              </Space>
              {msg.role === 'assistant' && msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                <Collapse
                  ghost
                  activeKey={msg.thinkingExpanded ? ['thinking'] : []}
                  onChange={(keys) => {
                    const expanded = Array.isArray(keys) ? keys.includes('thinking') : keys === 'thinking'
                    onToggleThinking?.(index, expanded)
                  }}
                >
                  <Panel header={`思考过程 (${msg.thinkingSteps.length}步)`} key="thinking">
                    <Space direction="vertical" style={{ width: '100%' }} size="small">
                      {msg.thinkingSteps.map((step, idx) => (
                        <Text key={idx} type="secondary">
                          {sanitizeDisplayText(step)}
                        </Text>
                      ))}
                    </Space>
                  </Panel>
                </Collapse>
              )}

              {!(mode === 'agent' && msg.role === 'assistant' && msg.report) && (
                <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p className="md-p" style={{ margin: '0 0 6px 0' }}>{children}</p>,
                      ul: ({ children }) => (
                        <ul className="md-list md-unordered">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ul className="md-list md-ordered">
                          {children}
                        </ul>
                      ),
                      li: ({ children }) => <li className="md-li">{children}</li>,
                    }}
                  >
                    {sanitizeDisplayText(msg.content, {
                      // Disable tail trimming to avoid any end-of-stream flicker.
                      trimIncompleteTail: false,
                      maxRef: msg.sources?.length ?? 0,
                      remapRefs: true,
                    })}
                  </ReactMarkdown>
                </Paragraph>
              )}

              {mode === 'agent' && msg.role === 'assistant' && msg.report && (
                <>
                  <Divider style={{ margin: '8px 0' }} />
                  <Space direction="vertical" style={{ width: '100%' }} size={8}>
                    <Card size="small" type="inner" title="紧急程度">
                      <Space size={8} wrap>
                        <Tag color={triageColor(msg.report.triage?.severity_level)}>
                          {msg.report.triage?.severity_level ?? 'routine'}
                        </Tag>
                        <Text type="secondary">{sanitizeDisplayText(msg.report.triage?.why ?? '')}</Text>
                      </Space>
                      {msg.report.triage?.red_flags?.length ? (
                        <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
                          <Text strong>红旗征：</Text>{' '}
                          {msg.report.triage.red_flags.map((x) => sanitizeDisplayText(x)).join('；')}
                        </Paragraph>
                      ) : null}
                    </Card>

                    <Card size="small" type="inner" title="就诊科室">
                      <Descriptions size="small" column={1}>
                        <Descriptions.Item label="推荐">
                          {(msg.report.department?.recommended ?? []).join('、') || '—'}
                        </Descriptions.Item>
                        <Descriptions.Item label="备选">
                          {(msg.report.department?.alternatives ?? []).join('、') || '—'}
                        </Descriptions.Item>
                        <Descriptions.Item label="理由">
                          {sanitizeDisplayText(msg.report.department?.reason || '—')}
                        </Descriptions.Item>
                      </Descriptions>
                    </Card>

                    <Card size="small" type="inner" title="下一步举措">
                      <Descriptions size="small" column={1}>
                        <Descriptions.Item label="立即措施">
                          {(msg.report.next_steps?.immediate_actions ?? []).map((x) => sanitizeDisplayText(x)).join('；') || '—'}
                        </Descriptions.Item>
                        <Descriptions.Item label="建议检查">
                          {(msg.report.next_steps?.recommended_tests ?? []).map((x) => sanitizeDisplayText(x)).join('；') || '—'}
                        </Descriptions.Item>
                        <Descriptions.Item label="就医时机">
                          {(msg.report.next_steps?.when_to_seek_care ?? []).map((x) => sanitizeDisplayText(x)).join('；') || '—'}
                        </Descriptions.Item>
                      </Descriptions>
                    </Card>

                    <Card size="small" type="inner" title="摘要">
                      {(() => {
                        const p = parsedAgentSummary(msg.report?.summary)
                        return (
                          <Space direction="vertical" style={{ width: '100%' }} size={6}>
                            <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
                              <ReactMarkdown
                                components={{
                                  p: ({ children }) => <p className="md-p" style={{ margin: '0 0 6px 0' }}>{children}</p>,
                                  ul: ({ children }) => <ul className="md-list md-unordered">{children}</ul>,
                                  ol: ({ children }) => <ul className="md-list md-ordered">{children}</ul>,
                                  li: ({ children }) => <li className="md-li">{children}</li>,
                                }}
                              >
                                {sanitizeDisplayText(p.summary || '—')}
                              </ReactMarkdown>
                            </Paragraph>
                            {p.disclaimer ? (
                              <Text type="secondary">免责声明：{sanitizeDisplayText(p.disclaimer)}</Text>
                            ) : null}
                          </Space>
                        )
                      })()}
                    </Card>

                    <Collapse ghost>
                      <Panel header="结构化 JSON（可选）" key="report_json">
                        <pre
                          style={{
                            margin: 0,
                            padding: 12,
                            background: 'rgba(0,0,0,0.02)',
                            border: '1px solid rgba(0,0,0,0.06)',
                            borderRadius: 6,
                            overflowX: 'auto',
                            whiteSpace: 'pre',
                            fontSize: 12,
                            lineHeight: 1.5,
                          }}
                        >
                          {JSON.stringify(msg.report, null, 2)}
                        </pre>
                      </Panel>
                    </Collapse>
                  </Space>
                </>
              )}

              {msg.sources && msg.sources.length > 0 && (
                <Collapse ghost>
                  <Panel header={`参考文献 (${msg.sources.length})`} key="sources">
                    <Space direction="vertical" style={{ width: '100%' }} size={6}>
                      {reorderSourcesByRemappedRefs(msg.content, msg.sources).map((source, idx) => (
                        <Card key={idx} size="small" type="inner">
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                            <Text strong style={{ marginBottom: 0 }}>{source.title}</Text>
                            {source.retrieval_source && (
                              <Tag
                                color={(sourceTagStyle[source.retrieval_source] || sourceTagStyle.unknown).color}
                                style={{ marginInlineEnd: 0 }}
                              >
                                {(sourceTagStyle[source.retrieval_source] || sourceTagStyle.unknown).label}
                              </Tag>
                            )}
                            {pageText(source.page) && (
                              <Tag color="blue" style={{ marginInlineEnd: 0 }}>
                                {pageText(source.page)}
                              </Tag>
                            )}
                          </div>
                          <Paragraph
                            ellipsis={{ rows: 3, expandable: true }}
                            style={{ marginTop: 6, marginBottom: 0 }}
                          >
                            {sanitizeDisplayText(source.content)}
                          </Paragraph>
                        </Card>
                      ))}
                    </Space>
                  </Panel>
                </Collapse>
              )}

              {mode === 'agent' && msg.trace && msg.trace.length > 0 && (
                <Collapse ghost>
                  <Panel header={`推理过程 (${msg.trace.length}步)`} key="trace">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {msg.trace.map((item, idx) => (
                        <Card key={idx} size="small" type="inner">
                          <Tag color="purple">{item.agent}</Tag>
                          <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
                            {item.message}
                          </Paragraph>
                        </Card>
                      ))}
                    </Space>
                  </Panel>
                </Collapse>
              )}
            </Space>
          </Card>
        ))}
        <div ref={bottomRef} />
      </Space>
    </div>
  )
}

export default MessageList
