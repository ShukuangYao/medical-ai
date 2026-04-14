import { Card, Typography, Collapse, Tag, Space, Descriptions, Divider } from 'antd'
import { UserOutlined, RobotOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import { useEffect, useRef } from 'react'
import type { Message, ChatMode } from '../types/shared'

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
  const triageColor = (level?: string) => {
    if (level === 'emergency') return 'red'
    if (level === 'urgent') return 'orange'
    return 'green'
  }

  // 切换 tab 或消息变化时自动滚动到底部
  useEffect(() => {
    // 等待 DOM 更新后再滚动更稳定
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    })
  }, [mode, messages.length])

  return (
    <div style={{ flex: 1, overflowY: 'auto', paddingRight: 8 }}>
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
                          {step}
                        </Text>
                      ))}
                    </Space>
                  </Panel>
                </Collapse>
              )}

              <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p style={{ margin: '0 0 6px 0' }}>{children}</p>,
                    ul: ({ children }) => <ul style={{ margin: '0 0 6px 18px', padding: 0 }}>{children}</ul>,
                    ol: ({ children }) => <ol style={{ margin: '0 0 6px 18px', padding: 0 }}>{children}</ol>,
                    li: ({ children }) => <li style={{ margin: '2px 0' }}>{children}</li>,
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </Paragraph>

              {mode === 'agent' && msg.role === 'assistant' && msg.report && (
                <>
                  <Divider style={{ margin: '8px 0' }} />
                  <Space direction="vertical" style={{ width: '100%' }} size={8}>
                    <Card size="small" type="inner" title="紧急程度">
                      <Space size={8} wrap>
                        <Tag color={triageColor(msg.report.triage?.severity_level)}>
                          {msg.report.triage?.severity_level ?? 'routine'}
                        </Tag>
                        <Text type="secondary">{msg.report.triage?.why}</Text>
                      </Space>
                      {msg.report.triage?.red_flags?.length ? (
                        <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
                          <Text strong>红旗征：</Text> {msg.report.triage.red_flags.join('；')}
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
                          {msg.report.department?.reason || '—'}
                        </Descriptions.Item>
                      </Descriptions>
                    </Card>

                    <Card size="small" type="inner" title="下一步举措">
                      <Descriptions size="small" column={1}>
                        <Descriptions.Item label="立即措施">
                          {(msg.report.next_steps?.immediate_actions ?? []).join('；') || '—'}
                        </Descriptions.Item>
                        <Descriptions.Item label="建议检查">
                          {(msg.report.next_steps?.recommended_tests ?? []).join('；') || '—'}
                        </Descriptions.Item>
                        <Descriptions.Item label="就医时机">
                          {(msg.report.next_steps?.when_to_seek_care ?? []).join('；') || '—'}
                        </Descriptions.Item>
                      </Descriptions>
                    </Card>
                  </Space>
                </>
              )}

              {msg.sources && msg.sources.length > 0 && (
                <Collapse ghost>
                  <Panel header={`参考文献 (${msg.sources.length})`} key="sources">
                    <Space direction="vertical" style={{ width: '100%' }} size={6}>
                      {msg.sources.map((source, idx) => (
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
                            {source.content}
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
