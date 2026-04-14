import { useState } from 'react'
import { Button, Dropdown, Input, List, Modal, Space, Typography } from 'antd'
import type { MenuProps } from 'antd'
import { MoreOutlined } from '@ant-design/icons'
import type { ChatMode } from '../types/shared'
import type { SessionListItem } from '../services/sessionsApi'
import { useChatStore } from '../store/chatStore'

const { Text } = Typography

export default function SessionSidebar({ mode }: { mode: ChatMode }) {
  const {
    sessions,
    activeSessionIdByMode,
    hydrateSessions,
    setActiveSession,
    createNewSession,
    renameSession,
    archiveSession,
  } = useChatStore()
  const list = sessions[mode]
  const active = activeSessionIdByMode[mode]

  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [renameOpen, setRenameOpen] = useState(false)
  const [renameTarget, setRenameTarget] = useState<SessionListItem | null>(null)
  const [renameTitle, setRenameTitle] = useState('')
  const [renameLoading, setRenameLoading] = useState(false)

  const openRename = (item: SessionListItem) => {
    setRenameTarget(item)
    setRenameTitle(item.title || '新会话')
    setRenameOpen(true)
  }

  const submitRename = async () => {
    if (!renameTarget) return
    const t = renameTitle.trim()
    if (!t) return
    setRenameLoading(true)
    try {
      await renameSession(mode, renameTarget.sessionId, t)
      setRenameOpen(false)
      setRenameTarget(null)
    } finally {
      setRenameLoading(false)
    }
  }

  const confirmArchive = (item: SessionListItem) => {
    const name = (item.title || '新会话').trim() || '新会话'
    Modal.confirm({
      title: `删除会话「${name}」`,
      content: `将删除（归档）会话「${name}」，并从列表中移除。确定吗？`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: () => archiveSession(mode, item.sessionId),
    })
  }

  const rowMenu = (item: SessionListItem): MenuProps => ({
    items: [
      {
        key: 'rename',
        label: '重命名',
        onClick: ({ domEvent }) => {
          domEvent.stopPropagation()
          openRename(item)
        },
      },
      {
        key: 'archive',
        label: '删除',
        danger: true,
        onClick: ({ domEvent }) => {
          domEvent.stopPropagation()
          confirmArchive(item)
        },
      },
    ],
  })

  return (
    <div style={{ padding: 12 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Button type="primary" block onClick={() => createNewSession(mode)}>
          + 新会话
        </Button>

        <Button block onClick={() => hydrateSessions(mode)}>
          刷新列表
        </Button>

        <List
          size="small"
          dataSource={list}
          locale={{ emptyText: '暂无会话' }}
          renderItem={(item) => (
            <List.Item
              style={{
                cursor: 'pointer',
                borderRadius: 6,
                padding: '8px 10px',
                background: item.sessionId === active ? '#e6f7ff' : undefined,
              }}
              onMouseEnter={() => setHoveredId(item.sessionId)}
              onMouseLeave={() => setHoveredId(null)}
            >
              <div
                style={{ display: 'flex', alignItems: 'flex-start', gap: 8, width: '100%', minWidth: 0 }}
              >
                <div
                  style={{ flex: 1, minWidth: 0 }}
                  onClick={() => setActiveSession(mode, item.sessionId)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setActiveSession(mode, item.sessionId)
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <Text strong style={{ display: 'block' }}>
                    {item.title || '新会话'}
                  </Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {new Date(item.updatedAt).toLocaleString()}
                  </Text>
                </div>
                <Dropdown menu={rowMenu(item)} trigger={['click']} placement="bottomRight">
                  <Button
                    type="text"
                    size="small"
                    icon={<MoreOutlined />}
                    aria-label="会话操作"
                    style={{
                      flexShrink: 0,
                      opacity: hoveredId === item.sessionId ? 1 : 0,
                      pointerEvents: hoveredId === item.sessionId ? 'auto' : 'none',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                </Dropdown>
              </div>
            </List.Item>
          )}
        />
      </Space>

      <Modal
        title="重命名会话"
        open={renameOpen}
        onOk={submitRename}
        onCancel={() => {
          setRenameOpen(false)
          setRenameTarget(null)
        }}
        confirmLoading={renameLoading}
        destroyOnClose
      >
        <Input
          value={renameTitle}
          onChange={(e) => setRenameTitle(e.target.value)}
          placeholder="会话标题"
          onPressEnter={submitRename}
          maxLength={120}
          showCount
        />
      </Modal>
    </div>
  )
}
