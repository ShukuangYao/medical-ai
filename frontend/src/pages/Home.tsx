import { useEffect } from 'react'
import { Layout, Menu, Card, Typography } from 'antd'
import { MessageOutlined, ExperimentOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import ChatBox from '../components/ChatBox'
import SessionSidebar from '../components/SessionSidebar'
import { useChatStore } from '../store/chatStore'

const { Header, Content, Sider } = Layout
const { Title } = Typography

function Home() {
  const navigate = useNavigate()
  const { mode, setMode, hydrateSessions } = useChatStore()

  // 关键：刷新/直达路由时，强制把 store.mode 与页面一致
  useEffect(() => {
    if (mode !== 'rag') setMode('rag')
  }, [mode, setMode])

  useEffect(() => {
    hydrateSessions('rag')
  }, [hydrateSessions])

  const handleMenuClick = (key: string) => {
    if (key === 'agent') {
      navigate('/analysis')
    } else {
      setMode('rag')
    }
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
            医疗AI辅助诊断系统
          </Title>
          <Menu
            mode="horizontal"
            selectedKeys={[mode]}
            onClick={({ key }) => handleMenuClick(key)}
            items={[
              {
                key: 'rag',
                icon: <MessageOutlined />,
                label: '普通问答',
              },
              {
                key: 'agent',
                icon: <ExperimentOutlined />,
                label: '病历分析',
              },
            ]}
          />
        </div>
      </Header>
      <Layout>
        <Sider width={260} theme="light" style={{ borderRight: '1px solid #f0f0f0' }}>
          <SessionSidebar mode="rag" />
        </Sider>
        <Content style={{ padding: '24px', background: '#f0f2f5' }}>
          <Card style={{ maxWidth: 1200, margin: '0 auto' }}>
            <ChatBox mode="rag" />
          </Card>
        </Content>
      </Layout>
    </Layout>
  )
}

export default Home
