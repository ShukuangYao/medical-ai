import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from 'antd'
import Home from './pages/Home'
import Analysis from './pages/Analysis'
import './App.css'

function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Routes>
        <Route path="/" element={<Navigate to="/home" replace />} />
        <Route path="/home" element={<Home />} />
        <Route path="/analysis" element={<Analysis />} />
      </Routes>
    </Layout>
  )
}

export default App
