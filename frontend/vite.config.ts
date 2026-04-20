import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        // Agent `full` can exceed default proxy/socket limits (align with axios 600s).
        timeout: 600000,
        proxyTimeout: 600000,
      }
    }
  }
})
