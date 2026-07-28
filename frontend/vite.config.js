import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxy = process.env.VITE_API_PROXY || 'http://localhost:8000'
const hmrHost = process.env.VITE_HMR_HOST || 'localhost'
const hmrClientPort = Number(process.env.VITE_HMR_CLIENT_PORT || 5173)

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    watch: {
      usePolling: true,
      interval: 200,
    },
    hmr: {
      protocol: 'ws',
      host: hmrHost,
      clientPort: hmrClientPort,
    },
    proxy: {
      '/api': {
        target: apiProxy,
        changeOrigin: true,
      },
      '/health': {
        target: apiProxy,
        changeOrigin: true,
      },
    },
  },
})
