/* global process */
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'


const backendApiUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
const backendWsUrl = process.env.BACKEND_WS_URL || 'ws://localhost:8000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    watch: {
      usePolling: true,
    },
    proxy: {
      '/api': {
        target: backendApiUrl,
        changeOrigin: true,
      },
      '/ws': {
        target: backendWsUrl,
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
