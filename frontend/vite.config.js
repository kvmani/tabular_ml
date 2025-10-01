import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/data': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/preprocess': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/model': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/visualization': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/system': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
});
