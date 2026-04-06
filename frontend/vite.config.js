import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  // ✅ Tell Vite NOT to bundle onnxruntime-web — let it load its own wasm files
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },

  server: {
    headers: {
      // ✅ Required for SharedArrayBuffer (ONNX multi-thread mode)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      // ✅ Allow Vite to serve files from node_modules
      allow: ['..'],
    },
  },

  build: {
    rollupOptions: {
      external: [],
    },
  },
})