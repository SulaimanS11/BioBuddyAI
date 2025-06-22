import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/BioBuddyReact/', // MUST match your repo name
  build: {
    outDir: 'dist', // Changed from 'docs' to 'dist'
    emptyOutDir: true,
  }
})