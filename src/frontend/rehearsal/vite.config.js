// Import Vite's React plugin and Tailwind plugin
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'url'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(), // enables Tailwind utility classes
  ],
  // Shared .env lives at the monorepo root, not per-app
  envDir: path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..'),
})