import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, host: '127.0.0.1' },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Keep three in its own chunk so the game code stays cache-friendly during iteration.
        manualChunks: { three: ['three'] },
      },
    },
  },
  // Shaders live in .ts files as template literals — no plugin, no loader, no external asset.
});
