import { defineConfig } from "vite";

// PixiJS v8 setup uses top-level await, so target modern browsers.
export default defineConfig({
  build: { target: "es2022" },
  // In development, the game server runs separately on :8000.
  server: { proxy: { "/ws": { target: "ws://localhost:8000", ws: true }, "/api": "http://localhost:8000" } },
});
