import { defineConfig } from "vite";

// PixiJS v8 setup uses top-level await, so target modern browsers.
export default defineConfig({
  build: { target: "es2022" },
});
