import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Tauri 开发时固定端口；端口被占用直接失败，不要悄悄换端口
export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  server: { port: 1430, strictPort: true },
});
