import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/tabular_ml/assets/",
  build: {
    outDir: "../src/tabular_ml_service/static",
    emptyOutDir: false,
    sourcemap: false,
    rollupOptions: {
      input: "./src/main.tsx",
      output: {
        entryFileNames: "ui/app.js",
        chunkFileNames: "ui/[name]-[hash].js",
        assetFileNames: "ui/[name][extname]",
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/tabular_ml/api": "http://127.0.0.1:5070",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
  },
});
