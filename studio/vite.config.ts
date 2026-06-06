import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built SPA is emitted INTO the Python package (himmy/api/_studio_static) so it
// ships as package data and `himmy studio` serves it. `outDir` is resolved relative
// to this config's directory (studio/). In dev, Vite runs on :5173 and proxies the
// API to the FastAPI BFF on :8000, so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../himmy/api/_studio_static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
