import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/** FastAPI dev server (match PORT in app/core/config.py, default 7862). Use 127.0.0.1 to avoid IPv6 localhost issues. */
const API_DEV = "http://127.0.0.1:7862";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/preview/",
  server: {
    middlewareMode: false,
    plugins: [
      {
        name: "redirect-preview",
        configureServer(server) {
          server.middlewares.use((req, res, next) => {
            if (req.url === "/preview") {
              res.writeHead(301, { Location: "/preview/" });
              res.end();
              return;
            }
            // SPA fallback: rewrite /preview/* (no static file) to / so Vite serves index.html
            const url = req.url?.split("?")[0] || "";
            if (url.startsWith("/preview") && !/\.(js|css|ico|png|svg|json|map)(\?|$)/i.test(url)) {
              req.url = "/";
              return next();
            }
            next();
          });
        },
      },
    ],
    proxy: {
      "/api": API_DEV,
      "/adventure": API_DEV,
      "/ai": API_DEV,
      "/rag": API_DEV,
      "/brain": API_DEV,
      "/tts": API_DEV,
      "/voices": API_DEV,
      "/npc": API_DEV,
      "/npcs": API_DEV,
      "/config": API_DEV,
      "/campaign": API_DEV,
      "/campaigns": API_DEV,
      "/session": API_DEV,
      "/sessions": API_DEV,
      "/session-assistant": API_DEV,
      "/scene": API_DEV,
      "/encounter": API_DEV,
      "/jobs": API_DEV,
      "/campaign-assets": API_DEV,
      "/static": API_DEV,
      "/ws": {
        target: "ws://127.0.0.1:7862",
        ws: true,
      },
    },
  },
  build: {
    outDir: "../static/frontend",
    emptyOutDir: true,
  },
});
