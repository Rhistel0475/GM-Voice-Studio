import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

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
            next();
          });
        },
      },
    ],
    proxy: {
      "/api": "http://localhost:7862",
      "/adventure": "http://localhost:7862",
      "/ai": "http://localhost:7862",
      "/rag": "http://localhost:7862",
      "/brain": "http://localhost:7862",
      "/tts": "http://localhost:7862",
      "/voices": "http://localhost:7862",
      "/npc": "http://localhost:7862",
      "/campaign-assets": "http://localhost:7862",
      "/static": "http://localhost:7862",
      "/ws": {
        target: "ws://localhost:7862",
        ws: true,
      },
    },
  },
  build: {
    outDir: "../static/preview-react",
    emptyOutDir: true,
  },
});
