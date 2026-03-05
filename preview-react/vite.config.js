import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/preview/",
  server: {
    proxy: {
      "/adventure": "http://localhost:7862",
      "/ai": "http://localhost:7862",
      "/rag": "http://localhost:7862",
      "/brain": "http://localhost:7862",
      "/tts": "http://localhost:7862",
      "/voices": "http://localhost:7862",
      "/npc": "http://localhost:7862",
      "/campaign-assets": "http://localhost:7862",
    },
  },
  build: {
    outDir: "../static/preview-react",
    emptyOutDir: true,
  },
});
