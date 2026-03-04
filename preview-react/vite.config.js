import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/preview/",
  build: {
    outDir: "../static/preview-react",
    emptyOutDir: true,
  },
});
