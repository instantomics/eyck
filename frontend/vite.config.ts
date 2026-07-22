import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/assets/",
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL("../src/eyck/web/assets", import.meta.url)),
    assetsDir: "",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "eyck-[hash].js",
        assetFileNames: "eyck-[hash][extname]",
        chunkFileNames: "eyck-[hash].js",
        manualChunks(id) {
          if (id.includes("/node_modules/regl/")) return "regl";
          if (id.includes("/node_modules/react/") || id.includes("/node_modules/react-dom/")) {
            return "react";
          }
        }
      }
    }
  }
});
