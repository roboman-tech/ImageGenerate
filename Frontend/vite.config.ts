import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8081,
    host: "0.0.0.0"
  },
  preview: {
    port: 4173,
  }
});
