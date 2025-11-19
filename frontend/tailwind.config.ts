import type { Config } from "tailwindcss";
export default {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: { brand: { 400: "#3b82f6", 500: "#60a5fa", 600: "#1e40af" } },
      boxShadow: { glow: "0 0 20px rgba(59,130,246,0.4)" }
    },
  },
  plugins: [],
} satisfies Config;
