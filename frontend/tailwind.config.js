/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        panel: "#0f172a",   // slate-900-ish base for panels
      },
    },
  },
  plugins: [],
};
