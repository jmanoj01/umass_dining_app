/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      // Only custom colors below; use Tailwind's built-in gray/neutral palette
      colors: {
        maroon: '#881c1c',
        orange: '#FFA94D',
        yellow: '#FFD166',
        green: '#8DC63F',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
      },
    },
  },
  plugins: [],
}
