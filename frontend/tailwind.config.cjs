/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: '#4F46E5',
          dark: '#312E81',
          light: '#6366F1'
        }
      },
      boxShadow: {
        panel: '0 20px 45px rgba(15, 23, 42, 0.35)'
      }
    }
  },
  plugins: []
};

