/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './assets/js/**/*.js'],
  darkMode: 'media',
  theme: {
    extend: {
      colors: {
        ink:   { 50:'#f6f7f9', 100:'#eceef2', 200:'#d5dae2', 300:'#b0b9c8', 400:'#8492a8',
                 500:'#65748d', 600:'#505d74', 700:'#424c5e', 800:'#39414f', 900:'#333944', 950:'#1c2029' },
        brand: { 50:'#f4f2ff', 100:'#ebe7ff', 200:'#d9d2ff', 300:'#bcadff', 400:'#9a7dff',
                 500:'#7c4dff', 600:'#6d2ff5', 700:'#5d1fd8', 800:'#4d1cb0', 900:'#41198f', 950:'#280e61' },
        calm:  { 400:'#34b98a', 500:'#159e73', 600:'#0d7f5c' },
        mid:   { 400:'#e0a72c', 500:'#c78f16', 600:'#a17311' },
        hot:   { 400:'#ef6a5a', 500:'#dc4b3c', 600:'#b93a2d' },
      },
      fontFamily: {
        sans: ['Inter','ui-sans-serif','system-ui','-apple-system','Segoe UI','Roboto','Helvetica Neue','Arial','sans-serif'],
        serif: ['"Source Serif 4"','Iowan Old Style','Palatino','Georgia','ui-serif','serif'],
        mono: ['ui-monospace','SFMono-Regular','Menlo','Consolas','monospace'],
      },
      maxWidth: { prose: '68ch', page: '78rem' },
    },
  },
  plugins: [],
}
