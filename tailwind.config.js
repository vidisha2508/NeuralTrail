/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkvibe: {
          bg: '#0c0418',
          bgSurface: '#120624',
          panel: '#170930',
          card: '#1e0c3d',
          cardHover: '#26104e',
          border: 'rgba(124, 77, 255, 0.2)',
          borderBright: 'rgba(124, 77, 255, 0.45)',
          
          // Semantic & Contrasting Neon Accents
          neonGreen: '#00ff88',
          mint: '#05ffa1',
          acidGreen: '#39ff14',
          neonPurple: '#d500f9',
          purple: '#7c4dff',
          lavender: '#b388ff',
          neonPink: '#ff007f',
          magenta: '#ff2a6d',
          cyan: '#00f0ff',
          amber: '#ffb300',
          yellow: '#ffe600',
        }
      },
      fontFamily: {
        sans: ['"Inter"', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        display: ['"Space Grotesk"', '"Orbitron"', 'sans-serif'],
        retro: ['"VT323"', 'monospace'],
        tech: ['"Share Tech Mono"', 'monospace'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      boxShadow: {
        'mac': '0 10px 30px -5px rgba(0, 0, 0, 0.8), 0 0 15px rgba(124, 77, 255, 0.35)',
        'neon-green': '0 0 14px rgba(0, 255, 136, 0.45)',
        'neon-purple': '0 0 16px rgba(213, 0, 249, 0.45)',
        'neon-pink': '0 0 16px rgba(255, 0, 127, 0.45)',
        'neon-cyan': '0 0 14px rgba(0, 240, 255, 0.45)',
      }
    },
  },
  plugins: [],
}
