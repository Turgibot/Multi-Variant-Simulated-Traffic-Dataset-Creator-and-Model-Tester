/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      screens: {
        // Mobile-first approach with comprehensive breakpoints
        'xs': '320px',      // Extra small phones
        'sm': '640px',      // Small tablets
        'md': '768px',      // Tablets
        'lg': '1024px',     // Laptops
        'xl': '1280px',     // Desktops
        '2xl': '1536px',    // Large desktops
        '3xl': '1920px',    // Ultra-wide monitors
        '4xl': '2560px',    // 4K displays
        '5xl': '3440px',    // Ultra-wide 4K
        '6xl': '3840px',    // 4K+ displays
        
        // Height-based breakpoints for landscape/portrait considerations
        'h-xs': { 'raw': '(max-height: 480px)' },
        'h-sm': { 'raw': '(max-height: 640px)' },
        'h-md': { 'raw': '(max-height: 768px)' },
        'h-lg': { 'raw': '(max-height: 1024px)' },
        
        // Aspect ratio breakpoints
        'aspect-mobile': { 'raw': '(max-aspect-ratio: 3/4)' },
        'aspect-tablet': { 'raw': '(min-aspect-ratio: 3/4) and (max-aspect-ratio: 16/9)' },
        'aspect-desktop': { 'raw': '(min-aspect-ratio: 16/9)' },
        'aspect-ultrawide': { 'raw': '(min-aspect-ratio: 21/9)' },
        
        // Orientation breakpoints
        'portrait': { 'raw': '(orientation: portrait)' },
        'landscape': { 'raw': '(orientation: landscape)' },
        
        // High DPI displays
        'retina': { 'raw': '(-webkit-min-device-pixel-ratio: 2)' },
        'high-dpi': { 'raw': '(min-resolution: 192dpi)' },
      },
      spacing: {
        // Extended spacing scale for ultra-wide layouts
        '18': '4.5rem',
        '88': '22rem',
        '96': '24rem',
        '128': '32rem',
        '144': '36rem',
        '160': '40rem',
        '176': '44rem',
        '192': '48rem',
        '208': '52rem',
        '224': '56rem',
        '240': '60rem',
        '256': '64rem',
        '288': '72rem',
        '320': '80rem',
        '384': '96rem',
        '448': '112rem',
        '512': '128rem',
      },
      fontSize: {
        // Extended font size scale
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }],
        '7xl': ['4.5rem', { lineHeight: '1' }],
        '8xl': ['6rem', { lineHeight: '1' }],
        '9xl': ['8rem', { lineHeight: '1' }],
        '10xl': ['10rem', { lineHeight: '1' }],
      },
      maxWidth: {
        // Extended max-width scale
        'xs': '20rem',
        'sm': '24rem',
        'md': '28rem',
        'lg': '32rem',
        'xl': '36rem',
        '2xl': '42rem',
        '3xl': '48rem',
        '4xl': '56rem',
        '5xl': '64rem',
        '6xl': '72rem',
        '7xl': '80rem',
        '8xl': '88rem',
        '9xl': '96rem',
        '10xl': '104rem',
        '11xl': '112rem',
        '12xl': '120rem',
        '13xl': '128rem',
        '14xl': '136rem',
        '15xl': '144rem',
        '16xl': '152rem',
        '17xl': '160rem',
        '18xl': '168rem',
        '19xl': '176rem',
        '20xl': '184rem',
        '21xl': '192rem',
        '22xl': '200rem',
        '23xl': '208rem',
        '24xl': '216rem',
        '25xl': '224rem',
        '26xl': '232rem',
        '27xl': '240rem',
        '28xl': '248rem',
        '29xl': '256rem',
        '30xl': '264rem',
        'none': 'none',
        'full': '100%',
        'min': 'min-content',
        'max': 'max-content',
        'fit': 'fit-content',
        'prose': '65ch',
        'screen-sm': '640px',
        'screen-md': '768px',
        'screen-lg': '1024px',
        'screen-xl': '1280px',
        'screen-2xl': '1536px',
        'screen-3xl': '1920px',
        'screen-4xl': '2560px',
        'screen-5xl': '3440px',
        'screen-6xl': '3840px',
      },
    },
  },
  plugins: [],
}
