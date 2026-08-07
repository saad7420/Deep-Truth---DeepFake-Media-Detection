/* Builds the three extension entry points into the parent directory, which is
   the folder Chrome loads. CSS imported from popup/index.tsx is emitted
   alongside as popup.css, which popup.html links.

   The PostCSS/Tailwind pass that used to run here is gone: Popup.css is now
   plain CSS with no directives to expand, so the build no longer needs
   tailwindcss, postcss or autoprefixer at all. */

const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

const isWatch = process.argv.includes('--watch');

const loadEnv = () => {
  const envPath = path.resolve(__dirname, '.env');
  const env = {};
  if (fs.existsSync(envPath)) {
    fs.readFileSync(envPath, 'utf8')
      .split('\n')
      .forEach((line) => {
        if (line.trim().startsWith('#')) return;
        const match = line.match(/^([^=]+)=(.*)$/);
        if (match) {
          env[match[1].trim()] = match[2].trim().replace(/^["']|["']$/g, '');
        }
      });
  }
  return env;
};

const env = loadEnv();

const buildOptions = {
  entryPoints: {
    popup: './src/popup/index.tsx',
    content: './src/content/content.tsx',
    background: './src/background/background.ts',
  },
  bundle: true,
  outdir: '..',
  format: 'iife',
  target: ['chrome102'],
  loader: {
    '.tsx': 'tsx',
    '.ts': 'ts',
    '.css': 'css',
  },
  minify: !isWatch,
  sourcemap: isWatch,
  logLevel: 'info',
  define: {
    'process.env.NODE_ENV': isWatch ? '"development"' : '"production"',
    'process.env.DEEP_TRUTH_API_URL': JSON.stringify(
      env.DEEP_TRUTH_API_URL || 'http://localhost:8000',
    ),
  },
};

async function build() {
  try {
    if (isWatch) {
      const ctx = await esbuild.context(buildOptions);
      await ctx.watch();
      console.log('Watching for changes...');
    } else {
      await esbuild.build(buildOptions);
      console.log('Build complete.');
    }
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

build();
