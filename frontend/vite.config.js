import { defineConfig } from 'vite';
import fs from 'fs';

const pkg = JSON.parse(fs.readFileSync('./package.json', 'utf-8'));
const today = new Date();
const releaseDate = today.getFullYear() + '-' + String(today.getMonth() + 1).padStart(2, '0') + '-' + String(today.getDate()).padStart(2, '0');

export default defineConfig({
  plugins: [
    {
      name: 'serve-wasm-static',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/wasm/')) {
            const cleanUrl = req.url.split('?')[0];
            const filePath = `./public${cleanUrl}`;
            if (fs.existsSync(filePath)) {
              const ext = cleanUrl.split('.').pop();
              let contentType = 'application/octet-stream';
              if (ext === 'js' || ext === 'mjs') {
                contentType = 'application/javascript';
              } else if (ext === 'wasm') {
                contentType = 'application/wasm';
              }
              res.setHeader('Content-Type', contentType);
              res.setHeader('Access-Control-Allow-Origin', '*');
              res.end(fs.readFileSync(filePath));
              return;
            }
          }
          next();
        });
      }
    }
  ],
  server: {
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:18000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __RELEASE_DATE__: JSON.stringify(releaseDate),
  }
});
