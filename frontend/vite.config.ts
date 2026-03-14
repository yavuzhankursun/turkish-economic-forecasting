import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiBase = env.VITE_API_BASE_URL || 'http://localhost:5000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: apiBase,
          changeOrigin: true,
          secure: false,
          timeout: 600000, // 10 dakika timeout
          proxyTimeout: 600000, // Proxy timeout
          ws: true, // WebSocket desteği
          xfwd: true, // X-Forwarded-* header'ları
          configure: (proxy, _options) => {
            // Socket timeout'u artır
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              console.log('Sending request to the target:', req.method, req.url);
              proxyReq.setHeader('Connection', 'keep-alive');
              proxyReq.setTimeout(600000); // 10 dakika
            });
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('Received response from the target:', proxyRes.statusCode, req.url);
              proxyRes.headers['connection'] = 'keep-alive';
              proxyRes.headers['keep-alive'] = 'timeout=600';
            });
            proxy.on('error', (err, req, res) => {
              console.error('Proxy error:', err);
              if (!res.headersSent) {
                res.writeHead(500, {
                  'Content-Type': 'text/plain'
                });
                res.end('Proxy error: ' + err.message);
              }
            });
          }
        }
      }
    },
    preview: {
      port: 4173
    },
    build: {
      sourcemap: true
    }
  };
});

