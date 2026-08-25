#!/usr/bin/env node
/**
 * Zero-dependency static server for local preview.
 *
 * The explorer fetches JSON, so the page cannot be opened over file:// —
 * it needs a real origin. Usage: `npm run serve` or `node tools/serve.mjs 8731`.
 */
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { dirname, join, resolve, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 8731);
const HOST = process.env.HOST ?? '127.0.0.1';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
  '.pdf': 'application/pdf',
};

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname.endsWith('/')) pathname += 'index.html';

  // Resolve inside ROOT only — never serve above the site directory.
  const target = join(ROOT, normalize(pathname).replace(/^(\.\.[/\\])+/, ''));
  if (!target.startsWith(ROOT)) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  if (!existsSync(target) || !statSync(target).isFile()) {
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    res.end(`404 Not Found — ${pathname}`);
    console.log(`  404  ${pathname}`);
    return;
  }

  const { size } = statSync(target);
  res.writeHead(200, {
    'content-type': MIME[extname(target).toLowerCase()] ?? 'application/octet-stream',
    'content-length': size,
    // Always revalidate: this server exists for iterating on the page.
    'cache-control': 'no-cache, no-store, must-revalidate',
  });
  if (req.method === 'HEAD') return res.end();
  createReadStream(target).pipe(res);
});

server.on('error', err => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Try: npm run serve -- ${PORT + 1}`);
    process.exit(1);
  }
  throw err;
});

server.listen(PORT, HOST, () => {
  console.log(`ODRA project page → http://${HOST}:${PORT}/`);
  console.log(`serving ${ROOT}`);
  console.log('Ctrl+C to stop.');
});
