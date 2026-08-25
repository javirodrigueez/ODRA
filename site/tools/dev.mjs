#!/usr/bin/env node
/**
 * Dev loop: Tailwind in watch mode plus the static server, in one terminal.
 *
 * Spawns the Tailwind binary directly rather than through `npx`, so there is no
 * intermediate `npm exec` process to outlive Ctrl+C. Both children are killed
 * together on exit.
 *
 * Watch mode writes an UNMINIFIED stylesheet. Run `npm run build` before
 * committing so the deployed `assets/css/style.css` stays minified.
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(HERE, '..');
const bin = join(SITE, 'node_modules', '.bin', process.platform === 'win32' ? 'tailwindcss.cmd' : 'tailwindcss');

if (!existsSync(bin)) {
  console.error('Tailwind is not installed. Run `npm install` first.');
  process.exit(1);
}

const children = [
  spawn(bin, ['-i', './src/input.css', '-o', './assets/css/style.css', '--watch'],
        { cwd: SITE, stdio: 'inherit' }),
  spawn(process.execPath, [join(HERE, 'serve.mjs'), ...process.argv.slice(2)],
        { cwd: SITE, stdio: 'inherit' }),
];

let closing = false;
function shutdown(code = 0) {
  if (closing) return;
  closing = true;
  for (const c of children) if (!c.killed) c.kill('SIGTERM');
  console.log('\nStopped. Run `npm run build` to re-minify the stylesheet before committing.');
  process.exit(code);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
for (const c of children) {
  c.on('error', err => { console.error(err.message); shutdown(1); });
  c.on('exit', code => shutdown(code ?? 0));
}
