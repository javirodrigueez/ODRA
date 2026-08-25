#!/usr/bin/env node
/**
 * Render the 1200x630 Open Graph card. Pure vector text — no embedded raster,
 * so it stays sharp at the thumbnail sizes social clients actually show.
 *
 * macOS quirk: qlmanage always emits a SQUARE thumbnail and scales an SVG to
 * fit its height. So the card is authored on a 1200x1200 canvas with the real
 * 1200x630 artwork centred, then cropped back out with sips.
 */
import { execFileSync } from 'node:child_process';
import { writeFileSync, existsSync, rmSync, mkdtempSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(HERE, '..');
const OUT = join(SITE, 'assets', 'img', 'social-card.jpg');
const SANS = 'Helvetica Neue,Helvetica,Arial,sans-serif';
const PAD = (1200 - 630) / 2; // letterbox above and below the artwork

const chip = (x, value, label) => `
  <rect x="${x}" y="438" width="316" height="106" rx="12" fill="#ffffff" opacity=".07"/>
  <text x="${x + 24}" y="492" font-family="${SANS}" font-size="38" font-weight="700" fill="#ffffff">${value}</text>
  <text x="${x + 24}" y="523" font-family="${SANS}" font-size="17" fill="#a79ddb">${label}</text>`;

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200"
     viewBox="0 ${-PAD} 1200 1200">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#191d26"/><stop offset="1" stop-color="#2b1f52"/>
    </linearGradient>
  </defs>
  <rect x="0" y="${-PAD}" width="1200" height="1200" fill="url(#bg)"/>
  <rect x="0" y="0" width="9" height="630" fill="#7c4dff"/>

  <text x="64" y="120" font-family="Georgia,serif" font-size="70" font-weight="700"
        fill="#ffffff" letter-spacing="1">ODRA</text>

  <text x="64" y="202" font-family="${SANS}" font-size="36" font-weight="500" fill="#ece9fb">
    Synthesizing CBT sessions with structured
  </text>
  <text x="64" y="250" font-family="${SANS}" font-size="36" font-weight="500" fill="#ece9fb">
    chain-of-thought and dynamic patient resistance
  </text>

  <text x="64" y="322" font-family="${SANS}" font-size="20" fill="#9b92cf">
    Rodriguez-Juan &#183; Arnaout &#183; Garcia-Rodriguez &#183; Tom&#225;s &#183; Gurevych
  </text>
  <text x="64" y="356" font-family="${SANS}" font-size="20" fill="#9b92cf">
    University of Alicante &#183; UKP Lab, TU Darmstadt &#183; arXiv:2608.04524
  </text>
  ${chip(64, '150', 'CBT sessions')}
  ${chip(442, '18,496', 'reasoning traces')}
  ${chip(820, '1.96/2', 'resistance alignment')}
</svg>`;

if (process.platform !== 'darwin') {
  console.error('This script needs macOS (qlmanage/sips).');
  process.exit(1);
}

const tmp = mkdtempSync(join(tmpdir(), 'odra-card-'));
try {
  const svgPath = join(tmp, 'card.svg');
  writeFileSync(svgPath, svg);
  execFileSync('qlmanage', ['-t', '-s', '1200', '-o', tmp, svgPath], { stdio: 'ignore' });

  const rendered = join(tmp, 'card.svg.png');
  if (!existsSync(rendered)) {
    console.error('qlmanage could not render the card SVG');
    process.exit(1);
  }
  // JPEG: the background gradient compresses poorly as PNG (700KB -> ~95KB).
  execFileSync('sips', ['-s', 'format', 'jpeg', '-s', 'formatOptions', '88',
                        '-c', '630', '1200', rendered, '--out', OUT], { stdio: 'ignore' });

  const dims = execFileSync('sips', ['-g', 'pixelWidth', '-g', 'pixelHeight', OUT], { encoding: 'utf8' })
    .match(/pixel(?:Width|Height): (\d+)/g).map(m => m.split(': ')[1]);
  console.log(`  social-card.jpg  ${dims.join('x')}  ${(statSync(OUT).size / 1024).toFixed(0)} KB`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
