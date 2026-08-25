#!/usr/bin/env node
/**
 * Rasterise the paper's PDF figures for the web.
 *
 * macOS only: uses the system `qlmanage` (render) and `sips` (convert).
 * On Linux install poppler-utils and swap RENDER for:
 *   pdftocairo -png -r 200 -singlefile <in>.pdf <out>
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, existsSync, rmSync, mkdtempSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(HERE, '..');
const FIG = resolve(SITE, '..', 'latex', 'figures');
const OUT = join(SITE, 'assets', 'img');

// Sized for a ~1200px CSS container at ~1.5x for retina; larger is wasted bytes.
const FIGURES = [
  ['graphical_abstract', 'teaser', 1700],
  ['new-odra-method', 'architecture', 2000],
  ['example_models', 'cot-example', 1600],
];

if (process.platform !== 'darwin') {
  console.error('This script needs macOS (qlmanage/sips). See the header for the Linux command.');
  process.exit(1);
}

mkdirSync(OUT, { recursive: true });
const tmp = mkdtempSync(join(tmpdir(), 'odra-fig-'));

try {
  for (const [src, name, size] of FIGURES) {
    const pdf = join(FIG, `${src}.pdf`);
    if (!existsSync(pdf)) {
      console.error(`  MISSING ${pdf}`);
      process.exitCode = 1;
      continue;
    }
    execFileSync('qlmanage', ['-t', '-s', String(size), '-o', tmp, pdf], { stdio: 'ignore' });

    const rendered = join(tmp, `${src}.pdf.png`);
    if (!existsSync(rendered)) {
      console.error(`  FAILED to render ${src}`);
      process.exitCode = 1;
      continue;
    }

    const dest = join(OUT, `${name}.png`);
    execFileSync('sips', ['-s', 'format', 'png', rendered, '--out', dest], { stdio: 'ignore' });

    const dims = execFileSync('sips', ['-g', 'pixelWidth', '-g', 'pixelHeight', dest], { encoding: 'utf8' })
      .match(/pixel(?:Width|Height): (\d+)/g)
      .map(m => m.split(': ')[1]);
    console.log(`  ${name}.png  ${dims.join('x')}  ${(statSync(dest).size / 1024).toFixed(0)} KB`);
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
