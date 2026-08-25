/* Result tabs, the alignment bar chart, and BibTeX copy. */
(function () {
  'use strict';
  const { esc } = window.ODRA;

  function initTabs() {
    const bar = document.getElementById('res-tabs');
    if (!bar) return;
    const tabs = [...bar.querySelectorAll('[role="tab"]')];

    const select = tab => {
      tabs.forEach(t => {
        const on = t === tab;
        t.setAttribute('aria-selected', String(on));
        const panel = document.getElementById(t.getAttribute('aria-controls'));
        if (panel) panel.hidden = !on;
      });
    };

    tabs.forEach((t, i) => {
      t.addEventListener('click', () => select(t));
      t.addEventListener('keydown', e => {
        const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!d) return;
        e.preventDefault();
        const next = tabs[(i + d + tabs.length) % tabs.length];
        next.focus();
        select(next);
      });
    });
  }

  /* Resistance alignment is the one number that separates the methods, so it
     gets a chart rather than another row of digits. */
  function initAlignChart() {
    const host = document.getElementById('align-chart');
    if (!host) return;
    const rows = [
      { m: 'CACTUS',  v: 0.70, ours: false },
      { m: 'MAGNeT',  v: 1.19, ours: false },
      { m: 'MIRROR',  v: 1.01, ours: false },
      { m: 'ODRA-NR', v: 1.18, ours: false },
      { m: 'ODRA',    v: 1.96, ours: true  },
    ];
    host.innerHTML =
      `<p class="eyebrow mb-3">Resistance alignment (0–2)</p>` +
      rows.map(r => `
        <div class="mb-2 flex items-center gap-3">
          <span class="w-[5.5rem] shrink-0 text-[12.5px] ${r.ours ? 'font-semibold' : ''}">${esc(r.m)}</span>
          <span class="h-5 flex-1 overflow-hidden rounded" style="background:var(--bg-sunk)">
            <span class="block h-full rounded" style="width:${(r.v / 2 * 100).toFixed(1)}%;background:${r.ours ? 'var(--accent)' : 'var(--fg-faint)'}"></span>
          </span>
          <span class="w-9 shrink-0 text-right font-mono text-[12px] ${r.ours ? 'font-semibold' : ''}" style="${r.ours ? '' : 'color:var(--fg-faint)'}">${r.v.toFixed(2)}</span>
        </div>`).join('');
  }

  function initCopy() {
    const btn = document.getElementById('copy-bib');
    const src = document.getElementById('bibtex');
    if (!btn || !src) return;
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(src.textContent);
        btn.textContent = 'Copied';
      } catch {
        btn.textContent = 'Press ⌘C';
        const r = document.createRange();
        r.selectNodeContents(src);
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initAlignChart();
    initCopy();
  });
})();
