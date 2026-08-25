/* Shared helpers: fetching, escaping, formatting. */
(function (w) {
  'use strict';

  const cache = new Map();

  async function getJSON(url) {
    if (cache.has(url)) return cache.get(url);
    const p = fetch(url).then(r => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
      return r.json();
    });
    cache.set(url, p);
    return p;
  }

  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ESC[c]);

  const num = (v, d = 2) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');

  /** Signed number, for showing deltas: +0.15 / −0.10 (true minus sign). */
  const signed = (v, d = 2) => {
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    const s = Math.abs(v).toFixed(d);
    return v < 0 ? '−' + s : '+' + s;
  };

  const band = r => (r <= 0.33 ? 'positive' : r <= 0.66 ? 'neutral' : 'negative');

  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  w.ODRA = { getJSON, esc, num, signed, band, el };
})(window);
