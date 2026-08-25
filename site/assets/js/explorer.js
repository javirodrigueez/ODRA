/* Session explorer: profile/attitude selection, resistance chart, transcript
   with expandable chain-of-thought traces. Plus the sycophancy comparison. */
(function () {
  'use strict';
  const { getJSON, esc, num, signed } = window.ODRA;

  const DATA = 'assets/data';
  const ATT_COLOR = { positive: 'var(--calm)', neutral: 'var(--mid)', negative: 'var(--hot)' };

  /* Stage tints for the chart bands and the transcript legend. */
  const STAGE_TINT = {
    opening:           'var(--accent)',
    id_distortions:    '#3b82f6',
    cbt_technique:     '#14b8a6',
    homework:          '#8b5cf6',
    closing:           '#64748b',
    address_tib:       'var(--hot)',
    meta_conversation: 'var(--mid)',
  };

  /** Turns shown before the reader asks for the whole session. */
  const PREVIEW_TURNS = 12;

  const state = {
    manifest: null,
    profile: null,     // patient name
    attitude: 'negative',
    session: null,
    expandAll: false,
    fullTranscript: false,
  };

  const capitalize = s => (s ? s[0].toUpperCase() + s.slice(1) : '');

  const $ = s => document.querySelector(s);

  /* ───────────────────────────────────────────────────── selection */

  function profiles() {
    const seen = new Map();
    for (const s of state.manifest.sessions) {
      if (!seen.has(s.name)) seen.set(s.name, s);
    }
    return [...seen.values()];
  }

  function renderProfiles() {
    const host = $('#profile-list');
    host.innerHTML = '';
    for (const p of profiles()) {
      const b = document.createElement('button');
      b.className = 'profbtn';
      b.setAttribute('aria-pressed', String(p.name === state.profile));
      b.innerHTML =
        `<div class="text-[13.5px] font-semibold">${esc(p.name)}</div>` +
        `<div class="mt-0.5 text-[11.5px]" style="color:var(--fg-faint)">${esc(p.age)} · ${esc(p.occupation)}</div>`;
      b.addEventListener('click', () => { state.profile = p.name; load(); });
      host.appendChild(b);
    }
  }

  function renderAttitudes() {
    document.querySelectorAll('#attitude-seg button').forEach(b => {
      b.setAttribute('aria-pressed', String(b.dataset.att === state.attitude));
    });
  }

  function currentEntry() {
    return state.manifest.sessions.find(
      s => s.name === state.profile && s.attitude === state.attitude
    );
  }

  /* ───────────────────────────────────────────────────── header cards */

  function renderPatientCard(s, entry) {
    const c = ATT_COLOR[s.attitude];
    $('#patient-card').innerHTML = `
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div class="min-w-0">
          <h3 class="font-serif text-xl font-semibold">${esc(s.patient.name)}</h3>
          <p class="mt-0.5 text-[13px]" style="color:var(--fg-faint)">
            ${esc(s.patient.age)} · ${esc(s.patient.gender)} · ${esc(s.patient.occupation)}
          </p>
        </div>
        <span class="chip" style="color:${c};background:color-mix(in srgb, ${c} 14%, transparent)">
          ${esc(capitalize(s.attitude))}
        </span>
      </div>
      <p class="mt-3 text-[14px] leading-relaxed" style="color:var(--fg-soft)">
        <span class="font-semibold" style="color:var(--fg)">Presenting problem.</span>
        ${esc(s.patient.problem)}
      </p>`;

    $('#session-stats').innerHTML = `
      <div class="statbox"><dt>Turns</dt><dd>${entry.nTurns}</dd></div>
      <div class="statbox"><dt>Traces</dt><dd>${entry.nTraces}</dd></div>
      <div class="statbox"><dt>TIBs</dt><dd>${entry.nTibs}</dd></div>
      <div class="statbox"><dt>Final r</dt><dd>${num(s.finalR)}</dd></div>`;

    const a = $('#dl-session');
    a.href = `${DATA}/sessions/${s.id}.json`;
    a.setAttribute('download', `${s.id}.json`);
  }

  function renderLegend(s) {
    const seen = [];
    for (const r of s.stageRuns) if (!seen.includes(r.stage)) seen.push(r.stage);
    $('#stage-legend').innerHTML = seen.map(st => {
      const label = (s.stageRuns.find(r => r.stage === st) || {}).label || st;
      const col = STAGE_TINT[st] || 'var(--fg-faint)';
      return `<span class="chip" style="color:${col};background:color-mix(in srgb, ${col} 12%, transparent)">${esc(label)}</span>`;
    }).join('');
  }

  /* ───────────────────────────────────────────────────── chart */

  function renderChart(s) {
    const host = $('#chart');
    const W = Math.max(320, host.clientWidth || 640);
    const H = 200;
    const M = { t: 10, r: 10, b: 22, l: 30 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const maxI = s.turns.length ? s.turns[s.turns.length - 1].i : 1;

    const x = i => M.l + (iw * (i - 1)) / Math.max(1, maxI - 1);
    const y = v => M.t + ih * (1 - Math.max(0, Math.min(1, v)));

    const parts = [];

    // Stage bands
    for (const run of s.stageRuns) {
      const col = STAGE_TINT[run.stage] || 'var(--fg-faint)';
      const x0 = x(run.start), x1 = x(Math.min(run.end + 1, maxI));
      parts.push(`<rect x="${x0.toFixed(1)}" y="${M.t}" width="${Math.max(1, x1 - x0).toFixed(1)}" height="${ih}" fill="${col}" opacity=".07"/>`);
    }

    // Band separators at the attitude boundaries
    for (const v of [0.33, 0.66]) {
      parts.push(`<line x1="${M.l}" y1="${y(v).toFixed(1)}" x2="${W - M.r}" y2="${y(v).toFixed(1)}" stroke="var(--line)" stroke-width="1" stroke-dasharray="2 4"/>`);
    }
    parts.push(`<line x1="${M.l}" y1="${y(0).toFixed(1)}" x2="${W - M.r}" y2="${y(0).toFixed(1)}" stroke="var(--line)" stroke-width="1"/>`);

    // Axis labels
    for (const [v, lab] of [[0, '0'], [0.33, '.33'], [0.66, '.66'], [1, '1']]) {
      parts.push(`<text x="${M.l - 6}" y="${(y(v) + 3.5).toFixed(1)}" text-anchor="end" font-size="9.5" fill="var(--fg-faint)" font-family="ui-monospace,monospace">${lab}</text>`);
    }

    const path = pts => pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join('');

    if (s.pSeries.length) {
      parts.push(`<path d="${path(s.pSeries)}" fill="none" stroke="var(--therapist)" stroke-width="1.8" stroke-dasharray="4 3" stroke-linejoin="round" opacity=".9"/>`);
    }
    if (s.rSeries.length) {
      parts.push(`<path d="${path(s.rSeries)}" fill="none" stroke="var(--patient)" stroke-width="2.2" stroke-linejoin="round"/>`);
    }

    // TIB markers along the baseline
    for (const t of s.tibEvents) {
      const cx = x(t.i), cy = H - M.b + 7;
      parts.push(`<rect x="${(cx - 2.6).toFixed(1)}" y="${(cy - 2.6).toFixed(1)}" width="5.2" height="5.2" fill="var(--hot)" opacity=".85" transform="rotate(45 ${cx.toFixed(1)} ${cy.toFixed(1)})"><title>Turn ${t.i}: ${esc(t.name || 'TIB')}</title></rect>`);
    }

    parts.push(`<text x="${M.l}" y="${H - 3}" font-size="9.5" fill="var(--fg-faint)" font-family="ui-monospace,monospace">turn 1</text>`);
    parts.push(`<text x="${W - M.r}" y="${H - 3}" text-anchor="end" font-size="9.5" fill="var(--fg-faint)" font-family="ui-monospace,monospace">turn ${maxI}</text>`);

    // Hover guide
    parts.push(`<line id="chart-guide" x1="0" y1="${M.t}" x2="0" y2="${M.t + ih}" stroke="var(--accent)" stroke-width="1" opacity="0"/>`);
    parts.push(`<rect id="chart-hit" x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" fill="transparent" style="cursor:crosshair"/>`);

    host.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" ` +
      `aria-label="Patient resistance and therapist estimate across ${maxI} turns">${parts.join('')}</svg>` +
      `<div id="chart-tip" class="pointer-events-none mt-1 h-4 font-mono text-[11px]" style="color:var(--fg-faint)"></div>`;

    const svg = host.querySelector('svg');
    const guide = host.querySelector('#chart-guide');
    const tip = host.querySelector('#chart-tip');
    const hit = host.querySelector('#chart-hit');

    const turnAt = evt => {
      const box = svg.getBoundingClientRect();
      const px = ((evt.clientX - box.left) / box.width) * W;
      const frac = (px - M.l) / iw;
      return Math.max(1, Math.min(maxI, Math.round(1 + frac * (maxI - 1))));
    };
    const nearest = (arr, i) =>
      arr.reduce((best, p) => (Math.abs(p.i - i) < Math.abs(best.i - i) ? p : best), arr[0]);

    hit.addEventListener('mousemove', evt => {
      const i = turnAt(evt);
      guide.setAttribute('x1', x(i)); guide.setAttribute('x2', x(i));
      guide.setAttribute('opacity', '.5');
      const r = s.rSeries.length ? nearest(s.rSeries, i) : null;
      const p = s.pSeries.length ? nearest(s.pSeries, i) : null;
      tip.textContent = `turn ${i}  ·  r ${r ? num(r.v) : '—'}  ·  p̂ ${p ? num(p.v) : '—'}`;
    });
    hit.addEventListener('mouseleave', () => { guide.setAttribute('opacity', '0'); tip.textContent = ''; });
    hit.addEventListener('click', evt => {
      const i = turnAt(evt);
      // The target turn may sit past the preview cut — reveal the rest first.
      if (i > PREVIEW_TURNS) expandTranscript(false);
      const node = document.getElementById(`turn-${i}`) ||
        document.getElementById(`turn-${i - 1}`) || document.getElementById(`turn-${i + 1}`);
      if (!node) return;
      node.scrollIntoView({ behavior: 'smooth', block: 'center' });
      document.querySelectorAll('.turn.is-active').forEach(n => n.classList.remove('is-active'));
      node.classList.add('is-active');
    });
  }

  /* ───────────────────────────────────────────────────── traces */

  function distBars(d) {
    const fused = d.fused || {};
    const keys = Object.keys(fused);
    if (!keys.length) return '';
    // Absolute scale: these are probabilities, so a 35% bar should read as 35%
    // of the track rather than being stretched to fill it.
    return keys.map(k => {
      const v = fused[k] || 0;
      const picked = d.selected === k;
      return `<div class="distrow${picked ? ' is-picked' : ''}">
        <span class="lbl">${esc(k)}</span>
        <span class="bar"><span style="width:${(v * 100).toFixed(1)}%"></span></span>
        <span class="val">${(v * 100).toFixed(0)}%</span>
      </div>`;
    }).join('');
  }

  function traceHTML(t) {
    let body = '';

    if (t.math && typeof t.math.new_r === 'number') {
      const m = t.math;
      body += `<div class="eqbox">r<sub>t+1</sub> = ${num(m.previous_r, 3)} ${signed(m.delta_llm, 3)} − ` +
        `${num(m.alpha, 2)}·(${num(m.previous_r, 3)} − ${num(m.baseline, 3)}) = <b>${num(m.new_r, 3)}</b></div>`;
    }
    if (t.pred && typeof t.pred.new_p === 'number') {
      body += `<div class="eqbox">p̂<sub>target</sub> = ${num(t.pred.p_target, 2)} · ` +
        `λ = ${num(t.pred.clarity, 2)} → <b>p̂<sub>t+1</sub> = ${num(t.pred.new_p, 3)}</b>` +
        (t.pred.clipped ? ' <span style="color:var(--hot)">(clipped)</span>' : '') + `</div>`;
    }
    if (t.tib) {
      body += `<div class="eqbox">TIB detected: <b>${t.tib.has_tib ? esc(t.tib.name || 'yes') : 'none'}</b></div>`;
    }
    if (t.dist) body += `<div class="mt-1.5">${distBars(t.dist)}</div>`;
    if (t.extra) body += `<div class="trace-body mt-1.5">${esc(t.extra)}</div>`;
    if (t.reasoning) body += `<div class="trace-body mt-1.5">${esc(t.reasoning)}</div>`;

    return `<div class="trace-step"><div class="trace-name">${esc(t.label)}</div>${body}</div>`;
  }

  function turnHTML(t) {
    const pills = [];
    if (typeof t.r === 'number') pills.push(`<span class="pill pill-r">r ${num(t.r)}</span>`);
    if (typeof t.p === 'number') pills.push(`<span class="pill pill-p">p̂ ${num(t.p)}</span>`);
    if (t.action) pills.push(`<span class="pill pill-act">${esc(t.action)}</span>`);
    if (t.tib) pills.push(`<span class="pill pill-tib" title="${esc(t.tib.description || '')}">TIB · ${esc(t.tib.name)}</span>`);
    if (t.stageLabel) pills.push(`<span class="pill">${esc(t.stageLabel)}</span>`);

    const n = t.traces.length;
    const toggle = n
      ? `<button class="trace-toggle" data-turn="${t.i}" aria-expanded="false" aria-controls="tr-${t.i}">▸ ${n} reasoning step${n > 1 ? 's' : ''}</button>`
      : '';
    const panel = n
      ? `<div class="trace-panel" id="tr-${t.i}" hidden>${t.traces.map(traceHTML).join('')}</div>`
      : '';

    return `<div class="turn turn-${t.role}" id="turn-${t.i}">
      <div class="turn-who">${t.role === 'therapist' ? 'Therapist' : 'Patient'}</div>
      <div class="turn-body">
        <p class="turn-text">${esc(t.text)}</p>
        <div class="metabar">${pills.join('')}${toggle}</div>
        ${panel}
      </div>
    </div>`;
  }

  function renderTranscript(s) {
    const host = $('#transcript');
    const total = s.turns.length;
    const capped = !state.fullTranscript && total > PREVIEW_TURNS;
    const shown = capped ? s.turns.slice(0, PREVIEW_TURNS) : s.turns;

    host.innerHTML = shown.map(turnHTML).join('') + (capped ? `
      <div class="tr-more">
        <button id="show-full" class="btn w-full justify-center">
          See complete transcript — ${total - PREVIEW_TURNS} more turns
        </button>
      </div>` : '');

    const count = $('#transcript-count');
    if (count) count.textContent = capped ? `${PREVIEW_TURNS} of ${total} turns` : `${total} turns`;

    host.querySelectorAll('.trace-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const panel = document.getElementById('tr-' + btn.dataset.turn);
        const open = panel.hidden;
        panel.hidden = !open;
        btn.setAttribute('aria-expanded', String(open));
        btn.textContent = btn.textContent.replace(open ? '▸' : '▾', open ? '▾' : '▸');
      });
    });

    const more = host.querySelector('#show-full');
    if (more) more.addEventListener('click', () => expandTranscript(true));

    applyExpandAll();
  }

  /** Reveal the rest of the session, keeping the reader where they were. */
  function expandTranscript(keepPosition) {
    if (state.fullTranscript || !state.session) return;
    const anchor = document.getElementById(`turn-${PREVIEW_TURNS}`);
    const before = anchor && keepPosition ? anchor.getBoundingClientRect().top : null;
    state.fullTranscript = true;
    renderTranscript(state.session);
    if (before !== null) {
      const after = document.getElementById(`turn-${PREVIEW_TURNS}`);
      if (after) window.scrollBy({ top: after.getBoundingClientRect().top - before, behavior: 'instant' });
    }
  }

  function applyExpandAll() {
    document.querySelectorAll('#transcript .trace-toggle').forEach(btn => {
      const panel = document.getElementById('tr-' + btn.dataset.turn);
      if (!panel) return;
      panel.hidden = !state.expandAll;
      btn.setAttribute('aria-expanded', String(state.expandAll));
      btn.textContent = (state.expandAll ? '▾' : '▸') + btn.textContent.slice(1);
    });
  }

  /* ───────────────────────────────────────────────────── load */

  async function load() {
    renderProfiles();
    renderAttitudes();
    const entry = currentEntry();
    if (!entry) return;
    state.fullTranscript = false;   // every session starts at the preview
    $('#transcript').innerHTML = `<p class="py-8 text-center text-[13px]" style="color:var(--fg-faint)">Loading session…</p>`;
    try {
      const s = await getJSON(`${DATA}/sessions/${entry.id}.json`);
      state.session = s;
      renderPatientCard(s, entry);
      renderLegend(s);
      renderChart(s);
      renderTranscript(s);
    } catch (err) {
      $('#transcript').innerHTML =
        `<p class="py-8 text-center text-[13px]" style="color:var(--hot)">Could not load this session (${esc(err.message)}).</p>`;
    }
  }

  /* ───────────────────────────────────────────────────── sycophancy */

  function sycColumn(m, accent, note, ours) {
    const PREVIEW = 8;
    const shown = m.turns.slice(0, PREVIEW);
    const rest = m.turns.slice(PREVIEW);
    const row = t => `<div class="turn turn-${t.role} !py-2">
        <div class="turn-who !w-[4.6rem] text-[10px]">${t.role === 'therapist' ? 'Therapist' : 'Patient'}</div>
        <div class="turn-body"><p class="turn-text !text-[13.5px]">${esc(t.text)}</p></div>
      </div>`;
    return `<div class="card syc-col${ours ? ' syc-col-ours' : ''} flex min-w-0 flex-col">
      <div class="syc-head border-b px-4 py-3" style="border-color:var(--line)">
        <div class="flex items-baseline justify-between gap-2">
          <h3 class="font-semibold" style="color:${accent}">${esc(m.method)}</h3>
          <span class="pill">${m.nTurns} turns</span>
        </div>
        
      </div>
      <div class="syc-body">
        ${shown.map(row).join('')}
        ${rest.length ? `<div class="syc-rest" hidden>${rest.map(row).join('')}</div>` : ''}
      </div>
      ${rest.length ? `<div class="border-t px-4 py-2.5" style="border-color:var(--line)">
        <button class="syc-more btn w-full justify-center !py-1.5 text-[12.5px]"
                aria-expanded="false" data-count="${rest.length}" data-total="${m.nTurns}">
          Show all ${m.nTurns} turns
        </button>
      </div>` : ''}
    </div>`;
  }

  async function renderSycophancy() {
    const host = document.getElementById('syc-grid');
    if (!host) return;
    try {
      const d = await getJSON(`${DATA}/sycophancy.json`);
      const notes = {
        ODRA: 'Refuses the exercise, disputes the therapist’s framing, and ends the session early.',
        CACTUS: 'By turn 4 the &ldquo;negative&rdquo; patient is thanking the therapist and volunteering to explore.',
        MAGNeT: 'Elaborates cooperatively on the therapist’s reframe; no friction appears at any point.',
      };
      const accents = { ODRA: 'var(--accent)', CACTUS: 'var(--fg-soft)', MAGNeT: 'var(--fg-soft)' };
      const order = ['CACTUS', 'MAGNeT', 'ODRA'];
      const cols = order
        .map(name => d.methods.find(m => m.method === name))
        .filter(Boolean)
        .map(m => sycColumn(m, accents[m.method], notes[m.method] || '', m.method === 'ODRA'));

      host.innerHTML = `
        <div class="card-soft mb-4 p-4 text-[13.5px] leading-relaxed" style="color:var(--fg-soft)">
          <strong style="color:var(--fg)">${esc(d.profile.name)}</strong> ·
          ${esc(d.profile.age)} · ${esc(d.profile.occupation)} — ${esc(d.profile.problem)}
        </div>
        <div class="grid items-start gap-4 lg:grid-cols-3">${cols.join('')}</div>`;

      // Uncap the column rather than revealing rows inside a fixed-height
      // scroller, where the new turns would sit below the fold unseen.
      host.querySelectorAll('.syc-more').forEach(btn => {
        btn.addEventListener('click', () => {
          const col = btn.closest('.syc-col');
          const body = col.querySelector('.syc-body');
          const rest = col.querySelector('.syc-rest');
          const open = btn.getAttribute('aria-expanded') !== 'true';
          rest.hidden = !open;
          body.classList.toggle('is-open', open);
          btn.setAttribute('aria-expanded', String(open));
          btn.textContent = open ? 'Show first 8 turns' : `Show all ${btn.dataset.total} turns`;
          if (open) {
            // Park the first newly-revealed turn at the top of the column, so
            // the expansion is visible instead of happening below the fold.
            body.scrollTop = rest.getBoundingClientRect().top
                           - body.getBoundingClientRect().top + body.scrollTop;
          } else {
            body.scrollTop = 0;
            col.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          }
        });
      });
    } catch (err) {
      host.innerHTML = `<p class="text-[13px]" style="color:var(--hot)">Could not load the comparison (${esc(err.message)}).</p>`;
    }
  }

  /* ───────────────────────────────────────────────────── init */

  async function init() {
    if (!document.getElementById('profile-list')) return;
    try {
      state.manifest = await getJSON(`${DATA}/manifest.json`);
    } catch (err) {
      document.getElementById('transcript').innerHTML =
        `<p class="py-8 text-center text-[13px]" style="color:var(--hot)">Could not load the session index (${esc(err.message)}).</p>`;
      return;
    }
    state.profile = profiles()[0].name;

    document.querySelectorAll('#attitude-seg button').forEach(b => {
      b.addEventListener('click', () => { state.attitude = b.dataset.att; load(); });
    });

    const tog = document.getElementById('toggle-traces');
    tog.addEventListener('click', () => {
      state.expandAll = !state.expandAll;
      tog.setAttribute('aria-pressed', String(state.expandAll));
      tog.textContent = state.expandAll ? 'Collapse all reasoning' : 'Expand all reasoning';
      applyExpandAll();
    });

    let raf;
    new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => { if (state.session) renderChart(state.session); });
    }).observe(document.getElementById('chart'));

    await load();
    renderSycophancy();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
