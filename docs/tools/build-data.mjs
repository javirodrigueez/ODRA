#!/usr/bin/env node
/**
 * Convert raw ODRA simulation dumps into compact JSON for the project page.
 *
 * Reads  ../project-page/*.json             (3 profiles x 3 attitudes)
 *        ../project-page-sycophancy/*.json  (ODRA vs CACTUS vs MAGNeT)
 * Writes assets/data/sessions/<id>.json, manifest.json, sycophancy.json
 */
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(HERE, '..');
const ROOT = resolve(SITE, '..');
const SRC_SESS = join(ROOT, 'project-page');
const SRC_SYC = join(ROOT, 'project-page-sycophancy');
const OUT = join(SITE, 'assets', 'data');

/** Human-readable labels for every reasoning step the framework emits. */
const STEP_LABELS = {
  resistance_update: 'Resistance update',
  patient_action_selection: 'Action selection',
  tib_check: 'TIB check',
  resistance_prediction: 'Resistance estimate',
  address_tib_enter: 'TIB — enter handler',
  address_tib_plan_generation: 'TIB — plan',
  address_tib_exit: 'TIB — exit handler',
  address_tib_repeat_inline_suppressed: 'TIB — repeat suppressed',
  opening_info_update: 'Opening — info update',
  opening_readiness_check: 'Opening — readiness check',
  initial_cognitive_model_bootstrap: 'Cognitive model — bootstrap',
  cognitive_gap_planning: 'Cognitive model — gap plan',
  cognitive_gap_response_review: 'Cognitive model — gap review',
  cognitive_refinement_planning: 'Cognitive model — refine plan',
  cognitive_refinement_response_review: 'Cognitive model — refine review',
  core_belief_probability: 'Core belief inference',
  distortion_probability: 'Distortion inference',
  id_distortions_completion_check: 'Conceptualization — stage check',
  tib_avoidance_new_cognitive_set: 'Avoidance — new cognitive set',
  technique_selection: 'Technique selection',
  technique_fit_check: 'Technique fit check',
  technique_plan: 'Technique plan',
  technique_completion_check: 'Technique — stage check',
  technique_evaluation: 'Technique evaluation',
  homework_task_proposal: 'Homework proposal',
  homework_agreement_check: 'Homework agreement check',
  meta_conversation_end_session_redirect: 'Meta — end-session redirect',
  meta_conversation_turn_cap: 'Meta — turn cap',
};

const STAGE_LABELS = {
  opening: 'Opening',
  id_distortions: 'Cognitive Conceptualization',
  cbt_technique: 'CBT Work',
  homework: 'Homework',
  closing: 'Closing',
  address_tib: 'TIB Interruption',
  meta_conversation: 'Meta-conversation',
};

const jload = p => JSON.parse(readFileSync(p, 'utf8'));

const titleCase = s =>
  s.split('_').filter(Boolean).map(w => w[0].toUpperCase() + w.slice(1)).join(' ');

const stepLabel = step =>
  STEP_LABELS[step] ?? (step ? step.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase()) : '');

/** Trace outputs are JSON strings, except for a handful of plain-text notes. */
function parseOutput(raw) {
  try {
    const v = JSON.parse(raw);
    return v && typeof v === 'object' && !Array.isArray(v) ? v : { note: String(v) };
  } catch {
    return { note: raw };
  }
}

function clean(text, limit = 1600) {
  if (typeof text !== 'string') return text;
  const t = text.trim();
  return t.length <= limit ? t : t.slice(0, limit - 1).replace(/\s+$/, '') + '…';
}

const isNum = v => typeof v === 'number' && Number.isFinite(v);

function buildSession(path) {
  const d = jload(path);
  const sid = basename(path, '.json');
  const prof = d.patient_profile ?? {};
  const info = prof.personal_info ?? {};

  // Index traces by utterance so each message carries its own reasoning.
  const byIdx = new Map();
  for (const entry of d.reasoning_traces ?? []) {
    const list = byIdx.get(entry.utterance_index) ?? [];
    list.push(...(entry.traces ?? []));
    byIdx.set(entry.utterance_index, list);
  }

  const turns = [], rSeries = [], pSeries = [];
  let lastR = d.resistance_baseline ?? null;
  let lastP = null;

  for (const msg of d.messages ?? []) {
    const idx = msg.utterance_index;
    const role = msg.role;
    const tracesOut = [];
    let tib = null, action = null;

    for (const t of byIdx.get(idx) ?? []) {
      const step = t.step ?? '';
      const out = parseOutput(t.output ?? '');
      const rec = { step, label: stepLabel(step), reasoning: clean(out.reasoning ?? out.note ?? '') };

      if (step === 'resistance_update') {
        rec.math = {
          previous_r: out.previous_r ?? null,
          delta_llm: out.delta_llm ?? null,
          baseline: out.baseline ?? null,
          alpha: out.alpha ?? null,
          decay: out.decay_vector ?? null,
          new_r: out.new_r ?? null,
        };
        if (isNum(out.new_r)) lastR = out.new_r;
      } else if (step === 'patient_action_selection') {
        rec.dist = {
          prior: out.prior_distribution ?? {},
          context: out.context_distribution ?? {},
          fused: out.action_distribution ?? {},
          selected: out.selected_action ?? null,
        };
        action = out.selected_action ?? null;
      } else if (step === 'resistance_prediction') {
        rec.pred = {
          p_target: out.p_target ?? null,
          clarity: out.signal_clarity ?? null,
          new_p: out.new_p ?? null,
          clipped: Boolean(out.delta_clip_applied),
          mode: out.prediction_mode ?? null,
        };
        if (isNum(out.new_p)) lastP = out.new_p;
      } else if (step === 'tib_check') {
        if (out.has_tib) tib = { name: out.tib_name ?? null, description: clean(out.tib_description, 400) };
        rec.tib = { has_tib: Boolean(out.has_tib), name: out.tib_name ?? null };
      } else if (step === 'technique_plan' || step === 'technique_selection') {
        rec.extra = clean(out.plan ?? out.technique ?? '', 700);
      } else if (step === 'homework_task_proposal') {
        rec.extra = clean(out.task_title ?? '', 200);
      }

      tracesOut.push(rec);
    }

    const stage = msg.stage ?? '';
    const turn = {
      i: idx,
      role,
      text: msg.content ?? '',
      stage,
      stageLabel: STAGE_LABELS[stage] ?? titleCase(stage),
      substage: msg.substage ?? '',
      traces: tracesOut,
    };
    if (role === 'patient') {
      turn.r = lastR;
      if (action) turn.action = action;
      if (lastR !== null) rSeries.push({ i: idx, v: lastR });
    } else {
      turn.p = lastP;
      if (tib) turn.tib = tib;
      if (lastP !== null) pSeries.push({ i: idx, v: lastP });
    }
    turns.push(turn);
  }

  // Contiguous stage runs drive the timeline bands under the chart.
  const stageRuns = [];
  for (const t of turns) {
    const last = stageRuns.at(-1);
    if (last && last.stage === t.stage) last.end = t.i;
    else stageRuns.push({ stage: t.stage, label: t.stageLabel, start: t.i, end: t.i });
  }

  const tibEvents = turns.filter(t => t.tib).map(t => ({ i: t.i, name: t.tib.name ?? null }));

  return {
    id: sid,
    attitude: prof.attitude ?? null,
    patient: {
      name: info.name ?? null,
      age: info.age ?? null,
      gender: info.gender ?? null,
      occupation: info.occupation ?? null,
      problem: clean(prof.presenting_problem, 900),
      reason: clean(prof.reason_seeking_counseling, 500),
    },
    baseline: d.resistance_baseline ?? null,
    alpha: d.resistance_recovery_rate ?? null,
    finalR: d.resistance ?? null,
    endedReason: d.session_ended_reason ?? null,
    turns,
    stageRuns,
    rSeries,
    pSeries,
    tibEvents,
    cognitiveModels: (d.cognitive_model_sets ?? []).map(c => ({
      situation: clean(c.situation, 300),
      automatic_thoughts: c.automatic_thoughts ?? [],
      emotions: c.emotions ?? [],
      behaviors: c.behaviors ?? [],
      core_belief: c.core_belief ?? null,
      distortion: c.distortion ?? null,
    })),
    techniques: (d.selected_techniques ?? []).map(t => ({
      name: t.name ?? null,
      target: t.target_distortion ?? null,
      rationale: clean(t.rationale, 600),
    })),
    homework: (d.proposed_tasks ?? []).map(t => ({
      title: t.task_title ?? null,
      description: clean(t.task_description, 700),
    })),
  };
}

/** CACTUS and MAGNeT store their dialogue under different keys and role names. */
function normBaseline(path, method) {
  const d = jload(path);
  const roleMap = { counselor: 'therapist', client: 'patient', therapist: 'therapist', patient: 'patient' };
  const raw = d.messages ?? d.dialogue ?? d.history ?? [];
  const turns = raw.map((m, n) => ({
    i: m.utterance_index ?? n + 1,
    role: roleMap[m.role] ?? m.role ?? '',
    text: m.content ?? m.message ?? '',
  }));
  return { method, turns, nTurns: turns.length };
}

function main() {
  mkdirSync(join(OUT, 'sessions'), { recursive: true });

  const files = readdirSync(SRC_SESS).filter(f => /^simulation_.*\.json$/.test(f)).sort();
  const manifest = [];

  for (const f of files) {
    const s = buildSession(join(SRC_SESS, f));
    const dest = join(OUT, 'sessions', `${s.id}.json`);
    writeFileSync(dest, JSON.stringify(s));
    const nTraces = s.turns.reduce((a, t) => a + t.traces.length, 0);
    manifest.push({
      id: s.id,
      name: s.patient.name,
      age: s.patient.age,
      occupation: s.patient.occupation,
      attitude: s.attitude,
      baseline: s.baseline,
      finalR: s.finalR,
      nTurns: s.turns.length,
      nTraces,
      nTibs: s.tibEvents.length,
      problem: s.patient.problem,
      bytes: statSync(dest).size,
    });
    console.log(
      `  ${s.id.padEnd(28)} ${String(s.attitude).padEnd(8)} turns=${String(s.turns.length).padStart(3)} ` +
      `traces=${String(nTraces).padStart(3)} tibs=${String(s.tibEvents.length).padStart(2)} ` +
      `${(statSync(dest).size / 1024).toFixed(1).padStart(7)} KB`
    );
  }

  writeFileSync(join(OUT, 'manifest.json'), JSON.stringify({ sessions: manifest }));

  // Sycophancy trio: one patient, one attitude, three generation methods.
  const odra = buildSession(join(SRC_SYC, 'simulation_134_negative.json'));
  const syc = {
    profile: odra.patient,
    methods: [
      {
        method: 'ODRA',
        nTurns: odra.turns.length,
        turns: odra.turns.map(t => ({ i: t.i, role: t.role, text: t.text })),
      },
      normBaseline(join(SRC_SYC, 'cactus_session_135_negative_jessica_hughes.json'), 'CACTUS'),
      normBaseline(join(SRC_SYC, 'magnet_session_135_negative_jessica_hughes.json'), 'MAGNeT'),
    ],
  };
  writeFileSync(join(OUT, 'sycophancy.json'), JSON.stringify(syc));
  console.log('  sycophancy.json  ' + syc.methods.map(m => `${m.method}=${m.nTurns}t`).join(', '));
}

main();
