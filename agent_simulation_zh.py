"""Agent simulation CLI variant that uses a Chinese therapist model (SoulChat2.0 / Psyche-R1) through translation.

Identical to agent_simulation.py except for the therapist side, which is handled by
graph/zh_therapist.py: the therapist model is served with vLLM and talks Chinese, while a
TranslateGemma vLLM server translates patient utterances EN->ZH and therapist utterances ZH->EN.
Config needs `therapist.zh_variant: soulchat2 | psyche_r1` and a `translator:` block
(see configs/zh_baselines/).

We monkeypatch graph.agent_graph.AgentTherapyGraphBuilder at import time because
agent_simulation.run_simulation() resolves that name from the module at call time, and
multiprocessing on Linux forks, so workers inherit the patched module.
"""

from __future__ import annotations

import argparse

import agent_simulation
import graph.agent_graph
from graph.zh_therapist import ZhTherapistGraphBuilder

graph.agent_graph.AgentTherapyGraphBuilder = ZhTherapistGraphBuilder


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run CBT therapy simulation with a Chinese therapist model (SoulChat2.0 / Psyche-R1) via translation")
  parser.add_argument("--cfg", type=str, default="configs/zh_baselines/soulchat2_wor.yaml", help="YAML configuration file.")
  parser.add_argument("--profiles-path", type=str, default=None, help="Path to patient profiles JSON/JSONL. Overrides patient.profiles in the config.")
  parser.add_argument("--n-profiles", type=int, default=None, help="Run only N new profiles from the loaded profile set.")
  parser.add_argument("--resume-from-json", type=str, default=None, help="Resume a batch from an execution manifest JSON.")
  parser.add_argument("--resume-input-dir", type=str, default=None, help="Resume direct *.json files in this directory whose session_ended_reason is max_turns, overwriting each JSON in place.")
  parser.add_argument("--eval", action="store_true", help="Evaluate all included attitudes after simulations finish.")
  parser.add_argument("--eval-max-iter", type=int, default=1, help="Evaluator samples per metric prompt.")
  parser.add_argument("--multiprocessing", action="store_true", dest="use_multiprocessing", help="Run selected batch profiles in parallel.")
  parser.add_argument("--n-processes", type=int, default=None, help="Number of worker processes to use with --multiprocessing.")
  parser.add_argument("--reasoning", action="store_true", dest="include_reasoning", help="Include therapist reasoning traces (Chinese) in TXT transcripts.")
  parser.add_argument("--cactus-patient", action="store_true", default=None, help="Use cactus patient utterance prompts for agent simulations.")
  parser.add_argument("--retry-failed", action="store_true", help="Re-execute profiles with success=false in the resume manifest.")
  args = parser.parse_args()
  agent_simulation.main(
    args.cfg,
    profiles_path=args.profiles_path,
    n_profiles=args.n_profiles,
    resume_from_json=args.resume_from_json,
    resume_input_dir=args.resume_input_dir,
    run_eval=args.eval,
    eval_max_iter=args.eval_max_iter,
    use_multiprocessing=args.use_multiprocessing,
    n_processes=args.n_processes,
    include_reasoning=args.include_reasoning,
    cactus_patient=args.cactus_patient,
    retry_failed=args.retry_failed,
  )
