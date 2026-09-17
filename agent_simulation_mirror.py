"""Agent simulation CLI variant that replaces only the patient (client) modeling with MIRROR's.

This is otherwise identical to agent_simulation.py: our therapist opens and closes the session as
usual, and the full simulation/manifest/multiprocessing machinery in agent_simulation.py / simulation.py
is reused unchanged. The only difference is the patient side, which follows the MIRROR paper's client
agent (see graph/mirror_patient.py, and mirror/llm_therapist/src/agents/client.py /
mirror/llm_therapist/src/run.py for the reference implementation).

We monkeypatch graph.agent_graph.AgentTherapyGraphBuilder at import time because
agent_simulation.run_simulation() does `from graph.agent_graph import AgentTherapyGraphBuilder`
inside the function body at call time, so it always resolves the name currently bound on the
graph.agent_graph module. Multiprocessing on Linux uses fork, so worker processes inherit this
patched module state from the parent process.
"""

from __future__ import annotations

import argparse

import agent_simulation
import graph.agent_graph
from graph.mirror_patient import MirrorAgentTherapyGraphBuilder

graph.agent_graph.AgentTherapyGraphBuilder = MirrorAgentTherapyGraphBuilder


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run single-call therapist CBT therapy simulation with a MIRROR-style patient")
  parser.add_argument("--cfg", type=str, default="configs/bedrock.yaml", help="YAML configuration file.")
  parser.add_argument("--profiles-path", type=str, default=None, help="Path to patient profiles JSON/JSONL. Overrides patient.profiles in the config.")
  parser.add_argument("--n-profiles", type=int, default=None, help="Run only N new profiles from the loaded profile set.")
  parser.add_argument("--resume-from-json", type=str, default=None, help="Resume a batch from an execution manifest JSON.")
  parser.add_argument("--resume-input-dir", type=str, default=None, help="Resume direct *.json files in this directory whose session_ended_reason is max_turns, overwriting each JSON in place.")
  parser.add_argument("--eval", action="store_true", help="Evaluate all included attitudes after simulations finish.")
  parser.add_argument("--eval-max-iter", type=int, default=1, help="Evaluator samples per metric prompt.")
  parser.add_argument("--multiprocessing", action="store_true", dest="use_multiprocessing", help="Run selected batch profiles in parallel.")
  parser.add_argument("--n-processes", type=int, default=None, help="Number of worker processes to use with --multiprocessing.")
  parser.add_argument("--reasoning", action="store_true", dest="include_reasoning", help="Include therapist reasoning traces in TXT transcripts.")
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
    retry_failed=args.retry_failed,
  )
