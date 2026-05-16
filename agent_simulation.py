from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import multiprocessing
import os
from pathlib import Path
import queue
from typing import Any, cast

from graph.agent_graph import format_agent_trace_block

from simulation import (
  _advance_progress_bar,
  _deduplicate_profiles,
  _drain_progress_events,
  _group_profiles_by_attitude,
  _initialize_manifest,
  _load_profiles,
  _load_tqdm,
  _manifest_path,
  _profile_attitude,
  _profile_idx,
  _profile_run_batches,
  _prompt_ablation_from_cfg_path,
  _prompt_ablation_from_patient_config,
  _record_simulation_result,
  _resolve_profiles_path,
  _run_evaluation_for_attitude,
  _select_profiles_to_execute,
  _simulation_output_paths,
  _validate_multiprocessing_options,
  _write_manifest,
  load_dotenv,
)


def _prompt_ablation_with_cactus_patient(prompt_ablation: dict[str, str], cactus_patient: bool) -> dict[str, str]:
  metadata = dict(prompt_ablation)
  if cactus_patient:
    metadata["cactus_patient"] = "true"
  return metadata


def _prompt_ablation_from_cfg_path_with_cactus_patient(cfg_path: str | None, cactus_patient: bool) -> dict[str, str]:
  return _prompt_ablation_with_cactus_patient(
    _prompt_ablation_from_cfg_path(cfg_path),
    cactus_patient,
  )


def _therapist_reasoning_for_utterance(result: dict[str, Any], utterance_index: int) -> str:
  for trace_group in result.get("reasoning_traces", []):
    if not isinstance(trace_group, dict):
      continue
    if int(trace_group.get("utterance_index", 0)) != int(utterance_index):
      continue
    if str(trace_group.get("speaker", "")).strip().lower() != "therapist":
      continue
    traces = trace_group.get("traces")
    if not isinstance(traces, list):
      return ""
    return format_agent_trace_block([trace for trace in traces if isinstance(trace, dict)])
  return ""


def _write_agent_transcript(path: Path, result: dict[str, Any], include_reasoning: bool) -> None:
  messages = cast(list[dict[str, Any]], result.get("messages", []))
  with path.open("w", encoding="utf-8") as f:
    for message in messages:
      role = str(message.get("role", "")).capitalize() or "Unknown"
      if include_reasoning and str(message.get("role", "")).strip().lower() == "therapist":
        reasoning = _therapist_reasoning_for_utterance(result, int(message.get("utterance_index", 0)))
        if reasoning:
          f.write(f"--> Therapist Reasoning:\n<think>{reasoning}</think>\n\n")
      f.write(f'--> {role}: {message.get("content", "")}\n\n')


def _write_agent_simulation_summary(result: dict[str, Any], writer: Callable[[str], None] = print) -> None:
  messages = [message for message in result.get("messages", []) if isinstance(message, dict)]
  therapist_turns = sum(1 for message in messages if message.get("role") == "therapist")
  patient_turns = sum(1 for message in messages if message.get("role") == "patient")
  writer(f'  Total utterances: {result.get("utterance_index", 0)}')
  writer(f"  Therapist utterances: {therapist_turns}")
  writer(f"  Patient utterances: {patient_turns}")
  writer(f'  End reason: {result.get("session_ended_reason", "unknown")}')
  writer("")


def _print_agent_simulation_header(
  run_index: int,
  total_to_run: int,
  profile: dict[str, Any],
  *,
  writer: Callable[[str], None] = print,
) -> None:
  attitude = _profile_attitude(profile)
  writer(f'{"="*60}')
  writer(f"Simulation {run_index}/{total_to_run} — idx: {_profile_idx(profile)} — attitude: {attitude}")
  writer(f'{"="*60}')


def run_simulation(
  cfg_path: str,
  profile: dict[str, Any],
  max_turns: int,
  min_turns: int,
  output_dir: str = "",
  verbose: bool = False,
  resume_state: dict[str, Any] | None = None,
  *,
  progress_callback: Callable[[int], None] | None = None,
  suppress_output: bool = False,
  print_saved_paths: bool = True,
  include_reasoning: bool = False,
  cactus_patient: bool = False,
) -> dict[str, Any]:
  result: dict[str, Any] = {
    "messages": [],
    "reasoning_traces": [],
    "utterance_index": 0,
    "session_ended": True,
    "session_ended_reason": "",
    "conversation_error": "",
    "patient_profile": profile,
    "prompt_ablation": _prompt_ablation_from_cfg_path_with_cactus_patient(cfg_path, cactus_patient),
  }
  error_printed = False
  last_reported_utterance = 0

  def report_progress(utterance_index: Any) -> None:
    nonlocal last_reported_utterance
    if progress_callback is None:
      return
    try:
      current = max(0, int(utterance_index or 0))
    except (TypeError, ValueError):
      return
    if current <= last_reported_utterance:
      return
    progress_callback(current)
    last_reported_utterance = current

  with ExitStack() as stack:
    if suppress_output:
      devnull = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
      stack.enter_context(redirect_stdout(devnull))
      stack.enter_context(redirect_stderr(devnull))

    try:
      from graph.agent_graph import AgentTherapyGraphBuilder

      if resume_state is not None:
        raise ValueError("agent_simulation.py does not support partial-session resume state.")
      builder = AgentTherapyGraphBuilder(cfg_path, profile, cactus_patient=cactus_patient)
      chain = builder.create_chain()
      initial_state = builder.initial_state(max_turns=max_turns, min_turns=min_turns)
      result = initial_state
      report_progress(initial_state.get("utterance_index", 0))

      if verbose or progress_callback is not None:
        if verbose:
          print()
        last_msg_count = len(initial_state.get("messages", []))
        for chunk in chain.stream(initial_state, stream_mode="values"):
          messages = chunk.get("messages", [])
          if verbose:
            for msg in messages[last_msg_count:]:
              resistance = msg.get("resistance")
              resistance_text = f" ({float(resistance):.2f})" if resistance is not None else ""
              print(f'    {msg["role"].capitalize()}{resistance_text}: {msg["content"]}')
          last_msg_count = len(messages)
          result = chunk
          report_progress(result.get("utterance_index", 0))
        if verbose and result.get("conversation_error"):
          print(f'ERROR: {result["conversation_error"]}')
          error_printed = True
        if verbose:
          print()
      else:
        result = chain.invoke(initial_state)
        report_progress(result.get("utterance_index", 0))
    except Exception as exc:
      result = dict(result)
      result["session_ended"] = True
      result["session_ended_reason"] = result.get("session_ended_reason") or "conversation_error"
      result["conversation_error"] = str(exc)
      result.setdefault("patient_profile", profile)
      if verbose:
        print(f'ERROR: {result["conversation_error"]}')
        error_printed = True

    report_progress(result.get("utterance_index", 0))

    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
      transcript_file, json_file = _simulation_output_paths(output_dir, profile)
      _write_agent_transcript(transcript_file, result, include_reasoning)
      with json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
      if print_saved_paths:
        print(f"  Saved transcript: {transcript_file}")
        print(f"  Saved state JSON: {json_file}")

    if verbose and result.get("conversation_error") and not error_printed:
      print(f'ERROR: {result["conversation_error"]}')

  return result


def _run_simulation_batch_worker(
  args: tuple[int, str, dict[str, Any], int, int, str, bool, bool, Any],
) -> dict[str, Any]:
  run_index, cfg_path, profile, max_turns, min_turns, output_dir, include_reasoning, cactus_patient, progress_queue = args

  def progress_callback(utterance_index: int) -> None:
    progress_queue.put((run_index, int(utterance_index)))

  result = run_simulation(
    cfg_path,
    profile,
    max_turns,
    min_turns,
    output_dir,
    verbose=False,
    progress_callback=progress_callback,
    suppress_output=True,
    print_saved_paths=False,
    include_reasoning=include_reasoning,
    cactus_patient=cactus_patient,
  )
  progress_callback(int(result.get("utterance_index", 0) or 0))
  return {
    "run_index": run_index,
    "profile": profile,
    "result": result,
  }


def _handle_completed_multiprocessing_result(
  payload: dict[str, Any],
  *,
  total_to_run: int,
  output_dir: str | Path,
  manifest: dict[str, Any],
  writer: Callable[[str], None],
) -> None:
  run_index = int(payload["run_index"])
  profile = cast(dict[str, Any], payload["profile"])
  result = cast(dict[str, Any], payload["result"])
  _print_agent_simulation_header(run_index, total_to_run, profile, writer=writer)
  _write_agent_simulation_summary(result, writer=writer)
  _record_simulation_result(manifest, profile, output_dir, result)


def _run_batch_sequential(
  *,
  cfg_path: str,
  profiles_to_run: list[dict[str, Any]],
  manifest: dict[str, Any],
  max_turns: int,
  min_turns: int,
  output_dir: str,
  verbose: bool,
  include_reasoning: bool,
  cactus_patient: bool,
) -> None:
  total_to_run = len(profiles_to_run)
  for run_index, profile in enumerate(profiles_to_run, start=1):
    _print_agent_simulation_header(run_index, total_to_run, profile)
    result = run_simulation(
      cfg_path,
      profile,
      max_turns,
      min_turns,
      output_dir,
      verbose,
      include_reasoning=include_reasoning,
      cactus_patient=cactus_patient,
    )
    _write_agent_simulation_summary(result)
    _record_simulation_result(manifest, profile, output_dir, result)


def _run_batch_multiprocessing(
  *,
  cfg_path: str,
  profiles_to_run: list[dict[str, Any]],
  manifest: dict[str, Any],
  max_turns: int,
  min_turns: int,
  output_dir: str,
  n_processes: int,
  include_reasoning: bool,
  cactus_patient: bool,
) -> None:
  if not profiles_to_run:
    return

  tqdm = _load_tqdm()
  total_to_run = len(profiles_to_run)

  with multiprocessing.Manager() as manager:
    progress_queue = manager.Queue()
    with multiprocessing.Pool(processes=n_processes) as pool:
      for batch_start, batch_profiles in _profile_run_batches(profiles_to_run, n_processes):
        bars: dict[int, Any] = {}
        progress_counts = {
          batch_start + batch_offset: 0
          for batch_offset in range(len(batch_profiles))
        }
        completion_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def on_success(payload: dict[str, Any]) -> None:
          completion_queue.put(("result", payload))

        def on_error(exc: BaseException) -> None:
          completion_queue.put(("error", exc))

        try:
          for batch_offset, profile in enumerate(batch_profiles):
            run_index = batch_start + batch_offset
            desc = f"{run_index}/{total_to_run} idx:{_profile_idx(profile)} {_profile_attitude(profile)}"
            bars[run_index] = tqdm(total=max_turns, desc=desc, position=run_index - 1, leave=True, unit="utt")
            pool.apply_async(
              _run_simulation_batch_worker,
              args=((run_index, cfg_path, profile, max_turns, min_turns, output_dir, include_reasoning, cactus_patient, progress_queue),),
              callback=on_success,
              error_callback=on_error,
            )

          completed = 0
          while completed < len(batch_profiles):
            _drain_progress_events(progress_queue, progress_counts, bars, max_turns)
            try:
              kind, payload = completion_queue.get(timeout=0.1)
            except queue.Empty:
              continue

            _drain_progress_events(progress_queue, progress_counts, bars, max_turns)
            if kind == "error":
              raise RuntimeError(f"Multiprocessing simulation worker failed: {payload}") from payload

            result_payload = cast(dict[str, Any], payload)
            run_index = int(result_payload["run_index"])
            result = cast(dict[str, Any], result_payload["result"])
            _advance_progress_bar(
              progress_counts,
              bars,
              run_index,
              int(result.get("utterance_index", 0) or 0),
              max_turns,
            )
            status = "error" if result.get("conversation_error") else "done"
            bars[run_index].set_postfix_str(status)
            bars[run_index].refresh()

            _handle_completed_multiprocessing_result(
              result_payload,
              total_to_run=total_to_run,
              output_dir=output_dir,
              manifest=manifest,
              writer=tqdm.write,
            )
            completed += 1

          _drain_progress_events(progress_queue, progress_counts, bars, max_turns)
        finally:
          for bar in bars.values():
            bar.close()


def main(
  cfg_path: str,
  *,
  profiles_path: str | None = None,
  n_profiles: int | None = None,
  resume_from_json: str | None = None,
  run_eval: bool = False,
  eval_max_iter: int = 1,
  use_multiprocessing: bool = False,
  n_processes: int | None = None,
  include_reasoning: bool = False,
  cactus_patient: bool = False,
) -> None:
  from runtime_config import load_runtime_config

  load_dotenv()

  cfg = load_runtime_config(cfg_path)

  max_turns = cfg.env.max_turns
  min_turns = cfg.env.get("min_turns", 0)
  output_dir = str(cfg.env.output_dir or "")
  verbose = bool(cfg.env.verbose)
  resume_file = cfg.env.get("resume_file")
  prompt_ablation = _prompt_ablation_with_cactus_patient(
    _prompt_ablation_from_patient_config(cfg.patient),
    cactus_patient,
  )

  if resume_file:
    raise ValueError("agent_simulation.py does not support cfg.env.resume_file partial-session resume.")
  if n_profiles is not None and n_profiles < 1:
    raise ValueError("--n-profiles must be a positive integer.")
  if eval_max_iter < 1:
    raise ValueError("--eval-max-iter must be a positive integer.")

  _validate_multiprocessing_options(use_multiprocessing, n_processes)

  if not output_dir:
    raise ValueError("cfg.env.output_dir is required for profile batch simulation manifests.")

  profiles_path = _resolve_profiles_path(profiles_path, cfg.patient.profiles)
  profiles, duplicate_idxs = _deduplicate_profiles(_load_profiles(profiles_path))
  grouped = _group_profiles_by_attitude(profiles)
  manifest = _initialize_manifest(
    cfg_path=cfg_path,
    profiles_path=profiles_path,
    output_dir=output_dir,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
    resume_from_json=resume_from_json,
    prompt_ablation=prompt_ablation,
  )

  print(f"Loaded {len(profiles)} unique profiles from {profiles_path}")
  if duplicate_idxs:
    print(f"Omitted duplicate profile idxs: {', '.join(duplicate_idxs)}")
  print(f"Included attitudes: {', '.join(sorted(grouped))}\n")

  _write_manifest(_manifest_path(output_dir), manifest)

  profiles_to_run = _select_profiles_to_execute(
    profiles,
    manifest,
    n_profiles=n_profiles,
  )
  print(f"Profiles selected for execution: {len(profiles_to_run)}\n")

  if use_multiprocessing:
    print(f"Multiprocessing enabled with {n_processes} processes.\n")
    _run_batch_multiprocessing(
      cfg_path=cfg_path,
      profiles_to_run=profiles_to_run,
      manifest=manifest,
      max_turns=max_turns,
      min_turns=min_turns,
      output_dir=output_dir,
      n_processes=cast(int, n_processes),
      include_reasoning=include_reasoning,
      cactus_patient=cactus_patient,
    )
  else:
    _run_batch_sequential(
      cfg_path=cfg_path,
      profiles_to_run=profiles_to_run,
      manifest=manifest,
      max_turns=max_turns,
      min_turns=min_turns,
      output_dir=output_dir,
      verbose=verbose,
      include_reasoning=include_reasoning,
      cactus_patient=cactus_patient,
    )

  if run_eval:
    for attitude in sorted(grouped):
      print(f'{"="*60}')
      print(f"Evaluating attitude: {attitude}")
      print(f'{"="*60}')
      scores = _run_evaluation_for_attitude(output_dir, attitude, eval_max_iter)
      manifest["scores"][attitude] = scores
      _write_manifest(_manifest_path(output_dir), manifest)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run single-call therapist CBT therapy simulation")
  parser.add_argument("--cfg", type=str, default="configs/bedrock.yaml", help="YAML configuration file.")
  parser.add_argument("--profiles-path", type=str, default=None, help="Path to patient profiles JSON/JSONL. Overrides patient.profiles in the config.")
  parser.add_argument("--n-profiles", type=int, default=None, help="Run only N new profiles from the loaded profile set.")
  parser.add_argument("--resume-from-json", type=str, default=None, help="Resume a batch from an execution manifest JSON.")
  parser.add_argument("--eval", action="store_true", help="Evaluate all included attitudes after simulations finish.")
  parser.add_argument("--eval-max-iter", type=int, default=1, help="Evaluator samples per metric prompt.")
  parser.add_argument("--multiprocessing", action="store_true", dest="use_multiprocessing", help="Run selected batch profiles in parallel.")
  parser.add_argument("--n-processes", type=int, default=None, help="Number of worker processes to use with --multiprocessing.")
  parser.add_argument("--reasoning", action="store_true", dest="include_reasoning", help="Include therapist reasoning traces in TXT transcripts.")
  parser.add_argument("--cactus-patient", action="store_true", help="Use cactus patient utterance prompts for agent simulations.")
  args = parser.parse_args()
  main(
    args.cfg,
    profiles_path=args.profiles_path,
    n_profiles=args.n_profiles,
    resume_from_json=args.resume_from_json,
    run_eval=args.eval,
    eval_max_iter=args.eval_max_iter,
    use_multiprocessing=args.use_multiprocessing,
    n_processes=args.n_processes,
    include_reasoning=args.include_reasoning,
    cactus_patient=args.cactus_patient,
  )
