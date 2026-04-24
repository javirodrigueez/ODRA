from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from collections import defaultdict
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
from typing import Any, cast

from graph.resistance import resolve_prompt_ablation_settings

try:
  from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
except ImportError:
  def load_dotenv(*args: Any, **kwargs: Any) -> None:
    return None


def _load_profiles(profiles_path: str) -> list[dict[str, Any]]:
  if profiles_path.endswith(".jsonl"):
    profiles = []
    with open(profiles_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        profiles.append(json.loads(line))
    return profiles

  with open(profiles_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  if isinstance(payload, list):
    return payload
  if isinstance(payload, dict):
    return [payload]
  raise ValueError(f"Unsupported profiles payload in {profiles_path}")


def _resolve_profiles_path(cli_profiles_path: str | None, cfg_profiles_path: Any) -> str:
  profiles_path = str(cli_profiles_path or cfg_profiles_path or "").strip()
  if not profiles_path:
    raise ValueError("Patient profiles path is required. Pass --profiles-path or set patient.profiles in the config.")
  return profiles_path


def _load_resume_state(resume_path: str) -> dict[str, Any]:
  with open(resume_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise ValueError(f"Unsupported resume payload in {resume_path}: expected a JSON object.")
  profile = payload.get("patient_profile")
  if not isinstance(profile, dict):
    raise ValueError(f"Resume file {resume_path} is missing a valid patient_profile object.")
  return payload


def _profile_idx(profile: dict[str, Any]) -> str:
  return str(profile.get("idx", "unknown"))


def _profile_attitude(profile: dict[str, Any]) -> str:
  return str(profile.get("attitude") or "unknown").strip().lower() or "unknown"


def _prompt_ablation_from_patient_config(patient_cfg: Any) -> dict[str, str]:
  if isinstance(patient_cfg, dict):
    return resolve_prompt_ablation_settings(
      patient_cfg.get("prompt_variant"),
    )
  return resolve_prompt_ablation_settings(
    getattr(patient_cfg, "prompt_variant", None),
  )


def _prompt_ablation_from_cfg_path(cfg_path: str | None) -> dict[str, str]:
  if not cfg_path:
    return resolve_prompt_ablation_settings()
  try:
    from runtime_config import load_runtime_config

    cfg = load_runtime_config(cfg_path)
  except Exception:
    return resolve_prompt_ablation_settings()
  return _prompt_ablation_from_patient_config(getattr(cfg, "patient", {}))


def _output_dir_manifest_prefix(output_dir: str | Path) -> str:
  normalized = Path(output_dir).as_posix().strip("/")
  return normalized.replace("/", "_") or "output"


def _manifest_path(output_dir: str | Path) -> Path:
  prefix = _output_dir_manifest_prefix(output_dir)
  return Path(output_dir) / f"{prefix}.json"


def _simulation_output_paths(output_dir: str | Path, profile: dict[str, Any]) -> tuple[Path, Path]:
  idx = _profile_idx(profile)
  attitude = _profile_attitude(profile)
  base = Path(output_dir)
  base_json = base / "json" 
  base_json.mkdir(parents=True, exist_ok=True)
  base_json = base_json / f"simulation_{idx}_{attitude}.json"
  base_txt = base / "transcripts" 
  base_txt.mkdir(parents=True, exist_ok=True)
  base_txt = base_txt / f"simulation_{idx}_{attitude}.txt"

  return (
    base_txt,
    base_json,
  )


def _deduplicate_profiles(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
  seen: set[str] = set()
  deduplicated: list[dict[str, Any]] = []
  duplicate_idxs: list[str] = []
  for i, raw_profile in enumerate(profiles):
    profile = dict(raw_profile)
    if "idx" not in profile:
      profile["idx"] = i + 1
    idx = _profile_idx(profile)
    if idx in seen:
      duplicate_idxs.append(idx)
      continue
    seen.add(idx)
    deduplicated.append(profile)
  return deduplicated, duplicate_idxs


def _group_profiles_by_attitude(profiles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for profile in profiles:
    grouped[_profile_attitude(profile)].append(profile)
  return dict(grouped)


def _default_profile_status(output_dir: str | Path, profile: dict[str, Any]) -> dict[str, Any]:
  transcript_file, json_file = _simulation_output_paths(output_dir, profile)
  return {
    "attitude": _profile_attitude(profile),
    "executed": False,
    "success": False,
    "output_json": str(json_file),
    "output_transcript": str(transcript_file),
    "error": "",
  }


def _new_manifest(
  *,
  cfg_path: str,
  profiles_path: str,
  output_dir: str | Path,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
  prompt_ablation: dict[str, str] | None = None,
) -> dict[str, Any]:
  attitudes = sorted(_group_profiles_by_attitude(profiles))
  return {
    "profiles": {
      _profile_idx(profile): _default_profile_status(output_dir, profile)
      for profile in profiles
    },
    "scores": {},
    "metadata": {
      "cfg": cfg_path,
      "profiles_path": profiles_path,
      "output_dir": str(output_dir),
      "attitudes": attitudes,
      "total_profiles": len(profiles),
      "executed_count": 0,
      "success_count": 0,
      "duplicate_idxs_omitted": duplicate_idxs,
      "prompt_ablation": dict(prompt_ablation or resolve_prompt_ablation_settings()),
      "updated_at": "",
    },
  }


def _load_manifest(path: str | Path) -> dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise ValueError(f"Manifest must be a JSON object: {path}")
  if not isinstance(payload.get("profiles"), dict):
    raise ValueError(f"Manifest is missing a profiles object: {path}")
  payload.setdefault("scores", {})
  payload.setdefault("metadata", {})
  return payload


def _merge_manifest_with_profiles(
  existing: dict[str, Any] | None,
  *,
  cfg_path: str,
  profiles_path: str,
  output_dir: str | Path,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
  prompt_ablation: dict[str, str] | None = None,
) -> dict[str, Any]:
  fresh = _new_manifest(
    cfg_path=cfg_path,
    profiles_path=profiles_path,
    output_dir=output_dir,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
    prompt_ablation=prompt_ablation,
  )
  if existing is None:
    return fresh

  existing_profiles = existing.get("profiles", {})
  if isinstance(existing_profiles, dict):
    for profile in profiles:
      idx = _profile_idx(profile)
      existing_status = existing_profiles.get(idx)
      if isinstance(existing_status, dict):
        status = dict(fresh["profiles"][idx])
        status.update(existing_status)
        fresh["profiles"][idx] = status

  if isinstance(existing.get("scores"), dict):
    fresh["scores"] = existing["scores"]
  return fresh


def _refresh_manifest_metadata(manifest: dict[str, Any]) -> None:
  profiles = manifest.get("profiles", {})
  if not isinstance(profiles, dict):
    profiles = {}
  executed_count = 0
  success_count = 0
  for status in profiles.values():
    if not isinstance(status, dict):
      continue
    if status.get("executed"):
      executed_count += 1
    if status.get("success"):
      success_count += 1

  metadata = manifest.setdefault("metadata", {})
  metadata["total_profiles"] = len(profiles)
  metadata["executed_count"] = executed_count
  metadata["success_count"] = success_count
  metadata["updated_at"] = datetime.now(timezone.utc).isoformat()


def _write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
  _refresh_manifest_metadata(manifest)
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=True)


def _initialize_manifest(
  *,
  cfg_path: str,
  profiles_path: str,
  output_dir: str | Path,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
  resume_from_json: str | None,
  prompt_ablation: dict[str, str] | None = None,
) -> dict[str, Any]:
  existing = _load_manifest(Path(resume_from_json).resolve()) if resume_from_json else None
  return _merge_manifest_with_profiles(
    existing,
    cfg_path=cfg_path,
    profiles_path=profiles_path,
    output_dir=output_dir,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
    prompt_ablation=prompt_ablation,
  )


def _select_profiles_to_execute(
  profiles: list[dict[str, Any]],
  manifest: dict[str, Any],
  *,
  n_profiles: int | None,
) -> list[dict[str, Any]]:
  selected: list[dict[str, Any]] = []
  for profile in profiles:
    idx = _profile_idx(profile)
    status = manifest["profiles"].get(idx, {})
    if isinstance(status, dict) and status.get("executed"):
      continue
    selected.append(profile)
    if n_profiles is not None and len(selected) >= n_profiles:
      break
  return selected


def _mark_profile_result(
  manifest: dict[str, Any],
  profile: dict[str, Any],
  output_dir: str | Path,
  result: dict[str, Any],
) -> None:
  idx = _profile_idx(profile)
  transcript_file, json_file = _simulation_output_paths(output_dir, profile)
  error = str(result.get("conversation_error") or "")
  manifest["profiles"][idx] = {
    "attitude": _profile_attitude(profile),
    "executed": True,
    "success": not bool(error),
    "output_json": str(json_file),
    "output_transcript": str(transcript_file),
    "error": error,
  }


def _matching_simulation_files(output_dir: str | Path, attitude: str) -> list[Path]:
  return sorted(Path(output_dir).glob(f"simulation_*_{attitude}.json"))


def _run_evaluation_for_attitude(output_dir: str | Path, attitude: str, eval_max_iter: int) -> dict[str, Any]:
  matching_files = _matching_simulation_files(output_dir, attitude)
  if not matching_files:
    raise RuntimeError(f"No simulation JSON files found for attitude '{attitude}' in {output_dir}")

  repo_root = Path(__file__).resolve().parent
  eval_dir = Path(output_dir) / "eval" / attitude
  eval_dir.mkdir(parents=True, exist_ok=True)

  with tempfile.TemporaryDirectory(prefix=f"reasoncbt-eval-{attitude}-") as tmp:
    tmp_path = Path(tmp)
    for source in matching_files:
      shutil.copy2(source, tmp_path / source.name)

    subprocess.run(
      [
        "bash",
        str(repo_root / "evaluation" / "run_all_metrics.sh"),
        str(tmp_path),
        str(eval_dir),
        str(eval_max_iter),
      ],
      cwd=repo_root,
      check=True,
    )

  rating_cmd = [
    sys.executable,
    str(repo_root / "evaluation" / "rating_metrics.py"),
    "--eval-dir",
    str(eval_dir),
  ]
  completed = subprocess.run(
    rating_cmd,
    cwd=repo_root,
    check=True,
    text=True,
    capture_output=True,
  )
  if completed.stderr:
    print(completed.stderr, file=sys.stderr, end="")
  print(completed.stdout, end="")
  return cast(dict[str, Any], json.loads(completed.stdout))


def _print_simulation_summary(result: dict[str, Any]) -> None:
  _write_simulation_summary(result)


def _write_simulation_summary(result: dict[str, Any], writer: Callable[[str], None] = print) -> None:
  exchange_counts = cast(dict[str, list[int]], result.get("exchange_counts", {}))
  exchanges = sum(item for sublist in exchange_counts.values() for item in sublist)
  writer(f'  Total utterances: {result.get("utterance_index", 0)}')
  writer(f"  Total exchanges: {exchanges}")
  writer(f'  End reason: {result.get("session_ended_reason", "unknown")}')
  writer("")


def _print_simulation_header(
  run_index: int,
  total_to_run: int,
  profile: dict[str, Any],
  *,
  writer: Callable[[str], None] = print,
) -> None:
  attitude = _profile_attitude(profile)
  writer(f'{"="*60}')
  writer(f"Simulation {run_index}/{total_to_run} \u2014 idx: {_profile_idx(profile)} \u2014 attitude: {attitude}")
  writer(f'{"="*60}')


def _record_simulation_result(
  manifest: dict[str, Any],
  profile: dict[str, Any],
  output_dir: str | Path,
  result: dict[str, Any],
) -> None:
  _mark_profile_result(manifest, profile, output_dir, result)
  _write_manifest(_manifest_path(output_dir), manifest)


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
) -> dict[str, Any]:
  """Run a single simulation for a given profile."""
  result: dict[str, Any] = {
    "messages": [],
    "exchange_counts": {},
    "utterance_index": 0,
    "session_ended": True,
    "session_ended_reason": "",
    "conversation_error": "",
    "patient_profile": profile,
    "prompt_ablation": _prompt_ablation_from_cfg_path(cfg_path),
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
      from graph.graph import TherapyGraphBuilder

      builder = TherapyGraphBuilder(cfg_path, profile)
      chain = builder.create_chain()
      if resume_state is None:
        initial_state = builder.initial_state(max_turns=max_turns, min_turns=min_turns)
      else:
        initial_state = builder.resume_state(deepcopy(resume_state), max_turns=max_turns, min_turns=min_turns)
      result = initial_state
      report_progress(initial_state.get("utterance_index", 0))

      if verbose or progress_callback is not None:
        if verbose:
          print()
        last_msg_count = len(initial_state.get("messages", []))
        for chunk in chain.stream(initial_state, stream_mode="values"):
          messages = chunk.get("messages", [])
          current_stage = chunk.get("current_stage", "N/A")
          r = chunk.get("resistance", "N/A")
          pred_r = chunk.get("predicted_resistance", "N/A")
          if verbose:
            for msg in messages[last_msg_count:]:
              r_print = pred_r if msg["role"] == "therapist" else r
              print(f'    {msg["role"].capitalize()} ({r_print}) [{current_stage}]: {msg["content"]}')
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
      with transcript_file.open("w", encoding="utf-8") as f:
        messages = cast(list[dict[str, Any]], result.get("messages", []))
        for message in messages:
          f.write(f'--> {message["role"].capitalize()} [{message["stage"]}]: {message["content"]}\n\n')
      with json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
      if print_saved_paths:
        print(f"  Saved transcript: {transcript_file}")
        print(f"  Saved state JSON: {json_file}")

    if verbose and result.get("conversation_error") and not error_printed:
      print(f'ERROR: {result["conversation_error"]}')

  return result


def _validate_multiprocessing_options(use_multiprocessing: bool, n_processes: int | None) -> None:
  if not use_multiprocessing:
    return
  if n_processes is None or n_processes < 1:
    raise ValueError("--n-processes must be a positive integer when --multiprocessing is enabled.")


def _run_simulation_batch_worker(args: tuple[int, str, dict[str, Any], int, int, str, Any]) -> dict[str, Any]:
  run_index, cfg_path, profile, max_turns, min_turns, output_dir, progress_queue = args

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
  )
  progress_callback(int(result.get("utterance_index", 0) or 0))
  return {
    "run_index": run_index,
    "profile": profile,
    "result": result,
  }


def _load_tqdm() -> Any:
  try:
    from tqdm import tqdm  # pyright: ignore[reportMissingImports]
  except ImportError as exc:
    raise RuntimeError("Multiprocessing progress bars require tqdm. Install tqdm or run without --multiprocessing.") from exc
  return tqdm


def _advance_progress_bar(
  progress_counts: dict[int, int],
  bars: dict[int, Any],
  run_index: int,
  utterance_index: int,
  max_turns: int,
) -> None:
  bar = bars.get(run_index)
  if bar is None:
    return

  current = progress_counts.get(run_index, 0)
  if utterance_index <= current:
    return

  current_capped = min(current, max_turns)
  next_capped = min(utterance_index, max_turns)
  delta = next_capped - current_capped
  if delta > 0:
    bar.update(delta)
  progress_counts[run_index] = utterance_index


def _drain_progress_events(progress_queue: Any, progress_counts: dict[int, int], bars: dict[int, Any], max_turns: int) -> None:
  while True:
    try:
      run_index, utterance_index = progress_queue.get_nowait()
    except queue.Empty:
      return
    except (EOFError, OSError):
      return
    _advance_progress_bar(progress_counts, bars, int(run_index), int(utterance_index), max_turns)


def _profile_run_batches(
  profiles_to_run: list[dict[str, Any]],
  batch_size: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
  for start in range(0, len(profiles_to_run), batch_size):
    yield start + 1, profiles_to_run[start : start + batch_size]


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
  _print_simulation_header(run_index, total_to_run, profile, writer=writer)
  _write_simulation_summary(result, writer=writer)
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
) -> None:
  total_to_run = len(profiles_to_run)
  for run_index, profile in enumerate(profiles_to_run, start=1):
    _print_simulation_header(run_index, total_to_run, profile)

    result = run_simulation(cfg_path, profile, max_turns, min_turns, output_dir, verbose)
    _print_simulation_summary(result)
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
              args=((run_index, cfg_path, profile, max_turns, min_turns, output_dir, progress_queue),),
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
) -> None:
  from runtime_config import load_runtime_config

  load_dotenv()

  cfg = load_runtime_config(cfg_path)

  max_turns = cfg.env.max_turns
  min_turns = cfg.env.get("min_turns", 0)
  output_dir = str(cfg.env.output_dir or "")
  verbose = bool(cfg.env.verbose)
  resume_file = cfg.env.get("resume_file")
  prompt_ablation = _prompt_ablation_from_patient_config(cfg.patient)

  if resume_file and resume_from_json:
    raise ValueError("env.resume_file cannot be combined with --resume-from-json.")
  if n_profiles is not None and n_profiles < 1:
    raise ValueError("--n-profiles must be a positive integer.")
  if eval_max_iter < 1:
    raise ValueError("--eval-max-iter must be a positive integer.")

  if resume_file:
    resume_state = _load_resume_state(resume_file)
    profile = deepcopy(resume_state["patient_profile"])
    profile.setdefault("idx", "unknown")
    print(f"Resuming simulation from {resume_file}\n")
    result = run_simulation(
      cfg_path,
      profile,
      max_turns,
      min_turns,
      output_dir,
      verbose,
      resume_state=resume_state,
    )
    _print_simulation_summary(result)
    return

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
  parser = argparse.ArgumentParser(description="Run CBT therapy simulation")
  parser.add_argument("--cfg", type=str, default="configs/bedrock.yaml", help="YAML configuration file.")
  parser.add_argument("--profiles-path", type=str, default=None, help="Path to patient profiles JSON/JSONL. Overrides patient.profiles in the config.")
  parser.add_argument("--n-profiles", type=int, default=None, help="Run only N new profiles from the loaded profile set.")
  parser.add_argument("--resume-from-json", type=str, default=None, help="Resume a batch from an execution manifest JSON.")
  parser.add_argument("--eval", action="store_true", help="Evaluate all included attitudes after simulations finish.")
  parser.add_argument("--eval-max-iter", type=int, default=1, help="Evaluator samples per metric prompt.")
  parser.add_argument("--multiprocessing", action="store_true", dest="use_multiprocessing", help="Run selected batch profiles in parallel.")
  parser.add_argument("--n-processes", type=int, default=None, help="Number of worker processes to use with --multiprocessing.")
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
  )
