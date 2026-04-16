from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, cast

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


def _output_dir_manifest_prefix(output_dir: str | Path) -> str:
  normalized = Path(output_dir).as_posix().strip("/")
  return normalized.replace("/", "_") or "output"


def _manifest_path(output_dir: str | Path, attitude: str) -> Path:
  prefix = _output_dir_manifest_prefix(output_dir)
  return Path(output_dir) / f"{prefix}_{attitude}.json"


def _simulation_output_paths(output_dir: str | Path, profile: dict[str, Any]) -> tuple[Path, Path]:
  idx = _profile_idx(profile)
  attitude = _profile_attitude(profile)
  base = Path(output_dir)
  base_json = base / "json" / f"simulation_{idx}_{attitude}.json"
  base_json.mkdir(parents=True, exist_ok=True)
  base_txt = base / "transcripts" / f"simulation_{idx}_{attitude}.txt"
  base_txt.parent.mkdir(parents=True, exist_ok=True)

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
  attitude: str,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
) -> dict[str, Any]:
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
      "attitude": attitude,
      "total_profiles": len(profiles),
      "executed_count": 0,
      "success_count": 0,
      "duplicate_idxs_omitted": duplicate_idxs,
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
  attitude: str,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
) -> dict[str, Any]:
  fresh = _new_manifest(
    cfg_path=cfg_path,
    profiles_path=profiles_path,
    output_dir=output_dir,
    attitude=attitude,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
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


def _initialize_manifests(
  *,
  cfg_path: str,
  profiles_path: str,
  output_dir: str | Path,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
  resume_from_json: str | None,
) -> dict[str, dict[str, Any]]:
  grouped = _group_profiles_by_attitude(profiles)
  existing_by_attitude: dict[str, dict[str, Any]] = {}
  provided_resume_path = Path(resume_from_json).resolve() if resume_from_json else None

  if provided_resume_path is not None:
    loaded = _load_manifest(provided_resume_path)
    resume_attitude = str(loaded.get("metadata", {}).get("attitude") or "").strip().lower()
    if not resume_attitude and len(grouped) == 1:
      resume_attitude = next(iter(grouped))
    if not resume_attitude:
      raise ValueError(
        "--resume-from-json must include metadata.attitude when the loaded profiles contain multiple attitudes."
      )
    existing_by_attitude[resume_attitude] = loaded

  manifests: dict[str, dict[str, Any]] = {}
  for attitude, attitude_profiles in grouped.items():
    existing = existing_by_attitude.get(attitude)
    default_path = _manifest_path(output_dir, attitude)
    if existing is None and resume_from_json and default_path.exists():
      existing = _load_manifest(default_path)

    manifests[attitude] = _merge_manifest_with_profiles(
      existing,
      cfg_path=cfg_path,
      profiles_path=profiles_path,
      output_dir=output_dir,
      attitude=attitude,
      profiles=attitude_profiles,
      duplicate_idxs=duplicate_idxs,
    )
  return manifests


def _select_profiles_to_execute(
  profiles: list[dict[str, Any]],
  manifests: dict[str, dict[str, Any]],
  *,
  n_profiles: int | None,
) -> list[dict[str, Any]]:
  selected: list[dict[str, Any]] = []
  for profile in profiles:
    attitude = _profile_attitude(profile)
    idx = _profile_idx(profile)
    status = manifests[attitude]["profiles"].get(idx, {})
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
  exchange_counts = cast(dict[str, list[int]], result.get("exchange_counts", {}))
  exchanges = sum(item for sublist in exchange_counts.values() for item in sublist)
  print(f'  Total utterances: {result.get("utterance_index", 0)}')
  print(f"  Total exchanges: {exchanges}")
  print(f'  End reason: {result.get("session_ended_reason", "unknown")}\n')


def run_simulation(
  cfg_path: str,
  profile: dict[str, Any],
  max_turns: int,
  min_turns: int,
  output_dir: str = "",
  verbose: bool = False,
  resume_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Run a single simulation for a given profile."""
  from graph.graph import TherapyGraphBuilder

  result: dict[str, Any] = {
    "messages": [],
    "exchange_counts": {},
    "utterance_index": 0,
    "session_ended": True,
    "session_ended_reason": "",
    "conversation_error": "",
    "patient_profile": profile,
  }
  error_printed = False
  try:
    builder = TherapyGraphBuilder(cfg_path, profile)
    chain = builder.create_chain()
    if resume_state is None:
      initial_state = builder.initial_state(max_turns=max_turns, min_turns=min_turns)
    else:
      initial_state = builder.resume_state(deepcopy(resume_state), max_turns=max_turns, min_turns=min_turns)
    result = initial_state

    if verbose:
      print()
      last_msg_count = len(initial_state.get("messages", []))
      for chunk in chain.stream(initial_state, stream_mode="values"):
        messages = chunk.get("messages", [])
        current_stage = chunk.get("current_stage", "N/A")
        r = chunk.get("resistance", "N/A")
        pred_r = chunk.get("predicted_resistance", "N/A")
        for msg in messages[last_msg_count:]:
          r_print = pred_r if msg["role"] == "therapist" else r
          print(f'    {msg["role"].capitalize()} ({r_print}) [{current_stage}]: {msg["content"]}')
        last_msg_count = len(messages)
        result = chunk
      if result.get("conversation_error"):
        print(f'ERROR: {result["conversation_error"]}')
        error_printed = True
      print()
    else:
      result = chain.invoke(initial_state)
  except Exception as exc:
    result = dict(result)
    result["session_ended"] = True
    result["session_ended_reason"] = result.get("session_ended_reason") or "conversation_error"
    result["conversation_error"] = str(exc)
    result.setdefault("patient_profile", profile)
    if verbose:
      print(f'ERROR: {result["conversation_error"]}')
      error_printed = True

  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    transcript_file, json_file = _simulation_output_paths(output_dir, profile)
    with transcript_file.open("w", encoding="utf-8") as f:
      messages = cast(list[dict[str, Any]], result.get("messages", []))
      for message in messages:
        f.write(f'--> {message["role"].capitalize()} [{message["stage"]}]: {message["content"]}\n\n')
    with json_file.open("w", encoding="utf-8") as f:
      json.dump(result, f, indent=2, ensure_ascii=True)
    print(f"  Saved transcript: {transcript_file}")
    print(f"  Saved state JSON: {json_file}")

  if verbose and result.get("conversation_error") and not error_printed:
    print(f'ERROR: {result["conversation_error"]}')

  return result


def main(
  cfg_path: str,
  *,
  profiles_path: str | None = None,
  n_profiles: int | None = None,
  resume_from_json: str | None = None,
  run_eval: bool = False,
  eval_max_iter: int = 1,
) -> None:
  from runtime_config import load_runtime_config

  load_dotenv()

  cfg = load_runtime_config(cfg_path)

  max_turns = cfg.env.max_turns
  min_turns = cfg.env.get("min_turns", 0)
  output_dir = str(cfg.env.output_dir or "")
  verbose = bool(cfg.env.verbose)
  resume_file = cfg.env.get("resume_file")

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

  if not output_dir:
    raise ValueError("cfg.env.output_dir is required for profile batch simulation manifests.")

  profiles_path = _resolve_profiles_path(profiles_path, cfg.patient.profiles)
  profiles, duplicate_idxs = _deduplicate_profiles(_load_profiles(profiles_path))
  grouped = _group_profiles_by_attitude(profiles)
  manifests = _initialize_manifests(
    cfg_path=cfg_path,
    profiles_path=profiles_path,
    output_dir=output_dir,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
    resume_from_json=resume_from_json,
  )

  print(f"Loaded {len(profiles)} unique profiles from {profiles_path}")
  if duplicate_idxs:
    print(f"Omitted duplicate profile idxs: {', '.join(duplicate_idxs)}")
  print(f"Included attitudes: {', '.join(sorted(grouped))}\n")

  for attitude, manifest in manifests.items():
    _write_manifest(_manifest_path(output_dir, attitude), manifest)

  profiles_to_run = _select_profiles_to_execute(
    profiles,
    manifests,
    n_profiles=n_profiles,
  )
  print(f"Profiles selected for execution: {len(profiles_to_run)}\n")

  total_to_run = len(profiles_to_run)
  for run_index, profile in enumerate(profiles_to_run, start=1):
    attitude = _profile_attitude(profile)
    print(f'{"="*60}')
    print(f'Simulation {run_index}/{total_to_run} — idx: {_profile_idx(profile)} — attitude: {attitude}')
    print(f'{"="*60}')

    result = run_simulation(cfg_path, profile, max_turns, min_turns, output_dir, verbose)
    _print_simulation_summary(result)
    manifest = manifests[attitude]
    _mark_profile_result(manifest, profile, output_dir, result)
    _write_manifest(_manifest_path(output_dir, attitude), manifest)

  if run_eval:
    for attitude in sorted(grouped):
      print(f'{"="*60}')
      print(f"Evaluating attitude: {attitude}")
      print(f'{"="*60}')
      scores = _run_evaluation_for_attitude(output_dir, attitude, eval_max_iter)
      manifests[attitude]["scores"] = scores
      _write_manifest(_manifest_path(output_dir, attitude), manifests[attitude])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run CBT therapy simulation")
  parser.add_argument("--cfg", type=str, default="configs/bedrock.yaml", help="YAML configuration file.")
  parser.add_argument("--profiles-path", type=str, default=None, help="Path to patient profiles JSON/JSONL. Overrides patient.profiles in the config.")
  parser.add_argument("--n-profiles", type=int, default=None, help="Run only N new profiles from the loaded profile set.")
  parser.add_argument("--resume-from-json", type=str, default=None, help="Resume a batch from an execution manifest JSON.")
  parser.add_argument("--eval", action="store_true", help="Evaluate all included attitudes after simulations finish.")
  parser.add_argument("--eval-max-iter", type=int, default=1, help="Evaluator samples per metric prompt.")
  args = parser.parse_args()
  main(
    args.cfg,
    profiles_path=args.profiles_path,
    n_profiles=args.n_profiles,
    resume_from_json=args.resume_from_json,
    run_eval=args.eval,
    eval_max_iter=args.eval_max_iter,
  )
