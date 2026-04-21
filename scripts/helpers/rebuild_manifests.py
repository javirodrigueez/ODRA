from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from simulation import (
  _deduplicate_profiles,
  _group_profiles_by_attitude,
  _initialize_manifests,
  _load_profiles,
  _manifest_path,
  _mark_profile_result,
  _profile_attitude,
  _profile_idx,
  _resolve_profiles_path,
  _simulation_output_paths,
  _write_manifest,
)


def _load_result_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise ValueError(f"Simulation result must be a JSON object: {path}")
  return cast(dict[str, Any], payload)


def _mark_existing_outputs(
  manifests: dict[str, dict[str, Any]],
  profiles: list[dict[str, Any]],
  output_dir: str | Path,
) -> tuple[int, list[str]]:
  restored = 0
  unreadable: list[str] = []

  for profile in profiles:
    attitude = _profile_attitude(profile)
    _transcript_file, json_file = _simulation_output_paths(output_dir, profile)
    if not json_file.exists():
      continue

    try:
      result = _load_result_json(json_file)
    except Exception as exc:
      unreadable.append(f"{json_file}: {exc}")
      idx = _profile_idx(profile)
      transcript_file, _json_file = _simulation_output_paths(output_dir, profile)
      manifests[attitude]["profiles"][idx] = {
        "executed": True,
        "success": False,
        "output_json": str(json_file),
        "output_transcript": str(transcript_file),
        "error": f"Failed to read existing result JSON: {exc}",
      }
      continue

    _mark_profile_result(manifests[attitude], profile, output_dir, result)
    restored += 1

  return restored, unreadable


def rebuild_manifests(
  cfg_path: str,
  *,
  profiles_path: str | None = None,
  output_dir: str | None = None,
  resume_from_json: str | None = None,
  scan_existing: bool = True,
) -> list[Path]:
  from runtime_config import load_runtime_config

  cfg = load_runtime_config(cfg_path)
  resolved_profiles_path = _resolve_profiles_path(profiles_path, cfg.patient.profiles)
  resolved_output_dir = str(output_dir or cfg.env.output_dir or "").strip()
  if not resolved_output_dir:
    raise ValueError("Output directory is required. Pass --output-dir or set env.output_dir in the config.")

  profiles, duplicate_idxs = _deduplicate_profiles(_load_profiles(resolved_profiles_path))
  grouped = _group_profiles_by_attitude(profiles)
  manifests = _initialize_manifests(
    cfg_path=cfg_path,
    profiles_path=resolved_profiles_path,
    output_dir=resolved_output_dir,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
    resume_from_json=resume_from_json,
  )

  restored = 0
  unreadable: list[str] = []
  if scan_existing:
    restored, unreadable = _mark_existing_outputs(manifests, profiles, resolved_output_dir)

  written_paths: list[Path] = []
  for attitude in sorted(grouped):
    manifest_path = _manifest_path(resolved_output_dir, attitude)
    _write_manifest(manifest_path, manifests[attitude])
    written_paths.append(manifest_path)

  print(f"Loaded {len(profiles)} unique profiles from {resolved_profiles_path}")
  if duplicate_idxs:
    print(f"Omitted duplicate profile idxs: {', '.join(duplicate_idxs)}")
  print(f"Included attitudes: {', '.join(sorted(grouped))}")
  if scan_existing:
    print(f"Restored executed status from existing result JSON files: {restored}")
  if unreadable:
    print("Unreadable result JSON files:")
    for item in unreadable:
      print(f"  {item}")
  print("Wrote manifests:")
  for path in written_paths:
    print(f"  {path}")

  return written_paths


def main() -> None:
  parser = argparse.ArgumentParser(description="Rebuild ReasonCBT batch manifests without running simulations.")
  parser.add_argument("--cfg", type=str, default="configs/bedrock.yaml", help="YAML configuration file.")
  parser.add_argument(
    "--profiles-path",
    type=str,
    default=None,
    help="Path to patient profiles JSON/JSONL. Overrides patient.profiles in the config.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory where manifests and simulation outputs live. Overrides env.output_dir in the config.",
  )
  parser.add_argument(
    "--resume-from-json",
    type=str,
    default=None,
    help="Optional existing manifest JSON to merge before rewriting manifests.",
  )
  parser.add_argument(
    "--no-scan-existing",
    action="store_true",
    help="Create fresh manifests without marking profiles from existing simulation JSON files.",
  )
  args = parser.parse_args()

  rebuild_manifests(
    args.cfg,
    profiles_path=args.profiles_path,
    output_dir=args.output_dir,
    resume_from_json=args.resume_from_json,
    scan_existing=not args.no_scan_existing,
  )


if __name__ == "__main__":
  main()
