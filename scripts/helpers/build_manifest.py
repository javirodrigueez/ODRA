from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from simulation import (  # noqa: E402
  _deduplicate_profiles,
  _load_profiles,
  _profile_attitude,
  _profile_idx,
  _prompt_ablation_from_cfg_path,
  _refresh_manifest_metadata,
  _write_manifest,
)


MatchKey = str


def _load_json_payload(path: str | Path) -> Any:
  path = Path(path)
  if path.suffix == ".jsonl":
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          records.append(json.loads(line))
    return records

  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _load_already_executed_records(path: str | Path) -> list[Any]:
  payload = _load_json_payload(path)
  if isinstance(payload, list):
    return payload
  if isinstance(payload, dict):
    profiles = payload.get("profiles")
    if isinstance(profiles, list):
      return profiles
    if isinstance(profiles, dict):
      records: list[dict[str, Any]] = []
      for idx, status in profiles.items():
        if isinstance(status, dict) and not status.get("executed", True):
          continue
        record = dict(status) if isinstance(status, dict) else {}
        record["idx"] = idx
        records.append(record)
      return records
    return [payload]
  raise ValueError(f"Unsupported already-executed payload in {path}: expected JSON object, array, or JSONL records.")


def _profile_identity_key(profile: dict[str, Any]) -> str:
  payload = dict(profile)
  payload.pop("idx", None)
  return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _record_idx(record: Any, position: int) -> str:
  if isinstance(record, dict):
    return str(record.get("idx", position + 1))
  if isinstance(record, (str, int)):
    return str(record)
  raise ValueError(f"Cannot read idx from already-executed record at position {position}: {record!r}")


def _build_target_lookup(
  profiles: list[dict[str, Any]],
  match_key: MatchKey,
) -> dict[str, list[str]]:
  lookup: dict[str, list[str]] = {}
  for profile in profiles:
    if match_key == "profile":
      key = _profile_identity_key(profile)
    elif match_key == "idx":
      key = _profile_idx(profile)
    else:
      raise ValueError(f"Unsupported match key: {match_key}")
    lookup.setdefault(key, []).append(_profile_idx(profile))
  return lookup


def _already_record_key(record: Any, position: int, match_key: MatchKey) -> str:
  if match_key == "idx":
    return _record_idx(record, position)
  if match_key == "profile":
    if not isinstance(record, dict):
      raise ValueError("--match-key profile requires profile objects in --profiles-already-exec.")
    return _profile_identity_key(record)
  raise ValueError(f"Unsupported match key: {match_key}")


def _resolve_output_dir(output_dir: str | None, cfg_path: str | None) -> str:
  if output_dir:
    return output_dir
  if not cfg_path:
    return ""

  try:
    from runtime_config import load_runtime_config

    cfg = load_runtime_config(cfg_path)
    resolved = str(cfg.env.output_dir or "").strip()
  except Exception as exc:
    resolved = _read_output_dir_from_cfg_text(cfg_path)
    if not resolved:
      raise ValueError("Could not derive output_dir from --cfg. Pass --output-dir explicitly.") from exc

  return resolved


def _strip_yaml_scalar(value: str) -> str:
  value = value.split("#", 1)[0].strip()
  if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
    return value[1:-1]
  return value


def _read_output_dir_from_cfg_text(cfg_path: str | Path) -> str:
  in_env = False
  env_indent = -1
  with Path(cfg_path).open("r", encoding="utf-8") as f:
    for raw_line in f:
      stripped = raw_line.strip()
      if not stripped or stripped.startswith("#"):
        continue

      indent = len(raw_line) - len(raw_line.lstrip())
      if stripped == "env:":
        in_env = True
        env_indent = indent
        continue
      if in_env and indent <= env_indent:
        in_env = False
      if not in_env or not stripped.startswith("output_dir:"):
        continue

      return _strip_yaml_scalar(stripped.split(":", 1)[1])
  return ""


def _simulation_output_path_strings(output_dir: str | Path, profile: dict[str, Any]) -> tuple[str, str]:
  idx = _profile_idx(profile)
  attitude = _profile_attitude(profile)
  base = Path(output_dir)
  transcript_file = base / "transcripts" / f"simulation_{idx}_{attitude}.txt"
  json_file = base / "json" / f"simulation_{idx}_{attitude}.json"
  return str(transcript_file), str(json_file)


def _default_profile_status(output_dir: str | Path, profile: dict[str, Any]) -> dict[str, Any]:
  transcript_file, json_file = _simulation_output_path_strings(output_dir, profile)
  return {
    "attitude": _profile_attitude(profile),
    "executed": False,
    "success": False,
    "output_json": json_file,
    "output_transcript": transcript_file,
    "error": "",
  }


def _new_manifest(
  *,
  cfg_path: str,
  profiles_path: str,
  output_dir: str | Path,
  profiles: list[dict[str, Any]],
  duplicate_idxs: list[str],
) -> dict[str, Any]:
  attitudes = sorted({_profile_attitude(profile) for profile in profiles})
  prompt_ablation = _prompt_ablation_from_cfg_path(cfg_path)
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
      "prompt_ablation": dict(prompt_ablation),
      "updated_at": "",
    },
  }


def build_manifest(
  *,
  profiles_to_exec_path: str,
  already_exec_paths: list[str],
  output_dir: str = "",
  cfg_path: str = "",
  match_key: MatchKey = "profile",
  only_executed: bool = False,
  fail_on_unmatched: bool = False,
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
  profiles, duplicate_idxs = _deduplicate_profiles(_load_profiles(profiles_to_exec_path))
  manifest = _new_manifest(
    cfg_path=cfg_path,
    profiles_path=profiles_to_exec_path,
    output_dir=output_dir,
    profiles=profiles,
    duplicate_idxs=duplicate_idxs,
  )

  target_lookup = _build_target_lookup(profiles, match_key)
  matched_idxs: set[str] = set()
  idx_mapping: dict[str, str] = {}
  unmatched: list[str] = []

  for already_exec_path in already_exec_paths:
    records = _load_already_executed_records(already_exec_path)
    for position, record in enumerate(records):
      old_idx = _record_idx(record, position)
      key = _already_record_key(record, position, match_key)
      target_idxs = target_lookup.get(key)
      if not target_idxs:
        unmatched.append(f"{already_exec_path}:{position}")
        continue
      matched_idxs.update(target_idxs)
      idx_mapping[old_idx] = target_idxs[0]

  if fail_on_unmatched and unmatched:
    raise ValueError("Some already-executed records did not match target profiles:\n" + "\n".join(unmatched))

  for idx in matched_idxs:
    status = manifest["profiles"][idx]
    status["executed"] = True
    status["success"] = True
    status["error"] = ""

  if only_executed:
    manifest["profiles"] = {
      idx: manifest["profiles"][idx]
      for idx in sorted(matched_idxs, key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    }

  metadata = manifest.setdefault("metadata", {})
  metadata["already_exec_profiles_paths"] = list(already_exec_paths)
  metadata["match_key"] = match_key
  metadata["matched_already_exec_count"] = len(matched_idxs)
  metadata["unmatched_already_exec_count"] = len(unmatched)
  metadata["idx_mapping_count"] = len(idx_mapping)
  _refresh_manifest_metadata(manifest)

  return manifest, unmatched, idx_mapping


def _write_manifest_output(path: str | None, manifest: dict[str, Any]) -> None:
  if path:
    _write_manifest(path, manifest)
    return

  _refresh_manifest_metadata(manifest)
  json.dump(manifest, sys.stdout, indent=2, ensure_ascii=True)
  sys.stdout.write("\n")


def _default_idx_mapping_output_path(manifest_output_path: str) -> str:
  path = Path(manifest_output_path)
  suffix = path.suffix or ".json"
  return str(path.with_name(f"{path.stem}_idx_mapping{suffix}"))


def _write_idx_mapping_output(path: str, idx_mapping: dict[str, str]) -> None:
  output_path = Path(path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as f:
    json.dump(idx_mapping, f, indent=2, ensure_ascii=True)
    f.write("\n")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Build a simulation.py --resume-from-json manifest from target profiles and already-executed profiles."
  )
  parser.add_argument(
    "--profiles-to-exec",
    required=True,
    help="JSON/JSONL profile list that will be passed to simulation.py via --profiles-path.",
  )
  parser.add_argument(
    "--profiles-already-exec",
    action="append",
    required=True,
    help="JSON/JSONL profile list or manifest containing profiles that have already executed. Can be passed multiple times.",
  )
  parser.add_argument(
    "--output-dir",
    default=None,
    help="Simulation output directory to record in manifest paths. If omitted, --cfg env.output_dir is used when available.",
  )
  parser.add_argument(
    "--cfg",
    default="",
    help="Optional simulation config path to record in manifest metadata and use for output_dir lookup.",
  )
  parser.add_argument(
    "--output",
    "-o",
    default="",
    help="Manifest JSON path to write. Defaults to stdout.",
  )
  parser.add_argument(
    "--idx-mapping-output",
    default=None,
    help="Old reduced-set idx to new complete-set idx JSON path. Defaults to <output>_idx_mapping.json when --output is set.",
  )
  parser.add_argument(
    "--match-key",
    choices=["profile", "idx"],
    default="profile",
    help="How to match already-executed records to target profiles. 'profile' ignores top-level idx.",
  )
  parser.add_argument(
    "--only-executed",
    action="store_true",
    help="Emit only matched executed profiles instead of a full target manifest.",
  )
  parser.add_argument(
    "--fail-on-unmatched",
    action="store_true",
    help="Exit with an error if any already-executed record does not match a target profile.",
  )
  args = parser.parse_args()

  output_dir = _resolve_output_dir(args.output_dir, args.cfg)
  manifest, unmatched, idx_mapping = build_manifest(
    profiles_to_exec_path=args.profiles_to_exec,
    already_exec_paths=cast(list[str], args.profiles_already_exec),
    output_dir=output_dir,
    cfg_path=args.cfg,
    match_key=args.match_key,
    only_executed=args.only_executed,
    fail_on_unmatched=args.fail_on_unmatched,
  )
  _write_manifest_output(args.output, manifest)
  idx_mapping_output = args.idx_mapping_output
  if idx_mapping_output is None and args.output:
    idx_mapping_output = _default_idx_mapping_output_path(args.output)
  if idx_mapping_output:
    _write_idx_mapping_output(idx_mapping_output, idx_mapping)

  executed_count = manifest.get("metadata", {}).get("executed_count", 0)
  total_profiles = manifest.get("metadata", {}).get("total_profiles", 0)
  print(f"Marked executed profiles: {executed_count}/{total_profiles}", file=sys.stderr)
  print(f"Mapped old idxs to new idxs: {len(idx_mapping)}", file=sys.stderr)
  if unmatched:
    print(f"Unmatched already-executed records: {len(unmatched)}", file=sys.stderr)
  if args.output:
    print(f"Wrote manifest: {args.output}", file=sys.stderr)
  if idx_mapping_output:
    print(f"Wrote idx mapping: {idx_mapping_output}", file=sys.stderr)


if __name__ == "__main__":
  main()
