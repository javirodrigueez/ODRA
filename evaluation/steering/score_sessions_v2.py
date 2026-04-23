#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
  from .steering_v2 import (
    DEFAULT_INPUT_COST_PER_1M,
    DEFAULT_MAX_ITER,
    DEFAULT_OUTPUT_COST_PER_1M,
    SteeringV2EvaluationError,
    atomic_write_json,
    attach_judgements,
    load_request_map,
    parse_all_shard_outputs,
    parse_batch_outputs,
    read_json,
    session_metrics,
    summary_from_sessions,
    write_session_outputs,
  )
except ImportError:  # pragma: no cover - supports direct script execution.
  from steering_v2 import (  # type: ignore[no-redef]
    DEFAULT_INPUT_COST_PER_1M,
    DEFAULT_MAX_ITER,
    DEFAULT_OUTPUT_COST_PER_1M,
    SteeringV2EvaluationError,
    atomic_write_json,
    attach_judgements,
    load_request_map,
    parse_all_shard_outputs,
    parse_batch_outputs,
    read_json,
    session_metrics,
    summary_from_sessions,
    write_session_outputs,
  )


def iter_session_files(sessions_dir: Path) -> list[Path]:
  return sorted(path for path in sessions_dir.glob("*.json") if path.is_file())


def read_session_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8-sig") as f:
    return json.load(f)


def read_json_object(path: Path) -> dict[str, Any]:
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise SteeringV2EvaluationError(f"Expected JSON object at {path}")
  return payload


def read_prepared_sessions(path: Path) -> list[dict[str, Any]]:
  payload = read_json(path)
  if not isinstance(payload, list):
    raise SteeringV2EvaluationError(f"Prepared sessions must be a JSON list: {path}")
  sessions: list[dict[str, Any]] = []
  for index, session in enumerate(payload):
    if not isinstance(session, dict):
      raise SteeringV2EvaluationError(f"Prepared session at index {index} is not an object: {path}")
    sessions.append(session)
  return sessions


def positive_int(value: Any, field_name: str) -> int:
  try:
    parsed = int(value)
  except (TypeError, ValueError) as exc:
    raise SteeringV2EvaluationError(f"{field_name} must be a positive integer, got {value!r}") from exc
  if parsed < 1:
    raise SteeringV2EvaluationError(f"{field_name} must be a positive integer, got {parsed}")
  return parsed


def files_seen_from_manifest(manifest: dict[str, Any], fallback: int) -> int:
  try:
    files_seen = int(manifest.get("files_seen", fallback))
  except (TypeError, ValueError):
    return fallback
  return max(0, files_seen)


def default_output_dir_for_artifact(artifact_dir: Path) -> Path:
  if artifact_dir.parent.name == "_batch_artifacts":
    return artifact_dir.parent.parent
  return artifact_dir


def artifact_dir_from_batch_jsonl(output_jsonl: Path) -> Path | None:
  shard_dir = output_jsonl.parent
  shards_dir = shard_dir.parent
  if shards_dir.name == "shards":
    return shards_dir.parent
  if (shard_dir / "manifest.json").is_file() and shard_dir.parent.name == "_batch_artifacts":
    return shard_dir
  return None


def default_companion_paths_for_batch_jsonl(
  output_jsonl: Path,
  request_map_path: str | Path | None,
  prepared_sessions_path: str | Path | None,
) -> tuple[Path, Path, Path | None]:
  request_map = Path(request_map_path).resolve() if request_map_path is not None else output_jsonl.parent / "request_map.json"
  artifact_dir = artifact_dir_from_batch_jsonl(output_jsonl)
  prepared_sessions = (
    Path(prepared_sessions_path).resolve()
    if prepared_sessions_path is not None
    else (artifact_dir / "prepared_sessions.json" if artifact_dir is not None else output_jsonl.parent / "prepared_sessions.json")
  )
  if not request_map.is_file():
    raise SteeringV2EvaluationError(f"Missing request_map.json: {request_map}")
  if not prepared_sessions.is_file():
    raise SteeringV2EvaluationError(f"Missing prepared_sessions.json: {prepared_sessions}")
  return request_map, prepared_sessions, artifact_dir


def score_sessions_folder(
  sessions_dir: str | Path,
  output_dir: str | Path | None = None,
) -> dict[str, Any]:
  sessions_path = Path(sessions_dir).resolve()
  if not sessions_path.is_dir():
    raise SteeringV2EvaluationError(f"sessions_dir does not exist or is not a directory: {sessions_path}")

  output_path = Path(output_dir).resolve() if output_dir is not None else sessions_path.parent
  output_sessions_path = output_path / "sessions"
  output_sessions_path.mkdir(parents=True, exist_ok=True)

  files = iter_session_files(sessions_path)
  session_results: list[dict[str, Any]] = []
  skipped_files: list[dict[str, Any]] = []

  for path in files:
    try:
      payload = read_session_json(path)
    except Exception as exc:
      skipped_files.append({"file": str(path), "reason": f"could not read JSON: {exc}"})
      continue
    if not isinstance(payload, dict):
      skipped_files.append({"file": str(path), "reason": "top-level JSON is not an object"})
      continue

    fragments = payload.get("evaluated_fragments")
    if not isinstance(fragments, list):
      payload = {
        **payload,
        "evaluated_fragments": [],
        "skipped_turns": list(payload.get("skipped_turns") or []) + [{"reason": "missing evaluated_fragments list"}],
      }
    payload["metrics"] = session_metrics([
      fragment for fragment in payload.get("evaluated_fragments", []) if isinstance(fragment, dict)
    ])
    payload.setdefault("input_filename", path.name)
    payload.setdefault("input_file", str(path))
    payload["rescored_from_v2_session_outputs"] = True
    atomic_write_json(output_sessions_path / path.name, payload)
    session_results.append(payload)

  summary = summary_from_sessions(
    sessions_path,
    output_path,
    len(files),
    session_results,
    skipped_files,
    artifact_dir=None,
    parse_report=None,
  )
  summary["source_sessions_dir"] = str(sessions_path)
  summary["rescored_from_v2_session_outputs"] = True
  atomic_write_json(output_path / "summary.json", summary)
  return summary


def score_sessions_from_artifacts(
  artifact_dir: str | Path,
  output_dir: str | Path | None = None,
  *,
  max_iter: int | None = None,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
) -> dict[str, Any]:
  artifact_path = Path(artifact_dir).resolve()
  if not artifact_path.is_dir():
    raise SteeringV2EvaluationError(f"artifact_dir does not exist or is not a directory: {artifact_path}")
  manifest_path = artifact_path / "manifest.json"
  if not manifest_path.is_file():
    raise SteeringV2EvaluationError(f"Missing v2 manifest: {manifest_path}")

  manifest = read_json_object(manifest_path)
  prepared_sessions_path = Path(str(manifest["prepared_sessions"]))
  prepared_sessions = read_prepared_sessions(prepared_sessions_path)
  expected_choices = (
    positive_int(max_iter, "--max-iter")
    if max_iter is not None
    else positive_int(manifest.get("max_iter") or DEFAULT_MAX_ITER, "manifest max_iter")
  )

  parsed_results, parse_report = parse_all_shard_outputs(
    manifest,
    expected_choices,
    input_cost_per_1m,
    output_cost_per_1m,
  )
  parse_report["source_artifact_dir"] = str(artifact_path)
  parse_report["source_prepared_sessions"] = str(prepared_sessions_path)

  output_path = Path(output_dir).resolve() if output_dir is not None else default_output_dir_for_artifact(artifact_path)
  output_path.mkdir(parents=True, exist_ok=True)
  scored_sessions = attach_judgements(prepared_sessions, parsed_results, artifact_path)
  write_session_outputs(output_path, scored_sessions)

  source_input_dir = Path(str(manifest.get("input_dir"))) if manifest.get("input_dir") else artifact_path
  summary = summary_from_sessions(
    source_input_dir,
    output_path,
    files_seen_from_manifest(manifest, len(prepared_sessions)),
    scored_sessions,
    [],
    artifact_path,
    parse_report,
    request_count=int(manifest.get("request_count") or 0),
    shard_count=int(manifest.get("shard_count") or 0),
  )
  summary["source_artifact_dir"] = str(artifact_path)
  summary["rescored_from_v2_artifacts"] = True
  summary["matched_concepts"] = bool(manifest.get("matched_concepts"))
  summary["unified_concepts"] = bool(manifest.get("unified_concepts"))
  if "fragmenting" in manifest:
    summary["fragmenting"] = manifest["fragmenting"]
  atomic_write_json(output_path / "summary.json", summary)
  return summary


def score_sessions_from_batch_jsonl(
  output_jsonl: str | Path,
  output_dir: str | Path | None = None,
  *,
  request_map_path: str | Path | None = None,
  prepared_sessions_path: str | Path | None = None,
  max_iter: int | None = None,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
) -> dict[str, Any]:
  batch_output_path = Path(output_jsonl).resolve()
  if not batch_output_path.is_file():
    raise SteeringV2EvaluationError(f"batch output JSONL does not exist or is not a file: {batch_output_path}")

  request_map_file, prepared_sessions_file, artifact_dir = default_companion_paths_for_batch_jsonl(
    batch_output_path,
    request_map_path,
    prepared_sessions_path,
  )
  artifact_manifest = read_json_object(artifact_dir / "manifest.json") if artifact_dir and (artifact_dir / "manifest.json").is_file() else {}
  shard_manifest = read_json_object(batch_output_path.parent / "manifest.json") if (batch_output_path.parent / "manifest.json").is_file() else {}
  expected_choices = (
    positive_int(max_iter, "--max-iter")
    if max_iter is not None
    else positive_int(artifact_manifest.get("max_iter") or DEFAULT_MAX_ITER, "manifest max_iter")
  )

  request_map = load_request_map(request_map_file)
  prepared_sessions = read_prepared_sessions(prepared_sessions_file)
  parsed_results, parse_report = parse_batch_outputs(
    batch_output_path,
    request_map,
    expected_choices,
    input_cost_per_1m,
    output_cost_per_1m,
  )
  parse_report["source_batch_jsonl"] = str(batch_output_path)
  parse_report["source_request_map"] = str(request_map_file)
  parse_report["source_prepared_sessions"] = str(prepared_sessions_file)
  if shard_manifest:
    parse_report["shard_index"] = shard_manifest.get("shard_index")

  output_path = (
    Path(output_dir).resolve()
    if output_dir is not None
    else (default_output_dir_for_artifact(artifact_dir) if artifact_dir is not None else batch_output_path.parent)
  )
  output_path.mkdir(parents=True, exist_ok=True)
  scored_sessions = attach_judgements(prepared_sessions, parsed_results, artifact_dir or batch_output_path.parent)
  write_session_outputs(output_path, scored_sessions)

  source_input_dir = (
    Path(str(artifact_manifest.get("input_dir")))
    if artifact_manifest.get("input_dir")
    else batch_output_path
  )
  summary = summary_from_sessions(
    source_input_dir,
    output_path,
    files_seen_from_manifest(artifact_manifest, len(prepared_sessions)),
    scored_sessions,
    [],
    artifact_dir or batch_output_path.parent,
    parse_report,
    request_count=len(request_map),
    shard_count=1,
  )
  summary["source_batch_jsonl"] = str(batch_output_path)
  summary["source_request_map"] = str(request_map_file)
  summary["source_prepared_sessions"] = str(prepared_sessions_file)
  summary["rescored_from_v2_batch_jsonl"] = True
  if artifact_manifest:
    summary["matched_concepts"] = bool(artifact_manifest.get("matched_concepts"))
    summary["unified_concepts"] = bool(artifact_manifest.get("unified_concepts"))
    if "fragmenting" in artifact_manifest:
      summary["fragmenting"] = artifact_manifest["fragmenting"]
  atomic_write_json(output_path / "summary.json", summary)
  return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Recompute steering v2 metrics from session outputs, a v2 artifact directory, or raw OpenAI Batch output JSONL."
  )
  parser.add_argument(
    "-i",
    "--sessions_dir",
    type=Path,
    required=True,
    help=(
      "Directory containing v2 per-session JSON files. With --from-artifacts, this is the run artifact directory. "
      "With --from-batch-jsonl, this is the raw batch output JSONL path."
    ),
  )
  parser.add_argument(
    "-o",
    "--output_dir",
    type=Path,
    default=None,
    help="Output directory. Defaults to the source run output directory when discoverable.",
  )
  parser.add_argument(
    "-a",
    "--from-artifacts",
    action="store_true",
    help="Parse all shard output JSONLs from a v2 run artifact directory.",
  )
  parser.add_argument(
    "-b",
    "--from-batch-jsonl",
    action="store_true",
    help="Score from a single raw OpenAI Batch output JSONL.",
  )
  parser.add_argument(
    "--request-map",
    type=Path,
    default=None,
    help="Request map JSON to use with --from-batch-jsonl. Default: sibling request_map.json.",
  )
  parser.add_argument(
    "--prepared-sessions",
    type=Path,
    default=None,
    help="Prepared sessions JSON to use with --from-batch-jsonl. Default: parent run artifact prepared_sessions.json.",
  )
  parser.add_argument(
    "--max-iter",
    type=int,
    default=None,
    help=f"Expected choices per batch request. Default: manifest max_iter, or {DEFAULT_MAX_ITER}.",
  )
  parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
  parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    if args.from_artifacts and args.from_batch_jsonl:
      raise SteeringV2EvaluationError("--from-artifacts and --from-batch-jsonl are mutually exclusive")
    if args.from_artifacts:
      if args.request_map or args.prepared_sessions:
        raise SteeringV2EvaluationError("--request-map and --prepared-sessions require --from-batch-jsonl")
      summary = score_sessions_from_artifacts(
        args.sessions_dir,
        args.output_dir,
        max_iter=args.max_iter,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
      )
    elif args.from_batch_jsonl:
      summary = score_sessions_from_batch_jsonl(
        args.sessions_dir,
        args.output_dir,
        request_map_path=args.request_map,
        prepared_sessions_path=args.prepared_sessions,
        max_iter=args.max_iter,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
      )
    else:
      if args.request_map or args.prepared_sessions or args.max_iter is not None:
        raise SteeringV2EvaluationError(
          "--request-map, --prepared-sessions, and --max-iter require --from-batch-jsonl or --from-artifacts"
        )
      summary = score_sessions_folder(args.sessions_dir, args.output_dir)
  except SteeringV2EvaluationError as exc:
    print(f"Steering v2 session scoring failed: {exc}")
    return 1

  print("Steering v2 scoring completed.")
  print(f"Session outputs: {Path(summary['output_dir']) / 'sessions'}")
  print(f"Summary: {Path(summary['output_dir']) / 'summary.json'}")
  print(f"Fragments evaluated: {summary['total_fragments_evaluated']}")
  if summary["total_fragments_evaluated"] == 0:
    return 1
  if summary.get("mean_harmonic_score") is not None and math.isnan(float(summary["mean_harmonic_score"])):
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
