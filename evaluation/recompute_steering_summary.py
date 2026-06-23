#!/usr/bin/env python3
"""Recompute steering v2 summary.json from existing scored session outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation.steering.steering_v2 import (  # noqa: E402
  CLASS_ORDER,
  SteeringV2EvaluationError,
  apply_cross_validation_to_fragment,
  atomic_write_json,
  coerce_float,
  fragment_label,
  harmonic_mean,
  mean,
  normalize_attitude_class,
  required_fragment_concept_scores,
  session_metrics,
  summary_from_sessions,
)


AGGREGATE_SUMMARY_KEYS = {
  "version",
  "input_dir",
  "output_dir",
  "batch_artifacts",
  "total_files_seen",
  "total_files_evaluated",
  "total_fragments_evaluated",
  "total_turns_evaluated",
  "mean_judge_scores",
  "mean_harmonic_score",
  "resistance_interval_accuracy",
  "shard_count",
  "total_requests",
  "total_estimated_cost",
  "skipped_files",
  "skipped_turns",
  "parse_report",
}


def read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8-sig") as f:
    return json.load(f)


def read_optional_summary(path: Path) -> dict[str, Any]:
  if not path.is_file():
    return {}
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise SteeringV2EvaluationError(f"Existing summary is not a JSON object: {path}")
  return payload


def iter_session_files(sessions_dir: Path) -> list[Path]:
  return sorted(path for path in sessions_dir.glob("simulation_*.json") if path.is_file())


def path_from_summary(value: Any) -> Path | None:
  if value in (None, ""):
    return None
  return Path(str(value))


def parse_attitudes(value: str) -> set[str]:
  attitudes = {item.strip().lower() for item in value.split(",") if item.strip()}
  unknown = sorted(attitudes - set(CLASS_ORDER))
  if unknown:
    raise SteeringV2EvaluationError(
      f"Unknown attitude(s) for cross-validation: {', '.join(unknown)}. "
      f"Expected comma-separated values from: {', '.join(CLASS_ORDER)}"
    )
  return attitudes or set(CLASS_ORDER)


def restore_unpenalized_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
  scored = dict(fragment)
  expected_score = coerce_float(scored.get("unpenalized_expected_resistance_concept_score"))

  if expected_score is None:
    expected_interval = normalize_attitude_class(scored.get("expected_resistance_interval"))
    if expected_interval is not None:
      try:
        expected_scores = required_fragment_concept_scores(scored, expected_interval)
        expected_score = mean(expected_scores)
      except SteeringV2EvaluationError:
        expected_score = coerce_float(scored.get("expected_resistance_concept_score"))

  if expected_score is not None:
    context_score = coerce_float(scored.get("context_relevance"))
    realism_score = coerce_float(scored.get("patient_like_realism"))
    scored["expected_resistance_concept_score"] = float(expected_score)
    if context_score is not None and realism_score is not None:
      scored["harmonic_mean"] = harmonic_mean([float(expected_score), context_score, realism_score])

  scored.pop("unpenalized_expected_resistance_concept_score", None)
  scored.pop("cross_validation_penalized", None)
  return scored


def apply_cross_validation_with_attitudes(
  fragment: dict[str, Any],
  *,
  matched_concepts: bool,
  cross_validate: bool,
  cross_validation_attitudes: set[str],
) -> dict[str, Any]:
  if not cross_validate or matched_concepts:
    return restore_unpenalized_fragment(fragment)

  if cross_validation_attitudes == set(CLASS_ORDER):
    return apply_cross_validation_to_fragment(
      fragment,
      matched_concepts=matched_concepts,
      cross_validate=True,
    )

  expected_interval = normalize_attitude_class(fragment.get("expected_resistance_interval"))
  if expected_interval is None:
    raise SteeringV2EvaluationError(
      f"cross_validate requires expected_resistance_interval in {fragment_label(fragment)} to be one of {CLASS_ORDER}"
    )

  if expected_interval not in cross_validation_attitudes:
    return restore_unpenalized_fragment(fragment)

  expected_scores = required_fragment_concept_scores(fragment, expected_interval)
  expected_score = mean(expected_scores)
  if expected_score is None:
    raise SteeringV2EvaluationError(
      f"cross_validate could not compute expected concept score for {fragment_label(fragment)}"
    )

  competing_scores = [
    score
    for class_name in CLASS_ORDER
    if class_name != expected_interval and class_name in cross_validation_attitudes
    for score in required_fragment_concept_scores(fragment, class_name)
  ]
  penalized = any(score > expected_score for score in competing_scores)

  context_score = coerce_float(fragment.get("context_relevance"))
  realism_score = coerce_float(fragment.get("patient_like_realism"))
  if context_score is None or realism_score is None:
    raise SteeringV2EvaluationError(
      f"cross_validate requires numeric context_relevance and patient_like_realism in {fragment_label(fragment)}"
    )

  scored = dict(fragment)
  scored["unpenalized_expected_resistance_concept_score"] = float(expected_score)
  scored["cross_validation_penalized"] = penalized
  scored["expected_resistance_concept_score"] = 0.0 if penalized else float(expected_score)
  scored["harmonic_mean"] = harmonic_mean(
    [float(scored["expected_resistance_concept_score"]), context_score, realism_score]
  )
  return scored


def recompute_session_payload(
  path: Path,
  *,
  force_cross_validate: bool,
  ignore_cross_validation: bool,
  cross_validation_attitudes: set[str],
) -> dict[str, Any]:
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise SteeringV2EvaluationError(f"Session output is not a JSON object: {path}")

  raw_fragments = payload.get("evaluated_fragments")
  if not isinstance(raw_fragments, list):
    payload = {
      **payload,
      "evaluated_fragments": [],
      "skipped_turns": list(payload.get("skipped_turns") or []) + [{"reason": "missing evaluated_fragments list"}],
    }
    raw_fragments = []

  matched_concepts = bool(payload.get("matched_concepts"))
  effective_cross_validate = (
    False
    if ignore_cross_validation
    else (bool(payload.get("cross_validate")) or force_cross_validate) and not matched_concepts
  )

  fragments: list[dict[str, Any]] = []
  for fragment in raw_fragments:
    if not isinstance(fragment, dict):
      continue
    fragments.append(
      apply_cross_validation_with_attitudes(
        fragment,
        matched_concepts=matched_concepts,
        cross_validate=effective_cross_validate,
        cross_validation_attitudes=cross_validation_attitudes,
      )
    )

  recomputed = dict(payload)
  recomputed["input_filename"] = str(recomputed.get("input_filename") or path.name)
  recomputed["input_file"] = str(recomputed.get("input_file") or path)
  recomputed["evaluated_fragments"] = fragments
  recomputed["metrics"] = session_metrics(fragments)
  recomputed["cross_validate"] = effective_cross_validate
  recomputed["cross_validation_attitudes"] = sorted(cross_validation_attitudes)
  return recomputed


def existing_metadata_flag(existing_summary: dict[str, Any], key: str) -> bool:
  return bool(existing_summary.get(key))


def recompute_summary(
  steering_results_dir: Path,
  *,
  sessions_dir: Path | None = None,
  output_json: Path | None = None,
  force_cross_validate: bool = False,
  ignore_cross_validation: bool = False,
  cross_validation_attitudes: set[str] | None = None,
  input_dir: Path | None = None,
  files_seen_override: int | None = None,
) -> dict[str, Any]:
  output_dir = steering_results_dir.resolve()
  sessions_path = (sessions_dir.resolve() if sessions_dir is not None else output_dir / "sessions")
  if not sessions_path.is_dir():
    raise SteeringV2EvaluationError(f"Sessions directory does not exist: {sessions_path}")

  summary_path = (output_json.resolve() if output_json is not None else output_dir / "summary.json")
  existing_summary = read_optional_summary(summary_path)
  session_files = iter_session_files(sessions_path)
  effective_cross_validation_attitudes = cross_validation_attitudes or set(CLASS_ORDER)

  sessions: list[dict[str, Any]] = []
  skipped_files: list[dict[str, Any]] = []
  for path in session_files:
    try:
      sessions.append(
        recompute_session_payload(
          path,
          force_cross_validate=force_cross_validate or existing_metadata_flag(existing_summary, "cross_validate"),
          ignore_cross_validation=ignore_cross_validation,
          cross_validation_attitudes=effective_cross_validation_attitudes,
        )
      )
    except Exception as exc:
      skipped_files.append({"file": str(path), "reason": f"could not read scored session JSON: {exc}"})

  source_input_dir = (
    input_dir.resolve()
    if input_dir is not None
    else path_from_summary(existing_summary.get("input_dir")) or sessions_path
  )
  files_seen = int(files_seen_override if files_seen_override is not None else existing_summary.get("total_files_seen") or len(session_files))
  artifact_dir = path_from_summary(existing_summary.get("batch_artifacts"))
  parse_report = existing_summary.get("parse_report") if isinstance(existing_summary.get("parse_report"), dict) else None

  summary = summary_from_sessions(
    source_input_dir,
    output_dir,
    files_seen,
    sessions,
    list(existing_summary.get("skipped_files") or []) + skipped_files,
    artifact_dir,
    parse_report,
    request_count=int(existing_summary.get("total_requests") or 0),
    shard_count=int(existing_summary.get("shard_count") or 0),
  )

  for key, value in existing_summary.items():
    if key not in AGGREGATE_SUMMARY_KEYS:
      summary[key] = value

  summary["source_sessions_dir"] = str(sessions_path)
  summary["recomputed_from_steering_session_outputs"] = True
  summary["cross_validate"] = any(bool(session.get("cross_validate")) for session in sessions)
  summary["cross_validation_attitudes"] = sorted(effective_cross_validation_attitudes)

  atomic_write_json(summary_path, summary)
  return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Recompute a steering v2 summary.json from already-scored sessions/*.json outputs."
  )
  parser.add_argument(
    "steering_results_dir",
    type=Path,
    help="Steering results directory, e.g. /outs/eval_full_catus/eval_steer_cross.",
  )
  parser.add_argument(
    "--sessions-dir",
    type=Path,
    default=None,
    help="Directory containing scored session JSON files. Default: <steering_results_dir>/sessions.",
  )
  parser.add_argument(
    "--output-json",
    type=Path,
    default=None,
    help="Summary JSON to write. Default: <steering_results_dir>/summary.json.",
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=None,
    help="Override summary input_dir. Default: existing summary input_dir, then sessions dir.",
  )
  parser.add_argument(
    "--cross-validate",
    action="store_true",
    help="Apply steering_v2 cross-validation penalty while recomputing fragment scores.",
  )
  parser.add_argument(
    "--ignore-cross-validation",
    action="store_true",
    help="Ignore cross-validation flags from the existing summary and session outputs while recomputing scores.",
  )
  parser.add_argument(
    "--cross-validation-attitudes",
    default=",".join(CLASS_ORDER),
    help=(
      "Comma-separated attitudes to consider when cross-validation is active. "
      f"Default: {','.join(CLASS_ORDER)}"
    ),
  )
  parser.add_argument(
    "--files-seen",
    type=int,
    default=None,
    help="Override total_files_seen in the recomputed summary. Default: existing summary value, then session count.",
  )
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    cross_validation_attitudes = parse_attitudes(args.cross_validation_attitudes)
    if args.cross_validate and args.ignore_cross_validation:
      raise SteeringV2EvaluationError("--cross-validate and --ignore-cross-validation cannot be combined.")
    summary = recompute_summary(
      args.steering_results_dir,
      sessions_dir=args.sessions_dir,
      output_json=args.output_json,
      force_cross_validate=args.cross_validate,
      ignore_cross_validation=args.ignore_cross_validation,
      cross_validation_attitudes=cross_validation_attitudes,
      input_dir=args.input_dir,
      files_seen_override=args.files_seen,
    )
  except SteeringV2EvaluationError as exc:
    print(f"Failed to recompute steering summary: {exc}", file=sys.stderr)
    return 1

  print(f"Summary: {Path(summary['output_dir']) / 'summary.json'}")
  print(f"Sessions: {summary['source_sessions_dir']}")
  print(f"Files evaluated: {summary['total_files_evaluated']}/{summary['total_files_seen']}")
  print(f"Fragments evaluated: {summary['total_fragments_evaluated']}")
  print(f"Turns evaluated: {summary['total_turns_evaluated']}")
  print(f"Mean harmonic score: {summary['mean_harmonic_score']}")
  print(f"Resistance interval accuracy: {summary['resistance_interval_accuracy']}")
  print("Mean judge scores:")
  for key, value in (summary.get("mean_judge_scores") or {}).items():
    print(f"  {key}: {value}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
