#!/usr/bin/env python3
"""Measure therapist resistance-prediction accuracy from simulation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Counter:
  correct: int = 0
  total: int = 0
  sessions: set[str] | None = None

  def add(self, is_correct: bool, session_name: str) -> None:
    if self.sessions is None:
      self.sessions = set()
    self.sessions.add(session_name)
    self.total += 1
    if is_correct:
      self.correct += 1

  @property
  def accuracy(self) -> float:
    if self.total == 0:
      return 0.0
    return self.correct / self.total * 100.0

  @property
  def session_count(self) -> int:
    return len(self.sessions or set())


@dataclass
class PairErrorStats:
  absolute_error_total: float = 0.0
  squared_error_total: float = 0.0
  total: int = 0

  def add(self, pair: "PredictionPair") -> None:
    error = pair.predicted_resistance - pair.actual_resistance
    self.absolute_error_total += abs(error)
    self.squared_error_total += error * error
    self.total += 1

  @property
  def mae(self) -> float | None:
    if self.total == 0:
      return None
    return self.absolute_error_total / self.total

  @property
  def rmse(self) -> float | None:
    if self.total == 0:
      return None
    return math.sqrt(self.squared_error_total / self.total)


@dataclass
class SessionResistanceStats:
  pair_difference_total: float = 0.0
  total: int = 0

  def add(self, pair: "PredictionPair") -> None:
    self.pair_difference_total += pair.predicted_resistance - pair.actual_resistance
    self.total += 1

  @property
  def avg_pair_predicted_minus_current(self) -> float | None:
    if self.total == 0:
      return None
    return self.pair_difference_total / self.total


@dataclass
class AttitudeStats:
  counter: Counter = field(default_factory=Counter)
  pair_errors: PairErrorStats = field(default_factory=PairErrorStats)
  max_prediction_change: float | None = None
  prediction_change_count: int = 0
  session_avg_pair_difference_total: float = 0.0
  session_avg_pair_difference_count: int = 0

  def add_prediction(
    self,
    is_correct: bool,
    session_name: str,
    pair: "PredictionPair",
  ) -> None:
    self.counter.add(is_correct, session_name)
    self.pair_errors.add(pair)

    if pair.previous_predicted_resistance is None:
      return

    change = abs(pair.predicted_resistance - pair.previous_predicted_resistance)
    if self.max_prediction_change is None or change > self.max_prediction_change:
      self.max_prediction_change = change
    self.prediction_change_count += 1

  def add_session(self, session_stats: SessionResistanceStats) -> None:
    difference = session_stats.avg_pair_predicted_minus_current
    if difference is None:
      return

    self.session_avg_pair_difference_total += difference
    self.session_avg_pair_difference_count += 1

  @property
  def avg_session_pair_predicted_minus_current(self) -> float | None:
    if self.session_avg_pair_difference_count == 0:
      return None
    return (
      self.session_avg_pair_difference_total
      / self.session_avg_pair_difference_count
    )


@dataclass(frozen=True)
class PredictionPair:
  patient_utterance_index: int | None
  therapist_utterance_index: int | None
  actual_resistance: float
  predicted_resistance: float
  previous_predicted_resistance: float | None


def resistance_interval(value: float) -> str:
  if value <= 0.33:
    return "positive"
  if value < 0.66:
    return "neutral"
  return "negative"


def load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def parse_trace_output(output: Any) -> dict[str, Any]:
  if isinstance(output, dict):
    return output
  if not isinstance(output, str):
    return {}

  text = output.strip()
  if not text:
    return {}

  try:
    parsed = json.loads(text)
  except json.JSONDecodeError:
    return {}
  return parsed if isinstance(parsed, dict) else {}


def coerce_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def coerce_int(value: Any) -> int | None:
  if value is None:
    return None
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def trace_payload(trace: dict[str, Any], step_name: str) -> dict[str, Any]:
  steps = trace.get("traces")
  if not isinstance(steps, list):
    return {}

  for step in steps:
    if not isinstance(step, dict):
      continue
    if step.get("step") != step_name:
      continue
    return parse_trace_output(step.get("output"))
  return {}


def iter_prediction_pairs(data: dict[str, Any]) -> Iterable[PredictionPair]:
  traces = data.get("reasoning_traces")
  if not isinstance(traces, list):
    return

  pending_actual: tuple[int | None, float] | None = None
  last_predicted: float | None = None
  for trace in traces:
    if not isinstance(trace, dict):
      continue

    speaker = str(trace.get("speaker") or "").strip().lower()
    utterance_index = coerce_int(trace.get("utterance_index"))

    if speaker == "patient":
      payload = trace_payload(trace, "resistance_update")
      actual = coerce_float(payload.get("new_r"))
      if actual is not None:
        pending_actual = (utterance_index, actual)
      continue

    if speaker != "therapist" or pending_actual is None:
      continue

    payload = trace_payload(trace, "resistance_prediction")
    predicted = coerce_float(payload.get("new_p"))
    if predicted is None:
      predicted = coerce_float(payload.get("p_target"))
    if predicted is None:
      continue
    previous_predicted = coerce_float(payload.get("previous_p"))
    if previous_predicted is None:
      previous_predicted = last_predicted

    patient_index, actual = pending_actual
    yield PredictionPair(
      patient_utterance_index=patient_index,
      therapist_utterance_index=utterance_index,
      actual_resistance=actual,
      predicted_resistance=predicted,
      previous_predicted_resistance=previous_predicted,
    )
    last_predicted = predicted
    pending_actual = None


def attitude_baseline_label(data: dict[str, Any]) -> str:
  profile = data.get("patient_profile")
  attitude = ""
  if isinstance(profile, dict):
    attitude = str(profile.get("attitude") or "").strip().lower()

  baseline = coerce_float(data.get("resistance_baseline"))
  if attitude and baseline is not None:
    return f"{attitude} (baseline={baseline:.2f})"
  if attitude:
    return attitude
  if baseline is not None:
    return f"baseline={baseline:.2f}"
  return "unknown"


def iter_json_files(input_dir: Path, recursive: bool) -> list[Path]:
  pattern = "**/*.json" if recursive else "*.json"
  return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def format_counter(counter: Counter) -> str:
  return f"{counter.accuracy:.2f}% ({counter.correct}/{counter.total})"


def format_optional_float(value: float | None) -> str:
  if value is None:
    return "n/a"
  return f"{value:.4f}"


def print_results(
  input_dir: Path,
  files_seen: int,
  files_used: int,
  overall: Counter,
  overall_pair_errors: PairErrorStats,
  by_attitude: dict[str, AttitudeStats],
  skipped: list[tuple[Path, str]],
) -> None:
  print("Resistance prediction accuracy")
  print(f"Input directory: {input_dir}")
  print(f"JSON files seen: {files_seen}")
  print(f"JSON files with paired turns: {files_used}")
  print(f"Paired turns evaluated: {overall.total}")
  print(f"Overall accuracy: {format_counter(overall)}")
  print(f"Overall MAE: {format_optional_float(overall_pair_errors.mae)}")
  print(f"Overall RMSE: {format_optional_float(overall_pair_errors.rmse)}")

  print("\nAccuracy by patient attitude baseline:")
  if by_attitude:
    for label, stats in sorted(by_attitude.items()):
      counter = stats.counter
      print(
        f"  {label}: {format_counter(counter)} "
        f"across {counter.session_count} session(s)"
      )
      print(
        "    Max absolute prediction change: "
        f"{format_optional_float(stats.max_prediction_change)} "
        f"({stats.prediction_change_count} change(s))"
      )
      print(f"    Pair MAE: {format_optional_float(stats.pair_errors.mae)}")
      print(f"    Pair RMSE: {format_optional_float(stats.pair_errors.rmse)}")
      print(
        "    Avg session pair Pred-R minus R difference: "
        f"{format_optional_float(stats.avg_session_pair_predicted_minus_current)}"
      )
  else:
    print("  No grouped results available.")

  if skipped:
    print("\nSkipped files:")
    for path, reason in skipped:
      print(f"  {path}: {reason}")


def json_results(
  input_dir: Path,
  files_seen: int,
  files_used: int,
  overall: Counter,
  overall_pair_errors: PairErrorStats,
  by_attitude: dict[str, AttitudeStats],
  skipped: list[tuple[Path, str]],
) -> dict[str, Any]:
  return {
    "input_dir": str(input_dir),
    "json_files_seen": files_seen,
    "json_files_with_paired_turns": files_used,
    "paired_turns_evaluated": overall.total,
    "overall": {
      "accuracy_percent": overall.accuracy,
      "correct": overall.correct,
      "total": overall.total,
      "mae": overall_pair_errors.mae,
      "rmse": overall_pair_errors.rmse,
    },
    "by_patient_attitude_baseline": {
      label: {
        "accuracy_percent": stats.counter.accuracy,
        "correct": stats.counter.correct,
        "total": stats.counter.total,
        "sessions": stats.counter.session_count,
        "mae": stats.pair_errors.mae,
        "rmse": stats.pair_errors.rmse,
        "max_prediction_change": stats.max_prediction_change,
        "prediction_change_count": stats.prediction_change_count,
        "avg_session_pair_predicted_minus_current_r": (
          stats.avg_session_pair_predicted_minus_current
        ),
        "session_avg_pair_difference_count": (
          stats.session_avg_pair_difference_count
        ),
      }
      for label, stats in sorted(by_attitude.items())
    },
    "skipped_files": [
      {"path": str(path), "reason": reason}
      for path, reason in skipped
    ],
  }


def evaluate_directory(
  input_dir: Path,
  recursive: bool,
) -> tuple[
  int,
  int,
  Counter,
  PairErrorStats,
  dict[str, AttitudeStats],
  list[tuple[Path, str]],
]:
  files = iter_json_files(input_dir, recursive)
  overall = Counter()
  overall_pair_errors = PairErrorStats()
  by_attitude: dict[str, AttitudeStats] = {}
  skipped: list[tuple[Path, str]] = []
  files_used = 0

  for path in files:
    try:
      data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
      skipped.append((path, f"could not parse JSON: {exc}"))
      continue

    if not isinstance(data, dict):
      skipped.append((path, "top-level JSON value is not an object"))
      continue

    pairs = list(iter_prediction_pairs(data))
    if not pairs:
      skipped.append((path, "no patient resistance / therapist prediction pairs found"))
      continue

    files_used += 1
    label = attitude_baseline_label(data)
    group_stats = by_attitude.setdefault(label, AttitudeStats())
    session_stats = SessionResistanceStats()
    session_name = path.name

    for pair in pairs:
      actual_interval = resistance_interval(pair.actual_resistance)
      predicted_interval = resistance_interval(pair.predicted_resistance)
      is_correct = actual_interval == predicted_interval
      overall.add(is_correct, session_name)
      overall_pair_errors.add(pair)
      group_stats.add_prediction(is_correct, session_name, pair)
      session_stats.add(pair)

    group_stats.add_session(session_stats)

  return len(files), files_used, overall, overall_pair_errors, by_attitude, skipped


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Compute therapist resistance-prediction accuracy from a directory of "
      "simulation JSON files."
    )
  )
  parser.add_argument(
    "input_dir",
    type=Path,
    help="Directory containing simulation .json files.",
  )
  parser.add_argument(
    "--recursive",
    action="store_true",
    help="Scan JSON files recursively under input_dir.",
  )
  parser.add_argument(
    "--json",
    action="store_true",
    help="Print machine-readable JSON instead of a text report.",
  )
  return parser


def main() -> int:
  parser = build_parser()
  args = parser.parse_args()

  input_dir = args.input_dir
  if not input_dir.is_dir():
    parser.error(f"input_dir does not exist or is not a directory: {input_dir}")

  (
    files_seen,
    files_used,
    overall,
    overall_pair_errors,
    by_attitude,
    skipped,
  ) = evaluate_directory(input_dir, recursive=args.recursive)

  if args.json:
    print(
      json.dumps(
        json_results(
          input_dir,
          files_seen,
          files_used,
          overall,
          overall_pair_errors,
          by_attitude,
          skipped,
        ),
        indent=2,
      )
    )
  else:
    print_results(
      input_dir,
      files_seen,
      files_used,
      overall,
      overall_pair_errors,
      by_attitude,
      skipped,
    )

  return 0 if overall.total else 1


if __name__ == "__main__":
  raise SystemExit(main())
