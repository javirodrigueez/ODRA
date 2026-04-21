#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CTRS_KEYS = [
  "general_1_understanding",
  "general_2_interpersonal_effectiveness",
  "general_3_collaboration",
  "CBT_1_guided_discovery",
  "CBT_2_focus",
  "CBT_3_strategy",
]

WAI_KEYS = [f"wai{i}" for i in range(1, 13)]

FEELINGS = [
  "interested",
  "excited",
  "strong",
  "enthusiastic",
  "proud",
  "alert",
  "inspired",
  "determined",
  "attentive",
  "active",
  "distressed",
  "upset",
  "guilty",
  "scared",
  "hostile",
  "irritable",
  "ashamed",
  "nervous",
  "jittery",
  "afraid",
]
POSITIVE_FEELINGS = FEELINGS[:10]
NEGATIVE_FEELINGS = FEELINGS[10:]


def _load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
  return payload if isinstance(payload, dict) else {}


def _json_files(path: Path) -> list[Path]:
  if not path.is_dir():
    return []
  return sorted(file for file in path.glob("*.json") if file.is_file() and 'negative' in file.name)


def _mean(values: list[float]) -> float | None:
  if not values:
    return None
  return sum(values) / len(values)


def _number(value: Any) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _score_value(value: Any) -> float | None:
  if isinstance(value, list):
    numeric_values = [_number(item) for item in value]
    return _mean([item for item in numeric_values if item is not None])
  return _number(value)


def _sum_cost(path: Path) -> float:
  total = 0.0
  for file in _json_files(path):
    total += float(_load_json(file).get("cost", 0.0) or 0.0)
  return total


def _average_keys(path: Path, keys: list[str]) -> tuple[dict[str, float], int]:
  totals = {key: 0.0 for key in keys}
  counts = {key: 0 for key in keys}
  file_count = 0
  for file in _json_files(path):
    data = _load_json(file)
    file_count += 1
    for key in keys:
      value = _number(data.get(key))
      if value is None:
        continue
      totals[key] += value
      counts[key] += 1

  averages = {
    key: totals[key] / counts[key]
    for key in keys
    if counts[key] > 0
  }
  return averages, file_count


def compute_ctrs(eval_dir: Path) -> dict[str, Any]:
  averages, file_count = _average_keys(eval_dir / "ctrs", CTRS_KEYS)
  return {
    **averages,
    "count": file_count,
  }


def compute_wai(eval_dir: Path) -> dict[str, Any]:
  items, file_count = _average_keys(eval_dir / "wai", WAI_KEYS)
  score_task = _mean([
    items[key]
    for key in ("wai1", "wai2", "wai8", "wai12")
    if key in items
  ])
  score_goal_values: list[float] = []
  if "wai4" in items:
    score_goal_values.append(8 - items["wai4"])
  if "wai6" in items:
    score_goal_values.append(items["wai6"])
  if "wai10" in items:
    score_goal_values.append(8 - items["wai10"])
  if "wai11" in items:
    score_goal_values.append(items["wai11"])
  score_goal = _mean(score_goal_values)
  score_bond = _mean([
    items[key]
    for key in ("wai3", "wai5", "wai7", "wai9")
    if key in items
  ])

  return {
    "items": items,
    "score_task": score_task,
    "score_goal": score_goal,
    "score_bond": score_bond,
    "count": file_count,
  }


def _collect_panas_scores(path: Path) -> dict[str, dict[str, list[float]]]:
  by_attitude: dict[str, dict[str, list[float]]] = {}
  for file in _json_files(path):
    data = _load_json(file)
    attitude = str(data.get("attitude") or "unknown").strip().lower() or "unknown"
    attitude_scores = by_attitude.setdefault(attitude, {feeling: [] for feeling in FEELINGS})
    for feeling in FEELINGS:
      value = _score_value(data.get(feeling))
      if value is not None:
        attitude_scores[feeling].append(value)
  return by_attitude


def _count_panas_files_by_attitude(path: Path) -> dict[str, int]:
  counts: dict[str, int] = {}
  for file in _json_files(path):
    data = _load_json(file)
    attitude = str(data.get("attitude") or "unknown").strip().lower() or "unknown"
    counts[attitude] = counts.get(attitude, 0) + 1
  return counts


def _average_panas_scores(raw: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
  averaged: dict[str, dict[str, float]] = {}
  for attitude, scores in raw.items():
    averaged[attitude] = {
      feeling: average
      for feeling, values in scores.items()
      if (average := _mean(values)) is not None
    }
  return averaged


def _emotion_group_change(
  before: dict[str, float],
  after: dict[str, float],
  feelings: list[str],
) -> float | None:
  changes = [
    after[feeling] - before[feeling]
    for feeling in feelings
    if feeling in before and feeling in after
  ]
  return _mean(changes)


def compute_panas(eval_dir: Path) -> dict[str, Any]:
  before = _average_panas_scores(_collect_panas_scores(eval_dir / "panas_before"))
  after = _average_panas_scores(_collect_panas_scores(eval_dir / "panas_after"))
  attitudes = sorted(set(before) | set(after))

  positive_change: dict[str, float | None] = {}
  negative_change: dict[str, float | None] = {}
  count_by_attitude = _count_panas_files_by_attitude(eval_dir / "panas_after")
  for attitude in attitudes:
    positive_change[attitude] = _emotion_group_change(
      before.get(attitude, {}),
      after.get(attitude, {}),
      POSITIVE_FEELINGS,
    )
    negative_change[attitude] = _emotion_group_change(
      before.get(attitude, {}),
      after.get(attitude, {}),
      NEGATIVE_FEELINGS,
    )
    count_by_attitude.setdefault(attitude, 0)

  return {
    "before": before,
    "after": after,
    "positive_change": positive_change,
    "negative_change": negative_change,
    "count_by_attitude": count_by_attitude,
  }


def compute_scores(eval_dir: str | Path) -> dict[str, Any]:
  eval_path = Path(eval_dir)
  cost = {
    "ctrs": _sum_cost(eval_path / "ctrs"),
    "wai": _sum_cost(eval_path / "wai"),
    "panas_before": _sum_cost(eval_path / "panas_before"),
    "panas_after": _sum_cost(eval_path / "panas_after"),
  }
  cost["total"] = sum(cost.values())

  return {
    "ctrs": compute_ctrs(eval_path),
    "wai": compute_wai(eval_path),
    "panas": compute_panas(eval_path),
    "cost": cost,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description="Aggregate CTRS, WAI, and PANAS evaluator outputs.")
  parser.add_argument("--eval-dir", type=Path, required=True, help="Directory containing ctrs/wai/panas_before/panas_after subdirs.")
  parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write aggregated score JSON.")
  args = parser.parse_args()

  scores = compute_scores(args.eval_dir)
  rendered = json.dumps(scores, indent=2, ensure_ascii=True)
  print(rendered)

  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
      f.write(rendered)
      f.write("\n")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
