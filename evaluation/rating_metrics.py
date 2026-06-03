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
KNOWN_ATTITUDES = ("very_positive", "very_negative", "positive", "neutral", "negative")
DELTA_SKIP_KEYS = {"count", "count_by_attitude"}


def _load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
  return payload if isinstance(payload, dict) else {}


def _json_files(path: Path) -> list[Path]:
  if not path.is_dir():
    return []
  return sorted(file for file in path.glob("*.json") if file.is_file())


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


def _normalize_attitude(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  attitude = value.strip().lower()
  return attitude or None


def _attitude_from_path(path: Path) -> str:
  stem = path.stem.lower()
  for attitude in KNOWN_ATTITUDES:
    if stem.endswith(f"_{attitude}") or stem == attitude:
      return attitude
  if "_" in stem:
    return stem.rsplit("_", 1)[1] or "unknown"
  return "unknown"


def _attitude_for_payload(data: dict[str, Any]) -> str | None:
  attitude = _normalize_attitude(data.get("attitude"))
  if attitude is not None:
    return attitude

  patient_profile = data.get("patient_profile")
  if isinstance(patient_profile, dict):
    attitude = _normalize_attitude(patient_profile.get("attitude"))
    if attitude is not None:
      return attitude

  return None


def _source_json_dir_for_eval_dir(eval_dir: Path) -> Path | None:
  if eval_dir.name == "eval":
    candidate = eval_dir.parent / "json"
  elif eval_dir.name.startswith("eval_"):
    candidate = eval_dir.parent / f"json_{eval_dir.name.removeprefix('eval_')}"
  else:
    candidate = eval_dir.parent / "json"

  return candidate if candidate.is_dir() else None


def _source_payload_for_file(path: Path, source_json_dir: Path | None) -> dict[str, Any]:
  if source_json_dir is None:
    return {}

  candidate = source_json_dir / path.name
  if not candidate.is_file():
    return {}

  return _load_json(candidate)


def _attitude_for_file(path: Path, data: dict[str, Any], source_json_dir: Path | None = None) -> str:
  attitude = _attitude_for_payload(data)
  if attitude is not None:
    return attitude

  source_attitude = _attitude_for_payload(_source_payload_for_file(path, source_json_dir))
  if source_attitude is not None:
    return source_attitude

  return _attitude_from_path(path)


def _payloads_by_attitude(path: Path, source_json_dir: Path | None = None) -> dict[str, list[dict[str, Any]]]:
  by_attitude: dict[str, list[dict[str, Any]]] = {}
  for file in _json_files(path):
    data = _load_json(file)
    attitude = _attitude_for_file(file, data, source_json_dir)
    by_attitude.setdefault(attitude, []).append(data)
  return by_attitude


def _sum_cost(path: Path) -> float:
  total = 0.0
  for file in _json_files(path):
    total += float(_load_json(file).get("cost", 0.0) or 0.0)
  return total


def _average_keys_from_payloads(payloads: list[dict[str, Any]], keys: list[str]) -> tuple[dict[str, float], int]:
  totals = {key: 0.0 for key in keys}
  counts = {key: 0 for key in keys}
  file_count = 0
  for data in payloads:
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


def _average_keys(path: Path, keys: list[str]) -> tuple[dict[str, float], int]:
  payloads = [_load_json(file) for file in _json_files(path)]
  return _average_keys_from_payloads(payloads, keys)


def _is_metric_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _delta_from_average(value: Any, average: Any) -> Any:
  if isinstance(value, dict) and isinstance(average, dict):
    deltas: dict[str, Any] = {}
    for key, item in value.items():
      if key in DELTA_SKIP_KEYS or key not in average:
        continue
      delta = _delta_from_average(item, average[key])
      if delta is not None:
        deltas[key] = delta
    return deltas

  if _is_metric_number(value) and _is_metric_number(average):
    return value - average

  return None


def _with_attitude_deltas(
  average: dict[str, Any],
  by_attitude: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  return {
    "average": average,
    "by_attitude": by_attitude,
    "delta_from_average": {
      attitude: _delta_from_average(metrics, average)
      for attitude, metrics in by_attitude.items()
    },
  }


def _flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
  if isinstance(value, dict):
    flattened: dict[str, float] = {}
    for key, item in value.items():
      metric_name = f"{prefix}.{key}" if prefix else str(key)
      flattened.update(_flatten_numeric_metrics(item, metric_name))
    return flattened

  if prefix and _is_metric_number(value):
    return {prefix: float(value)}

  return {}


def _format_metric_value(metric_name: str, value: float | None, *, signed: bool = False) -> str:
  if value is None:
    return ""
  if abs(value) < 0.00005:
    value = 0.0
  if signed:
    return f"{value:+.4f}"
  if "count" in metric_name and float(value).is_integer():
    return str(int(value))
  return f"{value:.4f}"


def _render_markdown_table(
  title: str,
  header: list[str],
  rows: list[list[str]],
  *,
  numeric_columns: set[int] | None = None,
) -> str:
  if not rows:
    return ""
  numeric_columns = numeric_columns or set()
  align = [
    "---:" if index in numeric_columns else "---"
    for index, _ in enumerate(header)
  ]
  lines = [
    title,
    "",
    "| " + " | ".join(header) + " |",
    "| " + " | ".join(align) + " |",
  ]
  lines.extend("| " + " | ".join(row) + " |" for row in rows)
  return "\n".join(lines)


def _render_numeric_metric_table(title: str, metrics: dict[str, Any]) -> str:
  flattened = _flatten_numeric_metrics(metrics)
  rows = [
    [metric_name, _format_metric_value(metric_name, value)]
    for metric_name, value in sorted(flattened.items())
  ]
  return _render_markdown_table(title, ["metric", "value"], rows, numeric_columns={1})


def _render_attitude_metric_table(
  title: str,
  by_attitude: dict[str, Any],
  *,
  signed: bool = False,
) -> str:
  attitudes = sorted(by_attitude)
  flattened_by_attitude = {
    attitude: _flatten_numeric_metrics(metrics)
    for attitude, metrics in by_attitude.items()
  }
  metric_keys = sorted({
    key
    for flattened in flattened_by_attitude.values()
    for key in flattened
  })
  if not attitudes or not metric_keys:
    return ""

  rows = [
    [
      metric_name,
      *[
        _format_metric_value(
          metric_name,
          flattened_by_attitude[attitude].get(metric_name),
          signed=signed,
        )
        for attitude in attitudes
      ],
    ]
    for metric_name in metric_keys
  ]
  return _render_markdown_table(
    title,
    ["metric", *attitudes],
    rows,
    numeric_columns=set(range(1, len(attitudes) + 1)),
  )


def _render_markdown_delta_table(metric_name: str, delta_by_attitude: dict[str, Any]) -> str:
  return _render_attitude_metric_table(
    f"{metric_name.upper()} delta from average",
    delta_by_attitude,
    signed=True,
  )


def render_delta_tables(scores: dict[str, Any]) -> str:
  tables = []
  for metric_name in ("ctrs", "wai", "panas"):
    metric_scores = scores.get(metric_name)
    if not isinstance(metric_scores, dict):
      continue
    delta_by_attitude = metric_scores.get("delta_from_average")
    if not isinstance(delta_by_attitude, dict):
      continue
    table = _render_markdown_delta_table(metric_name, delta_by_attitude)
    if table:
      tables.append(table)
  return "\n\n".join(tables)


def render_score_tables(scores: dict[str, Any]) -> str:
  tables = []
  for metric_name in ("ctrs", "wai", "panas"):
    metric_scores = scores.get(metric_name)
    if not isinstance(metric_scores, dict):
      continue

    average = metric_scores.get("average")
    by_attitude = metric_scores.get("by_attitude")
    delta_by_attitude = metric_scores.get("delta_from_average")
    if isinstance(average, dict) and isinstance(by_attitude, dict):
      average_table = _render_numeric_metric_table(
        f"{metric_name.upper()} average",
        average,
      )
      by_attitude_table = _render_attitude_metric_table(
        f"{metric_name.upper()} by attitude",
        by_attitude,
      )
      delta_table = _render_markdown_delta_table(metric_name, delta_by_attitude) if isinstance(delta_by_attitude, dict) else ""
      tables.extend(table for table in (average_table, by_attitude_table, delta_table) if table)
    else:
      table = _render_numeric_metric_table(f"{metric_name.upper()} scores", metric_scores)
      if table:
        tables.append(table)

  cost = scores.get("cost")
  if isinstance(cost, dict):
    table = _render_numeric_metric_table("Cost", cost)
    if table:
      tables.append(table)

  return "\n\n".join(tables)


def compute_ctrs(eval_dir: Path) -> dict[str, Any]:
  averages, file_count = _average_keys(eval_dir / "ctrs", CTRS_KEYS)
  return {
    **averages,
    "count": file_count,
  }


def compute_ctrs_by_attitude(eval_dir: Path) -> dict[str, Any]:
  by_attitude = {}
  source_json_dir = _source_json_dir_for_eval_dir(eval_dir)
  for attitude, payloads in sorted(_payloads_by_attitude(eval_dir / "ctrs", source_json_dir).items()):
    averages, file_count = _average_keys_from_payloads(payloads, CTRS_KEYS)
    by_attitude[attitude] = {
      **averages,
      "count": file_count,
    }
  return _with_attitude_deltas(compute_ctrs(eval_dir), by_attitude)


def _wai_result(items: dict[str, float], file_count: int) -> dict[str, Any]:
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


def compute_wai(eval_dir: Path) -> dict[str, Any]:
  items, file_count = _average_keys(eval_dir / "wai", WAI_KEYS)
  return _wai_result(items, file_count)


def compute_wai_by_attitude(eval_dir: Path) -> dict[str, Any]:
  by_attitude = {}
  source_json_dir = _source_json_dir_for_eval_dir(eval_dir)
  for attitude, payloads in sorted(_payloads_by_attitude(eval_dir / "wai", source_json_dir).items()):
    items, file_count = _average_keys_from_payloads(payloads, WAI_KEYS)
    by_attitude[attitude] = _wai_result(items, file_count)
  return _with_attitude_deltas(compute_wai(eval_dir), by_attitude)


def _collect_panas_scores(path: Path, source_json_dir: Path | None = None) -> dict[str, dict[str, list[float]]]:
  by_attitude: dict[str, dict[str, list[float]]] = {}
  for file in _json_files(path):
    data = _load_json(file)
    attitude = _attitude_for_file(file, data, source_json_dir)
    attitude_scores = by_attitude.setdefault(attitude, {feeling: [] for feeling in FEELINGS})
    for feeling in FEELINGS:
      value = _score_value(data.get(feeling))
      if value is not None:
        attitude_scores[feeling].append(value)
  return by_attitude


def _count_panas_files_by_attitude(path: Path, source_json_dir: Path | None = None) -> dict[str, int]:
  counts: dict[str, int] = {}
  for file in _json_files(path):
    data = _load_json(file)
    attitude = _attitude_for_file(file, data, source_json_dir)
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


def _average_all_panas_scores(raw: dict[str, dict[str, list[float]]]) -> dict[str, float]:
  combined = {feeling: [] for feeling in FEELINGS}
  for scores in raw.values():
    for feeling, values in scores.items():
      combined[feeling].extend(values)
  return {
    feeling: average
    for feeling, values in combined.items()
    if (average := _mean(values)) is not None
  }


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
  source_json_dir = _source_json_dir_for_eval_dir(eval_dir)
  before = _average_panas_scores(_collect_panas_scores(eval_dir / "panas_before", source_json_dir))
  after = _average_panas_scores(_collect_panas_scores(eval_dir / "panas_after", source_json_dir))
  attitudes = sorted(set(before) | set(after))

  positive_change: dict[str, float | None] = {}
  negative_change: dict[str, float | None] = {}
  count_by_attitude = _count_panas_files_by_attitude(eval_dir / "panas_after", source_json_dir)
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


def compute_panas_by_attitude(eval_dir: Path) -> dict[str, Any]:
  source_json_dir = _source_json_dir_for_eval_dir(eval_dir)
  raw_before = _collect_panas_scores(eval_dir / "panas_before", source_json_dir)
  raw_after = _collect_panas_scores(eval_dir / "panas_after", source_json_dir)
  before = _average_panas_scores(raw_before)
  after = _average_panas_scores(raw_after)
  attitudes = sorted(set(before) | set(after))
  count_by_attitude = _count_panas_files_by_attitude(eval_dir / "panas_after", source_json_dir)

  by_attitude: dict[str, dict[str, Any]] = {}
  for attitude in attitudes:
    by_attitude[attitude] = {
      "before": before.get(attitude, {}),
      "after": after.get(attitude, {}),
      "positive_change": _emotion_group_change(
        before.get(attitude, {}),
        after.get(attitude, {}),
        POSITIVE_FEELINGS,
      ),
      "negative_change": _emotion_group_change(
        before.get(attitude, {}),
        after.get(attitude, {}),
        NEGATIVE_FEELINGS,
      ),
      "count": count_by_attitude.get(attitude, 0),
    }

  average_before = _average_all_panas_scores(raw_before)
  average_after = _average_all_panas_scores(raw_after)
  average = {
    "before": average_before,
    "after": average_after,
    "positive_change": _emotion_group_change(
      average_before,
      average_after,
      POSITIVE_FEELINGS,
    ),
    "negative_change": _emotion_group_change(
      average_before,
      average_after,
      NEGATIVE_FEELINGS,
    ),
    "count": len(_json_files(eval_dir / "panas_after")),
  }
  return _with_attitude_deltas(average, by_attitude)


def compute_scores(
  eval_dir: str | Path,
  *,
  per_attitude: bool = False,
  include_ctrs: bool = True,
  include_wai: bool = True,
  include_panas: bool = True,
) -> dict[str, Any]:
  eval_path = Path(eval_dir)
  cost: dict[str, float] = {}
  if include_ctrs:
    cost["ctrs"] = _sum_cost(eval_path / "ctrs")
  if include_wai:
    cost["wai"] = _sum_cost(eval_path / "wai")
  if include_panas:
    cost["panas_before"] = _sum_cost(eval_path / "panas_before")
    cost["panas_after"] = _sum_cost(eval_path / "panas_after")
  cost["total"] = sum(cost.values())

  scores: dict[str, Any] = {}
  if include_ctrs:
    scores["ctrs"] = compute_ctrs_by_attitude(eval_path) if per_attitude else compute_ctrs(eval_path)
  if include_wai:
    scores["wai"] = compute_wai_by_attitude(eval_path) if per_attitude else compute_wai(eval_path)
  if include_panas:
    scores["panas"] = compute_panas_by_attitude(eval_path) if per_attitude else compute_panas(eval_path)
  scores["cost"] = cost
  return scores


def main() -> int:
  parser = argparse.ArgumentParser(description="Aggregate CTRS, WAI, and PANAS evaluator outputs.")
  parser.add_argument("--eval-dir", type=Path, required=True, help="Directory containing ctrs/wai/panas_before/panas_after subdirs.")
  parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write aggregated score JSON.")
  parser.add_argument("--per-attitude", action="store_true", help="Report metrics by attitude and include deltas from the overall average.")
  parser.add_argument("--exc-ctrs", action="store_true", help="Exclude CTRS metrics from the output.")
  parser.add_argument("--exc-wai", action="store_true", help="Exclude WAI metrics from the output.")
  parser.add_argument("--exc-panas", action="store_true", help="Exclude PANAS metrics from the output.")
  args = parser.parse_args()

  include_ctrs = not args.exc_ctrs
  include_wai = not args.exc_wai
  include_panas = not args.exc_panas
  if not (include_ctrs or include_wai or include_panas):
    parser.error("At least one metric must be included.")

  scores = compute_scores(
    args.eval_dir,
    per_attitude=args.per_attitude,
    include_ctrs=include_ctrs,
    include_wai=include_wai,
    include_panas=include_panas,
  )
  rendered = json.dumps(scores, indent=2, ensure_ascii=True)
  print(render_score_tables(scores))

  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
      f.write(rendered)
      f.write("\n")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
