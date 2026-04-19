#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
  from scripts.data.create_from_cactus import extract_line, extract_section
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
  from create_from_cactus import extract_line, extract_section


ATTITUDES = ("positive", "neutral", "negative")
DATASET_NAME = "LangAGI-Lab/cactus"


class CactusSetError(RuntimeError):
  pass


@dataclass(frozen=True)
class SourceRow:
  index: int
  data: dict[str, Any]


@dataclass(frozen=True)
class CompleteCase:
  case_id: int
  key: tuple[str, str, str]
  rows_by_attitude: dict[str, SourceRow]


def normalize_attitude(attitude: Any) -> str:
  return str(attitude or "").strip().lower()


def normalize_speaker(speaker: str) -> str:
  normalized = speaker.strip().lower()
  if normalized.startswith("counselor"):
    return "therapist"
  if normalized.startswith("client"):
    return "patient"
  raise CactusSetError(f"Unsupported dialogue speaker: {speaker!r}")


def parse_dialogue(dialogue: Any) -> list[dict[str, str]]:
  if not isinstance(dialogue, str) or not dialogue.strip():
    raise CactusSetError("Dialogue must be a non-empty string.")

  messages = []
  for line_number, line in enumerate(dialogue.splitlines(), start=1):
    stripped = line.strip()
    if not stripped:
      continue
    speaker, separator, content = stripped.partition(":")
    if not separator:
      raise CactusSetError(f"Dialogue line {line_number} is missing a speaker prefix: {line!r}")
    messages.append(
      {
        "role": normalize_speaker(speaker),
        "content": content.strip(),
      }
    )

  if not messages:
    raise CactusSetError("Dialogue did not contain any messages.")
  return messages


def complete_case_key(item: dict[str, Any]) -> tuple[str, str, str]:
  return (
    str(item.get("intake_form") or ""),
    str(item.get("thought") or ""),
    str(item.get("cbt_technique") or ""),
  )


def complete_case_groups(rows: Iterable[dict[str, Any]]) -> list[CompleteCase]:
  grouped: dict[tuple[str, str, str], dict[str, list[SourceRow]]] = {}

  for index, item in enumerate(rows):
    if not isinstance(item, dict):
      raise CactusSetError(f"Expected dataset row {index} to be an object.")
    attitude = normalize_attitude(item.get("attitude"))
    if attitude not in ATTITUDES:
      continue

    key = complete_case_key(item)
    if not all(key):
      continue
    grouped.setdefault(key, {}).setdefault(attitude, []).append(SourceRow(index=index, data=item))

  complete_cases = []
  for key, buckets in grouped.items():
    if not all(buckets.get(attitude) for attitude in ATTITUDES):
      continue
    rows_by_attitude = {
      attitude: sorted(buckets[attitude], key=lambda row: row.index)[0]
      for attitude in ATTITUDES
    }
    case_id = min(row.index for row in rows_by_attitude.values())
    complete_cases.append(
      CompleteCase(
        case_id=case_id,
        key=key,
        rows_by_attitude=rows_by_attitude,
      )
    )

  return sorted(complete_cases, key=lambda case: case.case_id)


def sample_complete_cases(
  complete_cases: list[CompleteCase],
  n_samples: int,
  seed: int,
) -> list[CompleteCase]:
  if n_samples < 1:
    raise CactusSetError(f"--n-samples must be positive, got: {n_samples}")
  if len(complete_cases) < n_samples:
    raise CactusSetError(
      f"Only found {len(complete_cases)} complete CACTUS cases, requested {n_samples}."
    )

  rng = random.Random(seed)
  return sorted(rng.sample(complete_cases, n_samples), key=lambda case: case.case_id)


def build_patient_profile(source_row: SourceRow) -> dict[str, Any]:
  item = source_row.data
  text = str(item.get("intake_form") or "")
  attitude = normalize_attitude(item.get("attitude"))

  return {
    "idx": source_row.index,
    "personal_info": {
      "name": extract_line("Name", text),
      "age": extract_line("Age", text),
      "gender": extract_line("Gender", text),
      "occupation": extract_line("Occupation", text),
      "education": extract_line("Education", text),
      "marital_status": extract_line("Marital Status", text),
      "family_details": extract_line("Family Details", text),
    },
    "presenting_problem": extract_section("2. Presenting Problem", text),
    "reason_seeking_counseling": extract_section("3. Reason for Seeking Counseling", text),
    "patterns": list(item.get("patterns") or []),
    "attitude": attitude,
    "emotional_recovery": "moderate",
    "more_information": {
      "past_history": extract_section("4. Past History", text),
      "academic_occupational_functioning": extract_section(
        "5. Academic/occupational functioning level",
        text,
      ),
      "social_support_system": extract_section("6. Social Support System", text),
    },
  }


def build_simulation(source_row: SourceRow) -> dict[str, Any]:
  return {
    "messages": parse_dialogue(source_row.data.get("dialogue")),
    "patient_profile": build_patient_profile(source_row),
  }


def clear_existing_simulations(output_dir: Path) -> int:
  removed = 0
  for path in output_dir.glob("simulation_*.json"):
    if path.is_file():
      path.unlink()
      removed += 1
  return removed


def write_simulation(path: Path, payload: dict[str, Any]) -> None:
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
    f.write("\n")


def write_cactus_set(
  rows: Iterable[dict[str, Any]],
  output_dir: Path,
  *,
  n_samples: int = 20,
  seed: int = 0,
) -> dict[str, Any]:
  if output_dir.exists() and not output_dir.is_dir():
    raise CactusSetError(f"Output path exists but is not a directory: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)

  complete_cases = complete_case_groups(rows)
  selected_cases = sample_complete_cases(complete_cases, n_samples, seed)
  removed_count = clear_existing_simulations(output_dir)

  written_files = []
  for case in selected_cases:
    for attitude in ATTITUDES:
      source_row = case.rows_by_attitude[attitude]
      filename = f"simulation_{case.case_id}_{attitude}.json"
      write_simulation(output_dir / filename, build_simulation(source_row))
      written_files.append(filename)

  return {
    "output_dir": str(output_dir),
    "complete_cases": len(complete_cases),
    "sampled_cases": len(selected_cases),
    "written_files": len(written_files),
    "removed_files": removed_count,
    "files": written_files,
  }


def load_cactus_rows() -> Any:
  try:
    from datasets import load_dataset
  except ImportError as exc:
    raise CactusSetError("The datasets package is required to load CACTUS.") from exc

  dataset = load_dataset(DATASET_NAME)
  return dataset["train"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Create a balanced run_batch_metrics-compatible simulation set from CACTUS."
  )
  parser.add_argument(
    "--ou-dir",
    "--out-dir",
    dest="out_dir",
    type=Path,
    required=True,
    help="Directory where simulation_*.json files will be written.",
  )
  parser.add_argument(
    "--n-samples",
    type=int,
    default=20,
    help="Number of complete base cases to sample. Writes 3 files per case. Default: 20",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Deterministic random seed for selecting complete cases. Default: 0",
  )
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    print("Loading CACTUS dataset...")
    summary = write_cactus_set(
      load_cactus_rows(),
      args.out_dir,
      n_samples=args.n_samples,
      seed=args.seed,
    )
  except CactusSetError as exc:
    print(f"Failed to create CACTUS set: {exc}", file=sys.stderr)
    return 1

  print(
    "Success: wrote "
    f"{summary['written_files']} files from {summary['sampled_cases']} complete cases "
    f"to {summary['output_dir']}."
  )
  if summary["removed_files"]:
    print(f"Removed {summary['removed_files']} existing simulation_*.json files first.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
