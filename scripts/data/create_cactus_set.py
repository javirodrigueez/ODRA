#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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


@dataclass(frozen=True)
class SeedReference:
  index: int
  name: str
  attitude: str


@dataclass(frozen=True)
class SeedMatch:
  reference: SeedReference
  source_row: SourceRow
  candidate_count: int


def normalize_attitude(attitude: Any) -> str:
  return str(attitude or "").strip().lower()


def extract_profile_name(intake_form: Any) -> str:
  return extract_line("Name", str(intake_form or "")).strip()


def normalize_speaker(speaker: str) -> str:
  normalized = speaker.strip().lower()
  if normalized.startswith("counselor") or normalized.startswith("therapist"):
    return "therapist"
  if normalized.startswith("client") or normalized.startswith("patient"):
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


def parse_history_messages(history: Any, *, source: str = "history") -> list[dict[str, str]]:
  if not isinstance(history, list):
    raise CactusSetError(f"{source} must be a list of message objects.")

  messages = []
  for index, item in enumerate(history):
    if not isinstance(item, dict):
      raise CactusSetError(f"{source}[{index}] must be an object.")
    if "role" not in item:
      raise CactusSetError(f"{source}[{index}] is missing role.")
    content = item.get("message", item.get("content"))
    if content is None:
      raise CactusSetError(f"{source}[{index}] is missing message/content.")
    messages.append(
      {
        "role": normalize_speaker(str(item["role"])),
        "content": str(content).strip(),
      }
    )

  if not messages:
    raise CactusSetError(f"{source} did not contain any messages.")
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


def load_seed_references(seed_file: Path) -> list[SeedReference]:
  try:
    with seed_file.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except FileNotFoundError as exc:
    raise CactusSetError(f"Seed file does not exist: {seed_file}") from exc
  except json.JSONDecodeError as exc:
    raise CactusSetError(f"Seed file is not valid JSON: {seed_file}: {exc}") from exc

  if not isinstance(payload, list):
    raise CactusSetError(f"Seed file must contain a JSON array: {seed_file}")

  references = []
  for index, item in enumerate(payload):
    if not isinstance(item, dict):
      raise CactusSetError(f"Seed entry {index} must be an object.")
    ai_client = item.get("AI_client")
    if not isinstance(ai_client, dict):
      raise CactusSetError(f"Seed entry {index} is missing AI_client object.")

    name = extract_profile_name(ai_client.get("intake_form"))
    if not name:
      raise CactusSetError(f"Seed entry {index} is missing a profile name in AI_client.intake_form.")

    attitude = normalize_attitude(ai_client.get("attitude"))
    if attitude not in ATTITUDES:
      raise CactusSetError(
        f"Seed entry {index} has unsupported AI_client.attitude {ai_client.get('attitude')!r}."
      )

    references.append(SeedReference(index=index, name=name, attitude=attitude))

  if not references:
    raise CactusSetError(f"Seed file has no entries: {seed_file}")
  return references


def match_seed_references(
  rows: Iterable[dict[str, Any]],
  references: list[SeedReference],
) -> list[SeedMatch]:
  rows_by_pair: dict[tuple[str, str], list[SourceRow]] = {}
  for index, item in enumerate(rows):
    if not isinstance(item, dict):
      raise CactusSetError(f"Expected dataset row {index} to be an object.")

    name = extract_profile_name(item.get("intake_form"))
    attitude = normalize_attitude(item.get("attitude"))
    if not name or attitude not in ATTITUDES:
      continue
    rows_by_pair.setdefault((name, attitude), []).append(SourceRow(index=index, data=item))

  missing = []
  matches = []
  for reference in references:
    candidates = sorted(
      rows_by_pair.get((reference.name, reference.attitude), []),
      key=lambda row: row.index,
    )
    if not candidates:
      missing.append(reference)
      continue
    matches.append(
      SeedMatch(
        reference=reference,
        source_row=candidates[0],
        candidate_count=len(candidates),
      )
    )

  if missing:
    examples = ", ".join(
      f"{reference.name!r}/{reference.attitude}" for reference in missing[:10]
    )
    if len(missing) > 10:
      examples += f", ... ({len(missing)} total)"
    raise CactusSetError(
      "Could not find CACTUS rows for seed references by (name, attitude): "
      f"{examples}"
    )

  return matches


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


def build_patient_profile_from_fields(
  idx: int,
  intake_form: Any,
  attitude: Any,
  patterns: Any = None,
) -> dict[str, Any]:
  text = str(intake_form or "")
  normalized_attitude = normalize_attitude(attitude)
  return {
    "idx": idx,
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
    "patterns": list(patterns or []),
    "attitude": normalized_attitude,
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


def build_patient_profile(source_row: SourceRow) -> dict[str, Any]:
  item = source_row.data
  return build_patient_profile_from_fields(
    source_row.index,
    item.get("intake_form"),
    item.get("attitude"),
    item.get("patterns"),
  )


def build_simulation(source_row: SourceRow) -> dict[str, Any]:
  return {
    "messages": parse_dialogue(source_row.data.get("dialogue")),
    "patient_profile": build_patient_profile(source_row),
  }


def read_json(path: Path) -> Any:
  try:
    with path.open("r", encoding="utf-8") as f:
      return json.load(f)
  except json.JSONDecodeError as exc:
    raise CactusSetError(f"Invalid JSON file: {path}: {exc}") from exc


def session_ai_client(payload: dict[str, Any], path: Path) -> dict[str, Any]:
  example = payload.get("example")
  if isinstance(example, dict) and isinstance(example.get("AI_client"), dict):
    return example["AI_client"]
  if isinstance(payload.get("AI_client"), dict):
    return payload["AI_client"]
  raise CactusSetError(f"Session file is missing example.AI_client: {path}")


def build_session_simulation(path: Path, payload: dict[str, Any], index: int) -> dict[str, Any]:
  ai_client = session_ai_client(payload, path)
  intake_form = ai_client.get("intake_form")
  attitude = normalize_attitude(ai_client.get("attitude"))
  if attitude not in ATTITUDES:
    raise CactusSetError(f"Session file has unsupported or missing attitude: {path}")

  history = payload.get("history")
  if isinstance(history, list):
    messages = parse_history_messages(history, source=f"{path.name}.history")
  else:
    dialogue = ai_client.get("dialogue_history") or ai_client.get("dialogue_history_init")
    messages = parse_dialogue(dialogue)

  example = payload.get("example")
  patterns = []
  if isinstance(example, dict):
    patterns = list(example.get("patterns") or [])
  return {
    "messages": messages,
    "patient_profile": build_patient_profile_from_fields(
      index,
      intake_form,
      attitude,
      patterns,
    ),
  }


def safe_filename_stem(stem: str) -> str:
  safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
  return safe or "session"


def session_output_filename(source_path: Path, used_filenames: set[str]) -> str:
  stem = safe_filename_stem(source_path.stem)
  if not stem.startswith("simulation_"):
    stem = f"simulation_{stem}"
  filename = f"{stem}.json"
  if filename not in used_filenames:
    return filename

  suffix = 2
  while f"{stem}_{suffix}.json" in used_filenames:
    suffix += 1
  return f"{stem}_{suffix}.json"


def write_cactus_set_from_json_source(source_dir: Path, output_dir: Path) -> dict[str, Any]:
  if not source_dir.is_dir():
    raise CactusSetError(f"JSON source must be a directory of session JSON files: {source_dir}")
  if output_dir.exists() and not output_dir.is_dir():
    raise CactusSetError(f"Output path exists but is not a directory: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)

  session_files = sorted(path for path in source_dir.glob("*.json") if path.is_file())
  if not session_files:
    raise CactusSetError(f"No session JSON files found in: {source_dir}")

  removed_count = clear_existing_simulations(output_dir)
  written_files = []
  used_filenames: set[str] = set()
  for index, source_path in enumerate(session_files):
    payload = read_json(source_path)
    if not isinstance(payload, dict):
      raise CactusSetError(f"Session file must contain a JSON object: {source_path}")
    filename = session_output_filename(source_path, used_filenames)
    used_filenames.add(filename)
    write_simulation(
      output_dir / filename,
      build_session_simulation(source_path, payload, index),
    )
    written_files.append(filename)

  return {
    "output_dir": str(output_dir),
    "json_source": str(source_dir),
    "source_files": len(session_files),
    "written_files": len(written_files),
    "removed_files": removed_count,
    "files": written_files,
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


def seed_match_filename(match: SeedMatch, used_filenames: set[str]) -> str:
  attitude = match.reference.attitude
  source_index = match.source_row.index
  filename = f"simulation_{source_index}_{attitude}.json"
  if filename not in used_filenames:
    return filename

  filename = f"simulation_{source_index}_{attitude}_seed_{match.reference.index}.json"
  suffix = 2
  while filename in used_filenames:
    filename = f"simulation_{source_index}_{attitude}_seed_{match.reference.index}_{suffix}.json"
    suffix += 1
  return filename


def write_cactus_set_from_seed_file(
  rows: Iterable[dict[str, Any]],
  output_dir: Path,
  seed_file: Path,
) -> dict[str, Any]:
  if output_dir.exists() and not output_dir.is_dir():
    raise CactusSetError(f"Output path exists but is not a directory: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)

  references = load_seed_references(seed_file)
  matches = match_seed_references(rows, references)
  removed_count = clear_existing_simulations(output_dir)

  written_files = []
  used_filenames: set[str] = set()
  ambiguous_matches = 0
  for match in matches:
    if match.candidate_count > 1:
      ambiguous_matches += 1
    filename = seed_match_filename(match, used_filenames)
    used_filenames.add(filename)
    write_simulation(output_dir / filename, build_simulation(match.source_row))
    written_files.append(filename)

  return {
    "output_dir": str(output_dir),
    "seed_file": str(seed_file),
    "seed_references": len(references),
    "written_files": len(written_files),
    "removed_files": removed_count,
    "ambiguous_matches": ambiguous_matches,
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
  parser.add_argument(
    "--seed-file",
    type=Path,
    help=(
      "Reference JSON file with AI_client.intake_form and AI_client.attitude. "
      "When provided, rows are selected by (profile name, attitude) instead of random sampling."
    ),
  )
  parser.add_argument(
    "--json-source",
    type=Path,
    help=(
      "Directory of generated session JSON files, each shaped like misc/session_1_cactus.json. "
      "When provided, each session is converted directly to the batch-metrics conversation format."
    ),
  )
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    if args.json_source and args.seed_file:
      raise CactusSetError("--json-source cannot be combined with --seed-file.")

    if args.json_source:
      summary = write_cactus_set_from_json_source(
        args.json_source,
        args.out_dir,
      )
    else:
      print("Loading CACTUS dataset...", flush=True)
      rows = load_cactus_rows()
      if args.seed_file:
        summary = write_cactus_set_from_seed_file(
          rows,
          args.out_dir,
          args.seed_file,
        )
      else:
        summary = write_cactus_set(
          rows,
          args.out_dir,
          n_samples=args.n_samples,
          seed=args.seed,
        )
  except CactusSetError as exc:
    print(f"Failed to create CACTUS set: {exc}", file=sys.stderr)
    return 1

  if args.json_source:
    print(
      "Success: wrote "
      f"{summary['written_files']} files from {summary['source_files']} session files "
      f"to {summary['output_dir']}."
    )
  elif args.seed_file:
    print(
      "Success: wrote "
      f"{summary['written_files']} files from {summary['seed_references']} seed references "
      f"to {summary['output_dir']}."
    )
    if summary["ambiguous_matches"]:
      print(
        f"Note: {summary['ambiguous_matches']} references matched multiple CACTUS rows; "
        "the earliest dataset row was used for each."
      )
  else:
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
