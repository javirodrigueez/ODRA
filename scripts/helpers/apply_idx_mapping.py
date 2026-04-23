from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_FILENAME_RE = re.compile(r"^simulation_(?P<idx>\d+)_(?P<label>.+)(?P<suffix>\.[^.]+)$")
_SUBDIR_SUFFIXES = {
  "json": ".json",
  "transcripts": ".txt",
}


@dataclass(frozen=True)
class RenamePlan:
  source: Path
  target: Path
  old_idx: str
  new_idx: str


@dataclass(frozen=True)
class RenameResult:
  plans: list[RenamePlan]
  missing_old_idxs: list[str]


def _coerce_idx(value: Any, description: str) -> str:
  if isinstance(value, bool):
    raise ValueError(f"{description} must be an integer or digit string, got {value!r}.")
  if isinstance(value, int):
    if value < 0:
      raise ValueError(f"{description} must be non-negative, got {value!r}.")
    return str(value)
  if isinstance(value, str):
    value = value.strip()
    if value.isdigit():
      return value
  raise ValueError(f"{description} must be an integer or digit string, got {value!r}.")


def load_idx_mapping(path: str | Path) -> dict[str, str]:
  mapping_path = Path(path)
  with mapping_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)

  if not isinstance(payload, dict):
    raise ValueError(f"Idx mapping must be a JSON object: {mapping_path}")

  idx_mapping: dict[str, str] = {}
  for old_idx, new_idx in payload.items():
    old_key = _coerce_idx(old_idx, "Idx mapping key")
    idx_mapping[old_key] = _coerce_idx(new_idx, f"Idx mapping value for {old_idx!r}")
  return idx_mapping


def _parse_simulation_filename(path: Path, suffix: str) -> tuple[str, str] | None:
  if path.suffix != suffix:
    return None
  match = _FILENAME_RE.match(path.name)
  if not match or match.group("suffix") != suffix:
    return None
  return match.group("idx"), match.group("label")


def _subdir(input_dir: Path, name: str) -> Path:
  path = input_dir / name
  if not path.is_dir():
    raise ValueError(f"Input directory must contain a {name!r} subdirectory: {input_dir}")
  return path


def build_rename_plan(input_dir: str | Path, idx_mapping: Mapping[str, str]) -> RenameResult:
  base = Path(input_dir)
  if not base.is_dir():
    raise ValueError(f"Input directory does not exist or is not a directory: {base}")

  normalized_mapping = {
    _coerce_idx(old_idx, "Idx mapping key"): _coerce_idx(new_idx, f"Idx mapping value for {old_idx!r}")
    for old_idx, new_idx in idx_mapping.items()
  }

  plans: list[RenamePlan] = []
  found_old_idxs: set[str] = set()
  for subdir_name, suffix in _SUBDIR_SUFFIXES.items():
    directory = _subdir(base, subdir_name)
    for source in sorted(directory.iterdir()):
      if not source.is_file():
        continue
      parsed = _parse_simulation_filename(source, suffix)
      if parsed is None:
        continue

      old_idx, label = parsed
      new_idx = normalized_mapping.get(old_idx)
      if new_idx is None:
        continue

      found_old_idxs.add(old_idx)
      if old_idx == new_idx:
        continue

      target = source.with_name(f"simulation_{new_idx}_{label}{suffix}")
      plans.append(RenamePlan(source=source, target=target, old_idx=old_idx, new_idx=new_idx))

  _validate_plan(plans)
  missing_old_idxs = sorted(
    set(normalized_mapping) - found_old_idxs,
    key=lambda item: int(item),
  )
  return RenameResult(plans=plans, missing_old_idxs=missing_old_idxs)


def _validate_plan(plans: list[RenamePlan]) -> None:
  sources = {plan.source for plan in plans}
  targets: dict[Path, Path] = {}
  for plan in plans:
    previous_source = targets.get(plan.target)
    if previous_source is not None:
      raise ValueError(
        f"Multiple files would be renamed to {plan.target}: {previous_source} and {plan.source}"
      )
    targets[plan.target] = plan.source

  for plan in plans:
    if plan.target.exists() and plan.target not in sources:
      raise ValueError(f"Refusing to overwrite existing file: {plan.target}")


def rename_simulation_idx_files(
  input_dir: str | Path,
  idx_mapping: Mapping[str, str],
  *,
  dry_run: bool = False,
  require_all: bool = False,
) -> RenameResult:
  result = build_rename_plan(input_dir, idx_mapping)
  if require_all and result.missing_old_idxs:
    missing = ", ".join(result.missing_old_idxs)
    raise ValueError(f"Mapping contains old idxs that were not found in filenames: {missing}")
  if dry_run or not result.plans:
    return result

  staged: list[tuple[RenamePlan, Path]] = []
  finalized: list[RenamePlan] = []
  try:
    for plan in result.plans:
      temp = _temporary_path(plan.source)
      plan.source.rename(temp)
      staged.append((plan, temp))

    for plan, temp in staged:
      temp.rename(plan.target)
      finalized.append(plan)
  except Exception:
    _rollback_renames(staged, finalized)
    raise

  return result


def _temporary_path(source: Path) -> Path:
  while True:
    candidate = source.with_name(f".{source.name}.idx-rename-{uuid.uuid4().hex}")
    if not candidate.exists():
      return candidate


def _rollback_renames(staged: list[tuple[RenamePlan, Path]], finalized: list[RenamePlan]) -> None:
  for plan in reversed(finalized):
    if plan.target.exists() and not plan.source.exists():
      plan.target.rename(plan.source)

  for plan, temp in reversed(staged):
    if temp.exists() and not plan.source.exists():
      temp.rename(plan.source)


def _print_result(result: RenameResult, *, dry_run: bool) -> None:
  verb = "Would rename" if dry_run else "Renamed"
  for plan in result.plans:
    print(f"{verb}: {plan.source} -> {plan.target}")

  print(f"{verb} files: {len(result.plans)}", file=sys.stderr)
  if result.missing_old_idxs:
    print(
      "Mapping old idxs not found in filenames: " + ", ".join(result.missing_old_idxs),
      file=sys.stderr,
    )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Rename simulation output files under json/ and transcripts/ using an old-idx to new-idx JSON mapping."
  )
  parser.add_argument(
    "--idx-mapping",
    required=True,
    help="JSON object mapping old idxs to new idxs, for example {\"0\": \"14296\"}.",
  )
  parser.add_argument(
    "--input-dir",
    required=True,
    help="Directory containing json/ and transcripts/ subdirectories.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the planned renames without changing files.",
  )
  parser.add_argument(
    "--require-all",
    action="store_true",
    help="Exit with an error if any old idx from the mapping is not found in either subdirectory.",
  )
  parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress normal rename output.",
  )
  args = parser.parse_args()

  try:
    idx_mapping = load_idx_mapping(args.idx_mapping)
    result = rename_simulation_idx_files(
      args.input_dir,
      idx_mapping,
      dry_run=args.dry_run,
      require_all=args.require_all,
    )
  except ValueError as exc:
    parser.exit(1, f"error: {exc}\n")

  if not args.quiet:
    _print_result(result, dry_run=args.dry_run)


if __name__ == "__main__":
  main()
