#!/usr/bin/env python3
"""Identify and rename synthetic session JSON files within set directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


METHODS = {"rcbt", "cactus", "magnet"}


class ValidationError(Exception):
    """Raised when a set directory cannot be safely processed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify session JSONs in each immediate set subdirectory and rename "
            "them to session_<n>_<method>.json. Defaults to dry-run."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory whose immediate subdirectories are set folders to process.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only print the planned changes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Write a JSON mapping from original session name to renamed session name. "
            "Ambiguous sessions are written with empty values."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"{path}: cannot read file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValidationError(f"{path}: expected a top-level JSON object")
    if "conversation" not in data:
        raise ValidationError(f"{path}: missing top-level 'conversation' field")
    if not isinstance(data["conversation"], list):
        raise ValidationError(f"{path}: expected 'conversation' to be a list")
    return data


def extract_session_number(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) != 3 or parts[0] != "session" or not parts[1]:
        raise ValidationError(
            f"{path}: expected filename format session_<n>_<suffix>.json"
        )
    return parts[1]


def identify_set(set_dir: Path) -> list[tuple[Path, Path, str]]:
    json_files = sorted(set_dir.glob("*.json"))
    if len(json_files) != 3:
        raise ValidationError(
            f"{set_dir}: expected exactly 3 JSON files, found {len(json_files)}"
        )

    session_numbers = {extract_session_number(path) for path in json_files}
    if len(session_numbers) != 1:
        numbers = ", ".join(sorted(session_numbers))
        raise ValidationError(f"{set_dir}: expected one shared session number, found {numbers}")
    session_number = next(iter(session_numbers))

    records = [(path, load_json(path)) for path in json_files]
    rcbt_records = [(path, data) for path, data in records if "reasonings" in data]
    if len(rcbt_records) != 1:
        raise ValidationError(
            f"{set_dir}: expected exactly one JSON with top-level 'reasonings', "
            f"found {len(rcbt_records)}"
        )

    assignments: dict[Path, str] = {rcbt_records[0][0]: "rcbt"}
    remaining = [(path, data) for path, data in records if path not in assignments]
    remaining_lengths = [(path, len(data["conversation"])) for path, data in remaining]
    remaining_lengths.sort(key=lambda item: item[1])

    if remaining_lengths[0][1] == remaining_lengths[1][1]:
        length = remaining_lengths[0][1]
        tied_paths = [path for path, _message_count in remaining_lengths]
        tied_names = ", ".join(path.name for path in tied_paths)
        print(
            f"warning: {set_dir}: cannot distinguish cactus and magnet; "
            f"{tied_names} both have {length} messages. Keeping them as-is.",
            file=sys.stderr,
        )
        for path in tied_paths:
            assignments[path] = ""
    else:
        assignments[remaining_lengths[0][0]] = "cactus"
        assignments[remaining_lengths[1][0]] = "magnet"

    plans: list[tuple[Path, Path, str]] = []
    for source, method in sorted(assignments.items()):
        if method:
            target = source.with_name(f"session_{session_number}_{method}.json")
            if target.exists() and target != source:
                raise ValidationError(f"{set_dir}: target already exists: {target}")
        else:
            target = source
        plans.append((source, target, method))
    return plans


def build_mapping(plans: list[tuple[Path, Path, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source, target, method in plans:
        key = source.stem
        if key in mapping:
            raise ValidationError(f"duplicate session name in output mapping: {key}")
        mapping[key] = target.stem if method else ""
    return mapping


def write_mapping(path: Path, mapping: dict[str, str]) -> None:
    if path.exists() and path.is_dir():
        raise ValidationError(f"{path}: output path is a directory")
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(mapping, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise ValidationError(f"{path}: cannot write output JSON: {exc}") from exc


def process_directory(root: Path, apply: bool, out: Path | None) -> None:
    if not root.exists():
        raise ValidationError(f"{root}: directory does not exist")
    if not root.is_dir():
        raise ValidationError(f"{root}: expected a directory")

    set_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not set_dirs:
        raise ValidationError(f"{root}: found no immediate set subdirectories")

    all_plans: list[tuple[Path, Path, str]] = []
    for set_dir in set_dirs:
        all_plans.extend(identify_set(set_dir))

    mapping = build_mapping(all_plans)

    mode = "APPLY" if apply else "DRY RUN"
    print(f"{mode}: {len(set_dirs)} set directories, {len(all_plans)} files")
    for source, target, method in all_plans:
        if method:
            action = "unchanged" if source == target else "rename"
            print(f"{action}: {source} -> {target} ({method})")
        else:
            print(f"ambiguous: {source} ->  (kept as-is)")

    if out is not None:
        write_mapping(out, mapping)
        print(f"mapping: {out}")

    if apply:
        for source, target, _method in all_plans:
            if source != target:
                source.rename(target)


def main() -> int:
    args = parse_args()
    try:
        process_directory(args.directory, args.apply, args.out)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
