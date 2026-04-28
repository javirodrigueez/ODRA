#!/usr/bin/env python3
"""Report conversation message counts for session JSONs in set directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """Raised when input data cannot be safely counted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count messages in the top-level 'conversation' list for each JSON file "
            "inside each immediate set subdirectory of the given directory."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory whose immediate subdirectories are set folders to inspect.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format. Default: table.",
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


def count_messages(root: Path) -> list[dict[str, str | int]]:
    if not root.exists():
        raise ValidationError(f"{root}: directory does not exist")
    if not root.is_dir():
        raise ValidationError(f"{root}: expected a directory")

    set_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not set_dirs:
        raise ValidationError(f"{root}: found no immediate set subdirectories")

    rows: list[dict[str, str | int]] = []
    for set_dir in set_dirs:
        json_files = sorted(set_dir.glob("*.json"))
        for json_file in json_files:
            data = load_json(json_file)
            rows.append(
                {
                    "set": set_dir.name,
                    "session": json_file.stem,
                    "file": str(json_file),
                    "messages": len(data["conversation"]),
                }
            )
    return rows


def print_table(rows: list[dict[str, str | int]]) -> None:
    columns = ("set", "session", "messages", "file")
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def print_csv(rows: list[dict[str, str | int]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=("set", "session", "messages", "file"))
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        rows = count_messages(args.directory)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.format == "csv":
        print_csv(rows)
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
