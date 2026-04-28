#!/usr/bin/env python3
"""Rename session references inside an evaluation JSON using a rename mapping."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SESSION_RE = re.compile(r"session_\d+_[A-Za-z0-9]+")


class ValidationError(Exception):
    """Raised when inputs cannot be safely rewritten."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename session references inside an eval JSON using a JSON mapping "
            "from old session names to new session names."
        )
    )
    parser.add_argument(
        "--in-renames",
        required=True,
        type=Path,
        help="JSON mapping from original session names to renamed session names.",
    )
    parser.add_argument(
        "--in-eval",
        required=True,
        type=Path,
        help="Evaluation JSON whose session references should be renamed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path. Defaults to overwriting --in-eval in place.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"{path}: cannot read file: {exc}") from exc


def load_renames(path: Path) -> dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: expected a JSON object")

    renames: dict[str, str] = {}
    for old, new in data.items():
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValidationError(f"{path}: rename keys and values must be strings")
        if not old:
            raise ValidationError(f"{path}: empty rename key is not allowed")
        if new:
            renames[old] = new
    return renames


def rename_session_refs(value: str, renames: dict[str, str]) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        old = match.group(0)
        new = renames.get(old)
        if new is None:
            return old
        count += 1
        return new

    return SESSION_RE.sub(replace, value), count


def rewrite_json(data: Any, renames: dict[str, str], path: str = "$") -> tuple[Any, int]:
    if isinstance(data, dict):
        rewritten: dict[str, Any] = {}
        total = 0
        for key, value in data.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path}: expected string object keys")
            new_key, key_count = rename_session_refs(key, renames)
            if new_key in rewritten:
                raise ValidationError(
                    f"{path}: duplicate key after renaming: {new_key!r}"
                )
            new_value, value_count = rewrite_json(value, renames, f"{path}.{new_key}")
            rewritten[new_key] = new_value
            total += key_count + value_count
        return rewritten, total

    if isinstance(data, list):
        rewritten_items = []
        total = 0
        for index, item in enumerate(data):
            new_item, count = rewrite_json(item, renames, f"{path}[{index}]")
            rewritten_items.append(new_item)
            total += count
        return rewritten_items, total

    if isinstance(data, str):
        return rename_session_refs(data, renames)

    return data, 0


def write_json(path: Path, data: Any) -> None:
    if path.exists() and path.is_dir():
        raise ValidationError(f"{path}: output path is a directory")
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ValidationError(f"{path}: cannot write output JSON: {exc}") from exc


def main() -> int:
    args = parse_args()
    out_path = args.out or args.in_eval

    try:
        renames = load_renames(args.in_renames)
        eval_data = load_json(args.in_eval)
        rewritten, count = rewrite_json(eval_data, renames)
        write_json(out_path, rewritten)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"renamed {count} session reference(s)")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
