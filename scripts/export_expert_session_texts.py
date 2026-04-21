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


METHOD_FIELDS = {
    "cactus": "history",
    "magnet": "dialogue",
    "rcbt": "messages",
}

PROFILE_FIELD_ORDER = (
    "attitude",
    "personal_info",
    "presenting_problem",
    "reason_seeking_counseling",
    "more_information",
)

EXCLUDED_PROFILE_FIELDS = {
    "emotional_recovery",
    "idx",
    "patterns",
}


class SessionTextExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConversationVariant:
    method: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class SessionSet:
    file_name: str
    stem: str
    set_dir_name: str
    variants: list[ConversationVariant]
    patient_profile: dict[str, Any]


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise SessionTextExportError(f"File does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SessionTextExportError(f"File is not valid JSON: {path}: {exc}") from exc


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def normalize_role(role: Any) -> str | None:
    normalized = str(role or "").strip().casefold()
    if normalized in {"therapist", "counselor"}:
        return "therapist"
    if normalized in {"patient", "client"}:
        return "patient"
    return None


def parse_messages(payload: Any, *, field: str, source: Path) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise SessionTextExportError(f"Session file must contain a JSON object: {source}")

    raw_messages = payload.get(field)
    if not isinstance(raw_messages, list):
        raise SessionTextExportError(f"{source} is missing list field {field!r}.")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise SessionTextExportError(f"{source}:{field}[{index}] must be an object.")
        if "role" not in item:
            raise SessionTextExportError(f"{source}:{field}[{index}] is missing role.")

        role = normalize_role(item["role"])
        if role is None:
            continue

        content = item.get("content")
        if content is None:
            content = item.get("message")
        if content is None:
            raise SessionTextExportError(
                f"{source}:{field}[{index}] is missing content/message."
            )

        content_text = normalize_text(content)
        if not content_text:
            raise SessionTextExportError(f"{source}:{field}[{index}] has empty content.")

        messages.append({"role": role, "content": content_text})

    if not messages:
        raise SessionTextExportError(f"{source}:{field} has no therapist/patient messages.")
    return messages


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def session_id_from_stem(stem: str) -> str:
    match = re.fullmatch(r"session_(.+)", stem)
    if match:
        return match.group(1)
    return stem


def method_json_files(input_dir: Path) -> dict[str, dict[str, Path]]:
    files_by_method: dict[str, dict[str, Path]] = {}
    for method in METHOD_FIELDS:
        method_dir = input_dir / method
        if not method_dir.is_dir():
            raise SessionTextExportError(f"Input directory is missing subdir: {method_dir}")

        files = {path.name: path for path in method_dir.glob("*.json") if path.is_file()}
        if not files:
            raise SessionTextExportError(f"Input subdir has no JSON files: {method_dir}")
        files_by_method[method] = files

    expected_names = set(files_by_method["cactus"])
    errors: list[str] = []
    for method, files in files_by_method.items():
        names = set(files)
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        if missing:
            errors.append(f"{method} is missing: {', '.join(missing)}")
        if extra:
            errors.append(f"{method} has extra files: {', '.join(extra)}")

    if errors:
        raise SessionTextExportError(
            "Method directories must contain matching JSON file names.\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    return files_by_method


def build_session_sets(input_dir: Path) -> list[SessionSet]:
    files_by_method = method_json_files(input_dir)
    session_sets: list[SessionSet] = []

    reference_files = sorted(files_by_method["cactus"].values(), key=natural_sort_key)
    for reference_file in reference_files:
        file_name = reference_file.name
        variants: list[ConversationVariant] = []
        rcbt_payload: dict[str, Any] | None = None

        for method, field in METHOD_FIELDS.items():
            source = files_by_method[method][file_name]
            payload = load_json(source)
            if method == "rcbt":
                if not isinstance(payload, dict):
                    raise SessionTextExportError(f"Session file must contain an object: {source}")
                rcbt_payload = payload

            variants.append(
                ConversationVariant(
                    method=method,
                    messages=parse_messages(payload, field=field, source=source),
                )
            )

        if rcbt_payload is None:
            raise SessionTextExportError(f"Missing rcbt payload for {file_name}")

        patient_profile = rcbt_payload.get("patient_profile")
        if not isinstance(patient_profile, dict):
            raise SessionTextExportError(
                f"{files_by_method['rcbt'][file_name]} is missing object field 'patient_profile'."
            )

        stem = reference_file.stem
        session_sets.append(
            SessionSet(
                file_name=file_name,
                stem=stem,
                set_dir_name=f"set_{session_id_from_stem(stem)}",
                variants=variants,
                patient_profile=patient_profile,
            )
        )

    return session_sets


def title_label(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def section_title(value: str) -> str:
    return title_label(value).upper()


def format_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip() or "None"


def format_key_value(key: str, value: Any) -> list[str]:
    label = title_label(key)
    if isinstance(value, dict):
        return format_dict_section(label, value)
    if isinstance(value, list):
        if not value:
            return [f"{label}: None"]
        return [f"{label}:", *[f"- {format_scalar(item)}" for item in value]]

    text = format_scalar(value)
    if "\n" in text:
        return [f"{label}:", text]
    return [f"{label}: {text}"]


def format_dict_section(title: str, values: dict[str, Any]) -> list[str]:
    lines = [section_title(title), "-" * len(section_title(title))]
    if not values:
        return [*lines, "None"]

    entries: list[list[str]] = []
    for key, value in values.items():
        if key in EXCLUDED_PROFILE_FIELDS:
            continue
        entries.append(format_key_value(key, value))

    if not entries:
        return [*lines, "None"]

    for index, entry in enumerate(entries):
        if index:
            lines.append("")
        lines.extend(entry)
    return lines


def format_compact_dict_section(title: str, values: dict[str, Any]) -> list[str]:
    lines = [section_title(title), "-" * len(section_title(title))]
    if not values:
        return [*lines, "None"]

    for key, value in values.items():
        if key in EXCLUDED_PROFILE_FIELDS:
            continue
        lines.extend(format_key_value(key, value))

    if len(lines) == 2:
        lines.append("None")
    return lines


def format_profile(profile: dict[str, Any]) -> str:
    lines = ["PATIENT PROFILE", "===============", ""]

    emitted = set(EXCLUDED_PROFILE_FIELDS)
    for key in PROFILE_FIELD_ORDER:
        emitted.add(key)
        if key not in profile or key in EXCLUDED_PROFILE_FIELDS:
            continue

        if key == "attitude":
            lines.append(f"ATTITUDE: {format_scalar(profile[key])}")
            lines.append("")
            continue

        if key == "personal_info" and isinstance(profile[key], dict):
            lines.extend(format_compact_dict_section(key, profile[key]))
        elif isinstance(profile[key], dict):
            lines.extend(format_dict_section(key, profile[key]))
        else:
            lines.extend(
                [
                    section_title(key),
                    "-" * len(section_title(key)),
                    format_scalar(profile[key]),
                ]
            )
        lines.append("")

    extra_fields = [
        (key, value)
        for key, value in profile.items()
        if key not in emitted and key not in EXCLUDED_PROFILE_FIELDS
    ]
    for index, (key, value) in enumerate(extra_fields):
        if index:
            lines.append("")
        if isinstance(value, dict):
            lines.extend(format_dict_section(key, value))
        else:
            lines.extend([section_title(key), "-" * len(section_title(key)), format_scalar(value)])

    return "\n".join(lines).rstrip() + "\n"


def format_conversation(messages: Iterable[dict[str, str]]) -> str:
    blocks = [
        f"> {message['role'].upper()}: {message['content']}"
        for message in messages
    ]
    return "\n\n".join(blocks).rstrip() + "\n"


def remove_previous_generated_files(set_dir: Path, stem: str) -> None:
    if not set_dir.exists():
        return
    if not set_dir.is_dir():
        raise SessionTextExportError(f"Output set path exists and is not a directory: {set_dir}")

    generated_pattern = re.compile(rf"{re.escape(stem)}_\d{{3}}\.txt\Z")
    for path in set_dir.iterdir():
        if path.name == "profile.txt" or generated_pattern.fullmatch(path.name):
            if path.is_file():
                path.unlink()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_session_set(output_dir: Path, session_set: SessionSet) -> None:
    set_dir = output_dir / session_set.set_dir_name
    remove_previous_generated_files(set_dir, session_set.stem)
    set_dir.mkdir(parents=True, exist_ok=True)

    variants = list(session_set.variants)
    random.shuffle(variants)
    random_ids = random.sample(range(101), len(variants))

    for variant, random_id in zip(variants, random_ids):
        path = set_dir / f"{session_set.stem}_{random_id:03d}.txt"
        write_text(path, format_conversation(variant.messages))

    write_text(set_dir / "profile.txt", format_profile(session_set.patient_profile))


def export_sessions(input_dir: Path, output_dir: Path) -> int:
    if not input_dir.is_dir():
        raise SessionTextExportError(f"Input path is not a directory: {input_dir}")

    session_sets = build_session_sets(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for session_set in session_sets:
        write_session_set(output_dir, session_set)
    return len(session_sets)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export matched cactus/magnet/rcbt JSON sessions as anonymized text variants."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing cactus, magnet, and rcbt JSON subdirectories.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where set_<n> folders will be written.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = export_sessions(args.input, args.output)
    except SessionTextExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Exported {count} session set(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
