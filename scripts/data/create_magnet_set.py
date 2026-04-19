#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class MagnetSetError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatientSeed:
    name: str
    name_key: str
    attitude: str


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_key(value: Any) -> str:
    return normalize_text(value).casefold()


def normalize_attitude(attitude: Any) -> str:
    return normalize_key(attitude)


def extract_line(key: str, full_text: Any) -> str:
    pattern = rf"{re.escape(key)}[:\n\s]*(.*?)(?=\n[A-Z]|\n\n|$)"
    match = re.search(pattern, str(full_text or ""), re.IGNORECASE | re.DOTALL)
    return normalize_text(match.group(1)) if match else ""


def extract_profile_name(intake_form: Any) -> str:
    return extract_line("Name", intake_form)


def extract_section(header: str, full_text: Any) -> str:
    pattern = rf"{re.escape(header)}\s*:?\s*(.*?)(?=\n\n\d\.|\n\n[A-Z]|$)"
    match = re.search(pattern, str(full_text or ""), re.IGNORECASE | re.DOTALL)
    return match.group(1).lstrip(": ").strip() if match else ""


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise MagnetSetError(f"File does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MagnetSetError(f"File is not valid JSON: {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def normalize_role(role: Any) -> str:
    normalized = normalize_key(role)
    if normalized.startswith("counselor") or normalized.startswith("therapist"):
        return "therapist"
    if normalized.startswith("client") or normalized.startswith("patient"):
        return "patient"
    raise MagnetSetError(f"Unsupported dialogue role: {role!r}")


def parse_dialogue_list(dialogue: Any, *, source: Path) -> list[dict[str, str]]:
    if not isinstance(dialogue, list):
        raise MagnetSetError(f"Session dialogue must be a list: {source}")

    messages = []
    for index, item in enumerate(dialogue):
        if not isinstance(item, dict):
            raise MagnetSetError(f"Invalid dialogue item at {source}:{index}")
        if "role" not in item:
            raise MagnetSetError(f"Dialogue item is missing role at {source}:{index}")
        content = item.get("message", item.get("content"))
        if content is None:
            raise MagnetSetError(f"Dialogue item is missing message/content at {source}:{index}")
        messages.append(
            {
                "role": normalize_role(item["role"]),
                "content": str(content).strip(),
            }
        )

    if not messages:
        raise MagnetSetError(f"Session dialogue is empty: {source}")
    return messages


def parse_dialogue_text(dialogue: Any, *, source: Path) -> list[dict[str, str]]:
    if not isinstance(dialogue, str) or not dialogue.strip():
        raise MagnetSetError(f"Session dialogue text is empty or missing: {source}")

    messages = []
    for line_number, line in enumerate(dialogue.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        role, separator, content = stripped.partition(":")
        if not separator:
            raise MagnetSetError(
                f"Dialogue line {line_number} is missing a role prefix in: {source}"
            )
        messages.append({"role": normalize_role(role), "content": content.strip()})

    if not messages:
        raise MagnetSetError(f"Session dialogue text has no messages: {source}")
    return messages


def build_patient_profile(payload: dict[str, Any], *, idx: int, source: Path) -> dict[str, Any]:
    ai_client = payload.get("AI_client")
    if not isinstance(ai_client, dict):
        raise MagnetSetError(f"Session is missing AI_client: {source}")

    intake_form = ai_client.get("intake_form", "")
    attitude = normalize_attitude(ai_client.get("attitude"))
    if not attitude:
        raise MagnetSetError(f"Session is missing AI_client.attitude: {source}")

    return {
        "idx": idx,
        "personal_info": {
            "name": extract_line("Name", intake_form),
            "age": extract_line("Age", intake_form),
            "gender": extract_line("Gender", intake_form),
            "occupation": extract_line("Occupation", intake_form),
            "education": extract_line("Education", intake_form),
            "marital_status": extract_line("Marital Status", intake_form),
            "family_details": extract_line("Family Details", intake_form),
        },
        "presenting_problem": extract_section("2. Presenting Problem", intake_form),
        "reason_seeking_counseling": extract_section("3. Reason for Seeking Counseling", intake_form),
        "patterns": list(payload.get("patterns") or []),
        "attitude": attitude,
        "emotional_recovery": "moderate",
        "more_information": {
            "past_history": extract_section("4. Past History", intake_form),
            "academic_occupational_functioning": extract_section(
                "5. Academic/occupational functioning level",
                intake_form,
            ),
            "social_support_system": extract_section("6. Social Support System", intake_form),
        },
    }


def build_evaluation_session(payload: dict[str, Any], *, idx: int, source: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise MagnetSetError(f"Session file must contain a JSON object: {source}")

    if isinstance(payload.get("dialogue"), list):
        messages = parse_dialogue_list(payload["dialogue"], source=source)
    else:
        ai_client = payload.get("AI_client") if isinstance(payload.get("AI_client"), dict) else {}
        dialogue_text = ai_client.get("dialogue_history") or ai_client.get("dialogue_history_init")
        messages = parse_dialogue_text(dialogue_text, source=source)

    return {
        "messages": messages,
        "patient_profile": build_patient_profile(payload, idx=idx, source=source),
    }


def iter_seed_entries(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from iter_seed_entries(item)
        return

    if not isinstance(payload, dict):
        return

    if isinstance(payload.get("AI_client"), dict):
        yield payload
        return

    example = payload.get("example")
    if isinstance(example, dict):
        yield from iter_seed_entries(example)
        return

    if isinstance(payload.get("personal_info"), dict):
        yield payload


def seed_from_entry(entry: dict[str, Any]) -> PatientSeed | None:
    ai_client = entry.get("AI_client")
    if isinstance(ai_client, dict):
        name = extract_profile_name(ai_client.get("intake_form"))
        if not name:
            return None
        return PatientSeed(
            name=name,
            name_key=normalize_key(name),
            attitude=normalize_attitude(ai_client.get("attitude")),
        )

    personal_info = entry.get("personal_info")
    if isinstance(personal_info, dict):
        name = normalize_text(personal_info.get("name"))
        if not name:
            return None
        return PatientSeed(
            name=name,
            name_key=normalize_key(name),
            attitude=normalize_attitude(entry.get("attitude")),
        )

    return None


def extract_patient_seeds(payload: Any) -> list[PatientSeed]:
    seeds = []
    seen = set()

    for entry in iter_seed_entries(payload):
        seed = seed_from_entry(entry)
        if seed is None:
            continue

        dedupe_key = (seed.name_key, seed.attitude)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        seeds.append(seed)

    return seeds


def make_seed_key(seed: PatientSeed, *, match_attitude: bool) -> tuple[str, ...]:
    if match_attitude:
        return (seed.name_key, seed.attitude)
    return (seed.name_key,)


def load_reference_keys(seed_file: Path, *, match_attitude: bool) -> set[tuple[str, ...]]:
    payload = load_json(seed_file)
    seeds = extract_patient_seeds(payload)

    if not seeds:
        raise MagnetSetError(f"No patient seeds found in reference JSON: {seed_file}")

    keys = {
        make_seed_key(seed, match_attitude=match_attitude)
        for seed in seeds
        if seed.name_key and (seed.attitude or not match_attitude)
    }

    if not keys:
        raise MagnetSetError(
            f"No usable patient seed keys found in reference JSON: {seed_file}"
        )

    return keys


def list_input_files(input_dir: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def filter_magnet_sessions(
    input_dir: Path,
    output_dir: Path,
    seed_file: Path,
    *,
    match_attitude: bool = False,
    recursive: bool = False,
) -> dict[str, int]:
    if not input_dir.is_dir():
        raise MagnetSetError(f"Input directory does not exist or is not a directory: {input_dir}")

    input_root = input_dir.resolve()
    output_root = output_dir.resolve()
    if input_root == output_root:
        raise MagnetSetError("--output-dir must be different from --input-dir.")

    reference_keys = load_reference_keys(seed_file, match_attitude=match_attitude)

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    without_seed = 0

    for idx, input_file in enumerate(list_input_files(input_dir, recursive=recursive)):
        resolved_input = input_file.resolve()
        if is_relative_to(resolved_input, output_root):
            continue

        payload = load_json(input_file)
        seeds = extract_patient_seeds(payload)
        if not seeds:
            without_seed += 1
            skipped += 1
            continue

        has_match = any(
            make_seed_key(seed, match_attitude=match_attitude) in reference_keys
            for seed in seeds
        )
        if not has_match:
            skipped += 1
            continue

        relative_path = input_file.relative_to(input_dir) if recursive else Path(input_file.name)
        output_file = output_dir / relative_path
        write_json(output_file, build_evaluation_session(payload, idx=idx, source=input_file))
        written += 1

    return {
        "reference_seeds": len(reference_keys),
        "scanned_files": written + skipped,
        "written_files": written,
        "skipped_files": skipped,
        "files_without_seed": without_seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter MAGNET session JSON files by patient seeds from a CACTUS-style "
            "reference JSON file and write batch-metrics-compatible outputs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing MAGNET session JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where matching sessions will be written in evaluation format.",
    )
    parser.add_argument(
        "--seed-file",
        "--reference-json",
        dest="seed_file",
        type=Path,
        required=True,
        help="CACTUS-style JSON file whose patient seeds should be kept.",
    )
    parser.add_argument(
        "--match-attitude",
        action="store_true",
        help="Require both patient name and attitude to match. Default matches by patient name only.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan --input-dir recursively and preserve relative paths in --output-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        summary = filter_magnet_sessions(
            args.input_dir,
            args.output_dir,
            args.seed_file,
            match_attitude=args.match_attitude,
            recursive=args.recursive,
        )
    except MagnetSetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    key_label = "name+attitude pairs" if args.match_attitude else "patient names"
    print(
        f"Loaded {summary['reference_seeds']} reference {key_label}. "
        f"Wrote {summary['written_files']} of {summary['scanned_files']} JSON files "
        f"to '{args.output_dir}'."
    )
    if summary["files_without_seed"]:
        print(f"Skipped {summary['files_without_seed']} files without patient seed data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
