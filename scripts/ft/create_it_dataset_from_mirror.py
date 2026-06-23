from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.ft.create_it_dataset_from_raw import (  # noqa: E402
  DatasetBuildError,
  ensure_str,
  require_dict,
  require_list,
)


def project_root() -> Path:
  return Path(__file__).resolve().parents[2]


def load_json_object(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{path}: failed to read file: {exc}") from exc
  return require_dict(payload, f"{path}: root")


def normalize_role(role: Any, source_label: str) -> str:
  normalized = ensure_str(role).strip().lower()
  if normalized in {"therapist", "counselor", "assistant"}:
    return "therapist"
  if normalized in {"patient", "client", "user"}:
    return "patient"
  raise DatasetBuildError(f"{source_label}: unsupported message role: {normalized!r}.")


def normalize_message(raw_message: Any, source_label: str, position: int) -> dict[str, Any]:
  message = require_dict(raw_message, source_label)
  content = ensure_str(message.get("content")).strip()
  if not content:
    raise DatasetBuildError(f"{source_label}: message content must be non-empty.")
  return {
    "role": normalize_role(message.get("role"), source_label),
    "content": content,
    "utterance_index": position,
  }


def normalize_key(value: Any) -> str:
  return ensure_str(value).strip().lower()


def patient_name_age(profile: dict[str, Any]) -> tuple[str, str]:
  personal_info = profile.get("personal_info")
  if not isinstance(personal_info, dict):
    raise DatasetBuildError("patient_profile.personal_info must be a JSON object.")
  name = ensure_str(personal_info.get("name")).strip()
  age = ensure_str(personal_info.get("age")).strip()
  if not name or not age:
    raise DatasetBuildError("patient_profile.personal_info must include non-empty name and age.")
  return name, age


def plan_lookup_key(profile: dict[str, Any]) -> tuple[str, str]:
  name, age = patient_name_age(profile)
  return normalize_key(name), normalize_key(age)


def format_personal_information(profile: dict[str, Any]) -> str:
  personal_info = require_dict(profile.get("personal_info"), "patient_profile.personal_info")
  lines = [
    f"{str(key).replace('_', ' ').title()}: {value}"
    for key, value in personal_info.items()
    if ensure_str(value).strip()
  ]
  return "\n".join(lines)


def load_llava_system_prompt(root: Path) -> str:
  prompt_path = root / "ft_prompts" / "llava.txt"
  if not prompt_path.is_file():
    raise DatasetBuildError(f"Missing LLaVA fine-tuning system prompt: {prompt_path}")
  prompt = prompt_path.read_text(encoding="utf-8").strip()
  if not prompt:
    raise DatasetBuildError(f"{prompt_path}: prompt must be non-empty.")
  return prompt


def load_cbt_plans(path: Path) -> dict[tuple[Any, ...], str]:
  payload = load_json_object_or_list(path)
  records = require_list(payload, f"{path}: root")
  plans: dict[tuple[Any, ...], str] = {}
  for idx, raw_record in enumerate(records, start=1):
    record = require_dict(raw_record, f"{path}: records[{idx}]")
    profile = {"personal_info": record.get("personal_info")}
    plan = ensure_str(record.get("cbt_plan")).strip()
    if not plan:
      continue
    if "idx" in record:
      plans[("idx", normalize_key(record.get("idx")))] = plan
    plans.setdefault(("profile", *plan_lookup_key(profile)), plan)
  if not plans:
    raise DatasetBuildError(f"{path}: no non-empty cbt_plan entries found.")
  return plans


def cbt_plan_for_profile(profile: dict[str, Any], cbt_plans: dict[tuple[Any, ...], str]) -> str:
  if "idx" in profile:
    plan = cbt_plans.get(("idx", normalize_key(profile.get("idx"))))
    if plan:
      return plan
  return cbt_plans.get(("profile", *plan_lookup_key(profile)), "")


def load_json_object_or_list(path: Path) -> Any:
  try:
    with path.open("r", encoding="utf-8") as f:
      return json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{path}: failed to read file: {exc}") from exc


def prefixed_history_message(message: dict[str, Any]) -> str:
  prefix = "Psychotherapist" if message["role"] == "therapist" else "Client"
  return f"{prefix}: {message['content']}"


def turn_output_name(path: Path, utterance_index: int) -> str:
  return f"{path.stem}.turn_{utterance_index:04d}{path.suffix}"


def build_turn_sessions(
  path: Path,
  system_prompt: str,
  cbt_plans: dict[tuple[Any, ...], str],
) -> list[tuple[str, dict[str, list[dict[str, str]]]]]:
  payload = load_json_object(path)
  patient_profile = require_dict(payload.get("patient_profile"), f"{path}: patient_profile")
  raw_messages = require_list(payload.get("messages"), f"{path}: messages")
  messages = [
    normalize_message(raw_message, f"{path}: messages[{idx}]", idx)
    for idx, raw_message in enumerate(raw_messages, start=1)
  ]

  system_message = {"role": "system", "content": system_prompt}
  personal_information = format_personal_information(patient_profile)
  reason = ensure_str(patient_profile.get("reason_seeking_counseling")).strip()
  cbt_plan = cbt_plan_for_profile(patient_profile, cbt_plans)
  if not cbt_plan:
    name, age = patient_name_age(patient_profile)
    raise DatasetBuildError(f"{path}: missing cbt_plan for patient {name!r}, age {age!r}.")
  rendered: list[tuple[str, dict[str, list[dict[str, str]]]]] = []

  for target_position, message in enumerate(messages):
    if message["role"] != "therapist":
      continue
    history = "\n".join(prefixed_history_message(context) for context in messages[:target_position])
    output_messages = [
      dict(system_message),
      {
        "role": "user",
        "content": (
          f"- Personal information: {personal_information}\n"
          f"- Reason for seeking counseling: {reason}\n"
          f"- CBT Technique: {cbt_plan}\n"
          "Below is a conversation between the client and the psychotherapist.\n"
          f"{history}\n\n"
          "Based on the conversation so far, respond as a psychotherapist conducting a CBT session."
        ),
      },
      {"role": "assistant", "content": message["content"]},
    ]
    rendered.append((turn_output_name(path, message["utterance_index"]), {"messages": output_messages}))
  return rendered


def build_dataset(
  input_dir: Path,
  output_dir: Path,
  root: Path | None = None,
  *,
  cactus_plan_path: Path | None = None,
) -> int:
  if not input_dir.is_dir():
    raise DatasetBuildError(f"--input must be a directory: {input_dir}")

  root = root or project_root()
  system_prompt = load_llava_system_prompt(root)
  cbt_plans = load_cbt_plans(cactus_plan_path or root / "data" / "processed_cactus_plan.json")
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")

  rendered_sessions: list[tuple[str, dict[str, list[dict[str, str]]]]] = []
  for input_file in input_files:
    rendered_sessions.extend(build_turn_sessions(input_file, system_prompt, cbt_plans))

  output_dir.mkdir(parents=True, exist_ok=True)
  for file_name, session in rendered_sessions:
    output_path = output_dir / file_name
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
      json.dump(session, f, ensure_ascii=False, indent=2)
      f.write("\n")

  return len(rendered_sessions)


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create one instruction-tuning ChatML file per therapist turn from mirror JSON sessions."
  )
  parser.add_argument("--input", required=True, help="Directory containing mirror JSON files.")
  parser.add_argument("--output", required=True, help="Directory to write one JSON file per therapist turn.")
  parser.add_argument(
    "--cactus-plan",
    default="data/processed_cactus_plan.json",
    help="Processed Cactus JSON file containing cbt_plan, matched to mirror patients by name and age.",
  )
  args = parser.parse_args()

  try:
    count = build_dataset(Path(args.input), Path(args.output), cactus_plan_path=Path(args.cactus_plan))
  except DatasetBuildError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {count} ChatML file(s) to {args.output}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
