from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.ft.create_reasoning_instruction_dataset import (  # noqa: E402
  DatasetBuildError,
  ensure_str,
  require_dict,
  require_list,
)


TraceKey = tuple[Any, str]
TraceFields = list[str] | None

STAGE_LABELS = {
  "opening": "Opening",
  "id_distortions": "Distortions Identification",
  "cbt_technique": "CBT Techniques",
  "homework": "Homework",
  "closing": "Closing",
  "anticipated_closing": "Closing",
  "forced_closing": "Closing",
  "address_tib": "Motivational Interviewing / Resistance Handling",
  "mi_techniques": "Motivational Interviewing / Resistance Handling",
  "meta_conversation": "Motivational Interviewing / Resistance Handling",
}

ROLE_MAP = {
  "patient": "user",
  "client": "user",
  "user": "user",
  "therapist": "assistant",
  "assistant": "assistant",
}

CUSTOM_TIB_MESSAGES = {
  "address_tib_enter": "As the therapy-interfering behavior {tib_name} was detected, I should address it",
  "address_tib_replaced": "As the therapy-interfering behavior {tib_name} was detected, I should address it",
  "address_tib_exit": "Previously identified therapy-interfering behavior {tib_name} was addressed",
  "address_tib_repeat_inline_suppressed": (
    "I already tried to address the therapy-interfering behavior {tib_name} in my last utterance. "
    "Now, I will address it again to demonstrate validation, while continuing with the main line of the session."
  ),
}

FIELD_ALIASES = {
  "previous_resistance": ("previous_resistance", "previous_p"),
  "predicted_resistance": ("predicted_resistance", "new_p"),
}


def project_root() -> Path:
  return Path(__file__).resolve().parents[2]


def load_therapist_system_prompt(root: Path) -> str:
  prompt_path = root / "agent_prompts" / "therapist_system_ft_2.txt"
  if not prompt_path.is_file():
    raise DatasetBuildError(f"Missing therapist fine-tuning system prompt: {prompt_path}")
  return prompt_path.read_text(encoding="utf-8")


def load_trace_field_config(root: Path) -> dict[str, TraceFields]:
  config_path = root / "data" / "knowledge" / "reasoning_trace_evaluation.json"
  if not config_path.is_file():
    raise DatasetBuildError(f"Missing reasoning trace field config: {config_path}")

  try:
    with config_path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{config_path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{config_path}: failed to read file: {exc}") from exc

  config = require_dict(payload, f"{config_path}: root")
  fields_by_step: dict[str, TraceFields] = {}

  def merge_fields(step: str, fields: Any) -> None:
    if not step:
      return
    if fields is None:
      fields_by_step[step] = None
      return
    raw_fields = require_list(fields, f"{config_path}: {step}.fields")
    normalized = [ensure_str(field).strip() for field in raw_fields if ensure_str(field).strip()]
    existing = fields_by_step.get(step)
    if existing is None and step in fields_by_step:
      return
    if existing is None:
      fields_by_step[step] = normalized
      return
    for field in normalized:
      if field not in existing:
        existing.append(field)

  stages = require_dict(config.get("stages"), f"{config_path}: stages")
  for stage_name, raw_stage in stages.items():
    stage = require_dict(raw_stage, f"{config_path}: stages.{stage_name}")
    steps = require_dict(stage.get("steps"), f"{config_path}: stages.{stage_name}.steps")
    for step_name, raw_step in steps.items():
      step_config = require_dict(raw_step, f"{config_path}: stages.{stage_name}.steps.{step_name}")
      merge_fields(ensure_str(step_name), step_config.get("fields"))

  global_steps = require_dict(config.get("global_steps"), f"{config_path}: global_steps")
  for step_name, raw_step in global_steps.items():
    step_config = require_dict(raw_step, f"{config_path}: global_steps.{step_name}")
    merge_fields(ensure_str(step_name), step_config.get("fields"))

  # User-requested dataset overrides.
  fields_by_step["resistance_prediction"] = [
    "previous_resistance",
    "predicted_resistance",
    "reasoning",
  ]
  fields_by_step["tib_check"] = ["has_tib", "tib_name", "reasoning"]
  fields_by_step["technique_plan"] = [
    "plan_description",
    "detailed_plan_description",
    "expected_outcome",
    "current_progress",
  ]

  for step in CUSTOM_TIB_MESSAGES:
    fields_by_step[step] = ["message"]

  return fields_by_step


def load_session(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{path}: failed to read file: {exc}") from exc
  return require_dict(payload, f"{path}: root")


def normalized_utterance_index(value: Any) -> Any:
  try:
    return int(value)
  except (TypeError, ValueError):
    return value


def trace_group_key(group: dict[str, Any], label: str) -> TraceKey:
  speaker = ensure_str(group.get("speaker")).strip().lower()
  if not speaker:
    raise DatasetBuildError(f"{label}: speaker must be non-empty.")
  if "utterance_index" not in group:
    raise DatasetBuildError(f"{label}: utterance_index is required.")
  return (normalized_utterance_index(group.get("utterance_index")), speaker)


def trace_groups_by_key(groups: list[Any], source_label: str) -> dict[TraceKey, list[Any]]:
  by_key: dict[TraceKey, list[Any]] = {}
  for idx, raw_group in enumerate(groups, start=1):
    group = require_dict(raw_group, f"{source_label}: reasoning_traces[{idx}]")
    key = trace_group_key(group, f"{source_label}: reasoning_traces[{idx}]")
    if key[1] != "therapist":
      continue
    if key in by_key:
      raise DatasetBuildError(
        f"{source_label}: duplicate reasoning trace group for utterance {key[0]} speaker {key[1]}."
      )
    by_key[key] = require_list(group.get("traces"), f"{source_label}: reasoning_traces[{idx}].traces")
  return by_key


def message_trace_key(message: dict[str, Any]) -> TraceKey:
  return (normalized_utterance_index(message.get("utterance_index")), "therapist")


def parse_trace_output(trace: dict[str, Any]) -> Any:
  output = trace.get("output")
  if isinstance(output, str):
    stripped = output.strip()
    if not stripped:
      return ""
    try:
      return json.loads(stripped)
    except json.JSONDecodeError:
      return stripped
  return output


def select_field(payload: dict[str, Any], field: str) -> tuple[bool, Any]:
  aliases = FIELD_ALIASES.get(field, (field,))
  for key in aliases:
    if key in payload:
      return True, payload[key]
  return False, None


def string_output_field(step: str, fields: TraceFields) -> str:
  if fields and len(fields) == 1:
    return fields[0]
  if step == "technique_evaluation":
    return "evaluation"
  return "message"


def tib_name_from_payload(payload: Any) -> str:
  if isinstance(payload, dict):
    tib_name = ensure_str(payload.get("tib_name")).strip()
    if tib_name:
      return tib_name
  return "unknown"


def selected_trace_payload(step: str, payload: Any, fields_by_step: dict[str, TraceFields]) -> Any:
  if step in CUSTOM_TIB_MESSAGES:
    return {"message": CUSTOM_TIB_MESSAGES[step].format(tib_name=tib_name_from_payload(payload))}

  fields = fields_by_step.get(step)
  if isinstance(payload, str):
    return {string_output_field(step, fields): payload}

  if not isinstance(payload, dict):
    if fields and len(fields) == 1:
      return {fields[0]: payload}
    return {"value": payload}

  if fields is None:
    return dict(payload)

  selected: dict[str, Any] = {}
  for field in fields:
    found, value = select_field(payload, field)
    if found:
      selected[field] = value

  if selected:
    return selected

  if len(fields) == 1:
    return {fields[0]: payload}
  return dict(payload)


def fenced_json(value: Any) -> str:
  return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def format_trace(trace: dict[str, Any], fields_by_step: dict[str, TraceFields]) -> str:
  step = ensure_str(trace.get("step")).strip() or "unknown"
  payload = parse_trace_output(trace)
  selected_payload = selected_trace_payload(step, payload, fields_by_step)
  return f"[{step}]\n{fenced_json(selected_payload)}"


def format_trace_block(traces: list[Any], source_label: str, fields_by_step: dict[str, TraceFields]) -> str:
  trace_lines: list[str] = []
  for idx, raw_trace in enumerate(traces, start=1):
    trace = require_dict(raw_trace, f"{source_label}: traces[{idx}]")
    trace_lines.append(format_trace(trace, fields_by_step))
  return "\n".join(trace_lines)


def human_stage_label(message: dict[str, Any]) -> str:
  stage = ensure_str(message.get("stage")).strip().lower()
  if not stage:
    return "Unknown"
  if stage in STAGE_LABELS:
    return STAGE_LABELS[stage]
  return stage.replace("_", " ").title()


def message_role(message: dict[str, Any]) -> str:
  return ensure_str(message.get("role")).strip().lower()


def message_content(message: dict[str, Any]) -> str:
  return ensure_str(message.get("content")).strip()


def chatml_role(message: dict[str, Any], source_label: str) -> str:
  role = message_role(message)
  mapped = ROLE_MAP.get(role)
  if mapped is None:
    raise DatasetBuildError(f"{source_label}: unsupported message role: {role!r}.")
  return mapped


def assistant_content(
  message: dict[str, Any],
  traces: list[Any],
  fields_by_step: dict[str, TraceFields],
  source_label: str,
) -> str:
  content = message_content(message)
  if not content:
    raise DatasetBuildError(f"{source_label}: therapist utterance must be non-empty.")
  if not traces:
    return content

  trace_text = format_trace_block(traces, source_label, fields_by_step)
  return f"<think>\n[stage] {human_stage_label(message)}\n{trace_text}\n</think>\n{content}"


def chatml_message(
  message: dict[str, Any],
  traces: list[Any],
  fields_by_step: dict[str, TraceFields],
  source_label: str,
) -> dict[str, str]:
  role = chatml_role(message, source_label)
  content = message_content(message)
  if not content:
    raise DatasetBuildError(f"{source_label}: message content must be non-empty.")
  if role == "assistant":
    content = assistant_content(message, traces, fields_by_step, source_label)
  return {"role": role, "content": content}


def build_chatml_session(
  path: Path,
  system_prompt: str,
  fields_by_step: dict[str, TraceFields],
) -> dict[str, list[dict[str, str]]]:
  payload = load_session(path)
  require_dict(payload.get("patient_profile"), f"{path}: patient_profile")
  messages = require_list(payload.get("messages"), f"{path}: messages")
  raw_trace_groups = require_list(payload.get("reasoning_traces"), f"{path}: reasoning_traces")
  trace_groups = trace_groups_by_key(raw_trace_groups, str(path))

  output_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
  first_therapist_message_seen = False
  for idx, raw_message in enumerate(messages, start=1):
    message = require_dict(raw_message, f"{path}: messages[{idx}]")
    if message_role(message) == "therapist" and not first_therapist_message_seen:
      first_therapist_message_seen = True
      content = message_content(message)
      if not content:
        raise DatasetBuildError(f"{path}: messages[{idx}]: message content must be non-empty.")
      output_messages.append({"role": "user", "content": content})
      continue

    traces = trace_groups.get(message_trace_key(message), []) if message_role(message) == "therapist" else []
    output_messages.append(
      chatml_message(
        message,
        traces,
        fields_by_step,
        f"{path}: messages[{idx}]",
      )
    )

  return {"messages": output_messages}


def build_dataset(input_dir: Path, output_dir: Path, root: Path | None = None) -> int:
  if not input_dir.is_dir():
    raise DatasetBuildError(f"--input-dir must be a directory: {input_dir}")

  root = root or project_root()
  system_prompt = load_therapist_system_prompt(root)
  fields_by_step = load_trace_field_config(root)
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")

  rendered_sessions = [
    (input_file.name, build_chatml_session(input_file, system_prompt, fields_by_step))
    for input_file in input_files
  ]

  output_dir.mkdir(parents=True, exist_ok=True)
  for file_name, session in rendered_sessions:
    output_path = output_dir / file_name
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
      json.dump(session, f, ensure_ascii=False, indent=2)
      f.write("\n")

  return len(rendered_sessions)


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create a ChatML-like session dataset from raw therapy simulation JSON files."
  )
  parser.add_argument("--input-dir", required=True, help="Directory containing raw simulation JSON files.")
  parser.add_argument("--output-dir", required=True, help="Directory to write one ChatML-like JSON file per session.")
  args = parser.parse_args()

  try:
    count = build_dataset(Path(args.input_dir), Path(args.output_dir))
  except DatasetBuildError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {count} ChatML session file(s) to {args.output_dir}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
