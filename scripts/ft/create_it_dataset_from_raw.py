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
PATIENT_INTAKE_FORM_PLACEHOLDER = "{patient_intake_form}"
GENERIC_WRAPPER_KEYS = {"items", "output", "result", "data", "payload"}
CORE_BELIEF_PROBABILITY_STEP = "core_belief_probability"
DISTORTION_PROBABILITY_STEP = "distortion_probability"
HOMEWORK_TASK_PROPOSAL_STEP = "homework_task_proposal"
HOMEWORK_AGREEMENT_CHECK_STEP = "homework_agreement_check"
END_SESSION_MARK = "[/END]"

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
  prompt = prompt_path.read_text(encoding="utf-8")
  if PATIENT_INTAKE_FORM_PLACEHOLDER not in prompt:
    raise DatasetBuildError(
      f"{prompt_path}: missing required {PATIENT_INTAKE_FORM_PLACEHOLDER} placeholder."
    )
  return prompt


def format_patient_intake_form(profile: dict[str, Any]) -> str:
  personal_info = profile.get("personal_info", {})
  more_info = profile.get("more_information", {})

  sections: list[str] = []
  if isinstance(personal_info, dict) and personal_info:
    personal_lines = [
      f"{str(key).replace('_', ' ').title()}: {value}"
      for key, value in personal_info.items()
    ]
    sections.append("Personal Info:\n" + "\n".join(personal_lines))

  reason = profile.get("reason_seeking_counseling", "")
  if reason:
    sections.append(f"Reason Seeking Counseling:\n{reason}")

  if isinstance(more_info, dict) and more_info:
    more_info_lines = [
      f"{str(key).replace('_', ' ').title()}:\n{value}"
      for key, value in more_info.items()
      if value
    ]
    if more_info_lines:
      sections.append("More Information:\n" + "\n\n".join(more_info_lines))

  return "\n\n".join(sections)


def render_system_prompt(system_prompt: str, patient_profile: dict[str, Any]) -> str:
  intake = format_patient_intake_form(patient_profile)
  return system_prompt.replace(PATIENT_INTAKE_FORM_PLACEHOLDER, intake)


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
  fields_by_step[HOMEWORK_TASK_PROPOSAL_STEP] = ["tasks"]
  fields_by_step[HOMEWORK_AGREEMENT_CHECK_STEP] = ["agreement_reached", "agreed_tasks"]

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
  payload = unwrap_generic_wrapper(payload)

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


def unwrap_generic_wrapper(value: Any) -> Any:
  while isinstance(value, dict) and len(value) == 1:
    key = next(iter(value))
    if key not in GENERIC_WRAPPER_KEYS:
      break
    value = value[key]
  return value


def field_label(key: Any) -> str:
  label = ensure_str(key).replace("_", " ").strip().title()
  return label or "Value"


def path_label(path: list[Any]) -> str:
  labels = [field_label(key) for key in path if field_label(key)]
  return " ".join(labels) if labels else "Value"


def ensure_terminal_punctuation(text: str) -> str:
  stripped = text.strip()
  if not stripped or stripped.endswith((".", "!", "?")):
    return stripped
  return stripped + "."


def join_plain_sentences(parts: list[str]) -> str:
  rendered: list[str] = []
  for part in parts:
    stripped = part.strip()
    if not stripped:
      continue
    if rendered:
      rendered[-1] = ensure_terminal_punctuation(rendered[-1])
    rendered.append(stripped)
  return " ".join(rendered)


def format_plain_scalar(value: Any) -> str:
  if value is None:
    return "none"
  if isinstance(value, str):
    stripped = value.strip()
    return stripped if stripped else "none"
  if isinstance(value, bool):
    return "yes" if value else "no"
  if isinstance(value, (int, float)):
    return str(value)
  return str(value)


def format_numbered_item(value: Any) -> str:
  value = unwrap_generic_wrapper(value)
  if isinstance(value, dict):
    return format_plain_text(value)
  if isinstance(value, list):
    return format_plain_list(value)
  return format_plain_scalar(value)


def format_plain_list(values: list[Any]) -> str:
  if not values:
    return "none"
  has_nested_value = any(isinstance(unwrap_generic_wrapper(value), (dict, list)) for value in values)
  if has_nested_value:
    return " ".join(
      ensure_terminal_punctuation(f"{idx}. {format_numbered_item(value)}")
      for idx, value in enumerate(values, start=1)
    )
  return ", ".join(format_plain_scalar(value) for value in values)


def flatten_plain_fields(value: dict[str, Any], path: list[Any] | None = None) -> list[tuple[str, str]]:
  path = path or []
  value = unwrap_generic_wrapper(value)
  if not isinstance(value, dict):
    return [(path_label(path), format_plain_value(value))]
  if not value:
    return [(path_label(path), "none")]

  fields: list[tuple[str, str]] = []
  for key, raw_field_value in value.items():
    field_value = unwrap_generic_wrapper(raw_field_value)
    field_path = path + [key]
    if isinstance(field_value, dict):
      fields.extend(flatten_plain_fields(field_value, field_path))
    else:
      fields.append((path_label(field_path), format_plain_value(field_value)))
  return fields


def format_plain_value(value: Any) -> str:
  value = unwrap_generic_wrapper(value)
  if isinstance(value, dict):
    return format_plain_text(value)
  if isinstance(value, list):
    return format_plain_list(value)
  return format_plain_scalar(value)


def format_plain_text(value: Any) -> str:
  value = unwrap_generic_wrapper(value)
  if isinstance(value, dict):
    if not value:
      return "none"
    fields = flatten_plain_fields(value)
    return join_plain_sentences([f"{label}: {text}" for label, text in fields])
  return format_plain_value(value)


def sorted_probability_items(value: Any) -> list[tuple[str, float]]:
  value = unwrap_generic_wrapper(value)
  if not isinstance(value, dict):
    return []

  items: list[tuple[str, float]] = []
  for key, raw_probability in value.items():
    if isinstance(raw_probability, bool):
      continue
    try:
      probability = float(raw_probability)
    except (TypeError, ValueError):
      continue
    items.append((ensure_str(key), probability))
  return sorted(items, key=lambda item: item[1], reverse=True)


def format_concept_name(name: str) -> str:
  return name.replace("_", " ").strip() or "none"


def format_probability_trace(step: str, value: Any) -> str | None:
  probability_items = sorted_probability_items(value)
  if step == CORE_BELIEF_PROBABILITY_STEP:
    belief = format_concept_name(probability_items[0][0]) if probability_items else "none"
    return f"Most likely core belief associated is: {belief}"
  if step == DISTORTION_PROBABILITY_STEP:
    distortions = [format_concept_name(name) for name, _ in probability_items[:3]]
    distortion_text = ", ".join(distortions) if distortions else "none"
    return f"Most likely distortions are: {distortion_text}"
  return None


def homework_task_title(value: Any) -> str:
  value = unwrap_generic_wrapper(value)
  if isinstance(value, dict):
    title = ensure_str(value.get("task_title")).strip()
    if title:
      return title
    title = ensure_str(value.get("title")).strip()
    if title:
      return title
  return format_plain_scalar(value)


def format_homework_agreed_tasks(value: Any) -> str:
  value = unwrap_generic_wrapper(value)
  if not isinstance(value, list) or not value:
    return "none"
  return " ".join(
    ensure_terminal_punctuation(f"{idx}. {homework_task_title(task)}")
    for idx, task in enumerate(value, start=1)
  )


def format_homework_agreement_trace(value: Any) -> str | None:
  value = unwrap_generic_wrapper(value)
  if not isinstance(value, dict):
    return None

  agreement_found, agreement_reached = select_field(value, "agreement_reached")
  if not agreement_found:
    agreement_found, agreement_reached = select_field(value, "agreement")
  tasks_found, agreed_tasks = select_field(value, "agreed_tasks")

  parts: list[str] = []
  if agreement_found:
    parts.append(f"Agreement Reached: {format_plain_scalar(agreement_reached)}")
  if tasks_found:
    parts.append(f"Agreed Tasks: {format_homework_agreed_tasks(agreed_tasks)}")
  return join_plain_sentences(parts) if parts else None


def format_step_specific_trace(step: str, value: Any) -> str | None:
  trace_text = format_probability_trace(step, value)
  if trace_text is not None:
    return trace_text
  if step == HOMEWORK_AGREEMENT_CHECK_STEP:
    return format_homework_agreement_trace(value)
  return None


def format_trace(trace: dict[str, Any], fields_by_step: dict[str, TraceFields]) -> str:
  step = ensure_str(trace.get("step")).strip() or "unknown"
  payload = parse_trace_output(trace)
  selected_payload = selected_trace_payload(step, payload, fields_by_step)
  trace_text = format_step_specific_trace(step, selected_payload)
  if trace_text is None:
    trace_text = format_plain_text(selected_payload)
  return f"[{step}] {trace_text}"


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


def remove_end_session_mark(content: str) -> tuple[str, bool]:
  if END_SESSION_MARK not in content:
    return content, False
  return content.replace(END_SESSION_MARK, "").strip(), True


def append_end_session_mark(content: str) -> str:
  stripped = content.rstrip()
  if stripped.endswith(END_SESSION_MARK):
    return stripped
  return f"{stripped} {END_SESSION_MARK}"


def move_end_session_mark_to_last_assistant(output_messages: list[dict[str, str]]) -> None:
  if not output_messages or output_messages[-1]["role"] != "user":
    return

  content, removed = remove_end_session_mark(output_messages[-1]["content"])
  if not removed:
    return
  output_messages[-1]["content"] = content

  for message in reversed(output_messages):
    if message["role"] == "assistant":
      message["content"] = append_end_session_mark(message["content"])
      return

  raise DatasetBuildError(f"{END_SESSION_MARK} found in final user message, but no therapist message exists.")


def build_chatml_session(
  path: Path,
  system_prompt: str,
  fields_by_step: dict[str, TraceFields],
) -> dict[str, list[dict[str, str]]]:
  payload = load_session(path)
  patient_profile = require_dict(payload.get("patient_profile"), f"{path}: patient_profile")
  messages = require_list(payload.get("messages"), f"{path}: messages")
  raw_trace_groups = require_list(payload.get("reasoning_traces"), f"{path}: reasoning_traces")
  trace_groups = trace_groups_by_key(raw_trace_groups, str(path))

  output_messages: list[dict[str, str]] = [
    {"role": "system", "content": render_system_prompt(system_prompt, patient_profile)}
  ]
  first_therapist_message_seen = False
  for idx, raw_message in enumerate(messages, start=1):
    message = require_dict(raw_message, f"{path}: messages[{idx}]")
    if message_role(message) == "therapist" and not first_therapist_message_seen:
      first_therapist_message_seen = True
      content = message_content(message)
      if not content:
        raise DatasetBuildError(f"{path}: messages[{idx}]: message content must be non-empty.")
      output_messages.append({"role": "assistant", "content": content})
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

  move_end_session_mark_to_last_assistant(output_messages)
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
