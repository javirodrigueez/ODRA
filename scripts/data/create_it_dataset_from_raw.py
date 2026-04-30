from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.data.create_reasoning_instruction_dataset import (  # noqa: E402
  DatasetBuildError,
  ensure_str,
  format_patient_intake_form,
  require_dict,
  require_list,
  trace_reasoning_text,
)


TraceKey = tuple[Any, str]


def project_root() -> Path:
  return Path(__file__).resolve().parents[2]


def load_therapist_system_prompt(root: Path) -> str:
  prompt_path = root / "agent_prompts" / "therapist_system.txt"
  if not prompt_path.is_file():
    raise DatasetBuildError(f"Missing therapist system prompt: {prompt_path}")
  return prompt_path.read_text(encoding="utf-8")


def render_instruction(system_prompt: str, patient_profile: dict[str, Any]) -> str:
  intake = format_patient_intake_form(patient_profile)
  return system_prompt.replace("{patient_intake_form}", intake)


def load_session(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{path}: failed to read file: {exc}") from exc
  return require_dict(payload, f"{path}: root")


def trace_group_key(group: dict[str, Any], label: str) -> TraceKey:
  speaker = ensure_str(group.get("speaker")).strip().lower()
  if not speaker:
    raise DatasetBuildError(f"{label}: speaker must be non-empty.")
  if "utterance_index" not in group:
    raise DatasetBuildError(f"{label}: utterance_index is required.")
  return (group.get("utterance_index"), speaker)


def trace_groups_by_key(groups: list[Any], source_label: str) -> dict[TraceKey, list[Any]]:
  by_key: dict[TraceKey, list[Any]] = {}
  for idx, raw_group in enumerate(groups, start=1):
    group = require_dict(raw_group, f"{source_label}: reasoning_traces[{idx}]")
    key = trace_group_key(group, f"{source_label}: reasoning_traces[{idx}]")
    if key in by_key:
      raise DatasetBuildError(
        f"{source_label}: duplicate reasoning trace group for utterance {key[0]} speaker {key[1]}."
      )
    by_key[key] = require_list(group.get("traces"), f"{source_label}: reasoning_traces[{idx}].traces")
  return by_key


def format_trace_block(traces: list[Any], source_label: str) -> str:
  trace_lines: list[str] = []
  for idx, raw_trace in enumerate(traces, start=1):
    trace = require_dict(raw_trace, f"{source_label}: traces[{idx}]")
    trace_lines.append(trace_reasoning_text(trace))
  return "\n".join(trace_lines)


def message_role(message: dict[str, Any]) -> str:
  return ensure_str(message.get("role")).strip().lower()


def message_content(message: dict[str, Any]) -> str:
  return ensure_str(message.get("content")).strip()


def format_context_message(
  message: dict[str, Any],
  trace_groups: dict[TraceKey, list[Any]],
  source_label: str,
) -> str:
  role = message_role(message)
  content = message_content(message)
  display_role = role.title() if role else "Unknown"

  if role != "therapist":
    return f"{display_role}: {content}"

  traces = trace_groups.get((message.get("utterance_index"), "therapist"), [])
  if not traces:
    return f"Therapist: {content}"

  trace_text = format_trace_block(
    traces,
    f"{source_label}: therapist utterance {message.get('utterance_index')}",
  )
  return f"Therapist: <think>{trace_text}</think> {content}"


def format_session_context(
  previous_messages: list[Any],
  trace_groups: dict[TraceKey, list[Any]],
  source_label: str,
) -> str:
  if not previous_messages:
    return "none"

  rendered: list[str] = []
  for idx, raw_message in enumerate(previous_messages, start=1):
    message = require_dict(raw_message, f"{source_label}: previous_messages[{idx}]")
    rendered.append(format_context_message(message, trace_groups, source_label))
  return "\n".join(rendered)


def build_sample(
  message: dict[str, Any],
  previous_messages: list[Any],
  traces: list[Any],
  trace_groups: dict[TraceKey, list[Any]],
  instruction: str,
  source_label: str,
) -> dict[str, str] | None:
  if not traces:
    return None

  therapist_utterance = message_content(message)
  if not therapist_utterance:
    raise DatasetBuildError(f"{source_label}: therapist utterance must be non-empty.")

  trace_text = format_trace_block(traces, source_label)
  context_text = format_session_context(previous_messages, trace_groups, source_label)
  input_text = f"<context>\n{context_text}\n</context>\nContinue with the ongoing therapy session."
  output_text = f"<think>{trace_text}</think>{therapist_utterance}"
  return {
    "instruction": instruction,
    "input": input_text,
    "output": output_text,
  }


def build_samples_for_file(path: Path, system_prompt: str) -> list[dict[str, str]]:
  payload = load_session(path)
  profile = require_dict(payload.get("patient_profile"), f"{path}: patient_profile")
  messages = require_list(payload.get("messages"), f"{path}: messages")
  raw_trace_groups = require_list(payload.get("reasoning_traces"), f"{path}: reasoning_traces")
  trace_groups = trace_groups_by_key(raw_trace_groups, str(path))
  instruction = render_instruction(system_prompt, profile)

  samples: list[dict[str, str]] = []
  for idx, raw_message in enumerate(messages):
    message = require_dict(raw_message, f"{path}: messages[{idx + 1}]")
    if message_role(message) != "therapist":
      continue

    key = (message.get("utterance_index"), "therapist")
    traces = trace_groups.get(key, [])
    previous_messages = messages[max(0, idx - 10):idx]
    sample = build_sample(
      message,
      previous_messages,
      traces,
      trace_groups,
      instruction,
      f"{path}: messages[{idx + 1}]",
    )
    if sample is not None:
      samples.append(sample)
  return samples


def build_dataset(input_dir: Path, output_file: Path, root: Path | None = None) -> int:
  if not input_dir.is_dir():
    raise DatasetBuildError(f"--input-dir must be a directory: {input_dir}")
  root = root or project_root()
  system_prompt = load_therapist_system_prompt(root)
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")

  samples: list[dict[str, str]] = []
  for input_file in input_files:
    samples.extend(build_samples_for_file(input_file, system_prompt))

  output_file.parent.mkdir(parents=True, exist_ok=True)
  with output_file.open("w", encoding="utf-8", newline="\n") as f:
    for sample in samples:
      f.write(json.dumps(sample, ensure_ascii=False))
      f.write("\n")
  return len(samples)


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create an instruction-tuning JSONL dataset from raw therapy simulation JSON files."
  )
  parser.add_argument("--input-dir", required=True, help="Directory containing raw simulation JSON files.")
  parser.add_argument("--output-file", required=True, help="Output JSONL file to write.")
  args = parser.parse_args()

  try:
    count = build_dataset(Path(args.input_dir), Path(args.output_file))
  except DatasetBuildError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {count} samples to {args.output_file}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
