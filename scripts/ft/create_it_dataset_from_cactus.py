from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Literal


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.ft.create_it_dataset_from_raw import (  # noqa: E402
  PATIENT_INTAKE_FORM_PLACEHOLDER,
  DatasetBuildError,
  OutputTemplate,
  ensure_str,
  render_output_template,
  render_plain_system_prompt,
  require_dict,
  require_list,
)


BuildMode = Literal["session", "turns"]
SourceFormat = Literal["cactus", "magnet"]


def project_root() -> Path:
  return Path(__file__).resolve().parents[2]


def load_system_prompt(root: Path, source_format: SourceFormat) -> str:
  prompt_path = root / "ft_prompts" / f"{source_format}.txt"
  if not prompt_path.is_file():
    raise DatasetBuildError(f"Missing {source_format} fine-tuning system prompt: {prompt_path}")
  prompt = prompt_path.read_text(encoding="utf-8").strip()
  if not prompt:
    raise DatasetBuildError(f"{prompt_path}: prompt must be non-empty.")
  return prompt


def load_json_object(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{path}: failed to read file: {exc}") from exc
  return require_dict(payload, f"{path}: root")


def patient_intake_form(payload: dict[str, Any], source_label: str) -> str:
  candidates: list[Any] = []
  example = payload.get("example")
  if isinstance(example, dict):
    ai_client = example.get("AI_client")
    if isinstance(ai_client, dict):
      candidates.append(ai_client.get("intake_form"))
  ai_client = payload.get("AI_client")
  if isinstance(ai_client, dict):
    candidates.append(ai_client.get("intake_form"))

  for candidate in candidates:
    intake = ensure_str(candidate).strip()
    if intake:
      return intake
  raise DatasetBuildError(f"{source_label}: missing AI_client.intake_form.")


def source_messages(payload: dict[str, Any], source_label: str) -> list[Any]:
  if isinstance(payload.get("messages"), list):
    return require_list(payload.get("messages"), f"{source_label}: messages")
  if isinstance(payload.get("dialogue"), list):
    return require_list(payload.get("dialogue"), f"{source_label}: dialogue")
  if isinstance(payload.get("history"), list):
    return require_list(payload.get("history"), f"{source_label}: history")
  raise DatasetBuildError(f"{source_label}: missing messages, dialogue, or history list.")


def detect_source_format(payload: dict[str, Any], source_label: str) -> SourceFormat:
  if isinstance(payload.get("dialogue"), list):
    return "magnet"
  if isinstance(payload.get("messages"), list) or isinstance(payload.get("history"), list):
    return "cactus"
  raise DatasetBuildError(f"{source_label}: cannot detect cactus or magnet format.")


def normalize_role(role: Any, source_label: str) -> str:
  normalized = ensure_str(role).strip().lower()
  if normalized in {"therapist", "counselor", "assistant"}:
    return "assistant"
  if normalized in {"patient", "client", "user"}:
    return "user"
  raise DatasetBuildError(f"{source_label}: unsupported message role: {normalized!r}.")


def normalize_message(raw_message: Any, source_label: str, fallback_index: int) -> dict[str, Any]:
  message = require_dict(raw_message, source_label)
  role = normalize_role(message.get("role"), source_label)
  content = message.get("content")
  if content is None:
    content = message.get("message")
  content_text = ensure_str(content).strip()
  if not content_text:
    raise DatasetBuildError(f"{source_label}: message content must be non-empty.")
  utterance_index = message.get("utterance_index", fallback_index)
  return {"role": role, "content": content_text, "utterance_index": utterance_index}


def load_cactus_session(path: Path) -> tuple[SourceFormat, str, list[dict[str, Any]]]:
  payload = load_json_object(path)
  source_format = detect_source_format(payload, str(path))
  intake = patient_intake_form(payload, str(path))
  raw_messages = source_messages(payload, str(path))
  messages = [
    normalize_message(raw_message, f"{path}: messages[{idx}]", idx)
    for idx, raw_message in enumerate(raw_messages, start=1)
  ]
  if not messages:
    raise DatasetBuildError(f"{path}: conversation has no messages.")
  return source_format, intake, messages


def render_system_prompt(system_prompt: str, intake_form: str) -> str:
  return system_prompt.replace(PATIENT_INTAKE_FORM_PLACEHOLDER, intake_form)


def prefixed_history_message(message: dict[str, Any]) -> str:
  prefix = "Therapist" if message["role"] == "assistant" else "Patient"
  return f"{prefix}: {message['content']}"


def turn_output_name(path: Path, utterance_index: Any) -> str:
  try:
    index = int(utterance_index)
  except (TypeError, ValueError) as exc:
    raise DatasetBuildError(f"{path}: therapist utterance_index must be an integer in turns mode.") from exc
  return f"{path.stem}.turn_{index:04d}{path.suffix}"


def build_chatml_session(
  path: Path,
  root: Path,
) -> dict[str, list[dict[str, str]]]:
  source_format, intake_form, messages = load_cactus_session(path)
  system_prompt = load_system_prompt(root, source_format)
  output_messages = [{"role": "system", "content": render_system_prompt(system_prompt, intake_form)}]
  output_messages.extend(
    {"role": message["role"], "content": message["content"]}
    for message in messages
  )
  return {"messages": output_messages}


def build_chatml_turn_sessions(
  path: Path,
  root: Path,
) -> list[tuple[str, dict[str, list[dict[str, str]]]]]:
  source_format, intake_form, messages = load_cactus_session(path)
  system_prompt = load_system_prompt(root, source_format)
  system_message = {"role": "system", "content": render_plain_system_prompt(system_prompt)}
  rendered: list[tuple[str, dict[str, list[dict[str, str]]]]] = []

  for target_position, message in enumerate(messages):
    if message["role"] != "assistant":
      continue
    history = "\n".join(prefixed_history_message(context) for context in messages[:target_position])
    output_messages = [
      dict(system_message),
      {
        "role": "user",
        "content": f"Dialogue History: {history}\nClient Profile: {intake_form}",
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
  mode: BuildMode = "session",
  template: OutputTemplate = "chatml",
) -> int:
  if not input_dir.is_dir():
    raise DatasetBuildError(f"--input-dir must be a directory: {input_dir}")
  if mode not in {"session", "turns"}:
    raise DatasetBuildError(f"Unsupported build mode: {mode!r}.")
  if template not in {"chatml", "sharegpt"}:
    raise DatasetBuildError(f"Unsupported output template: {template!r}.")

  root = root or project_root()
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")

  rendered_sessions: list[tuple[str, dict[str, list[dict[str, str]]]]] = []
  for input_file in input_files:
    if mode == "turns":
      rendered_sessions.extend(build_chatml_turn_sessions(input_file, root))
      continue
    rendered_sessions.append((input_file.name, build_chatml_session(input_file, root)))

  output_dir.mkdir(parents=True, exist_ok=True)
  for file_name, session in rendered_sessions:
    output_path = output_dir / file_name
    rendered = render_output_template(session, template, file_name)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
      json.dump(rendered, f, ensure_ascii=False, indent=2)
      f.write("\n")

  return len(rendered_sessions)


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create an instruction-tuning dataset from Cactus or Magnet therapy session JSON files."
  )
  parser.add_argument("--input-dir", required=True, help="Directory containing Cactus or Magnet JSON files.")
  parser.add_argument("--output-dir", required=True, help="Directory to write one JSON file per sample.")
  parser.add_argument(
    "--template",
    choices=["chatml", "sharegpt"],
    default="chatml",
    help="Output schema: ChatML {'messages': ...} or ShareGPT {'conversations': ...}.",
  )
  parser.add_argument(
    "--mode",
    choices=["session", "turns"],
    default="session",
    help="Create one sample per session or one sample per therapist turn.",
  )
  args = parser.parse_args()

  try:
    count = build_dataset(
      Path(args.input_dir),
      Path(args.output_dir),
      mode=args.mode,
      template=args.template,
    )
  except DatasetBuildError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {count} {args.template} file(s) to {args.output_dir}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
