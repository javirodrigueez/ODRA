from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Literal


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.ft.create_it_dataset_from_raw import (  # noqa: E402
  DatasetBuildError,
  TraceFields,
  format_trace,
  load_trace_field_config,
)


OutputFormat = Literal["chatml", "sharegpt"]
TraceOutputFormat = Literal["raw-json", "text"]
PROMPTS_DIR = ROOT / "prompts"
PROMPTS_TEXT_DIR = ROOT / "prompts_text"
THERAPIST_ROLE = "therapist"
PATIENT_ROLE = "patient"


class ReplayError(DatasetBuildError):
  pass


class _NoopLLM:
  def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
    raise ReplayError("Replay LLM should never be called directly.")


def ensure_str(value: Any) -> str:
  if value is None:
    return ""
  return str(value)


def load_json_object(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise ReplayError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise ReplayError(f"{path}: failed to read file: {exc}") from exc
  if not isinstance(payload, dict):
    raise ReplayError(f"{path}: root must be a JSON object.")
  return payload


def require_list(value: Any, label: str) -> list[Any]:
  if not isinstance(value, list):
    raise ReplayError(f"{label} must be a JSON list.")
  return value


def require_dict(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise ReplayError(f"{label} must be a JSON object.")
  return value


def normalize_role(role: Any) -> str:
  return ensure_str(role).strip().lower()


def coerce_int(value: Any, label: str) -> int:
  try:
    return int(value)
  except (TypeError, ValueError) as exc:
    raise ReplayError(f"{label} must be an integer.") from exc


def trace_groups_by_key(payload: dict[str, Any], source_label: str) -> dict[tuple[int, str], dict[str, Any]]:
  result: dict[tuple[int, str], dict[str, Any]] = {}
  for index, raw_group in enumerate(require_list(payload.get("reasoning_traces"), f"{source_label}: reasoning_traces"), start=1):
    group = require_dict(raw_group, f"{source_label}: reasoning_traces[{index}]")
    utterance_index = coerce_int(group.get("utterance_index"), f"{source_label}: reasoning_traces[{index}].utterance_index")
    speaker = normalize_role(group.get("speaker"))
    if not speaker:
      raise ReplayError(f"{source_label}: reasoning_traces[{index}].speaker must be non-empty.")
    key = (utterance_index, speaker)
    if key in result:
      raise ReplayError(f"{source_label}: duplicate reasoning trace group for {key}.")
    require_list(group.get("traces"), f"{source_label}: reasoning_traces[{index}].traces")
    result[key] = group
  return result


def parse_trace_output(trace: dict[str, Any], source_label: str) -> Any:
  output = trace.get("output")
  if not isinstance(output, str):
    return output
  stripped = output.strip()
  if not stripped:
    return ""
  try:
    return json.loads(stripped)
  except json.JSONDecodeError as exc:
    raise ReplayError(f"{source_label}: trace output is not valid JSON: {exc}") from exc


def is_request_trace(trace: dict[str, Any]) -> bool:
  output = trace.get("output")
  if not ensure_str(trace.get("input")).strip() or not ensure_str(output).strip():
    return False
  if not isinstance(output, str):
    return True
  try:
    json.loads(output.strip())
  except json.JSONDecodeError:
    return False
  return True


def prompt_path(prompt_root: Path, filename: str | Path) -> Path:
  path = Path(filename)
  if path.is_absolute():
    return path
  return prompt_root / path


def load_prompt_from_root(prompt_root: Path, filename: str | Path) -> str:
  path = prompt_path(prompt_root, filename)
  try:
    return path.read_text(encoding="utf-8").strip()
  except OSError as exc:
    raise ReplayError(f"Failed to read prompt {path}: {exc}") from exc


def render_prompt_from_root(prompt_root: Path, filename: str | Path, **values: Any) -> str:
  text = load_prompt_from_root(prompt_root, filename)
  for key, value in values.items():
    text = text.replace(f"{{{key}}}", "" if value is None else str(value))
  return text


def load_stage_prompt_from_root(self: Any, stage: str, name: str, **values: Any) -> str:
  if stage == "mi_topic_exploration":
    stage = "mi_techniques_with_te"
  filename = f"{stage}/{name}.txt"
  if values:
    return render_prompt_from_root(self.prompt_root, filename, **values)
  return load_prompt_from_root(self.prompt_root, filename)


def make_chatml_sample(messages: list[dict[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
  return {
    "messages": messages,
    "metadata": metadata,
  }


def make_sharegpt_sample(messages: list[dict[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
  role_map = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
  }
  conversations: list[dict[str, str]] = []
  for message in messages:
    role = message.get("role", "")
    if role not in role_map:
      raise ReplayError(f"Cannot convert unsupported ChatML role to ShareGPT: {role!r}.")
    conversations.append({"from": role_map[role], "value": message.get("content", "")})
  return {
    "conversations": conversations,
    "metadata": metadata,
  }


def make_sample(
  messages: list[dict[str, str]],
  metadata: dict[str, Any],
  output_format: OutputFormat,
) -> dict[str, Any]:
  if output_format == "chatml":
    return make_chatml_sample(messages, metadata)
  if output_format == "sharegpt":
    return make_sharegpt_sample(messages, metadata)
  raise ReplayError(f"Unsupported output format: {output_format!r}.")


class SampleCollector:
  def __init__(self, output_format: OutputFormat):
    self.output_format = output_format
    self.samples: list[dict[str, Any]] = []

  def add(self, messages: list[dict[str, str]], target: str, metadata: dict[str, Any]) -> None:
    normalized_messages = [dict(message) for message in messages]
    normalized_messages.append({"role": "assistant", "content": target})
    self.samples.append(make_sample(normalized_messages, dict(metadata), self.output_format))


class ReplayTherapist:
  def __init__(
    self,
    *,
    profile: dict[str, Any],
    prompt_root: Path,
    collector: SampleCollector,
    source_file: str,
    trace_output_format: TraceOutputFormat,
    fields_by_step: dict[str, TraceFields],
    rollup_summary: str,
  ):
    from agents.therapist import TherapistAgent

    self.agent = TherapistAgent(_NoopLLM(), profile)
    self.agent.system_prompt = self.agent._render_agent_system_prompt(prompt_root / "therapist_system.txt")
    self.agent.system_prompt_structured = self.agent._render_agent_system_prompt(prompt_root / "therapist_system_structured.txt")
    self.prompt_root = prompt_root
    self.collector = collector
    self.source_file = source_file
    self.trace_output_format = trace_output_format
    self.fields_by_step = fields_by_step
    self.rollup_summary = rollup_summary
    self.message: dict[str, Any] | None = None
    self.traces: list[dict[str, Any]] = []
    self.trace_index = 0
    self.last_prompt_context: dict[str, str] = {}
    self.utterance_sample_added = False

  def start_turn(self, message: dict[str, Any], trace_group: dict[str, Any]) -> None:
    self.message = message
    self.traces = [
      require_dict(trace, f"{self.source_file}: therapist trace")
      for trace in require_list(trace_group.get("traces"), f"{self.source_file}: therapist traces")
    ]
    self.trace_index = 0
    self.last_prompt_context = {}
    self.utterance_sample_added = False

  def finish_turn(self) -> None:
    remaining_request_traces = [trace for trace in self.traces[self.trace_index:] if is_request_trace(trace)]
    for trace in remaining_request_traces:
      self._add_fallback_trace_sample(trace)
    if not self.utterance_sample_added:
      self._add_fallback_utterance_sample()
    self.message = None
    self.traces = []
    self.trace_index = 0
    self.last_prompt_context = {}
    self.utterance_sample_added = False

  def build_structured_messages(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
  ) -> list[dict[str, str]]:
    suffix = load_prompt_from_root(self.prompt_root, "shared/structured_suffix.txt")
    user_prompt = self.agent._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      suffix=suffix,
    )
    return [
      {"role": "system", "content": self.agent.system_prompt_structured},
      {"role": "user", "content": user_prompt},
    ]

  def build_utterance_messages(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
  ) -> list[dict[str, str]]:
    suffix = load_prompt_from_root(self.prompt_root, "shared/utterance_suffix.txt")
    user_prompt = self.agent._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      suffix=suffix,
      utterance=True,
    )
    return [
      {"role": "system", "content": self.agent.system_prompt},
      {"role": "user", "content": user_prompt},
    ]

  def _next_trace(self) -> dict[str, Any]:
    while self.trace_index < len(self.traces):
      trace = self.traces[self.trace_index]
      self.trace_index += 1
      if is_request_trace(trace):
        return trace
    idx = self.message.get("utterance_index") if self.message else "unknown"
    steps = ", ".join(ensure_str(trace.get("step")) for trace in self.traces) or "none"
    raise ReplayError(
      f"{self.source_file}: graph requested more therapist structured outputs than recorded "
      f"for utterance {idx}; recorded request trace steps: {steps}."
    )

  def _remember_prompt_context(
    self,
    *,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
  ) -> None:
    self.last_prompt_context = {
      "full_conversation": full_conversation,
      "last_utterances": last_utterances,
      "last_reasoning": last_reasoning,
      "session_context": session_context,
    }

  def _add_fallback_trace_sample(self, trace: dict[str, Any]) -> None:
    if self.message is None:
      return
    context = self.last_prompt_context
    if not context:
      context = {
        "full_conversation": "(unavailable during replay fallback)",
        "last_utterances": "(unavailable during replay fallback)",
        "last_reasoning": "",
        "session_context": "{}",
      }
    instruction = ensure_str(trace.get("input")).strip()
    messages = self.build_structured_messages(
      full_conversation=context["full_conversation"],
      last_utterances=context["last_utterances"],
      last_reasoning=context["last_reasoning"],
      session_context=context["session_context"],
      instruction=instruction,
    )
    target = ensure_str(trace.get("output")).strip() if self.trace_output_format == "raw-json" else format_trace(trace, self.fields_by_step)
    self.collector.add(
      messages,
      target,
      {
        "source_file": self.source_file,
        "utterance_index": self.message.get("utterance_index"),
        "sample_kind": "reasoning_trace",
        "step": trace.get("step"),
        "stage": self.message.get("stage"),
        "substage": self.message.get("substage"),
        "trace_index": self.traces.index(trace) + 1,
        "replay_status": "fallback_unconsumed_trace",
      },
    )

  def _add_fallback_utterance_sample(self) -> None:
    if self.message is None:
      return
    context = self.last_prompt_context
    if not context:
      context = {
        "full_conversation": "(unavailable during replay fallback)",
        "last_utterances": "(unavailable during replay fallback)",
        "last_reasoning": "",
        "session_context": "{}",
      }
    instruction = "Continue with the next therapist utterance for the current therapy stage."
    messages = self.build_utterance_messages(
      full_conversation=context["full_conversation"],
      last_utterances=context["last_utterances"],
      last_reasoning=context["last_reasoning"],
      session_context=context["session_context"],
      instruction=instruction,
    )
    self.collector.add(
      messages,
      ensure_str(self.message.get("content")).strip(),
      {
        "source_file": self.source_file,
        "utterance_index": self.message.get("utterance_index"),
        "sample_kind": "utterance",
        "stage": self.message.get("stage"),
        "substage": self.message.get("substage"),
        "replay_status": "fallback_unemitted_utterance",
      },
    )

  def _is_rollup_instruction(self, instruction: str) -> bool:
    lowered = instruction.lower()
    return "conversation_summary" in lowered and "summarize the conversation" in lowered

  def generate_structured(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
    expected_type: type[Any] | tuple[type[Any], ...] | None = dict,
    temperature: float | None = None,
    **kwargs: Any,
  ) -> tuple[Any, dict[str, Any]]:
    self._remember_prompt_context(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
    )
    if self._is_rollup_instruction(instruction):
      return {
        "conversation_summary": self.rollup_summary or "Conversation summary unavailable in replay source."
      }, {"raw": "", "attempts": 1, "parse_mode": "replay_rollup"}

    trace = self._next_trace()
    output = ensure_str(trace.get("output")).strip()
    messages = self.build_structured_messages(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      instruction=instruction,
    )
    target = output if self.trace_output_format == "raw-json" else format_trace(trace, self.fields_by_step)
    message = self.message or {}
    self.collector.add(
      messages,
      target,
      {
        "source_file": self.source_file,
        "utterance_index": message.get("utterance_index"),
        "sample_kind": "reasoning_trace",
        "step": trace.get("step"),
        "stage": message.get("stage"),
        "substage": message.get("substage"),
        "trace_index": self.trace_index,
      },
    )
    parsed = parse_trace_output(trace, f"{self.source_file}: {trace.get('step')}")
    return parsed, {"raw": output, "attempts": 1, "parse_mode": "replay"}

  def generate_utterance(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
    **kwargs: Any,
  ) -> str:
    self._remember_prompt_context(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
    )
    if self.message is None:
      raise ReplayError(f"{self.source_file}: therapist utterance requested without an active replay message.")
    content = ensure_str(self.message.get("content")).strip()
    if not content:
      raise ReplayError(f"{self.source_file}: therapist replay message content is empty.")
    messages = self.build_utterance_messages(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      instruction=instruction,
    )
    self.collector.add(
      messages,
      content,
      {
        "source_file": self.source_file,
        "utterance_index": self.message.get("utterance_index"),
        "sample_kind": "utterance",
        "stage": self.message.get("stage"),
        "substage": self.message.get("substage"),
      },
    )
    self.utterance_sample_added = True
    return content


class ReplayPatient:
  def __init__(self, *, source_file: str):
    self.source_file = source_file
    self.message: dict[str, Any] | None = None
    self.traces: list[dict[str, Any]] = []
    self.trace_index = 0

  def start_turn(self, message: dict[str, Any], trace_group: dict[str, Any]) -> None:
    self.message = message
    self.traces = [
      require_dict(trace, f"{self.source_file}: patient trace")
      for trace in require_list(trace_group.get("traces", []), f"{self.source_file}: patient traces")
    ]
    self.trace_index = 0

  def finish_turn(self) -> None:
    self.message = None
    self.traces = []
    self.trace_index = 0

  def build_structured_messages(self, **kwargs: Any) -> list[dict[str, str]]:
    return [{"role": "user", "content": "patient replay structured request"}]

  def build_utterance_messages(self, **kwargs: Any) -> list[dict[str, str]]:
    return [{"role": "user", "content": "patient replay utterance request"}]

  def _next_trace(self, instruction: str) -> dict[str, Any]:
    while self.trace_index < len(self.traces):
      trace = self.traces[self.trace_index]
      self.trace_index += 1
      if ensure_str(trace.get("input")).strip() and ensure_str(trace.get("output")).strip():
        return trace
    raise ReplayError(f"{self.source_file}: graph requested more patient structured outputs than recorded.")

  def generate_structured(self, instruction: str, expected_type: type[Any] | tuple[type[Any], ...] | None = dict, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    trace = self._next_trace(instruction)
    parsed = parse_trace_output(trace, f"{self.source_file}: patient {trace.get('step')}")
    return parsed, {"raw": trace.get("output"), "attempts": 1, "parse_mode": "replay"}

  def generate_utterance(self, **kwargs: Any) -> str:
    if self.message is None:
      raise ReplayError(f"{self.source_file}: patient utterance requested without an active replay message.")
    content = ensure_str(self.message.get("content")).strip()
    if not content:
      raise ReplayError(f"{self.source_file}: patient replay message content is empty.")
    return content


def replay_builder(
  *,
  cfg_path: str,
  profile: dict[str, Any],
  disable_tib: bool,
  prompt_root: Path,
  collector: SampleCollector,
  source_file: str,
  trace_output_format: TraceOutputFormat,
  fields_by_step: dict[str, TraceFields],
  rollup_summary: str,
) -> Any:
  from graph.graph import TherapyGraphBuilder
  from runtime_config import load_runtime_config

  builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
  builder.cfg = load_runtime_config(cfg_path)
  builder.profile = profile
  builder.disable_tib = bool(disable_tib)
  builder.prompt_root = prompt_root
  builder._load_stage_prompt = MethodType(load_stage_prompt_from_root, builder)
  builder.therapist = ReplayTherapist(
    profile=profile,
    prompt_root=prompt_root,
    collector=collector,
    source_file=source_file,
    trace_output_format=trace_output_format,
    fields_by_step=fields_by_step,
    rollup_summary=rollup_summary,
  )
  builder.patient = ReplayPatient(source_file=source_file)
  builder.cbt_techniques = builder._load_techniques()
  builder.mi_techniques = builder._load_mi_techniques()
  return builder


def force_raw_message_state(state: dict[str, Any], messages_so_far: list[dict[str, Any]], message: dict[str, Any]) -> None:
  state["messages"] = deepcopy(messages_so_far)
  state["utterance_index"] = coerce_int(message.get("utterance_index"), "message.utterance_index")
  stage = ensure_str(message.get("stage")).strip()
  substage = ensure_str(message.get("substage")).strip()
  if stage:
    state["current_stage"] = stage
    state["active_exchange_stage"] = stage
  if substage:
    state["current_substage"] = substage
    state["active_exchange_substage"] = substage
  state["session_ended"] = False
  state["session_ended_reason"] = ""
  state["conversation_error"] = ""


def build_samples_for_session(
  path: Path,
  *,
  cfg_path: str,
  output_format: OutputFormat,
  trace_output_format: TraceOutputFormat,
  fields_by_step: dict[str, TraceFields],
) -> list[dict[str, Any]]:
  payload = load_json_object(path)
  profile = require_dict(payload.get("patient_profile"), f"{path}: patient_profile")
  raw_messages = require_list(payload.get("messages"), f"{path}: messages")
  messages = [require_dict(message, f"{path}: messages[{index}]") for index, message in enumerate(raw_messages, start=1)]
  trace_groups = trace_groups_by_key(payload, str(path))
  prompt_root = PROMPTS_TEXT_DIR if trace_output_format == "text" else PROMPTS_DIR
  if not prompt_root.is_dir():
    raise ReplayError(f"Prompt root is missing: {prompt_root}")

  collector = SampleCollector(output_format)
  max_utterance_index = max((coerce_int(message.get("utterance_index"), f"{path}: message utterance_index") for message in messages), default=0)
  builder = replay_builder(
    cfg_path=cfg_path,
    profile=profile,
    disable_tib=bool(payload.get("disable_tib", False)),
    prompt_root=prompt_root,
    collector=collector,
    source_file=path.name,
    trace_output_format=trace_output_format,
    fields_by_step=fields_by_step,
    rollup_summary=ensure_str(payload.get("conversation_rollup_summary")),
  )
  state = builder.initial_state(
    max_turns=int(payload.get("max_turns") or max_utterance_index + 2),
    min_turns=int(payload.get("min_turns") or 0),
  )
  messages_so_far: list[dict[str, Any]] = []

  for message in messages:
    role = normalize_role(message.get("role"))
    idx = coerce_int(message.get("utterance_index"), f"{path}: message.utterance_index")
    if role == THERAPIST_ROLE:
      trace_group = trace_groups.get((idx, THERAPIST_ROLE), {"traces": []})
      builder.therapist.start_turn(message, trace_group)
      state = builder.therapist_turn(state)
      builder.therapist.finish_turn()
      messages_so_far.append(deepcopy(message))
      force_raw_message_state(state, messages_so_far, message)
      continue
    if role == PATIENT_ROLE:
      trace_group = trace_groups.get((idx, PATIENT_ROLE), {"traces": []})
      builder.patient.start_turn(message, trace_group)
      state = builder.patient_turn(state)
      builder.patient.finish_turn()
      messages_so_far.append(deepcopy(message))
      force_raw_message_state(state, messages_so_far, message)
      continue
    raise ReplayError(f"{path}: unsupported message role {role!r}.")

  return collector.samples


def build_dataset(
  input_dir: Path,
  output: Path,
  *,
  cfg_path: str,
  output_format: OutputFormat,
  trace_output_format: TraceOutputFormat,
) -> int:
  if output_format not in {"chatml", "sharegpt"}:
    raise ReplayError(f"Unsupported --output-format: {output_format!r}.")
  if trace_output_format not in {"raw-json", "text"}:
    raise ReplayError(f"Unsupported --trace-output-format: {trace_output_format!r}.")
  if not input_dir.is_dir():
    raise ReplayError(f"--input-dir must be a directory: {input_dir}")
  if not Path(cfg_path).is_file():
    raise ReplayError(f"--cfg must be an existing file: {cfg_path}")

  fields_by_step = load_trace_field_config(ROOT)
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
  if not input_files:
    raise ReplayError(f"No .json files found in {input_dir}")

  output.parent.mkdir(parents=True, exist_ok=True)
  count = 0
  with output.open("w", encoding="utf-8", newline="\n") as f:
    for input_file in input_files:
      with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
          samples = build_samples_for_session(
            input_file,
            cfg_path=cfg_path,
            output_format=output_format,
            trace_output_format=trace_output_format,
            fields_by_step=fields_by_step,
          )
      for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False))
        f.write("\n")
        count += 1
  return count


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create one replayed graph-inference training sample per therapist reasoning request and utterance."
  )
  parser.add_argument("--input-dir", required=True, help="Directory containing raw simulation JSON files.")
  parser.add_argument("--output", required=True, help="JSONL output path.")
  parser.add_argument("--cfg", required=True, help="Runtime config used to generate the raw simulations.")
  parser.add_argument("--output-format", choices=["chatml", "sharegpt"], default="chatml")
  parser.add_argument("--trace-output-format", choices=["raw-json", "text"], default="raw-json")
  args = parser.parse_args()

  try:
    count = build_dataset(
      Path(args.input_dir),
      Path(args.output),
      cfg_path=args.cfg,
      output_format=args.output_format,
      trace_output_format=args.trace_output_format,
    )
  except DatasetBuildError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {count} replay sample(s) to {args.output}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
