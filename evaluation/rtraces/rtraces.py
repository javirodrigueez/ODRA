#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover - clearer CLI behavior if dotenv is absent.
  load_dotenv = None  # type: ignore[assignment]


MODEL_NAME = "gpt-4o"
BATCH_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
DEFAULT_MAX_ITER = 3
DEFAULT_CONTEXT_TURNS = 5
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_LOCAL_TIMEOUT_HOURS = 26.0
DEFAULT_INPUT_COST_PER_1M = 1.25
DEFAULT_OUTPUT_COST_PER_1M = 5.0
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024

SCRIPT_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = SCRIPT_ROOT / "prompts"
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_KNOWLEDGE_PATH = REPO_ROOT / "data" / "knowledge" / "reasoning_trace_evaluation.json"

FINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
FAILED_STATUSES = {"failed", "expired", "cancelled"}
FINAL_RATING_RE = re.compile(r"Rating:\s*\[?\[?\s*(-?\d+(?:\.\d+)?)\s*\]?\]?", re.IGNORECASE)
TRACE_NOT_ALLOWED = object()


class RTracesEvaluationError(RuntimeError):
  pass


@dataclass(frozen=True)
class MetricSpec:
  key: str
  prompt_file: str
  min_score: float
  max_score: float


@dataclass(frozen=True)
class TraceKnowledge:
  stage_steps: dict[str, dict[str, tuple[str, ...] | None]]
  global_steps: dict[str, tuple[str, ...] | None]
  excluded_for_experts: tuple[str, ...]
  stage_order: tuple[str, ...]
  min_traces_per_sample: int


@dataclass(frozen=True)
class EvaluableReasoning:
  turn_key: str
  stage: str
  substage: str | None
  therapist_utterance_index: int
  therapist_utterance: str
  context: str
  context_utterance_count: int
  traces: list[dict[str, Any]]
  reasoning_traces_txt: str


@dataclass(frozen=True)
class BatchRequest:
  custom_id: str
  metric: str
  input_file: str
  output_file: str
  turn_key: str
  stage: str
  therapist_utterance_index: int
  min_score: float
  max_score: float
  body: dict[str, Any]

  def jsonl_line(self) -> dict[str, Any]:
    return {
      "custom_id": self.custom_id,
      "method": "POST",
      "url": BATCH_ENDPOINT,
      "body": self.body,
    }

  def map_entry(self) -> dict[str, Any]:
    return {
      "custom_id": self.custom_id,
      "metric": self.metric,
      "input_file": self.input_file,
      "output_file": self.output_file,
      "turn_key": self.turn_key,
      "stage": self.stage,
      "therapist_utterance_index": self.therapist_utterance_index,
      "min_score": self.min_score,
      "max_score": self.max_score,
    }


@dataclass(frozen=True)
class BatchOutputRef:
  custom_id: str
  request_index: int
  filename_stem: str
  utterance_index: int
  metric: str


METRIC_SPECS = {
  "faithfulness": MetricSpec("faithfulness", "faithfulness.txt", 0.0, 2.0),
  "logic_consistency": MetricSpec("logic_consistency", "logic_consistency.txt", 0.0, 2.0),
  "answer_explanation_alignment": MetricSpec(
    "answer_explanation_alignment",
    "answer_explanation_alignment.txt",
    0.0,
    2.0,
  ),
}
METRIC_ORDER = tuple(METRIC_SPECS)
METRIC_INDEX = {metric: index for index, metric in enumerate(METRIC_ORDER)}


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def atomic_write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, ensure_ascii=True)
      f.write("\n")
    os.replace(tmp_name, path)
  except Exception:
    try:
      os.unlink(tmp_name)
    except FileNotFoundError:
      pass
    raise


def atomic_write_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(text)
    os.replace(tmp_name, path)
  except Exception:
    try:
      os.unlink(tmp_name)
    except FileNotFoundError:
      pass
    raise


def to_plain(value: Any) -> Any:
  if hasattr(value, "model_dump"):
    return value.model_dump()
  if isinstance(value, dict):
    return {key: to_plain(item) for key, item in value.items()}
  if isinstance(value, list):
    return [to_plain(item) for item in value]
  return value


def get_field(value: Any, key: str, default: Any = None) -> Any:
  if isinstance(value, dict):
    return value.get(key, default)
  return getattr(value, key, default)


def coerce_int(value: Any) -> int | None:
  try:
    if value is None or value == "":
      return None
    return int(value)
  except (TypeError, ValueError):
    return None


def mean(values: Iterable[float]) -> float | None:
  items = list(values)
  if not items:
    return None
  return sum(items) / len(items)


def harmonic_mean(scores: list[float]) -> float:
  if not scores or any(score <= 0 for score in scores):
    return 0.0
  return len(scores) / sum(1 / score for score in scores)


def parse_fields(raw_step: Any, *, step_name: str) -> tuple[str, ...] | None:
  if not isinstance(raw_step, dict):
    raise RTracesEvaluationError(f"Trace knowledge step {step_name!r} must be an object.")
  fields = raw_step.get("fields")
  if fields is None:
    return None
  if not isinstance(fields, list) or any(not isinstance(item, str) or not item for item in fields):
    raise RTracesEvaluationError(f"Trace knowledge step {step_name!r} has invalid fields.")
  return tuple(fields)


def load_trace_knowledge(path: Path = TRACE_KNOWLEDGE_PATH) -> TraceKnowledge:
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise RTracesEvaluationError(f"Trace knowledge file must contain a JSON object: {path}")

  raw_stages = payload.get("stages")
  if not isinstance(raw_stages, dict) or not raw_stages:
    raise RTracesEvaluationError(f"Trace knowledge file is missing non-empty 'stages': {path}")

  stage_steps: dict[str, dict[str, tuple[str, ...] | None]] = {}
  stage_order: list[str] = []
  for stage, raw_stage in raw_stages.items():
    if not isinstance(stage, str) or not isinstance(raw_stage, dict):
      raise RTracesEvaluationError(f"Invalid trace knowledge stage entry: {stage!r}")
    raw_steps = raw_stage.get("steps")
    if not isinstance(raw_steps, dict):
      raise RTracesEvaluationError(f"Trace knowledge stage {stage!r} is missing object 'steps'.")
    stage_order.append(stage)
    stage_steps[stage] = {
      str(step): parse_fields(raw_step, step_name=str(step))
      for step, raw_step in raw_steps.items()
    }

  raw_global_steps = payload.get("global_steps") or {}
  if not isinstance(raw_global_steps, dict):
    raise RTracesEvaluationError(f"Trace knowledge 'global_steps' must be an object: {path}")
  global_steps = {
    str(step): parse_fields(raw_step, step_name=str(step))
    for step, raw_step in raw_global_steps.items()
  }

  raw_excluded = payload.get("excluded_for_experts") or []
  if not isinstance(raw_excluded, list) or any(not isinstance(item, str) for item in raw_excluded):
    raise RTracesEvaluationError(f"Trace knowledge 'excluded_for_experts' must be a list of strings: {path}")

  min_traces = coerce_int(payload.get("min_traces_per_sample"))
  if min_traces is None or min_traces < 1:
    min_traces = 3

  return TraceKnowledge(
    stage_steps=stage_steps,
    global_steps=global_steps,
    excluded_for_experts=tuple(raw_excluded),
    stage_order=tuple(stage_order),
    min_traces_per_sample=min_traces,
  )


def trace_fields_for_step(
  knowledge: TraceKnowledge,
  stage: str | None,
  step: str,
) -> tuple[str, ...] | None | object:
  if stage and step in knowledge.stage_steps.get(stage, {}):
    return knowledge.stage_steps[stage][step]
  if step in knowledge.global_steps:
    return knowledge.global_steps[step]
  return TRACE_NOT_ALLOWED


def parsed_trace_output(raw_output: Any) -> Any:
  if not isinstance(raw_output, str):
    return raw_output
  text = raw_output.strip()
  if not text:
    return ""
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return raw_output


def compact_trace_output(step: str, raw_output: Any, fields: tuple[str, ...] | None) -> dict[str, Any]:
  output = parsed_trace_output(raw_output)

  if step == "resistance_prediction":
    if not isinstance(output, dict):
      return {"message": str(raw_output)}
    return {
      "previous_resistance": output.get("previous_p"),
      "predicted_resistance": output.get("new_p"),
      "reasoning": output.get("reasoning"),
    }

  if step == "technique_evaluation":
    if isinstance(output, str):
      return {"evaluation": output}

  if step == "technique_auto_complete":
    return {"message": output if isinstance(output, str) else raw_output}

  if not isinstance(output, dict):
    return {"message": str(output)}

  if fields is None:
    compact = dict(output)
  else:
    compact = {field: output[field] for field in fields if field in output}

  if "reasoning" in output and "reasoning" not in compact:
    compact["reasoning"] = output["reasoning"]
  return compact


def normalized_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
  raw_messages = data.get("messages")
  if not isinstance(raw_messages, list):
    return []

  normalized: list[dict[str, Any]] = []
  next_index = 1
  for item in raw_messages:
    if not isinstance(item, dict):
      continue
    role = str(item.get("role") or "").strip().lower()
    content = str(item.get("content") or "").strip()
    if role not in {"therapist", "patient"} or not content:
      continue

    utterance_index = coerce_int(item.get("utterance_index"))
    if utterance_index is None or utterance_index <= 0:
      utterance_index = next_index
    next_index = max(next_index + 1, utterance_index + 1)

    normalized.append(
      {
        "utterance_index": utterance_index,
        "role": role,
        "content": content,
        "stage": item.get("stage"),
        "substage": item.get("substage"),
      }
    )

  return sorted(normalized, key=lambda message: int(message["utterance_index"]))


def therapist_reasoning_traces(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
  raw_traces = data.get("reasoning_traces")
  if not isinstance(raw_traces, list):
    return {}

  indexed: dict[int, dict[str, Any]] = {}
  for raw_trace in raw_traces:
    if not isinstance(raw_trace, dict):
      continue
    if str(raw_trace.get("speaker") or "").strip().lower() != "therapist":
      continue
    utterance_index = coerce_int(raw_trace.get("utterance_index"))
    if utterance_index is None:
      continue
    raw_steps = raw_trace.get("traces")
    if not isinstance(raw_steps, list):
      continue
    target = indexed.setdefault(
      utterance_index,
      {"utterance_index": utterance_index, "speaker": "therapist", "traces": []},
    )
    target["traces"].extend(raw_steps)
  return indexed


def format_context(messages: list[dict[str, Any]]) -> str:
  if not messages:
    return "(No previous context available.)"

  lines = []
  for message in messages:
    role = str(message.get("role") or "").strip().title() or "Speaker"
    utterance_index = message.get("utterance_index")
    stage = message.get("stage") or "unknown"
    substage = message.get("substage") or "unknown"
    content = str(message.get("content") or "").strip()
    lines.append(f"{role} #{utterance_index} [stage={stage}, substage={substage}]: {content}")
  return "\n".join(lines)


def previous_context_messages(
  messages: list[dict[str, Any]],
  therapist_utterance_index: int,
  ctxt_turns: int,
) -> list[dict[str, Any]]:
  limit = max(0, ctxt_turns) * 2
  previous = [
    message
    for message in messages
    if int(message["utterance_index"]) < therapist_utterance_index
  ]
  if limit <= 0:
    return []
  return previous[-limit:]


def render_value(value: Any) -> str:
  if isinstance(value, (dict, list)):
    return json.dumps(value, ensure_ascii=True, indent=2)
  if value is None:
    return "null"
  return str(value)


def append_field(lines: list[str], label: str, value: Any) -> None:
  rendered = render_value(value)
  if "\n" in rendered:
    indented = "\n".join(f"  {line}" for line in rendered.splitlines())
    lines.append(f"{label}:\n{indented}")
  else:
    lines.append(f"{label}: {rendered}")


def format_reasoning_traces_txt(
  *,
  stage: str,
  substage: str | None,
  therapist_utterance_index: int,
  traces: list[dict[str, Any]],
) -> str:
  lines = [
    "Target:",
    f"  Therapist utterance index: {therapist_utterance_index}",
    f"  Stage: {stage}",
    f"  Substage: {substage or 'unknown'}",
    "",
  ]

  for trace_index, trace in enumerate(traces, start=1):
    lines.append(f"=== Trace {trace_index} ===")
    lines.append(f"Step: {trace.get('step')}")
    lines.append("Output Fields:")
    output = trace.get("output")
    if isinstance(output, dict) and output:
      for key, value in output.items():
        label = str(key).replace("_", " ").title()
        append_field(lines, label, value)
    elif isinstance(output, dict):
      lines.append("(empty object)")
    else:
      append_field(lines, "Message", output)
    lines.append("")

  return "\n".join(lines).rstrip() + "\n"


def make_turn_key(filename: str, therapist_utterance_index: int) -> str:
  return f"{filename}#therapist-{therapist_utterance_index:04d}"


def extract_evaluable_turns(
  data: dict[str, Any],
  filename: str,
  ctxt_turns: int,
  knowledge: TraceKnowledge,
) -> tuple[list[EvaluableReasoning], list[dict[str, Any]]]:
  messages = normalized_messages(data)
  if not messages:
    return [], [{"reason": "missing messages list"}]

  traces_by_index = therapist_reasoning_traces(data)
  if not traces_by_index:
    return [], [{"reason": "missing therapist reasoning_traces"}]

  turns: list[EvaluableReasoning] = []
  skipped: list[dict[str, Any]] = []
  messages_by_index = {int(message["utterance_index"]): message for message in messages}

  for utterance_index, reasoning_trace in sorted(traces_by_index.items()):
    message = messages_by_index.get(utterance_index)
    if not isinstance(message, dict):
      skipped.append({"utterance_index": utterance_index, "reason": "missing therapist message"})
      continue
    if message.get("role") != "therapist":
      skipped.append({"utterance_index": utterance_index, "reason": "trace is not attached to therapist message"})
      continue

    stage = str(message.get("stage") or "unknown")
    substage = str(message.get("substage") or "") or None
    filtered_traces: list[dict[str, Any]] = []
    for trace in reasoning_trace.get("traces") or []:
      if not isinstance(trace, dict):
        continue
      step = trace.get("step")
      if not isinstance(step, str) or not step:
        continue
      fields = trace_fields_for_step(knowledge, stage, step)
      if fields is TRACE_NOT_ALLOWED:
        continue
      filtered_traces.append(
        {
          "step": step,
          "output": compact_trace_output(step, trace.get("output"), fields),  # type: ignore[arg-type]
        }
      )

    if len(filtered_traces) < knowledge.min_traces_per_sample:
      skipped.append(
        {
          "utterance_index": utterance_index,
          "stage": stage,
          "reason": "insufficient included traces",
          "included_trace_count": len(filtered_traces),
          "min_traces_per_sample": knowledge.min_traces_per_sample,
        }
      )
      continue

    context_messages = previous_context_messages(messages, utterance_index, ctxt_turns)
    turns.append(
      EvaluableReasoning(
        turn_key=make_turn_key(filename, utterance_index),
        stage=stage,
        substage=substage,
        therapist_utterance_index=utterance_index,
        therapist_utterance=str(message.get("content") or ""),
        context=format_context(context_messages),
        context_utterance_count=len(context_messages),
        traces=filtered_traces,
        reasoning_traces_txt=format_reasoning_traces_txt(
          stage=stage,
          substage=substage,
          therapist_utterance_index=utterance_index,
          traces=filtered_traces,
        ),
      )
    )

  return turns, skipped


def sample_turns_by_stage(
  turns: list[EvaluableReasoning],
  n_traces: int,
  stage_order: tuple[str, ...],
) -> tuple[list[EvaluableReasoning], dict[str, int]]:
  candidates_by_stage: dict[str, list[EvaluableReasoning]] = {}
  for turn in turns:
    candidates_by_stage.setdefault(turn.stage, []).append(turn)

  ordered_stages = [
    *[stage for stage in stage_order if stage in candidates_by_stage],
    *sorted(stage for stage in candidates_by_stage if stage not in stage_order),
  ]
  selected: list[EvaluableReasoning] = []
  counts: dict[str, int] = {}
  for stage in ordered_stages:
    candidates = candidates_by_stage[stage]
    sample_size = min(max(0, n_traces), len(candidates))
    sampled = random.sample(candidates, sample_size) if sample_size else []
    selected.extend(sampled)
    counts[stage] = len(sampled)
  return selected, counts


def load_prompt_template(prompt_file: str) -> str:
  path = PROMPT_ROOT / prompt_file
  with path.open("r", encoding="utf-8") as f:
    return f.read()


def render_template(template: str, values: dict[str, str]) -> str:
  rendered = template
  for key, value in values.items():
    rendered = rendered.replace("{" + key + "}", value)
  return rendered


def render_metric_prompt(template: str, turn: EvaluableReasoning) -> str:
  return render_template(
    template,
    {
      "context": turn.context,
      "reasoning_traces": turn.reasoning_traces_txt,
      "therapist_utterance": turn.therapist_utterance,
    },
  )


def make_chat_body(prompt: str, model: str, max_iter: int) -> dict[str, Any]:
  return {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "n": max_iter,
  }


def safe_custom_id_part(value: str) -> str:
  safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
  return safe[:80] or "item"


def make_custom_id(index: int, filename: str, utterance_index: int, metric: str) -> str:
  name = safe_custom_id_part(Path(filename).stem)
  metric_part = safe_custom_id_part(metric)
  return f"rtraces:{index:08d}:{name}:u{utterance_index:04d}:{metric_part}"


def serialize_turn(turn: EvaluableReasoning) -> dict[str, Any]:
  return {
    "turn_key": turn.turn_key,
    "stage": turn.stage,
    "substage": turn.substage,
    "therapist_utterance_index": turn.therapist_utterance_index,
    "therapist_utterance": turn.therapist_utterance,
    "context_utterance_count": turn.context_utterance_count,
    "context": turn.context,
    "trace_count": len(turn.traces),
    "traces": turn.traces,
    "reasoning_traces_txt": turn.reasoning_traces_txt,
  }


def build_requests(
  files: list[Path],
  output_dir: Path,
  model: str,
  max_iter: int,
  n_traces: int,
  ctxt_turns: int,
  *,
  knowledge: TraceKnowledge | None = None,
) -> tuple[list[BatchRequest], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  active_knowledge = knowledge or load_trace_knowledge()
  templates = {key: load_prompt_template(spec.prompt_file) for key, spec in METRIC_SPECS.items()}
  requests: list[BatchRequest] = []
  sessions: list[dict[str, Any]] = []
  skipped_files: list[dict[str, Any]] = []
  request_index = 0
  sampling = {
    "n_traces_per_stage": n_traces,
    "ctxt_turns": ctxt_turns,
    "context_utterance_limit": ctxt_turns * 2,
    "min_traces_per_sample": active_knowledge.min_traces_per_sample,
    "source_evaluable_turns": 0,
    "selected_turns": 0,
  }

  for path in files:
    try:
      payload = read_json(path)
    except Exception as exc:
      skipped_files.append({"file": str(path), "reason": f"could not read JSON: {exc}"})
      continue

    output_file = Path("sessions") / path.name
    if not isinstance(payload, dict):
      reason = "top-level JSON is not an object"
      skipped_files.append({"file": str(path), "reason": reason})
      sessions.append(
        {
          "input_file": str(path),
          "input_filename": path.name,
          "model": model,
          "max_iter": max_iter,
          "n_traces": n_traces,
          "ctxt_turns": ctxt_turns,
          "evaluated_metrics": list(METRIC_SPECS),
          "evaluated_turns": [],
          "metrics": session_metrics([]),
          "skipped_turns": [{"reason": reason}],
        }
      )
      continue

    turns, skipped_turns = extract_evaluable_turns(payload, path.name, ctxt_turns, active_knowledge)
    selected_turns, selected_counts = sample_turns_by_stage(turns, n_traces, active_knowledge.stage_order)
    if not selected_turns:
      skipped_files.append({"file": str(path), "reason": "no sampled therapist reasoning turns"})

    sampling["source_evaluable_turns"] += len(turns)
    sampling["selected_turns"] += len(selected_turns)

    session = {
      "input_file": str(path),
      "input_filename": path.name,
      "model": model,
      "max_iter": max_iter,
      "n_traces": n_traces,
      "ctxt_turns": ctxt_turns,
      "trace_knowledge": str(TRACE_KNOWLEDGE_PATH),
      "evaluated_metrics": list(METRIC_SPECS),
      "evaluated_turns": [],
      "metrics": session_metrics([]),
      "skipped_turns": skipped_turns,
      "sampling": {
        "n_traces_per_stage": n_traces,
        "source_evaluable_turn_count": len(turns),
        "selected_turn_count": len(selected_turns),
        "selected_counts_by_stage": selected_counts,
        "min_traces_per_sample": active_knowledge.min_traces_per_sample,
      },
    }

    for turn in selected_turns:
      turn_payload = serialize_turn(turn)
      turn_payload["batch_custom_ids"] = {}
      for metric_key, spec in METRIC_SPECS.items():
        prompt = render_metric_prompt(templates[metric_key], turn)
        custom_id = make_custom_id(request_index, path.name, turn.therapist_utterance_index, metric_key)
        request_index += 1
        turn_payload["batch_custom_ids"][metric_key] = custom_id
        requests.append(
          BatchRequest(
            custom_id=custom_id,
            metric=metric_key,
            input_file=path.name,
            output_file=output_file.as_posix(),
            turn_key=turn.turn_key,
            stage=turn.stage,
            therapist_utterance_index=turn.therapist_utterance_index,
            min_score=spec.min_score,
            max_score=spec.max_score,
            body=make_chat_body(prompt, model, max_iter),
          )
        )
      session["evaluated_turns"].append(turn_payload)

    sessions.append(session)

  return requests, sessions, skipped_files, sampling


def request_map_from_specs(specs: list[BatchRequest]) -> dict[str, dict[str, Any]]:
  return {spec.custom_id: spec.map_entry() for spec in specs}


def jsonl_line_text(spec: BatchRequest) -> str:
  return json.dumps(spec.jsonl_line(), ensure_ascii=True)


def write_jsonl(path: Path, specs: list[BatchRequest]) -> None:
  lines = [jsonl_line_text(spec) for spec in specs]
  atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def split_request_shards(
  requests: list[BatchRequest],
  *,
  max_requests: int = MAX_BATCH_REQUESTS,
  max_bytes: int = MAX_BATCH_FILE_BYTES,
) -> list[list[BatchRequest]]:
  shards: list[list[BatchRequest]] = []
  current: list[BatchRequest] = []
  current_bytes = 0

  for request in requests:
    line_bytes = len(jsonl_line_text(request).encode("utf-8")) + 1
    if line_bytes > max_bytes:
      raise RTracesEvaluationError(
        f"Single batch request {request.custom_id} is {line_bytes} bytes; maximum input file size is {max_bytes}."
      )
    would_exceed_count = len(current) >= max_requests
    would_exceed_bytes = bool(current) and current_bytes + line_bytes > max_bytes
    if would_exceed_count or would_exceed_bytes:
      shards.append(current)
      current = []
      current_bytes = 0
    current.append(request)
    current_bytes += line_bytes

  if current:
    shards.append(current)
  return shards


def response_usage_cost(
  body: dict[str, Any],
  input_cost_per_1m: float,
  output_cost_per_1m: float,
) -> float:
  usage = body.get("usage") or {}
  prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
  completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
  return (
    (float(prompt_tokens) * input_cost_per_1m)
    + (float(completion_tokens) * output_cost_per_1m)
  ) / 1_000_000


def clamp_score(value: float, min_score: float, max_score: float) -> float | None:
  if min_score <= value <= max_score:
    return value
  return None


def parse_rating(text: str, min_score: float, max_score: float) -> tuple[float, str | None]:
  matches = FINAL_RATING_RE.findall(text)
  if not matches:
    return 0.0, "missing Rating line"
  raw_value = matches[-1]
  try:
    value = float(raw_value)
  except ValueError:
    return 0.0, f"invalid numeric rating: {raw_value!r}"
  clamped = clamp_score(value, min_score, max_score)
  if clamped is None:
    return 0.0, f"rating {value} outside [{min_score}, {max_score}]"
  return clamped, None


def parse_batch_output_lines(path: Path) -> dict[str, dict[str, Any]]:
  results: dict[str, dict[str, Any]] = {}
  with path.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
      if not line.strip():
        continue
      try:
        payload = json.loads(line)
      except json.JSONDecodeError as exc:
        raise RTracesEvaluationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
      custom_id = payload.get("custom_id")
      if not isinstance(custom_id, str) or not custom_id:
        raise RTracesEvaluationError(f"Batch output line is missing custom_id at {path}:{line_number}")
      if custom_id in results:
        raise RTracesEvaluationError(f"Duplicate custom_id in batch output: {custom_id}")
      results[custom_id] = payload
  return results


def parse_batch_output_ref(custom_id: str) -> BatchOutputRef:
  parts = custom_id.split(":")
  if len(parts) != 5 or parts[0] != "rtraces":
    raise RTracesEvaluationError(f"Unsupported rtraces batch custom_id: {custom_id}")
  try:
    request_index = int(parts[1])
    utterance_index = int(parts[3].removeprefix("u"))
  except ValueError as exc:
    raise RTracesEvaluationError(f"Invalid rtraces batch custom_id: {custom_id}") from exc
  metric = parts[4]
  if metric not in METRIC_SPECS:
    raise RTracesEvaluationError(f"Unknown rtraces metric in custom_id {custom_id}: {metric}")
  return BatchOutputRef(
    custom_id=custom_id,
    request_index=request_index,
    filename_stem=parts[2],
    utterance_index=utterance_index,
    metric=metric,
  )


def batch_output_turn_groups(
  output_jsonl: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
  raw_outputs = parse_batch_output_lines(output_jsonl)
  groups: dict[tuple[str, int], dict[str, Any]] = {}
  report = {
    "source_batch_output_jsonl": str(output_jsonl),
    "observed_response_count": len(raw_outputs),
    "observed_turn_count": 0,
    "inferred_missing_custom_ids": [],
  }

  for custom_id in sorted(raw_outputs):
    ref = parse_batch_output_ref(custom_id)
    key = (ref.filename_stem, ref.utterance_index)
    group = groups.setdefault(
      key,
      {
        "filename_stem": ref.filename_stem,
        "utterance_index": ref.utterance_index,
        "base_index_candidates": [],
        "custom_ids": {},
      },
    )
    custom_ids = group["custom_ids"]
    if ref.metric in custom_ids:
      raise RTracesEvaluationError(
        f"Duplicate metric {ref.metric!r} for {ref.filename_stem} utterance {ref.utterance_index}"
      )
    custom_ids[ref.metric] = ref.custom_id
    group["base_index_candidates"].append(ref.request_index - METRIC_INDEX[ref.metric])

  for group in groups.values():
    base_indexes = set(int(value) for value in group.pop("base_index_candidates"))
    if len(base_indexes) != 1:
      raise RTracesEvaluationError(
        "Inconsistent request indexes for "
        f"{group['filename_stem']} utterance {group['utterance_index']}: {sorted(base_indexes)}"
      )
    base_index = base_indexes.pop()
    group["base_index"] = base_index
    custom_ids = group["custom_ids"]
    filename = f"{group['filename_stem']}.json"
    for metric in METRIC_ORDER:
      if metric not in custom_ids:
        inferred = make_custom_id(
          base_index + METRIC_INDEX[metric],
          filename,
          int(group["utterance_index"]),
          metric,
        )
        custom_ids[metric] = inferred
        report["inferred_missing_custom_ids"].append(inferred)

  report["observed_turn_count"] = len(groups)
  return groups, report


def build_requests_from_batch_output(
  files: list[Path],
  output_dir: Path,
  output_jsonl: Path,
  model: str,
  max_iter: int,
  ctxt_turns: int,
  *,
  knowledge: TraceKnowledge | None = None,
) -> tuple[list[BatchRequest], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
  active_knowledge = knowledge or load_trace_knowledge()
  templates = {key: load_prompt_template(spec.prompt_file) for key, spec in METRIC_SPECS.items()}
  turn_groups, batch_selection = batch_output_turn_groups(output_jsonl)
  files_by_stem = {safe_custom_id_part(path.stem): path for path in files}
  missing_files = sorted(stem for stem, _ in turn_groups if stem not in files_by_stem)
  if missing_files:
    raise RTracesEvaluationError(
      "Batch output references input files not present in input_dir: "
      + ", ".join(f"{stem}.json" for stem in missing_files[:10])
    )

  requests: list[BatchRequest] = []
  sessions: list[dict[str, Any]] = []
  skipped_files: list[dict[str, Any]] = []
  selected_turn_count = 0
  source_evaluable_turn_count = 0
  selected_counts_by_stage_total: dict[str, int] = {}

  for path in files:
    try:
      payload = read_json(path)
    except Exception as exc:
      skipped_files.append({"file": str(path), "reason": f"could not read JSON: {exc}"})
      continue

    output_file = Path("sessions") / path.name
    if not isinstance(payload, dict):
      skipped_files.append({"file": str(path), "reason": "top-level JSON is not an object"})
      continue

    turns, skipped_turns = extract_evaluable_turns(payload, path.name, ctxt_turns, active_knowledge)
    source_evaluable_turn_count += len(turns)
    turns_by_index = {turn.therapist_utterance_index: turn for turn in turns}
    selected_groups = [
      group
      for (filename_stem, _), group in turn_groups.items()
      if filename_stem == safe_custom_id_part(path.stem)
    ]
    selected_groups.sort(key=lambda group: int(group["base_index"]))

    missing_turns = [
      int(group["utterance_index"])
      for group in selected_groups
      if int(group["utterance_index"]) not in turns_by_index
    ]
    if missing_turns:
      raise RTracesEvaluationError(
        f"Batch output references unavailable evaluable turns in {path.name}: {missing_turns[:10]}"
      )

    session = {
      "input_file": str(path),
      "input_filename": path.name,
      "model": model,
      "max_iter": max_iter,
      "n_traces": None,
      "ctxt_turns": ctxt_turns,
      "trace_knowledge": str(TRACE_KNOWLEDGE_PATH),
      "evaluated_metrics": list(METRIC_SPECS),
      "evaluated_turns": [],
      "metrics": session_metrics([]),
      "skipped_turns": skipped_turns,
      "sampling": {
        "selection_source": "batch_output_jsonl",
        "source_batch_output_jsonl": str(output_jsonl),
        "source_evaluable_turn_count": len(turns),
        "selected_turn_count": len(selected_groups),
        "selected_counts_by_stage": {},
        "min_traces_per_sample": active_knowledge.min_traces_per_sample,
      },
    }

    selected_counts_by_stage: dict[str, int] = {}
    for group in selected_groups:
      turn = turns_by_index[int(group["utterance_index"])]
      selected_turn_count += 1
      selected_counts_by_stage[turn.stage] = selected_counts_by_stage.get(turn.stage, 0) + 1
      selected_counts_by_stage_total[turn.stage] = selected_counts_by_stage_total.get(turn.stage, 0) + 1
      turn_payload = serialize_turn(turn)
      turn_payload["batch_custom_ids"] = {}
      for metric_key, spec in METRIC_SPECS.items():
        prompt = render_metric_prompt(templates[metric_key], turn)
        custom_id = str(group["custom_ids"][metric_key])
        turn_payload["batch_custom_ids"][metric_key] = custom_id
        requests.append(
          BatchRequest(
            custom_id=custom_id,
            metric=metric_key,
            input_file=path.name,
            output_file=output_file.as_posix(),
            turn_key=turn.turn_key,
            stage=turn.stage,
            therapist_utterance_index=turn.therapist_utterance_index,
            min_score=spec.min_score,
            max_score=spec.max_score,
            body=make_chat_body(prompt, model, max_iter),
          )
        )
      session["evaluated_turns"].append(turn_payload)

    session["sampling"]["selected_counts_by_stage"] = selected_counts_by_stage
    if not selected_groups:
      skipped_files.append({"file": str(path), "reason": "no turns referenced by batch output"})
    sessions.append(session)

  sampling = {
    "selection_source": "batch_output_jsonl",
    "source_batch_output_jsonl": str(output_jsonl),
    "ctxt_turns": ctxt_turns,
    "context_utterance_limit": ctxt_turns * 2,
    "min_traces_per_sample": active_knowledge.min_traces_per_sample,
    "source_evaluable_turns": source_evaluable_turn_count,
    "selected_turns": selected_turn_count,
    "selected_counts_by_stage": selected_counts_by_stage_total,
  }
  return requests, sessions, skipped_files, sampling, batch_selection


def empty_judge_result(score: float = 0.0, *, error: str | None = None) -> dict[str, Any]:
  result = {
    "score": score,
    "cost": 0.0,
    "choices": [],
  }
  if error:
    result["error"] = error
  return result


def parse_batch_outputs(
  output_jsonl: Path,
  request_map: dict[str, dict[str, Any]],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  raw_outputs = parse_batch_output_lines(output_jsonl)
  parsed: dict[str, dict[str, Any]] = {}
  report = {
    "request_count": len(request_map),
    "response_count": len(raw_outputs),
    "missing_custom_ids": [],
    "unexpected_custom_ids": [],
    "parse_errors": [],
    "choice_count_warnings": [],
    "batch_errors": [],
  }

  for custom_id in sorted(raw_outputs):
    if custom_id not in request_map:
      report["unexpected_custom_ids"].append(custom_id)

  for custom_id, entry in sorted(request_map.items()):
    payload = raw_outputs.get(custom_id)
    if payload is None:
      report["missing_custom_ids"].append(custom_id)
      parsed[custom_id] = empty_judge_result(error="missing batch output")
      continue

    if payload.get("error"):
      report["batch_errors"].append({"custom_id": custom_id, "error": payload.get("error")})
      parsed[custom_id] = empty_judge_result(error=f"batch error: {payload.get('error')}")
      continue

    response = payload.get("response") or {}
    if not isinstance(response, dict):
      report["batch_errors"].append({"custom_id": custom_id, "error": "response is not an object"})
      parsed[custom_id] = empty_judge_result(error="response is not an object")
      continue

    body = response.get("body") or {}
    if not isinstance(body, dict):
      report["batch_errors"].append({"custom_id": custom_id, "error": "response body is not an object"})
      parsed[custom_id] = empty_judge_result(error="response body is not an object")
      continue

    body_error = body.get("error")
    if body_error:
      report["batch_errors"].append({"custom_id": custom_id, "error": body_error})
      parsed[custom_id] = empty_judge_result(error=f"response body error: {body_error}")
      continue

    choices = body.get("choices") or []
    if not isinstance(choices, list):
      choices = []
    if len(choices) != max_iter:
      report["choice_count_warnings"].append(
        {"custom_id": custom_id, "expected": max_iter, "received": len(choices)}
      )

    min_score = float(entry["min_score"])
    max_score = float(entry["max_score"])
    parsed_choices = []
    for choice_index, choice in enumerate(choices):
      message = get_field(choice, "message", {})
      raw_text = str(get_field(message, "content", "") or "")
      parsed_score, parse_error = parse_rating(raw_text, min_score, max_score)
      if parse_error:
        report["parse_errors"].append(
          {"custom_id": custom_id, "choice_index": choice_index, "error": parse_error}
        )
      parsed_choices.append(
        {
          "raw_text": raw_text,
          "parsed_score": parsed_score,
          "parse_error": parse_error,
        }
      )

    avg_score = mean(float(choice["parsed_score"]) for choice in parsed_choices)
    parsed[custom_id] = {
      "score": avg_score if avg_score is not None else 0.0,
      "cost": response_usage_cost(body, input_cost_per_1m, output_cost_per_1m),
      "choices": parsed_choices,
    }

  return parsed, report


def score_turn(turn: dict[str, Any], parsed_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
  custom_ids = turn.get("batch_custom_ids") or {}
  judge_scores: dict[str, dict[str, Any]] = {}
  metric_scores: dict[str, float] = {}
  cost = 0.0

  for metric_key in METRIC_SPECS:
    result = parsed_results.get(
      str(custom_ids.get(metric_key) or ""),
      empty_judge_result(error=f"missing {metric_key} result"),
    )
    judge_scores[metric_key] = result
    metric_scores[metric_key] = float(result.get("score") or 0.0)
    cost += float(result.get("cost") or 0.0)

  scored = dict(turn)
  scored.update(metric_scores)
  scored["harmonic_mean"] = harmonic_mean([metric_scores[key] for key in METRIC_SPECS])
  scored["cost"] = cost
  scored["judge_scores"] = judge_scores
  scored["evaluated_metrics"] = list(METRIC_SPECS)
  return scored


def attach_judgements(
  sessions: list[dict[str, Any]],
  parsed_results: dict[str, dict[str, Any]],
  artifact_dir: Path | None,
) -> list[dict[str, Any]]:
  scored_sessions = []
  for session in sessions:
    scored_turns = []
    for turn in session.get("evaluated_turns", []):
      if isinstance(turn, dict):
        scored_turns.append(score_turn(turn, parsed_results))

    scored_session = dict(session)
    scored_session["evaluated_turns"] = scored_turns
    scored_session["metrics"] = session_metrics(scored_turns)
    if artifact_dir is not None:
      scored_session["batch_artifacts"] = str(artifact_dir)
    scored_sessions.append(scored_session)
  return scored_sessions


def mean_numeric_field(turns: list[dict[str, Any]], field: str) -> float | None:
  values = [
    float(turn[field])
    for turn in turns
    if turn.get(field) is not None
  ]
  return mean(values)


def session_metrics(turns: list[dict[str, Any]]) -> dict[str, Any]:
  if not turns:
    return {
      "turn_count": 0,
      "mean_harmonic_score": None,
      "mean_judge_scores": {metric: None for metric in METRIC_SPECS},
      "cost": 0.0,
    }

  return {
    "turn_count": len(turns),
    "mean_harmonic_score": mean_numeric_field(turns, "harmonic_mean"),
    "mean_judge_scores": {
      metric: mean_numeric_field(turns, metric)
      for metric in METRIC_SPECS
    },
    "cost": sum(float(turn.get("cost") or 0.0) for turn in turns),
  }


def summary_from_sessions(
  input_dir: Path,
  output_dir: Path,
  files_seen: int,
  session_results: list[dict[str, Any]],
  skipped_files: list[dict[str, Any]],
  artifact_dir: Path | None,
  parse_report: dict[str, Any] | None,
  *,
  request_count: int = 0,
  shard_count: int = 0,
) -> dict[str, Any]:
  all_turns = [
    turn
    for session in session_results
    for turn in session.get("evaluated_turns", [])
  ]
  skipped_turns = [
    {"file": session.get("input_file"), **skipped}
    for session in session_results
    for skipped in session.get("skipped_turns", [])
  ]

  return {
    "version": 1,
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "batch_artifacts": str(artifact_dir) if artifact_dir is not None else None,
    "trace_knowledge": str(TRACE_KNOWLEDGE_PATH),
    "total_files_seen": files_seen,
    "total_files_evaluated": sum(1 for session in session_results if session.get("evaluated_turns")),
    "total_turns_evaluated": len(all_turns),
    "mean_judge_scores": {
      metric: mean_numeric_field(all_turns, metric)
      for metric in METRIC_SPECS
    },
    "mean_harmonic_score": mean_numeric_field(all_turns, "harmonic_mean"),
    "shard_count": shard_count,
    "total_requests": request_count,
    "total_estimated_cost": sum(float(turn.get("cost") or 0.0) for turn in all_turns),
    "skipped_files": skipped_files,
    "skipped_turns": skipped_turns,
    "parse_report": parse_report,
  }


def iter_json_files(input_dir: Path, recursive: bool) -> list[Path]:
  pattern = "**/*.json" if recursive else "*.json"
  return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def write_session_outputs(output_dir: Path, sessions: list[dict[str, Any]]) -> None:
  for session in sessions:
    filename = str(session.get("input_filename") or Path(str(session.get("input_file"))).name)
    atomic_write_json(output_dir / "sessions" / filename, session)


def create_run_artifacts(
  output_dir: Path,
  requests: list[BatchRequest],
  sessions: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
  run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
  artifact_dir = output_dir / "_batch_artifacts" / run_id
  artifact_dir.mkdir(parents=True, exist_ok=False)

  request_map = request_map_from_specs(requests)
  request_map_path = artifact_dir / "request_map.json"
  prepared_sessions_path = artifact_dir / "prepared_sessions.json"
  manifest_path = artifact_dir / "manifest.json"
  atomic_write_json(request_map_path, request_map)
  atomic_write_json(prepared_sessions_path, sessions)

  shard_refs = []
  for shard_index, shard_requests in enumerate(split_request_shards(requests), start=1):
    shard_dir = artifact_dir / "shards" / f"shard_{shard_index:05d}"
    shard_dir.mkdir(parents=True, exist_ok=False)
    input_jsonl = shard_dir / "input.jsonl"
    shard_request_map_path = shard_dir / "request_map.json"
    shard_manifest_path = shard_dir / "manifest.json"
    write_jsonl(input_jsonl, shard_requests)
    atomic_write_json(shard_request_map_path, request_map_from_specs(shard_requests))
    shard_manifest = {
      "version": 1,
      "run_id": run_id,
      "shard_index": shard_index,
      "status": "prepared",
      "created_at": utc_now(),
      "updated_at": utc_now(),
      "endpoint": BATCH_ENDPOINT,
      "completion_window": COMPLETION_WINDOW,
      "request_count": len(shard_requests),
      "input_jsonl": str(input_jsonl),
      "request_map": str(shard_request_map_path),
    }
    atomic_write_json(shard_manifest_path, shard_manifest)
    shard_refs.append(
      {
        "shard_index": shard_index,
        "manifest": str(shard_manifest_path),
        "request_count": len(shard_requests),
        "status": "prepared",
      }
    )

  manifest = {
    "version": 1,
    "run_id": run_id,
    "status": "prepared",
    "created_at": utc_now(),
    "updated_at": utc_now(),
    "endpoint": BATCH_ENDPOINT,
    "completion_window": COMPLETION_WINDOW,
    "request_count": len(requests),
    "shard_count": len(shard_refs),
    "shards": shard_refs,
    "request_map": str(request_map_path),
    "prepared_sessions": str(prepared_sessions_path),
  }
  atomic_write_json(manifest_path, manifest)
  return artifact_dir, manifest


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
  manifest["updated_at"] = utc_now()
  atomic_write_json(manifest_path, manifest)


def save_parent_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
  save_manifest(artifact_dir / "manifest.json", manifest)


def make_openai_client() -> Any:
  if load_dotenv is not None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
  api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
  if not api_key:
    raise RTracesEvaluationError("Missing OpenAI API key. Set GPT_API_KEY or OPENAI_API_KEY.")
  try:
    from openai import OpenAI
  except ImportError as exc:
    raise RTracesEvaluationError("The openai Python package is required to run rtraces batches.") from exc
  return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


def content_response_to_text(response: Any) -> str:
  if hasattr(response, "text"):
    text_value = response.text
    if callable(text_value):
      text_value = text_value()
    if isinstance(text_value, str):
      return text_value
  if hasattr(response, "read"):
    data = response.read()
  elif hasattr(response, "content"):
    data = response.content
  else:
    data = response
  if isinstance(data, bytes):
    return data.decode("utf-8")
  return str(data)


def download_file(client: Any, file_id: str, destination: Path) -> None:
  content = client.files.content(file_id)
  atomic_write_text(destination, content_response_to_text(content))


def upload_and_create_batch(client: Any, shard_dir: Path, manifest: dict[str, Any]) -> None:
  if manifest.get("batch_id"):
    return
  input_jsonl = Path(str(manifest["input_jsonl"]))
  with input_jsonl.open("rb") as f:
    uploaded_file = client.files.create(file=f, purpose="batch")
  uploaded_file_id = get_field(uploaded_file, "id")
  if not uploaded_file_id:
    raise RTracesEvaluationError("OpenAI file upload did not return a file id.")

  manifest["input_file_id"] = uploaded_file_id
  manifest["uploaded_file"] = to_plain(uploaded_file)
  manifest["status"] = "uploaded"
  save_manifest(shard_dir / "manifest.json", manifest)

  batch = client.batches.create(
    input_file_id=uploaded_file_id,
    endpoint=BATCH_ENDPOINT,
    completion_window=COMPLETION_WINDOW,
    metadata={
      "description": "ReasonCBT reasoning trace judge evaluation",
      "shard_index": str(manifest.get("shard_index", "")),
    },
  )
  batch_id = get_field(batch, "id")
  if not batch_id:
    raise RTracesEvaluationError("OpenAI batch creation did not return a batch id.")
  manifest["batch_id"] = batch_id
  manifest["status"] = get_field(batch, "status", "submitted")
  manifest["batch"] = to_plain(batch)
  save_manifest(shard_dir / "manifest.json", manifest)


def download_batch_outputs(client: Any, shard_dir: Path, manifest: dict[str, Any]) -> None:
  output_file_id = manifest.get("output_file_id")
  error_file_id = manifest.get("error_file_id")
  if output_file_id and not manifest.get("output_jsonl"):
    output_path = shard_dir / "output.jsonl"
    download_file(client, str(output_file_id), output_path)
    manifest["output_jsonl"] = str(output_path)
  if error_file_id and not manifest.get("error_jsonl"):
    error_path = shard_dir / "error.jsonl"
    download_file(client, str(error_file_id), error_path)
    manifest["error_jsonl"] = str(error_path)


def poll_until_complete(
  client: Any,
  shard_dir: Path,
  manifest: dict[str, Any],
  poll_interval: float,
  local_timeout_hours: float,
) -> None:
  if manifest.get("output_jsonl"):
    return
  batch_id = str(manifest.get("batch_id") or "")
  if not batch_id:
    raise RTracesEvaluationError("Cannot poll before a batch is created.")

  deadline = time.monotonic() + (local_timeout_hours * 3600)
  while True:
    batch = client.batches.retrieve(batch_id)
    batch_plain = to_plain(batch)
    status = str(get_field(batch, "status", batch_plain.get("status", "")))
    manifest["status"] = status
    manifest["batch"] = batch_plain
    manifest["output_file_id"] = get_field(batch, "output_file_id", batch_plain.get("output_file_id"))
    manifest["error_file_id"] = get_field(batch, "error_file_id", batch_plain.get("error_file_id"))
    manifest["request_counts"] = to_plain(get_field(batch, "request_counts", batch_plain.get("request_counts")))
    save_manifest(shard_dir / "manifest.json", manifest)

    if status in FINAL_STATUSES:
      download_batch_outputs(client, shard_dir, manifest)
      save_manifest(shard_dir / "manifest.json", manifest)
      if status in FAILED_STATUSES:
        raise RTracesEvaluationError(f"Batch shard {manifest.get('shard_index')} ended with status {status}.")
      counts = manifest.get("request_counts") or {}
      failed_count = get_field(counts, "failed", 0) or 0
      if int(failed_count) > 0:
        raise RTracesEvaluationError(
          f"Batch shard {manifest.get('shard_index')} completed with {failed_count} failed requests."
        )
      if not manifest.get("output_jsonl"):
        raise RTracesEvaluationError(f"Batch shard {manifest.get('shard_index')} completed without an output file.")
      return

    if time.monotonic() >= deadline:
      raise RTracesEvaluationError(
        f"Timed out waiting for batch shard {manifest.get('shard_index')} after {local_timeout_hours} hours."
      )
    time.sleep(max(0.0, poll_interval))


def run_openai_batch(
  shard_dir: Path,
  manifest: dict[str, Any],
  poll_interval: float,
  local_timeout_hours: float,
  client: Any | None = None,
) -> dict[str, Any]:
  active_client = client or make_openai_client()
  upload_and_create_batch(active_client, shard_dir, manifest)
  poll_until_complete(active_client, shard_dir, manifest, poll_interval, local_timeout_hours)
  return manifest


def run_openai_batch_shards(
  artifact_dir: Path,
  manifest: dict[str, Any],
  poll_interval: float,
  local_timeout_hours: float,
  client: Any | None = None,
) -> dict[str, Any]:
  active_client = client or make_openai_client()
  shard_refs = manifest.get("shards") or []
  for shard_ref in shard_refs:
    if not isinstance(shard_ref, dict):
      continue
    shard_manifest_path = Path(str(shard_ref["manifest"]))
    shard_dir = shard_manifest_path.parent
    shard_manifest = read_json(shard_manifest_path)
    if not isinstance(shard_manifest, dict):
      raise RTracesEvaluationError(f"Shard manifest must be a JSON object: {shard_manifest_path}")
    run_openai_batch(shard_dir, shard_manifest, poll_interval, local_timeout_hours, client=active_client)
    shard_manifest = read_json(shard_manifest_path)
    shard_ref.update(
      {
        "status": shard_manifest.get("status"),
        "batch_id": shard_manifest.get("batch_id"),
        "output_jsonl": shard_manifest.get("output_jsonl"),
        "error_jsonl": shard_manifest.get("error_jsonl"),
      }
    )
    manifest["status"] = "running" if shard_manifest.get("status") not in FINAL_STATUSES else "shard_completed"
    save_parent_manifest(artifact_dir, manifest)

  manifest["status"] = "batches_completed"
  save_parent_manifest(artifact_dir, manifest)
  return manifest


def load_request_map(path: Path) -> dict[str, dict[str, Any]]:
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise RTracesEvaluationError(f"Request map must be a JSON object: {path}")
  return {
    str(key): value
    for key, value in payload.items()
    if isinstance(value, dict)
  }


def merge_parse_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
  merged = {
    "request_count": 0,
    "response_count": 0,
    "missing_custom_ids": [],
    "unexpected_custom_ids": [],
    "parse_errors": [],
    "choice_count_warnings": [],
    "batch_errors": [],
    "shards": [],
  }
  for report in reports:
    merged["request_count"] += int(report.get("request_count") or 0)
    merged["response_count"] += int(report.get("response_count") or 0)
    for key in ("missing_custom_ids", "unexpected_custom_ids", "parse_errors", "choice_count_warnings", "batch_errors"):
      items = report.get(key) or []
      if isinstance(items, list):
        merged[key].extend(items)
    merged["shards"].append(
      {
        "shard_index": report.get("shard_index"),
        "source_batch_jsonl": report.get("source_batch_jsonl"),
        "source_request_map": report.get("source_request_map"),
        "request_count": report.get("request_count"),
        "response_count": report.get("response_count"),
      }
    )
  return merged


def parse_all_shard_outputs(
  manifest: dict[str, Any],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  merged_results: dict[str, dict[str, Any]] = {}
  reports: list[dict[str, Any]] = []

  for shard_ref in manifest.get("shards") or []:
    if not isinstance(shard_ref, dict):
      continue
    shard_manifest_path = Path(str(shard_ref["manifest"]))
    shard_manifest = read_json(shard_manifest_path)
    if not isinstance(shard_manifest, dict):
      raise RTracesEvaluationError(f"Shard manifest must be a JSON object: {shard_manifest_path}")
    output_jsonl = shard_manifest.get("output_jsonl")
    if not output_jsonl:
      raise RTracesEvaluationError(f"Shard {shard_manifest.get('shard_index')} has no output_jsonl.")
    request_map_path = Path(str(shard_manifest["request_map"]))
    request_map = load_request_map(request_map_path)
    parsed, report = parse_batch_outputs(
      Path(str(output_jsonl)),
      request_map,
      max_iter,
      input_cost_per_1m,
      output_cost_per_1m,
    )
    for custom_id, result in parsed.items():
      if custom_id in merged_results:
        raise RTracesEvaluationError(f"Duplicate parsed custom_id across shards: {custom_id}")
      merged_results[custom_id] = result
    report["shard_index"] = shard_manifest.get("shard_index")
    report["source_batch_jsonl"] = str(output_jsonl)
    report["source_request_map"] = str(request_map_path)
    reports.append(report)

  return merged_results, merge_parse_reports(reports)


def run_rtraces_batch(
  input_dir: str | Path,
  output_dir: str | Path,
  max_iter: int = DEFAULT_MAX_ITER,
  *,
  model: str = MODEL_NAME,
  recursive: bool = False,
  n_traces: int,
  ctxt_turns: int = DEFAULT_CONTEXT_TURNS,
  poll_interval: float = DEFAULT_POLL_INTERVAL,
  local_timeout_hours: float = DEFAULT_LOCAL_TIMEOUT_HOURS,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
  prepare_only: bool = False,
  batch_output_jsonl: str | Path | None = None,
  client: Any | None = None,
) -> dict[str, Any]:
  if max_iter < 1:
    raise RTracesEvaluationError(f"max_iter must be positive, got {max_iter}")
  if n_traces < 1:
    raise RTracesEvaluationError(f"n_traces must be positive, got {n_traces}")
  if ctxt_turns < 0:
    raise RTracesEvaluationError(f"ctxt_turns must be non-negative, got {ctxt_turns}")

  input_path = Path(input_dir).resolve()
  output_path = Path(output_dir).resolve()
  if not input_path.is_dir():
    raise RTracesEvaluationError(f"input_dir does not exist or is not a directory: {input_path}")

  knowledge = load_trace_knowledge()
  files = iter_json_files(input_path, recursive)
  output_path.mkdir(parents=True, exist_ok=True)
  (output_path / "sessions").mkdir(parents=True, exist_ok=True)

  batch_selection_report: dict[str, Any] | None = None
  if batch_output_jsonl is None:
    requests, prepared_sessions, skipped_files, sampling = build_requests(
      files,
      output_path,
      model,
      max_iter,
      n_traces,
      ctxt_turns,
      knowledge=knowledge,
    )
  else:
    if prepare_only:
      raise RTracesEvaluationError("--prepare-only cannot be combined with --batch-output-jsonl")
    batch_output_path = Path(batch_output_jsonl).resolve()
    if not batch_output_path.is_file():
      raise RTracesEvaluationError(f"batch output JSONL does not exist: {batch_output_path}")
    requests, prepared_sessions, skipped_files, sampling, batch_selection_report = build_requests_from_batch_output(
      files,
      output_path,
      batch_output_path,
      model,
      max_iter,
      ctxt_turns,
      knowledge=knowledge,
    )

  artifact_dir, manifest = create_run_artifacts(output_path, requests, prepared_sessions)
  manifest["model"] = model
  manifest["max_iter"] = max_iter
  manifest["n_traces"] = n_traces
  manifest["ctxt_turns"] = ctxt_turns
  manifest["evaluated_metrics"] = list(METRIC_SPECS)
  manifest["trace_knowledge"] = str(TRACE_KNOWLEDGE_PATH)
  manifest["sampling"] = sampling
  manifest["input_dir"] = str(input_path)
  manifest["output_dir"] = str(output_path)
  manifest["files_seen"] = len(files)
  manifest["recursive"] = recursive
  if batch_selection_report is not None:
    manifest["batch_output_jsonl"] = str(Path(batch_output_jsonl).resolve())
    manifest["batch_selection_report"] = batch_selection_report
  save_parent_manifest(artifact_dir, manifest)

  if prepare_only:
    return manifest

  parse_report: dict[str, Any] | None = None
  if requests:
    try:
      if batch_output_jsonl is None:
        run_openai_batch_shards(artifact_dir, manifest, poll_interval, local_timeout_hours, client=client)
        manifest = read_json(artifact_dir / "manifest.json")
        parsed_results, parse_report = parse_all_shard_outputs(
          manifest,
          max_iter,
          input_cost_per_1m,
          output_cost_per_1m,
        )
      else:
        request_map = request_map_from_specs(requests)
        parsed_results, parse_report = parse_batch_outputs(
          Path(batch_output_jsonl).resolve(),
          request_map,
          max_iter,
          input_cost_per_1m,
          output_cost_per_1m,
        )
        parse_report["source_batch_jsonl"] = str(Path(batch_output_jsonl).resolve())
        parse_report["source_request_map"] = str(artifact_dir / "request_map.json")
        parse_report["batch_selection_report"] = batch_selection_report
      parse_report_path = artifact_dir / "parse_report.json"
      atomic_write_json(parse_report_path, parse_report)
      manifest["parse_report"] = str(parse_report_path)
      manifest["status"] = "parsed" if batch_output_jsonl is None else "parsed_from_batch_output"
      save_parent_manifest(artifact_dir, manifest)
    except Exception as exc:
      manifest["status"] = "failed"
      manifest["error"] = str(exc)
      atomic_write_json(
        artifact_dir / "failure_report.json",
        {"error": str(exc), "type": type(exc).__name__, "updated_at": utc_now()},
      )
      save_parent_manifest(artifact_dir, manifest)
      if isinstance(exc, RTracesEvaluationError):
        raise
      raise RTracesEvaluationError(str(exc)) from exc
  else:
    parsed_results = {}
    manifest["status"] = "no_requests"
    save_parent_manifest(artifact_dir, manifest)

  scored_sessions = attach_judgements(prepared_sessions, parsed_results, artifact_dir)
  write_session_outputs(output_path, scored_sessions)
  summary = summary_from_sessions(
    input_path,
    output_path,
    len(files),
    scored_sessions,
    skipped_files,
    artifact_dir,
    parse_report,
    request_count=len(requests),
    shard_count=int(manifest.get("shard_count") or 0),
  )
  summary["evaluated_metrics"] = list(METRIC_SPECS)
  summary["sampling"] = sampling
  atomic_write_json(output_path / "summary.json", summary)
  manifest["status"] = "completed"
  manifest["summary"] = str(output_path / "summary.json")
  save_parent_manifest(artifact_dir, manifest)
  return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate therapist reasoning traces with OpenAI Batch API judges."
  )
  parser.add_argument("-i", "--input_dir", type=Path, default=Path("."), help="Directory with simulation JSON files.")
  parser.add_argument("-o", "--output_dir", type=Path, default=Path("."), help="Directory for rtraces outputs.")
  parser.add_argument("-m_iter", "--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Judge samples per metric.")
  parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Judge model. Default: {MODEL_NAME}")
  parser.add_argument("--recursive", action="store_true", help="Scan JSON files recursively under input_dir.")
  parser.add_argument(
    "--n-traces",
    type=int,
    required=True,
    help="Sample up to this many eligible therapist reasoning turns per stage per session.",
  )
  parser.add_argument(
    "-ct",
    "--ctxt-turns",
    type=int,
    default=DEFAULT_CONTEXT_TURNS,
    help="Number of previous dialogue turns to include as context. Default: 5.",
  )
  parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
  parser.add_argument("--local-timeout-hours", type=float, default=DEFAULT_LOCAL_TIMEOUT_HOURS)
  parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
  parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
  parser.add_argument(
    "--prepare-only",
    action="store_true",
    help="Write batch artifacts without uploading or polling OpenAI batches.",
  )
  parser.add_argument(
    "--batch-output-jsonl",
    type=Path,
    help="Parse an existing OpenAI batch output JSONL and write session/summary outputs without sending requests.",
  )
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    manifest = run_rtraces_batch(
      args.input_dir,
      args.output_dir,
      args.max_iter,
      model=args.model,
      recursive=args.recursive,
      n_traces=args.n_traces,
      ctxt_turns=args.ctxt_turns,
      poll_interval=args.poll_interval,
      local_timeout_hours=args.local_timeout_hours,
      input_cost_per_1m=args.input_cost_per_1m,
      output_cost_per_1m=args.output_cost_per_1m,
      prepare_only=args.prepare_only,
      batch_output_jsonl=args.batch_output_jsonl,
    )
  except RTracesEvaluationError as exc:
    print(f"Reasoning trace evaluation failed: {exc}", file=os.sys.stderr)
    return 1

  artifact_dir = Path(manifest["output_dir"]) / "_batch_artifacts" / manifest["run_id"]
  if args.prepare_only:
    print("Reasoning trace batch artifacts prepared.")
    print(f"Run artifacts: {artifact_dir}")
    print(f"Request count: {manifest['request_count']}")
    print(f"Shard count: {manifest['shard_count']}")
    sampling = manifest.get("sampling") or {}
    print(
      "Turns: "
      f"{sampling.get('selected_turns')} from "
      f"{sampling.get('source_evaluable_turns')} evaluable turns"
    )
    return 0

  if args.batch_output_jsonl:
    print("Reasoning trace evaluation recovered from batch output.")
  else:
    print("Reasoning trace evaluation completed.")
  print(f"Run artifacts: {artifact_dir}")
  print(f"Session outputs: {Path(manifest['output_dir']) / 'sessions'}")
  print(f"Summary: {Path(manifest['output_dir']) / 'summary.json'}")
  print(f"Shard count: {manifest.get('shard_count')}")
  if args.batch_output_jsonl:
    selection = manifest.get("batch_selection_report") or {}
    print(f"Observed batch responses: {selection.get('observed_response_count')}")
    print(f"Observed batch turns: {selection.get('observed_turn_count')}")
    print(f"Inferred missing responses: {len(selection.get('inferred_missing_custom_ids') or [])}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
