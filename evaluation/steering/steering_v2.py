#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_LOCAL_TIMEOUT_HOURS = 26.0
DEFAULT_INPUT_COST_PER_1M = 1.25
DEFAULT_OUTPUT_COST_PER_1M = 5.0
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024

SCRIPT_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = SCRIPT_ROOT / "prompts_v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONCEPTS_PATH = REPO_ROOT / "data" / "knowledge" / "evaluation_concepts_general.json"

CLASS_ORDER = ("positive", "neutral", "negative")
FINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
FAILED_STATUSES = {"failed", "expired", "cancelled"}
FINAL_RATING_RE = re.compile(r"Rating:\s*\[?\[?\s*(-?\d+(?:\.\d+)?)\s*\]?\]?", re.IGNORECASE)
CURRENT_RESISTANCE_RE = re.compile(r"current resistance\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


class SteeringV2EvaluationError(RuntimeError):
  pass


@dataclass(frozen=True)
class MetricSpec:
  key: str
  prompt_file: str
  min_score: float
  max_score: float


@dataclass(frozen=True)
class EvaluableTurn:
  turn_number: int
  therapist_utterance_index: int
  patient_utterance_index: int
  stage: str | None
  substage: str | None
  therapist_utterance: str
  patient_utterance: str
  expected_resistance_0_1: float
  expected_resistance_interval: str
  resistance_source: str
  resistance_update_reasoning: str


@dataclass(frozen=True)
class ConversationFragment:
  fragment_index: int
  fragment_key: str
  expected_resistance_interval: str
  turns: list[EvaluableTurn]
  conversation_fragment: str


@dataclass(frozen=True)
class BatchRequest:
  custom_id: str
  metric: str
  input_file: str
  output_file: str
  fragment_key: str
  fragment_index: int
  min_score: float
  max_score: float
  body: dict[str, Any]
  concept_class: str | None = None
  concept_index: int | None = None
  concept: str | None = None
  concept_count: int | None = None
  unified_concepts: bool = False

  def jsonl_line(self) -> dict[str, Any]:
    return {
      "custom_id": self.custom_id,
      "method": "POST",
      "url": BATCH_ENDPOINT,
      "body": self.body,
    }

  def map_entry(self) -> dict[str, Any]:
    entry = {
      "custom_id": self.custom_id,
      "metric": self.metric,
      "input_file": self.input_file,
      "output_file": self.output_file,
      "fragment_key": self.fragment_key,
      "fragment_index": self.fragment_index,
      "min_score": self.min_score,
      "max_score": self.max_score,
    }
    if self.concept_class is not None:
      entry["concept_class"] = self.concept_class
      entry["concept_index"] = self.concept_index
      entry["concept"] = self.concept
      entry["concept_count"] = self.concept_count
      entry["unified_concepts"] = self.unified_concepts
    return entry


METRIC_SPECS = {
  "resistance_concept": MetricSpec("resistance_concept", "resistance_concept.txt", 0.0, 2.0),
  "context_relevance": MetricSpec("context_relevance", "context_relevance.txt", 0.0, 2.0),
  "patient_like_realism": MetricSpec("patient_like_realism", "patient_like_realism.txt", 0.0, 2.0),
}


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


def coerce_float(value: Any) -> float | None:
  try:
    if value is None or value == "":
      return None
    return float(value)
  except (TypeError, ValueError):
    return None


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


def resistance_interval(value_0_1: float) -> str:
  if value_0_1 <= 0.33:
    return "positive"
  if value_0_1 < 0.66:
    return "neutral"
  return "negative"


def parse_trace_output(output: Any) -> dict[str, Any]:
  if isinstance(output, dict):
    return output
  if not isinstance(output, str):
    return {}
  text = output.strip()
  if not text:
    return {}
  try:
    parsed = json.loads(text)
  except json.JSONDecodeError:
    return {}
  return parsed if isinstance(parsed, dict) else {}


def trace_payload(trace: dict[str, Any], step_name: str) -> tuple[dict[str, Any], str]:
  steps = trace.get("traces")
  if not isinstance(steps, list):
    return {}, ""
  for step in steps:
    if not isinstance(step, dict):
      continue
    if step.get("step") == step_name:
      return parse_trace_output(step.get("output")), str(step.get("input") or "")
  return {}, ""


def format_mapping(mapping: Any) -> str:
  if not isinstance(mapping, dict):
    return ""
  lines = []
  for key, value in mapping.items():
    label = str(key).replace("_", " ").title()
    if isinstance(value, (dict, list)):
      rendered = json.dumps(value, ensure_ascii=True)
    else:
      rendered = str(value)
    lines.append(f"{label}: {rendered}")
  return "\n".join(lines)


def format_patient_profile(data: dict[str, Any]) -> str:
  profile = data.get("patient_profile")
  if not isinstance(profile, dict):
    return ""

  sections = []
  personal = format_mapping(profile.get("personal_info"))
  if personal:
    sections.append("Personal information:\n" + personal)

  for key in ("presenting_problem", "reason_seeking_counseling", "patterns", "attitude"):
    value = profile.get(key)
    if value in (None, "", []):
      continue
    label = key.replace("_", " ").title()
    if isinstance(value, list):
      rendered = ", ".join(str(item) for item in value)
    else:
      rendered = str(value)
    sections.append(f"{label}:\n{rendered}")

  more_info = format_mapping(profile.get("more_information"))
  if more_info:
    sections.append("Additional background:\n" + more_info)

  return "\n\n".join(sections)


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
        "resistance": item.get("resistance"),
      }
    )

  return sorted(normalized, key=lambda message: int(message["utterance_index"]))


def previous_therapist_message(messages: list[dict[str, Any]], patient_index: int) -> dict[str, Any] | None:
  for message in reversed([item for item in messages if int(item["utterance_index"]) < patient_index]):
    if message.get("role") == "therapist":
      return message
  return None


def patient_resistance_traces(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
  traces = data.get("reasoning_traces")
  if not isinstance(traces, list):
    return {}

  indexed: dict[int, dict[str, Any]] = {}
  for trace in traces:
    if not isinstance(trace, dict):
      continue
    if str(trace.get("speaker") or "").strip().lower() != "patient":
      continue
    utterance_index = coerce_int(trace.get("utterance_index"))
    if utterance_index is not None:
      indexed[utterance_index] = trace
  return indexed


def resistance_value_from_update(
  update: dict[str, Any],
  step_input: str,
  message: dict[str, Any],
) -> tuple[float | None, str]:
  new_r = coerce_float(update.get("new_r"))
  if new_r is not None:
    return max(0.0, min(1.0, new_r)), "resistance_update.new_r"

  message_r = coerce_float(message.get("resistance"))
  if message_r is not None:
    return max(0.0, min(1.0, message_r)), "message.resistance"

  r_change = coerce_float(update.get("r_change"))
  if r_change is not None:
    match = CURRENT_RESISTANCE_RE.search(step_input)
    if match:
      previous_r = coerce_float(match.group(1))
      if previous_r is not None:
        return max(0.0, min(1.0, previous_r + r_change)), "resistance_update.r_change+input_current"

  return None, ""


def extract_evaluable_turns(data: dict[str, Any]) -> tuple[list[EvaluableTurn], list[dict[str, Any]]]:
  messages = normalized_messages(data)
  if not messages:
    return [], [{"reason": "missing messages list"}]

  traces = patient_resistance_traces(data)
  if not traces:
    return [], [{"reason": "missing patient resistance_update traces"}]

  turns: list[EvaluableTurn] = []
  skipped: list[dict[str, Any]] = []
  patient_turn_number = 0

  for message in messages:
    if message.get("role") != "patient":
      continue
    patient_turn_number += 1
    patient_index = int(message["utterance_index"])
    therapist = previous_therapist_message(messages, patient_index)
    if therapist is None:
      skipped.append({"utterance_index": patient_index, "reason": "missing previous therapist utterance"})
      continue

    trace = traces.get(patient_index)
    if trace is None:
      skipped.append({"utterance_index": patient_index, "reason": "missing patient reasoning trace"})
      continue

    update, step_input = trace_payload(trace, "resistance_update")
    if not update:
      skipped.append({"utterance_index": patient_index, "reason": "missing resistance_update payload"})
      continue

    resistance, resistance_source = resistance_value_from_update(update, step_input, message)
    if resistance is None:
      skipped.append({"utterance_index": patient_index, "reason": "missing resistance_update.new_r"})
      continue

    turns.append(
      EvaluableTurn(
        turn_number=patient_turn_number,
        therapist_utterance_index=int(therapist["utterance_index"]),
        patient_utterance_index=patient_index,
        stage=message.get("stage"),
        substage=message.get("substage"),
        therapist_utterance=str(therapist.get("content") or ""),
        patient_utterance=str(message.get("content") or ""),
        expected_resistance_0_1=resistance,
        expected_resistance_interval=resistance_interval(resistance),
        resistance_source=resistance_source,
        resistance_update_reasoning=str(update.get("reasoning") or ""),
      )
    )

  return turns, skipped


def format_conversation_fragment(turns: list[EvaluableTurn]) -> str:
  lines = []
  for turn in turns:
    lines.append(f"Therapist: {turn.therapist_utterance}")
    lines.append(f"Patient: {turn.patient_utterance}")
  return "\n".join(lines)


def make_fragment_key(filename: str, fragment_index: int) -> str:
  return f"{filename}#fragment-{fragment_index:04d}"


def build_conversation_fragments(
  filename: str,
  turns: list[EvaluableTurn],
  n_per_batch: int,
) -> list[ConversationFragment]:
  if n_per_batch < 1:
    raise SteeringV2EvaluationError(f"n_per_batch must be positive, got {n_per_batch}")

  fragments: list[ConversationFragment] = []
  current_run: list[EvaluableTurn] = []
  current_interval: str | None = None

  def flush_run() -> None:
    nonlocal current_run
    if not current_run:
      return
    for start in range(0, len(current_run), n_per_batch):
      chunk = current_run[start:start + n_per_batch]
      fragment_index = len(fragments) + 1
      fragments.append(
        ConversationFragment(
          fragment_index=fragment_index,
          fragment_key=make_fragment_key(filename, fragment_index),
          expected_resistance_interval=chunk[0].expected_resistance_interval,
          turns=chunk,
          conversation_fragment=format_conversation_fragment(chunk),
        )
      )
    current_run = []

  for turn in turns:
    if current_interval is None:
      current_interval = turn.expected_resistance_interval
    if turn.expected_resistance_interval != current_interval:
      flush_run()
      current_interval = turn.expected_resistance_interval
    current_run.append(turn)

  flush_run()
  return fragments


def load_prompt_template(prompt_file: str) -> str:
  path = PROMPT_ROOT / prompt_file
  with path.open("r", encoding="utf-8") as f:
    return f.read()


def render_template(template: str, values: dict[str, str]) -> str:
  rendered = template
  for key, value in values.items():
    rendered = rendered.replace("{" + key + "}", value)
  return rendered


def render_fragment_prompt(template: str, profile: str, fragment: ConversationFragment) -> str:
  return render_template(
    template,
    {
      "patient_profile": profile or "(No patient profile available.)",
      "conversation_fragment": fragment.conversation_fragment,
    },
  )


def render_resistance_prompt(
  template: str,
  profile: str,
  fragment: ConversationFragment,
  concept: str,
) -> str:
  return render_template(
    template,
    {
      "patient_profile": profile or "(No patient profile available.)",
      "conversation_fragment": fragment.conversation_fragment,
      "concept": concept,
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


def make_custom_id(index: int, filename: str, fragment_index: int, metric: str, suffix: str = "") -> str:
  name = safe_custom_id_part(Path(filename).stem)
  metric_part = safe_custom_id_part(metric)
  suffix_part = f":{safe_custom_id_part(suffix)}" if suffix else ""
  return f"steering_v2:{index:08d}:{name}:f{fragment_index:04d}:{metric_part}{suffix_part}"


def load_evaluation_concepts(path: Path = CONCEPTS_PATH) -> dict[str, list[str]]:
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise SteeringV2EvaluationError(f"Concept file must contain a JSON object: {path}")

  concepts: dict[str, list[str]] = {}
  for class_name in CLASS_ORDER:
    raw_items = payload.get(class_name)
    if not isinstance(raw_items, list):
      raise SteeringV2EvaluationError(f"Concept file is missing list key {class_name!r}: {path}")
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    if not items:
      raise SteeringV2EvaluationError(f"Concept list {class_name!r} is empty: {path}")
    concepts[class_name] = items
  return concepts


def unified_concept_text(concepts: list[str]) -> str:
  return ". ".join(concept.strip().rstrip(".") for concept in concepts if concept.strip())


def serialize_turn(turn: EvaluableTurn) -> dict[str, Any]:
  return {
    "turn_number": turn.turn_number,
    "therapist_utterance_index": turn.therapist_utterance_index,
    "patient_utterance_index": turn.patient_utterance_index,
    "stage": turn.stage,
    "substage": turn.substage,
    "therapist_utterance": turn.therapist_utterance,
    "patient_utterance": turn.patient_utterance,
    "expected_resistance_0_1": turn.expected_resistance_0_1,
    "expected_resistance_interval": turn.expected_resistance_interval,
    "resistance_source": turn.resistance_source,
    "resistance_update_reasoning": turn.resistance_update_reasoning,
  }


def serialize_fragment(fragment: ConversationFragment) -> dict[str, Any]:
  resistance_values = [turn.expected_resistance_0_1 for turn in fragment.turns]
  return {
    "fragment_index": fragment.fragment_index,
    "fragment_key": fragment.fragment_key,
    "turn_count": len(fragment.turns),
    "start_turn_number": fragment.turns[0].turn_number,
    "end_turn_number": fragment.turns[-1].turn_number,
    "start_patient_utterance_index": fragment.turns[0].patient_utterance_index,
    "end_patient_utterance_index": fragment.turns[-1].patient_utterance_index,
    "expected_resistance_interval": fragment.expected_resistance_interval,
    "mean_expected_resistance_0_1": mean(resistance_values),
    "conversation_fragment": fragment.conversation_fragment,
    "turns": [serialize_turn(turn) for turn in fragment.turns],
  }


def build_requests(
  files: list[Path],
  output_dir: Path,
  model: str,
  max_iter: int,
  n_per_batch: int,
  *,
  matched_concepts: bool = False,
  unified_concepts: bool = False,
  concepts: dict[str, list[str]] | None = None,
) -> tuple[list[BatchRequest], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  active_concepts = concepts or load_evaluation_concepts()
  templates = {key: load_prompt_template(spec.prompt_file) for key, spec in METRIC_SPECS.items()}
  requests: list[BatchRequest] = []
  sessions: list[dict[str, Any]] = []
  skipped_files: list[dict[str, Any]] = []
  request_index = 0
  fragmenting = {
    "n_per_batch": n_per_batch,
    "split_scope": "all_modes",
    "selected_fragments": 0,
    "source_evaluable_turns": 0,
    "matched_concepts": matched_concepts,
    "unified_concepts": unified_concepts,
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
          "matched_concepts": matched_concepts,
          "unified_concepts": unified_concepts,
          "evaluated_metrics": list(METRIC_SPECS),
          "evaluated_fragments": [],
          "metrics": session_metrics([]),
          "skipped_turns": [{"reason": reason}],
        }
      )
      continue

    turns, skipped_turns = extract_evaluable_turns(payload)
    if not turns:
      skipped_files.append({"file": str(path), "reason": "no evaluable patient turns"})

    fragments = build_conversation_fragments(path.name, turns, n_per_batch)
    profile = format_patient_profile(payload)
    fragmenting["source_evaluable_turns"] += len(turns)
    fragmenting["selected_fragments"] += len(fragments)

    session = {
      "input_file": str(path),
      "input_filename": path.name,
      "model": model,
      "max_iter": max_iter,
      "matched_concepts": matched_concepts,
      "unified_concepts": unified_concepts,
      "concepts_source": str(CONCEPTS_PATH),
      "evaluated_metrics": list(METRIC_SPECS),
      "evaluated_fragments": [],
      "metrics": session_metrics([]),
      "skipped_turns": skipped_turns,
      "fragmenting": {
        "n_per_batch": n_per_batch,
        "source_evaluable_turn_count": len(turns),
        "selected_fragment_count": len(fragments),
        "split_on_resistance_interval_changes": True,
      },
    }

    for fragment in fragments:
      fragment_payload = serialize_fragment(fragment)
      fragment_payload["batch_custom_ids"] = {
        "resistance_concepts": {class_name: [] for class_name in CLASS_ORDER},
      }

      concept_classes = (
        [fragment.expected_resistance_interval]
        if matched_concepts
        else list(CLASS_ORDER)
      )
      for class_name in concept_classes:
        class_concepts = active_concepts[class_name]
        rendered_concepts = (
          [(None, unified_concept_text(class_concepts), len(class_concepts), True)]
          if unified_concepts
          else [
            (concept_index, concept, 1, False)
            for concept_index, concept in enumerate(class_concepts)
          ]
        )
        for concept_index, concept, concept_count, is_unified in rendered_concepts:
          spec = METRIC_SPECS["resistance_concept"]
          prompt = render_resistance_prompt(templates["resistance_concept"], profile, fragment, concept)
          suffix = f"{class_name}-unified" if is_unified else f"{class_name}-{int(concept_index):03d}"
          custom_id = make_custom_id(
            request_index,
            path.name,
            fragment.fragment_index,
            "resistance_concept",
            suffix,
          )
          request_index += 1
          fragment_payload["batch_custom_ids"]["resistance_concepts"][class_name].append(
            {
              "concept_index": concept_index,
              "concept": concept,
              "concept_count": concept_count,
              "unified_concepts": is_unified,
              "custom_id": custom_id,
            }
          )
          requests.append(
            BatchRequest(
              custom_id=custom_id,
              metric="resistance_concept",
              input_file=path.name,
              output_file=output_file.as_posix(),
              fragment_key=fragment.fragment_key,
              fragment_index=fragment.fragment_index,
              min_score=spec.min_score,
              max_score=spec.max_score,
              body=make_chat_body(prompt, model, max_iter),
              concept_class=class_name,
              concept_index=concept_index,
              concept=concept,
              concept_count=concept_count,
              unified_concepts=is_unified,
            )
          )

      for metric_key in ("context_relevance", "patient_like_realism"):
        spec = METRIC_SPECS[metric_key]
        prompt = render_fragment_prompt(templates[metric_key], profile, fragment)
        custom_id = make_custom_id(request_index, path.name, fragment.fragment_index, metric_key)
        request_index += 1
        fragment_payload["batch_custom_ids"][metric_key] = custom_id
        requests.append(
          BatchRequest(
            custom_id=custom_id,
            metric=metric_key,
            input_file=path.name,
            output_file=output_file.as_posix(),
            fragment_key=fragment.fragment_key,
            fragment_index=fragment.fragment_index,
            min_score=spec.min_score,
            max_score=spec.max_score,
            body=make_chat_body(prompt, model, max_iter),
          )
        )

      session["evaluated_fragments"].append(fragment_payload)

    sessions.append(session)

  return requests, sessions, skipped_files, fragmenting


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
      raise SteeringV2EvaluationError(
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
        raise SteeringV2EvaluationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
      custom_id = payload.get("custom_id")
      if not isinstance(custom_id, str) or not custom_id:
        raise SteeringV2EvaluationError(f"Batch output line is missing custom_id at {path}:{line_number}")
      if custom_id in results:
        raise SteeringV2EvaluationError(f"Duplicate custom_id in batch output: {custom_id}")
      results[custom_id] = payload
  return results


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


def score_fragment(
  fragment: dict[str, Any],
  parsed_results: dict[str, dict[str, Any]],
  *,
  matched_concepts: bool = False,
) -> dict[str, Any]:
  custom_ids = fragment.get("batch_custom_ids") or {}
  concept_custom_ids = custom_ids.get("resistance_concepts") or {}
  concept_scores: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in CLASS_ORDER}

  for class_name in CLASS_ORDER:
    raw_items = concept_custom_ids.get(class_name) or []
    if not isinstance(raw_items, list):
      continue
    for item in raw_items:
      if not isinstance(item, dict):
        continue
      custom_id = str(item.get("custom_id") or "")
      result = parsed_results.get(custom_id, empty_judge_result(error="missing parsed result"))
      concept_scores[class_name].append(
        {
          "concept_index": item.get("concept_index"),
          "concept": item.get("concept"),
          "concept_count": item.get("concept_count"),
          "unified_concepts": bool(item.get("unified_concepts")),
          "score": float(result.get("score") or 0.0),
          "cost": float(result.get("cost") or 0.0),
          "choices": result.get("choices") or [],
          "error": result.get("error"),
        }
      )

  class_scores = {
    class_name: mean(item["score"] for item in items) if items else None
    for class_name, items in concept_scores.items()
  }
  expected_interval = str(fragment.get("expected_resistance_interval") or "")
  expected_score = class_scores.get(expected_interval)

  predicted_interval = None
  resistance_interval_match = None
  resistance_interval_tie = None
  if not matched_concepts:
    available = {
      class_name: score
      for class_name, score in class_scores.items()
      if score is not None
    }
    if available:
      max_score = max(float(score) for score in available.values())
      winners = [
        class_name
        for class_name in CLASS_ORDER
        if available.get(class_name) == max_score
      ]
      predicted_interval = winners[0]
      resistance_interval_tie = len(winners) > 1
      resistance_interval_match = predicted_interval == expected_interval

  context_result = parsed_results.get(
    str(custom_ids.get("context_relevance") or ""),
    empty_judge_result(error="missing context_relevance result"),
  )
  realism_result = parsed_results.get(
    str(custom_ids.get("patient_like_realism") or ""),
    empty_judge_result(error="missing patient_like_realism result"),
  )
  context_score = float(context_result.get("score") or 0.0)
  realism_score = float(realism_result.get("score") or 0.0)
  resistance_score = float(expected_score) if expected_score is not None else None
  harmonic = (
    harmonic_mean([resistance_score, context_score, realism_score])
    if resistance_score is not None
    else None
  )

  scored = dict(fragment)
  scored.update(
    {
      "resistance_concept_scores": concept_scores,
      "resistance_class_scores": class_scores,
      "expected_resistance_concept_score": resistance_score,
      "predicted_resistance_interval": predicted_interval,
      "resistance_interval_match": resistance_interval_match,
      "resistance_interval_tie": resistance_interval_tie,
      "context_relevance": context_score,
      "patient_like_realism": realism_score,
      "harmonic_mean": harmonic,
      "cost": (
        sum(float(item.get("cost") or 0.0) for items in concept_scores.values() for item in items)
        + float(context_result.get("cost") or 0.0)
        + float(realism_result.get("cost") or 0.0)
      ),
      "judge_scores": {
        "context_relevance": context_result,
        "patient_like_realism": realism_result,
      },
      "evaluated_metrics": list(METRIC_SPECS),
    }
  )
  return scored


def attach_judgements(
  sessions: list[dict[str, Any]],
  parsed_results: dict[str, dict[str, Any]],
  artifact_dir: Path | None,
) -> list[dict[str, Any]]:
  scored_sessions = []
  for session in sessions:
    matched_concepts = bool(session.get("matched_concepts"))
    scored_fragments = []
    for fragment in session.get("evaluated_fragments", []):
      if isinstance(fragment, dict):
        scored_fragments.append(score_fragment(fragment, parsed_results, matched_concepts=matched_concepts))

    scored_session = dict(session)
    scored_session["evaluated_fragments"] = scored_fragments
    scored_session["metrics"] = session_metrics(scored_fragments)
    if artifact_dir is not None:
      scored_session["batch_artifacts"] = str(artifact_dir)
    scored_sessions.append(scored_session)
  return scored_sessions


def mean_numeric_field(fragments: list[dict[str, Any]], field: str) -> float | None:
  values = [
    float(fragment[field])
    for fragment in fragments
    if fragment.get(field) is not None
  ]
  return mean(values)


def rate_bool_field(fragments: list[dict[str, Any]], field: str) -> float | None:
  values = [
    1.0 if fragment[field] else 0.0
    for fragment in fragments
    if fragment.get(field) is not None
  ]
  return mean(values)


def session_metrics(fragments: list[dict[str, Any]]) -> dict[str, Any]:
  if not fragments:
    return {
      "fragment_count": 0,
      "turn_count": 0,
      "mean_harmonic_score": None,
      "mean_expected_resistance_concept_score": None,
      "mean_context_relevance": None,
      "mean_patient_like_realism": None,
      "resistance_interval_accuracy": None,
      "cost": 0.0,
    }

  return {
    "fragment_count": len(fragments),
    "turn_count": sum(int(fragment.get("turn_count") or 0) for fragment in fragments),
    "mean_harmonic_score": mean_numeric_field(fragments, "harmonic_mean"),
    "mean_expected_resistance_concept_score": mean_numeric_field(fragments, "expected_resistance_concept_score"),
    "mean_context_relevance": mean_numeric_field(fragments, "context_relevance"),
    "mean_patient_like_realism": mean_numeric_field(fragments, "patient_like_realism"),
    "resistance_interval_accuracy": rate_bool_field(fragments, "resistance_interval_match"),
    "cost": sum(float(fragment.get("cost") or 0.0) for fragment in fragments),
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
  all_fragments = [
    fragment
    for session in session_results
    for fragment in session.get("evaluated_fragments", [])
  ]
  skipped_turns = [
    {"file": session.get("input_file"), **skipped}
    for session in session_results
    for skipped in session.get("skipped_turns", [])
  ]

  return {
    "version": 2,
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "batch_artifacts": str(artifact_dir) if artifact_dir is not None else None,
    "total_files_seen": files_seen,
    "total_files_evaluated": sum(1 for session in session_results if session.get("evaluated_fragments")),
    "total_fragments_evaluated": len(all_fragments),
    "total_turns_evaluated": sum(int(fragment.get("turn_count") or 0) for fragment in all_fragments),
    "mean_judge_scores": {
      "expected_resistance_concept_score": mean_numeric_field(all_fragments, "expected_resistance_concept_score"),
      "context_relevance": mean_numeric_field(all_fragments, "context_relevance"),
      "patient_like_realism": mean_numeric_field(all_fragments, "patient_like_realism"),
    },
    "mean_harmonic_score": mean_numeric_field(all_fragments, "harmonic_mean"),
    "resistance_interval_accuracy": rate_bool_field(all_fragments, "resistance_interval_match"),
    "shard_count": shard_count,
    "total_requests": request_count,
    "total_estimated_cost": sum(float(fragment.get("cost") or 0.0) for fragment in all_fragments),
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
      "version": 2,
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
    "version": 2,
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
    raise SteeringV2EvaluationError("Missing OpenAI API key. Set GPT_API_KEY or OPENAI_API_KEY.")
  try:
    from openai import OpenAI
  except ImportError as exc:
    raise SteeringV2EvaluationError("The openai Python package is required to run steering v2 batches.") from exc
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
    raise SteeringV2EvaluationError("OpenAI file upload did not return a file id.")

  manifest["input_file_id"] = uploaded_file_id
  manifest["uploaded_file"] = to_plain(uploaded_file)
  manifest["status"] = "uploaded"
  save_manifest(shard_dir / "manifest.json", manifest)

  batch = client.batches.create(
    input_file_id=uploaded_file_id,
    endpoint=BATCH_ENDPOINT,
    completion_window=COMPLETION_WINDOW,
    metadata={
      "description": "ReasonCBT steering v2 fragment judge evaluation",
      "shard_index": str(manifest.get("shard_index", "")),
    },
  )
  batch_id = get_field(batch, "id")
  if not batch_id:
    raise SteeringV2EvaluationError("OpenAI batch creation did not return a batch id.")
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
    raise SteeringV2EvaluationError("Cannot poll before a batch is created.")

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
        raise SteeringV2EvaluationError(f"Batch shard {manifest.get('shard_index')} ended with status {status}.")
      counts = manifest.get("request_counts") or {}
      failed_count = get_field(counts, "failed", 0) or 0
      if int(failed_count) > 0:
        raise SteeringV2EvaluationError(
          f"Batch shard {manifest.get('shard_index')} completed with {failed_count} failed requests."
        )
      if not manifest.get("output_jsonl"):
        raise SteeringV2EvaluationError(f"Batch shard {manifest.get('shard_index')} completed without an output file.")
      return

    if time.monotonic() >= deadline:
      raise SteeringV2EvaluationError(
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
      raise SteeringV2EvaluationError(f"Shard manifest must be a JSON object: {shard_manifest_path}")
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
    raise SteeringV2EvaluationError(f"Request map must be a JSON object: {path}")
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
      raise SteeringV2EvaluationError(f"Shard manifest must be a JSON object: {shard_manifest_path}")
    output_jsonl = shard_manifest.get("output_jsonl")
    if not output_jsonl:
      raise SteeringV2EvaluationError(f"Shard {shard_manifest.get('shard_index')} has no output_jsonl.")
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
        raise SteeringV2EvaluationError(f"Duplicate parsed custom_id across shards: {custom_id}")
      merged_results[custom_id] = result
    report["shard_index"] = shard_manifest.get("shard_index")
    report["source_batch_jsonl"] = str(output_jsonl)
    report["source_request_map"] = str(request_map_path)
    reports.append(report)

  return merged_results, merge_parse_reports(reports)


def run_steering_batch_v2(
  input_dir: str | Path,
  output_dir: str | Path,
  max_iter: int = DEFAULT_MAX_ITER,
  *,
  model: str = MODEL_NAME,
  recursive: bool = False,
  n_per_batch: int,
  matched_concepts: bool = False,
  unified_concepts: bool = False,
  poll_interval: float = DEFAULT_POLL_INTERVAL,
  local_timeout_hours: float = DEFAULT_LOCAL_TIMEOUT_HOURS,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
  prepare_only: bool = False,
  client: Any | None = None,
) -> dict[str, Any]:
  if max_iter < 1:
    raise SteeringV2EvaluationError(f"max_iter must be positive, got {max_iter}")
  if n_per_batch < 1:
    raise SteeringV2EvaluationError(f"n_per_batch must be positive, got {n_per_batch}")

  input_path = Path(input_dir).resolve()
  output_path = Path(output_dir).resolve()
  if not input_path.is_dir():
    raise SteeringV2EvaluationError(f"input_dir does not exist or is not a directory: {input_path}")

  concepts = load_evaluation_concepts()
  files = iter_json_files(input_path, recursive)
  output_path.mkdir(parents=True, exist_ok=True)
  (output_path / "sessions").mkdir(parents=True, exist_ok=True)

  requests, prepared_sessions, skipped_files, fragmenting = build_requests(
    files,
    output_path,
    model,
    max_iter,
    n_per_batch,
    matched_concepts=matched_concepts,
    unified_concepts=unified_concepts,
    concepts=concepts,
  )
  artifact_dir, manifest = create_run_artifacts(output_path, requests, prepared_sessions)
  manifest["model"] = model
  manifest["max_iter"] = max_iter
  manifest["matched_concepts"] = matched_concepts
  manifest["unified_concepts"] = unified_concepts
  manifest["evaluated_metrics"] = list(METRIC_SPECS)
  manifest["concepts_source"] = str(CONCEPTS_PATH)
  manifest["concept_counts"] = {class_name: len(items) for class_name, items in concepts.items()}
  manifest["fragmenting"] = fragmenting
  manifest["input_dir"] = str(input_path)
  manifest["output_dir"] = str(output_path)
  manifest["files_seen"] = len(files)
  manifest["recursive"] = recursive
  save_parent_manifest(artifact_dir, manifest)

  if prepare_only:
    return manifest

  parse_report: dict[str, Any] | None = None
  if requests:
    try:
      run_openai_batch_shards(artifact_dir, manifest, poll_interval, local_timeout_hours, client=client)
      manifest = read_json(artifact_dir / "manifest.json")
      parsed_results, parse_report = parse_all_shard_outputs(
        manifest,
        max_iter,
        input_cost_per_1m,
        output_cost_per_1m,
      )
      parse_report_path = artifact_dir / "parse_report.json"
      atomic_write_json(parse_report_path, parse_report)
      manifest["parse_report"] = str(parse_report_path)
      manifest["status"] = "parsed"
      save_parent_manifest(artifact_dir, manifest)
    except Exception as exc:
      manifest["status"] = "failed"
      manifest["error"] = str(exc)
      atomic_write_json(
        artifact_dir / "failure_report.json",
        {"error": str(exc), "type": type(exc).__name__, "updated_at": utc_now()},
      )
      save_parent_manifest(artifact_dir, manifest)
      if isinstance(exc, SteeringV2EvaluationError):
        raise
      raise SteeringV2EvaluationError(str(exc)) from exc
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
  summary["matched_concepts"] = matched_concepts
  summary["unified_concepts"] = unified_concepts
  summary["fragmenting"] = fragmenting
  atomic_write_json(output_path / "summary.json", summary)
  manifest["status"] = "completed"
  manifest["summary"] = str(output_path / "summary.json")
  save_parent_manifest(artifact_dir, manifest)
  return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate simulated patient steering with fragment-level AxBench-like judges through the OpenAI Batch API."
  )
  parser.add_argument("-i", "--input_dir", type=Path, default=Path("."), help="Directory with simulation JSON files.")
  parser.add_argument("-o", "--output_dir", type=Path, default=Path("."), help="Directory for steering v2 outputs.")
  parser.add_argument("-m_iter", "--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Judge samples per metric.")
  parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Judge model. Default: {MODEL_NAME}")
  parser.add_argument("--recursive", action="store_true", help="Scan JSON files recursively under input_dir.")
  parser.add_argument(
    "--n-per-batch",
    type=int,
    required=True,
    help="Maximum number of therapist/patient turns per evaluated conversation fragment.",
  )
  parser.add_argument(
    "-mc",
    "--matched-concepts",
    action="store_true",
    help="Evaluate resistance concepts only from the trace-derived expected resistance interval.",
  )
  parser.add_argument(
    "-u",
    "--unified",
    action="store_true",
    help="Evaluate one concatenated resistance concept set per class instead of one request per individual concept.",
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
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    manifest = run_steering_batch_v2(
      args.input_dir,
      args.output_dir,
      args.max_iter,
      model=args.model,
      recursive=args.recursive,
      n_per_batch=args.n_per_batch,
      matched_concepts=args.matched_concepts,
      unified_concepts=args.unified,
      poll_interval=args.poll_interval,
      local_timeout_hours=args.local_timeout_hours,
      input_cost_per_1m=args.input_cost_per_1m,
      output_cost_per_1m=args.output_cost_per_1m,
      prepare_only=args.prepare_only,
    )
  except SteeringV2EvaluationError as exc:
    print(f"Steering v2 evaluation failed: {exc}", file=os.sys.stderr)
    return 1

  artifact_dir = Path(manifest["output_dir"]) / "_batch_artifacts" / manifest["run_id"]
  if args.prepare_only:
    print("Steering v2 batch artifacts prepared.")
    print(f"Run artifacts: {artifact_dir}")
    print(f"Request count: {manifest['request_count']}")
    print(f"Shard count: {manifest['shard_count']}")
    fragmenting = manifest.get("fragmenting") or {}
    print(
      "Fragments: "
      f"{fragmenting.get('selected_fragments')} from "
      f"{fragmenting.get('source_evaluable_turns')} evaluable turns"
    )
    return 0

  print("Steering v2 evaluation completed.")
  print(f"Run artifacts: {artifact_dir}")
  print(f"Session outputs: {Path(manifest['output_dir']) / 'sessions'}")
  print(f"Summary: {Path(manifest['output_dir']) / 'summary.json'}")
  print(f"Shard count: {manifest.get('shard_count')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
