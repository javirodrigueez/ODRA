#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rtraces import (
  BATCH_ENDPOINT,
  DEFAULT_CONTEXT_TURNS,
  DEFAULT_INPUT_COST_PER_1M,
  DEFAULT_LOCAL_TIMEOUT_HOURS,
  DEFAULT_MAX_ITER,
  DEFAULT_OUTPUT_COST_PER_1M,
  DEFAULT_POLL_INTERVAL,
  METRIC_SPECS,
  MODEL_NAME,
  RTracesEvaluationError,
  atomic_write_json,
  create_run_artifacts,
  harmonic_mean,
  iter_json_files,
  make_chat_body,
  mean,
  parse_all_shard_outputs,
  parse_batch_outputs,
  read_json,
  render_value,
  run_openai_batch_shards,
  safe_custom_id_part,
  utc_now,
)


DEFAULT_MIN_PATIENT_TRACES = 2

PATIENT_PROMPTS = {
  "faithfulness": """[System]
Please act as an impartial judge and evaluate the faithfulness of the patient reasoning traces below. Focus solely on whether the reasoning traces are grounded in, and aligned with, the previous conversation context. Disregard whether the reasoning is internally logical and disregard whether it matches the final patient utterance.

Use this 0-2 scale:
0: The traces are mostly unfaithful to the previous context, invent important unsupported facts, contradict the context, or rely on claims not present in the prior conversation.
1: The traces are partly faithful but include some unsupported assumptions, weak grounding, or context mismatches.
2: The traces are well grounded in the prior context, avoid unsupported claims, and accurately use the available conversation.

Begin your evaluation with a brief explanation. After the explanation, provide your rating using this exact format: "Rating: [[score]]".

[Previous Context Start]
{context}
[Previous Context End]

[Reasoning Traces Start]
{reasoning_traces}
[Reasoning Traces End]
""",
  "logic_consistency": """[System]
Please act as an impartial judge and evaluate the internal logic consistency of the patient reasoning traces below. Focus solely on whether the traces are logically sound as a chain of intermediate decisions: the steps should be coherent with each other, avoid self-contradiction, and support later trace conclusions from earlier trace conclusions. Disregard whether the traces are faithful to the conversation context and disregard whether they match the final patient utterance.

Use this 0-2 scale:
0: The traces are logically incoherent, self-contradictory, or the later conclusions do not follow from the earlier trace content.
1: The traces are partly coherent but contain noticeable gaps, tension, or weak transitions between reasoning steps.
2: The traces are internally coherent, logically connected, and the reasoning steps support each other.

Begin your evaluation with a brief explanation. After the explanation, provide your rating using this exact format: "Rating: [[score]]".

[Reasoning Traces Start]
{reasoning_traces}
[Reasoning Traces End]
""",
  "answer_explanation_alignment": """[System]
Please act as an impartial judge and evaluate whether the patient reasoning traces align with the final patient utterance. Focus solely on whether the final utterance follows from the reasoning traces and reflects their stated decisions, plans, and conclusions. Disregard whether the traces are faithful to earlier context and disregard whether the traces are internally logical except as needed to compare them with the final utterance.

Use this 0-2 scale:
0: The final patient utterance is mostly disconnected from, or contradicts, the reasoning traces.
1: The final patient utterance is partly aligned but omits or shifts important trace conclusions, or only loosely follows the traces.
2: The final patient utterance clearly follows from and reflects the reasoning traces.

Begin your evaluation with a brief explanation. After the explanation, provide your rating using this exact format: "Rating: [[score]]".

[Reasoning Traces Start]
{reasoning_traces}
[Reasoning Traces End]

[Final Patient Utterance Start]
{patient_utterance}
[Final Patient Utterance End]
""",
}


@dataclass(frozen=True)
class TherapistSelection:
  input_filename: str
  therapist_turn_key: str
  therapist_utterance_index: int
  therapist_utterance: str
  stage: str
  substage: str | None


@dataclass(frozen=True)
class PatientReasoning:
  turn_key: str
  source_therapist_turn_key: str
  source_therapist_utterance_index: int
  source_therapist_utterance: str
  stage: str
  substage: str | None
  patient_utterance_index: int
  patient_utterance: str
  context: str
  context_utterance_count: int
  traces: list[dict[str, Any]]
  reasoning_traces_txt: str


@dataclass(frozen=True)
class PatientBatchRequest:
  custom_id: str
  metric: str
  input_file: str
  output_file: str
  turn_key: str
  stage: str
  source_therapist_utterance_index: int
  patient_utterance_index: int
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
      "source_therapist_utterance_index": self.source_therapist_utterance_index,
      "patient_utterance_index": self.patient_utterance_index,
      "min_score": self.min_score,
      "max_score": self.max_score,
    }


def coerce_int(value: Any) -> int | None:
  try:
    if value is None or value == "":
      return None
    return int(value)
  except (TypeError, ValueError):
    return None


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


def compact_patient_trace_output(step: str, raw_output: Any) -> dict[str, Any]:
  output = parsed_trace_output(raw_output)
  if not isinstance(output, dict):
    return {"message": str(output)}

  if step == "resistance_update":
    compact = {
      "previous_resistance": output.get("previous_r"),
      "resistance_change": output.get("r_change", output.get("delta_llm")),
      "predicted_resistance": output.get("new_r"),
      "reasoning": output.get("reasoning"),
    }
    return {key: value for key, value in compact.items() if value is not None}

  if step == "patient_action_selection":
    compact = {
      "prior_distribution": output.get("prior_distribution"),
      "context_distribution": output.get("context_distribution"),
      "action_distribution": output.get("action_distribution"),
      "selected_action": output.get("selected_action"),
    }
    return {key: value for key, value in compact.items() if value is not None}

  return dict(output)


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


def patient_reasoning_traces(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
  raw_traces = data.get("reasoning_traces")
  if not isinstance(raw_traces, list):
    return {}

  indexed: dict[int, dict[str, Any]] = {}
  for raw_trace in raw_traces:
    if not isinstance(raw_trace, dict):
      continue
    if str(raw_trace.get("speaker") or "").strip().lower() != "patient":
      continue
    utterance_index = coerce_int(raw_trace.get("utterance_index"))
    if utterance_index is None:
      continue
    raw_steps = raw_trace.get("traces")
    if not isinstance(raw_steps, list):
      continue
    target = indexed.setdefault(
      utterance_index,
      {"utterance_index": utterance_index, "speaker": "patient", "traces": []},
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
  patient_utterance_index: int,
  ctxt_turns: int,
) -> list[dict[str, Any]]:
  limit = max(0, ctxt_turns) * 2
  previous = [
    message
    for message in messages
    if int(message["utterance_index"]) < patient_utterance_index
  ]
  if limit <= 0:
    return []
  return previous[-limit:]


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
  source_therapist_utterance_index: int,
  patient_utterance_index: int,
  traces: list[dict[str, Any]],
) -> str:
  lines = [
    "Target:",
    f"  Source therapist utterance index: {source_therapist_utterance_index}",
    f"  Patient utterance index: {patient_utterance_index}",
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


def make_patient_turn_key(filename: str, patient_utterance_index: int) -> str:
  return f"{filename}#patient-{patient_utterance_index:04d}"


def natural_sort_key(path: Path) -> tuple[Any, ...]:
  return tuple(
    int(part) if part.isdigit() else part.lower()
    for part in re.split(r"(\d+)", path.name)
  )


def iter_therapist_turn_files(path: Path) -> list[Path]:
  if path.is_file():
    return [path]
  if not path.is_dir():
    raise RTracesEvaluationError(f"--therapist-turns does not exist: {path}")

  sessions_dir = path / "sessions"
  source_dir = sessions_dir if sessions_dir.is_dir() else path
  return sorted(
    (item for item in source_dir.glob("*.json") if item.is_file()),
    key=natural_sort_key,
  )


def load_therapist_selections(path: Path) -> tuple[dict[str, list[TherapistSelection]], list[dict[str, Any]]]:
  selections: dict[str, list[TherapistSelection]] = {}
  skipped: list[dict[str, Any]] = []
  seen: set[tuple[str, int]] = set()

  for file_path in iter_therapist_turn_files(path):
    try:
      payload = read_json(file_path)
    except Exception as exc:
      skipped.append({"file": str(file_path), "reason": f"could not read JSON: {exc}"})
      continue
    if not isinstance(payload, dict):
      skipped.append({"file": str(file_path), "reason": "top-level JSON is not an object"})
      continue

    filename = str(payload.get("input_filename") or file_path.name)
    raw_turns = payload.get("evaluated_turns")
    if not isinstance(raw_turns, list):
      skipped.append({"file": str(file_path), "reason": "missing evaluated_turns list"})
      continue

    for raw_turn in raw_turns:
      if not isinstance(raw_turn, dict):
        continue
      therapist_index = coerce_int(raw_turn.get("therapist_utterance_index"))
      if therapist_index is None:
        skipped.append(
          {
            "file": str(file_path),
            "reason": "evaluated turn missing therapist_utterance_index",
          }
        )
        continue
      dedupe_key = (filename, therapist_index)
      if dedupe_key in seen:
        continue
      seen.add(dedupe_key)
      selections.setdefault(filename, []).append(
        TherapistSelection(
          input_filename=filename,
          therapist_turn_key=str(raw_turn.get("turn_key") or f"{filename}#therapist-{therapist_index:04d}"),
          therapist_utterance_index=therapist_index,
          therapist_utterance=str(raw_turn.get("therapist_utterance") or ""),
          stage=str(raw_turn.get("stage") or "unknown"),
          substage=str(raw_turn.get("substage") or "") or None,
        )
      )

  for items in selections.values():
    items.sort(key=lambda item: item.therapist_utterance_index)
  return selections, skipped


def find_subsequent_patient_message(
  messages: list[dict[str, Any]],
  therapist_utterance_index: int,
) -> dict[str, Any] | None:
  for message in messages:
    if int(message["utterance_index"]) > therapist_utterance_index and message.get("role") == "patient":
      return message
    if int(message["utterance_index"]) > therapist_utterance_index and message.get("role") == "therapist":
      return None
  return None


def extract_patient_turns_for_selections(
  data: dict[str, Any],
  filename: str,
  selections: list[TherapistSelection],
  ctxt_turns: int,
  min_traces: int,
) -> tuple[list[PatientReasoning], list[dict[str, Any]]]:
  messages = normalized_messages(data)
  if not messages:
    return [], [{"reason": "missing messages list"}]

  traces_by_index = patient_reasoning_traces(data)
  if not traces_by_index:
    return [], [{"reason": "missing patient reasoning_traces"}]

  turns: list[PatientReasoning] = []
  skipped: list[dict[str, Any]] = []
  messages_by_index = {int(message["utterance_index"]): message for message in messages}

  for selection in selections:
    therapist_message = messages_by_index.get(selection.therapist_utterance_index)
    if not isinstance(therapist_message, dict) or therapist_message.get("role") != "therapist":
      skipped.append(
        {
          "source_therapist_utterance_index": selection.therapist_utterance_index,
          "source_therapist_turn_key": selection.therapist_turn_key,
          "reason": "missing source therapist message",
        }
      )
      continue

    patient_message = find_subsequent_patient_message(messages, selection.therapist_utterance_index)
    if not isinstance(patient_message, dict):
      skipped.append(
        {
          "source_therapist_utterance_index": selection.therapist_utterance_index,
          "source_therapist_turn_key": selection.therapist_turn_key,
          "reason": "missing subsequent patient message",
        }
      )
      continue

    patient_index = int(patient_message["utterance_index"])
    reasoning_trace = traces_by_index.get(patient_index)
    if not isinstance(reasoning_trace, dict):
      skipped.append(
        {
          "source_therapist_utterance_index": selection.therapist_utterance_index,
          "patient_utterance_index": patient_index,
          "source_therapist_turn_key": selection.therapist_turn_key,
          "reason": "missing patient reasoning trace",
        }
      )
      continue

    compact_traces = []
    for trace in reasoning_trace.get("traces") or []:
      if not isinstance(trace, dict):
        continue
      step = trace.get("step")
      if not isinstance(step, str) or not step:
        continue
      compact_traces.append(
        {
          "step": step,
          "output": compact_patient_trace_output(step, trace.get("output")),
        }
      )

    if len(compact_traces) < min_traces:
      skipped.append(
        {
          "source_therapist_utterance_index": selection.therapist_utterance_index,
          "patient_utterance_index": patient_index,
          "source_therapist_turn_key": selection.therapist_turn_key,
          "stage": patient_message.get("stage") or selection.stage,
          "reason": "insufficient included traces",
          "included_trace_count": len(compact_traces),
          "min_traces_per_sample": min_traces,
        }
      )
      continue

    stage = str(patient_message.get("stage") or selection.stage or "unknown")
    substage = str(patient_message.get("substage") or selection.substage or "") or None
    context_messages = previous_context_messages(messages, patient_index, ctxt_turns)
    turns.append(
      PatientReasoning(
        turn_key=make_patient_turn_key(filename, patient_index),
        source_therapist_turn_key=selection.therapist_turn_key,
        source_therapist_utterance_index=selection.therapist_utterance_index,
        source_therapist_utterance=str(therapist_message.get("content") or selection.therapist_utterance),
        stage=stage,
        substage=substage,
        patient_utterance_index=patient_index,
        patient_utterance=str(patient_message.get("content") or ""),
        context=format_context(context_messages),
        context_utterance_count=len(context_messages),
        traces=compact_traces,
        reasoning_traces_txt=format_reasoning_traces_txt(
          stage=stage,
          substage=substage,
          source_therapist_utterance_index=selection.therapist_utterance_index,
          patient_utterance_index=patient_index,
          traces=compact_traces,
        ),
      )
    )

  return turns, skipped


def render_metric_prompt(metric: str, turn: PatientReasoning) -> str:
  template = PATIENT_PROMPTS[metric]
  return (
    template
    .replace("{context}", turn.context)
    .replace("{reasoning_traces}", turn.reasoning_traces_txt)
    .replace("{patient_utterance}", turn.patient_utterance)
  )


def make_custom_id(index: int, filename: str, therapist_index: int, patient_index: int, metric: str) -> str:
  name = safe_custom_id_part(Path(filename).stem)
  metric_part = safe_custom_id_part(metric)
  return f"patient_rtraces:{index:08d}:{name}:t{therapist_index:04d}p{patient_index:04d}:{metric_part}"


def serialize_turn(turn: PatientReasoning) -> dict[str, Any]:
  return {
    "turn_key": turn.turn_key,
    "source_therapist_turn_key": turn.source_therapist_turn_key,
    "source_therapist_utterance_index": turn.source_therapist_utterance_index,
    "source_therapist_utterance": turn.source_therapist_utterance,
    "stage": turn.stage,
    "substage": turn.substage,
    "patient_utterance_index": turn.patient_utterance_index,
    "patient_utterance": turn.patient_utterance,
    "context_utterance_count": turn.context_utterance_count,
    "context": turn.context,
    "trace_count": len(turn.traces),
    "traces": turn.traces,
    "reasoning_traces_txt": turn.reasoning_traces_txt,
  }


def build_requests(
  files: list[Path],
  output_dir: Path,
  therapist_selections: dict[str, list[TherapistSelection]],
  model: str,
  max_iter: int,
  ctxt_turns: int,
  min_traces: int,
) -> tuple[list[PatientBatchRequest], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  requests: list[PatientBatchRequest] = []
  sessions: list[dict[str, Any]] = []
  skipped_files: list[dict[str, Any]] = []
  request_index = 0
  selected_turn_count = 0
  source_selection_count = 0

  files_by_name = {path.name: path for path in files}
  missing_input_files = sorted(filename for filename in therapist_selections if filename not in files_by_name)
  for filename in missing_input_files:
    skipped_files.append(
      {
        "file": filename,
        "reason": "therapist selection references file not present in input dir",
      }
    )

  for path in files:
    selections = therapist_selections.get(path.name, [])
    if not selections:
      skipped_files.append({"file": str(path), "reason": "no therapist turns selected for this file"})
      continue
    source_selection_count += len(selections)

    try:
      payload = read_json(path)
    except Exception as exc:
      skipped_files.append({"file": str(path), "reason": f"could not read JSON: {exc}"})
      continue
    if not isinstance(payload, dict):
      skipped_files.append({"file": str(path), "reason": "top-level JSON is not an object"})
      continue

    turns, skipped_turns = extract_patient_turns_for_selections(
      payload,
      path.name,
      selections,
      ctxt_turns,
      min_traces,
    )
    selected_turn_count += len(turns)
    if not turns:
      skipped_files.append({"file": str(path), "reason": "no selected patient reasoning turns"})

    output_file = Path("sessions") / path.name
    session = {
      "input_file": str(path),
      "input_filename": path.name,
      "model": model,
      "max_iter": max_iter,
      "ctxt_turns": ctxt_turns,
      "min_traces_per_sample": min_traces,
      "evaluated_speaker": "patient",
      "selection_source": "therapist_turns",
      "source_therapist_turn_count": len(selections),
      "evaluated_metrics": list(METRIC_SPECS),
      "evaluated_turns": [],
      "metrics": session_metrics([]),
      "skipped_turns": skipped_turns,
    }

    for turn in turns:
      turn_payload = serialize_turn(turn)
      turn_payload["batch_custom_ids"] = {}
      for metric_key, spec in METRIC_SPECS.items():
        prompt = render_metric_prompt(metric_key, turn)
        custom_id = make_custom_id(
          request_index,
          path.name,
          turn.source_therapist_utterance_index,
          turn.patient_utterance_index,
          metric_key,
        )
        request_index += 1
        turn_payload["batch_custom_ids"][metric_key] = custom_id
        requests.append(
          PatientBatchRequest(
            custom_id=custom_id,
            metric=metric_key,
            input_file=path.name,
            output_file=output_file.as_posix(),
            turn_key=turn.turn_key,
            stage=turn.stage,
            source_therapist_utterance_index=turn.source_therapist_utterance_index,
            patient_utterance_index=turn.patient_utterance_index,
            min_score=spec.min_score,
            max_score=spec.max_score,
            body=make_chat_body(prompt, model, max_iter),
          )
        )
      session["evaluated_turns"].append(turn_payload)

    sessions.append(session)

  sampling = {
    "selection_source": "therapist_turns",
    "source_therapist_turns": source_selection_count,
    "selected_patient_turns": selected_turn_count,
    "ctxt_turns": ctxt_turns,
    "context_utterance_limit": ctxt_turns * 2,
    "min_traces_per_sample": min_traces,
  }
  return requests, sessions, skipped_files, sampling


def empty_judge_result(score: float = 0.0, *, error: str | None = None) -> dict[str, Any]:
  result = {
    "score": score,
    "cost": 0.0,
    "choices": [],
  }
  if error:
    result["error"] = error
  return result


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


def write_session_outputs(output_dir: Path, sessions: list[dict[str, Any]]) -> None:
  for session in sessions:
    filename = str(session.get("input_filename") or Path(str(session.get("input_file"))).name)
    atomic_write_json(output_dir / "sessions" / filename, session)


def request_map_from_specs(specs: list[PatientBatchRequest]) -> dict[str, dict[str, Any]]:
  return {spec.custom_id: spec.map_entry() for spec in specs}


def summary_from_sessions(
  input_dir: Path,
  output_dir: Path,
  therapist_turns_path: Path,
  files_seen: int,
  session_results: list[dict[str, Any]],
  skipped_files: list[dict[str, Any]],
  skipped_therapist_turn_files: list[dict[str, Any]],
  artifact_dir: Path | None,
  parse_report: dict[str, Any] | None,
  sampling: dict[str, Any],
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
    "evaluated_speaker": "patient",
    "selection_source": "therapist_turns",
    "input_dir": str(input_dir),
    "therapist_turns": str(therapist_turns_path),
    "output_dir": str(output_dir),
    "batch_artifacts": str(artifact_dir) if artifact_dir is not None else None,
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
    "skipped_therapist_turn_files": skipped_therapist_turn_files,
    "skipped_turns": skipped_turns,
    "parse_report": parse_report,
    "evaluated_metrics": list(METRIC_SPECS),
    "sampling": sampling,
  }


def run_patient_rtraces_batch(
  input_dir: str | Path,
  output_dir: str | Path,
  therapist_turns: str | Path,
  max_iter: int = DEFAULT_MAX_ITER,
  *,
  model: str = MODEL_NAME,
  recursive: bool = False,
  ctxt_turns: int = DEFAULT_CONTEXT_TURNS,
  min_traces: int = DEFAULT_MIN_PATIENT_TRACES,
  poll_interval: float = DEFAULT_POLL_INTERVAL,
  local_timeout_hours: float = DEFAULT_LOCAL_TIMEOUT_HOURS,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
  prepare_only: bool = False,
  batch_output_jsonl: str | Path | None = None,
) -> dict[str, Any]:
  if max_iter < 1:
    raise RTracesEvaluationError(f"max_iter must be positive, got {max_iter}")
  if ctxt_turns < 0:
    raise RTracesEvaluationError(f"ctxt_turns must be non-negative, got {ctxt_turns}")
  if min_traces < 1:
    raise RTracesEvaluationError(f"min_traces must be positive, got {min_traces}")
  if prepare_only and batch_output_jsonl is not None:
    raise RTracesEvaluationError("--prepare-only cannot be combined with --batch-output-jsonl")

  input_path = Path(input_dir).resolve()
  output_path = Path(output_dir).resolve()
  therapist_turns_path = Path(therapist_turns).resolve()
  if not input_path.is_dir():
    raise RTracesEvaluationError(f"input dir does not exist or is not a directory: {input_path}")

  therapist_selections, skipped_therapist_turn_files = load_therapist_selections(therapist_turns_path)
  files = iter_json_files(input_path, recursive)
  output_path.mkdir(parents=True, exist_ok=True)
  (output_path / "sessions").mkdir(parents=True, exist_ok=True)

  requests, prepared_sessions, skipped_files, sampling = build_requests(
    files,
    output_path,
    therapist_selections,
    model,
    max_iter,
    ctxt_turns,
    min_traces,
  )

  artifact_dir, manifest = create_run_artifacts(output_path, requests, prepared_sessions)
  manifest["model"] = model
  manifest["max_iter"] = max_iter
  manifest["ctxt_turns"] = ctxt_turns
  manifest["min_traces"] = min_traces
  manifest["evaluated_speaker"] = "patient"
  manifest["evaluated_metrics"] = list(METRIC_SPECS)
  manifest["sampling"] = sampling
  manifest["input_dir"] = str(input_path)
  manifest["therapist_turns"] = str(therapist_turns_path)
  manifest["output_dir"] = str(output_path)
  manifest["files_seen"] = len(files)
  manifest["recursive"] = recursive
  manifest["skipped_therapist_turn_files"] = skipped_therapist_turn_files
  if batch_output_jsonl is not None:
    manifest["batch_output_jsonl"] = str(Path(batch_output_jsonl).resolve())
  atomic_write_json(artifact_dir / "manifest.json", manifest)

  if prepare_only:
    return manifest

  parse_report: dict[str, Any] | None = None
  if requests:
    try:
      if batch_output_jsonl is None:
        run_openai_batch_shards(artifact_dir, manifest, poll_interval, local_timeout_hours)
        manifest = read_json(artifact_dir / "manifest.json")
        parsed_results, parse_report = parse_all_shard_outputs(
          manifest,
          max_iter,
          input_cost_per_1m,
          output_cost_per_1m,
        )
      else:
        batch_output_path = Path(batch_output_jsonl).resolve()
        if not batch_output_path.is_file():
          raise RTracesEvaluationError(f"batch output JSONL does not exist: {batch_output_path}")
        parsed_results, parse_report = parse_batch_outputs(
          batch_output_path,
          request_map_from_specs(requests),
          max_iter,
          input_cost_per_1m,
          output_cost_per_1m,
        )
        parse_report["source_batch_jsonl"] = str(batch_output_path)
        parse_report["source_request_map"] = str(artifact_dir / "request_map.json")

      parse_report_path = artifact_dir / "parse_report.json"
      atomic_write_json(parse_report_path, parse_report)
      manifest["parse_report"] = str(parse_report_path)
      manifest["status"] = "parsed" if batch_output_jsonl is None else "parsed_from_batch_output"
      atomic_write_json(artifact_dir / "manifest.json", manifest)
    except Exception as exc:
      manifest["status"] = "failed"
      manifest["error"] = str(exc)
      atomic_write_json(
        artifact_dir / "failure_report.json",
        {"error": str(exc), "type": type(exc).__name__, "updated_at": utc_now()},
      )
      atomic_write_json(artifact_dir / "manifest.json", manifest)
      if isinstance(exc, RTracesEvaluationError):
        raise
      raise RTracesEvaluationError(str(exc)) from exc
  else:
    parsed_results = {}
    manifest["status"] = "no_requests"
    atomic_write_json(artifact_dir / "manifest.json", manifest)

  scored_sessions = attach_judgements(prepared_sessions, parsed_results, artifact_dir)
  write_session_outputs(output_path, scored_sessions)
  summary = summary_from_sessions(
    input_path,
    output_path,
    therapist_turns_path,
    len(files),
    scored_sessions,
    skipped_files,
    skipped_therapist_turn_files,
    artifact_dir,
    parse_report,
    sampling,
    request_count=len(requests),
    shard_count=int(manifest.get("shard_count") or 0),
  )
  atomic_write_json(output_path / "summary.json", summary)
  manifest["status"] = "completed"
  manifest["summary"] = str(output_path / "summary.json")
  atomic_write_json(artifact_dir / "manifest.json", manifest)
  return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate patient reasoning traces after previously evaluated therapist turns."
  )
  parser.add_argument("-i", "--input", "--input_dir", dest="input_dir", type=Path, required=True, help="Directory with original simulation JSON files.")
  parser.add_argument("-o", "--output", "--output_dir", dest="output_dir", type=Path, required=True, help="Directory for patient rtraces outputs.")
  parser.add_argument("--therapist-turns", type=Path, required=True, help="Therapist rtraces sessions directory, output directory containing sessions/, or a single session JSON file.")
  parser.add_argument("-m_iter", "--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Judge samples per metric.")
  parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Judge model. Default: {MODEL_NAME}")
  parser.add_argument("--recursive", action="store_true", help="Scan JSON files recursively under input dir.")
  parser.add_argument(
    "-ct",
    "--ctxt-turns",
    type=int,
    default=DEFAULT_CONTEXT_TURNS,
    help="Number of previous dialogue turns to include as context. Default: 5.",
  )
  parser.add_argument(
    "--min-traces",
    type=int,
    default=DEFAULT_MIN_PATIENT_TRACES,
    help=f"Minimum patient trace steps required for evaluation. Default: {DEFAULT_MIN_PATIENT_TRACES}.",
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
    manifest = run_patient_rtraces_batch(
      args.input_dir,
      args.output_dir,
      args.therapist_turns,
      args.max_iter,
      model=args.model,
      recursive=args.recursive,
      ctxt_turns=args.ctxt_turns,
      min_traces=args.min_traces,
      poll_interval=args.poll_interval,
      local_timeout_hours=args.local_timeout_hours,
      input_cost_per_1m=args.input_cost_per_1m,
      output_cost_per_1m=args.output_cost_per_1m,
      prepare_only=args.prepare_only,
      batch_output_jsonl=args.batch_output_jsonl,
    )
  except RTracesEvaluationError as exc:
    print(f"Patient reasoning trace evaluation failed: {exc}", file=os.sys.stderr)
    return 1

  artifact_dir = Path(manifest["output_dir"]) / "_batch_artifacts" / manifest["run_id"]
  if args.prepare_only:
    print("Patient reasoning trace batch artifacts prepared.")
    print(f"Run artifacts: {artifact_dir}")
    print(f"Request count: {manifest['request_count']}")
    print(f"Shard count: {manifest['shard_count']}")
    sampling = manifest.get("sampling") or {}
    print(
      "Turns: "
      f"{sampling.get('selected_patient_turns')} patient turns from "
      f"{sampling.get('source_therapist_turns')} selected therapist turns"
    )
    return 0

  if args.batch_output_jsonl:
    print("Patient reasoning trace evaluation recovered from batch output.")
  else:
    print("Patient reasoning trace evaluation completed.")
  print(f"Run artifacts: {artifact_dir}")
  print(f"Session outputs: {Path(manifest['output_dir']) / 'sessions'}")
  print(f"Summary: {Path(manifest['output_dir']) / 'summary.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
