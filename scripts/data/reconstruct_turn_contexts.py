from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


CORE_BELIEF_KEYS = ["helplessness", "unlovability", "worthlessness"]
DISTORTION_KEYS = [
  "mental_filtering",
  "fortune_telling",
  "mind_reading",
  "personalization",
  "catastrophizing",
  "all_or_nothing_thinking",
  "should_statements",
  "overgeneralization",
  "labeling_and_mislabeling",
  "discounting_the_positive",
]

DEFAULT_SUBSTAGE_BY_STAGE = {
  "opening": "open_initial",
  "mi_techniques_with_te": "mi_choose_topic",
  "mi_techniques": "mi_choose_technique",
  "meta_conversation": "mi_meta_open",
  "address_tib": "tib_plan",
  "id_distortions": "cognitive_review_active_question",
  "cbt_technique": "cbt_select_techniques",
  "homework": "homework_check_existing",
  "closing": "closing_share_summary",
  "anticipated_closing": "mi_end_session",
  "forced_closing": "forced_closing_end",
}

MIN_COGNITIVE_SET_EXCHANGES = 3


class ReconstructionError(ValueError):
  pass


def ensure_str(value: Any) -> str:
  if value is None:
    return ""
  return str(value)


def ensure_str_list(value: Any) -> list[str]:
  if not isinstance(value, list):
    return []
  return [ensure_str(item) for item in value if ensure_str(item)]


def new_cognitive_set(set_id: int) -> dict[str, Any]:
  return {
    "cognitive_model_set_id": set_id,
    "situation": "",
    "automatic_thoughts": [],
    "emotions": [],
    "behaviors": [],
    "core_belief": "",
    "core_belief_probabilities": {key: 0.0 for key in CORE_BELIEF_KEYS},
    "distortion": "",
    "distortion_probabilities": {key: 0.0 for key in DISTORTION_KEYS},
  }


def initial_state() -> dict[str, Any]:
  return {
    "current_stage": "opening",
    "current_substage": "open_initial",
    "opening_patient_mood": "",
    "opening_patient_problem": "",
    "last_reasoning": "",
    "predicted_resistance": None,
    "cognitive_model_sets": [],
    "working_cognitive_set": new_cognitive_set(1),
    "shelved_cognitive_sets": [],
    "cognitive_gap_plan": "",
    "active_cognitive_target": "",
    "active_cognitive_question": "",
    "active_cognitive_follow_up_attempts": 0,
    "working_cognitive_set_exchange_count": 0,
    "selected_techniques": [],
    "current_technique_index": 0,
    "current_technique_plan": "",
    "current_technique_expected_outcome": "",
    "current_technique_current_progress": "",
    "technique_evaluations": [],
    "techniques_completed": 0,
    "technique_turn_count": 0,
    "tasks": [],
    "homework_waiting_for_patient": False,
    "homework_proposal_attempts": 0,
    "homework_fallback_done": False,
    "homework_fallback_to_apply": False,
    "address_tib_name": "",
    "exchange_counts": {},
  }


def curated_context(state: dict[str, Any]) -> dict[str, Any]:
  return {
    "current_stage": state["current_stage"],
    "current_substage": state["current_substage"],
    "opening_patient_mood": state["opening_patient_mood"],
    "opening_patient_problem": state["opening_patient_problem"],
    "predicted_resistance": state["predicted_resistance"],
    "cognitive_model_sets": deepcopy(state["cognitive_model_sets"]),
    "working_cognitive_set": deepcopy(state["working_cognitive_set"]),
    "shelved_cognitive_sets": deepcopy(state["shelved_cognitive_sets"]),
    "cognitive_gap_plan": state["cognitive_gap_plan"],
    "active_cognitive_target": state["active_cognitive_target"],
    "active_cognitive_question": state["active_cognitive_question"],
    "selected_techniques": deepcopy(state["selected_techniques"]),
    "current_technique": deepcopy(current_technique(state)),
    "current_technique_plan": state["current_technique_plan"],
    "current_technique_expected_outcome": state["current_technique_expected_outcome"],
    "current_technique_current_progress": state["current_technique_current_progress"],
    "technique_evaluations": deepcopy(state["technique_evaluations"]),
    "tasks": deepcopy(state["tasks"]),
    "address_tib_name": state["address_tib_name"],
  }


def require_dict(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise ReconstructionError(f"{label} must be a JSON object.")
  return value


def require_list(value: Any, label: str) -> list[Any]:
  if not isinstance(value, list):
    raise ReconstructionError(f"{label} must be a JSON list.")
  return value


def normalize_probabilities(payload: dict[str, Any], keys: list[str], label: str) -> dict[str, float]:
  probs: dict[str, float] = {}
  for key in keys:
    if key not in payload:
      raise ReconstructionError(f"{label} is missing probability key {key!r}.")
    try:
      value = float(payload[key])
    except (TypeError, ValueError) as exc:
      raise ReconstructionError(f"{label}.{key} must be numeric.") from exc
    if value < 0.0 or value > 1.0:
      raise ReconstructionError(f"{label}.{key} must be in [0, 1].")
    probs[key] = value
  total = sum(probs.values())
  if total <= 0.0:
    raise ReconstructionError(f"{label} probabilities sum to zero.")
  return {key: value / total for key, value in probs.items()}


def best_key(probs: dict[str, float]) -> str:
  if not probs:
    return ""
  return max(probs.items(), key=lambda item: item[1])[0]


def parse_trace_output(trace: dict[str, Any]) -> Any:
  output = trace.get("output")
  if isinstance(output, str):
    stripped = output.strip()
    if not stripped:
      return ""
    try:
      return json.loads(stripped)
    except json.JSONDecodeError:
      return output
  return output


def merge_extracted_items(state: dict[str, Any], extracted: Any) -> None:
  if not isinstance(extracted, dict):
    return
  working = state["working_cognitive_set"]
  situation = ensure_str(extracted.get("situation"))
  thoughts = ensure_str_list(extracted.get("auto_thoughts"))
  emotions = ensure_str_list(extracted.get("emotions"))
  behaviors = ensure_str_list(extracted.get("behaviors"))

  if situation and not working.get("situation"):
    working["situation"] = situation

  for key, values in [
    ("automatic_thoughts", thoughts),
    ("emotions", emotions),
    ("behaviors", behaviors),
  ]:
    existing = list(working.get(key, []))
    for item in values:
      if item not in existing:
        existing.append(item)
    working[key] = existing


def working_set_base_complete(state: dict[str, Any]) -> bool:
  working = state["working_cognitive_set"]
  return bool(
    working.get("situation")
    and working.get("automatic_thoughts")
    and working.get("emotions")
    and working.get("behaviors")
  )


def working_set_full_complete(state: dict[str, Any]) -> bool:
  working = state["working_cognitive_set"]
  return working_set_base_complete(state) and bool(working.get("core_belief")) and bool(working.get("distortion"))


def working_set_has_content(state: dict[str, Any]) -> bool:
  working = state["working_cognitive_set"]
  return bool(
    working.get("situation")
    or working.get("automatic_thoughts")
    or working.get("emotions")
    or working.get("behaviors")
  )


def next_cognitive_set_id(state: dict[str, Any]) -> int:
  known_ids = [int(state["working_cognitive_set"].get("cognitive_model_set_id", 0))]
  known_ids.extend(int(item.get("cognitive_model_set_id", 0)) for item in state["cognitive_model_sets"])
  known_ids.extend(int(item.get("cognitive_model_set_id", 0)) for item in state["shelved_cognitive_sets"])
  return max(known_ids, default=0) + 1


def reset_cognitive_gap_tracking(state: dict[str, Any]) -> None:
  state["cognitive_gap_plan"] = ""
  state["active_cognitive_target"] = ""
  state["active_cognitive_question"] = ""
  state["active_cognitive_follow_up_attempts"] = 0


def is_cognitive_target_filled(state: dict[str, Any], target: str) -> bool:
  working = state["working_cognitive_set"]
  if target == "situation":
    return bool(working.get("situation"))
  if target == "auto_thoughts":
    return bool(working.get("automatic_thoughts"))
  if target == "emotions":
    return bool(working.get("emotions"))
  if target == "behaviors":
    return bool(working.get("behaviors"))
  return False


def ordered_cognitive_gap_targets(state: dict[str, Any]) -> list[str]:
  working = state["working_cognitive_set"]
  targets: list[str] = []
  if not working.get("situation"):
    targets.append("situation")
  if not working.get("automatic_thoughts"):
    targets.append("auto_thoughts")
  if not working.get("emotions"):
    targets.append("emotions")
  if not working.get("behaviors"):
    targets.append("behaviors")
  preferred = state["active_cognitive_target"]
  if preferred in targets:
    targets = [preferred] + [target for target in targets if target != preferred]
  return targets


def infer_gap_planning_target(state: dict[str, Any]) -> str:
  targets = ordered_cognitive_gap_targets(state)
  if not targets:
    raise ReconstructionError("cognitive_gap_planning appeared with no missing cognitive target.")
  return targets[0]


def commit_working_cognitive_set_if_complete(state: dict[str, Any]) -> None:
  if not working_set_full_complete(state):
    return
  state["cognitive_model_sets"].append(deepcopy(state["working_cognitive_set"]))
  state["working_cognitive_set"] = new_cognitive_set(next_cognitive_set_id(state))
  state["working_cognitive_set_exchange_count"] = 0
  reset_cognitive_gap_tracking(state)


def shelve_current_cognitive_set_and_begin_new(state: dict[str, Any]) -> None:
  if working_set_has_content(state):
    state["shelved_cognitive_sets"].append(deepcopy(state["working_cognitive_set"]))
  state["working_cognitive_set"] = new_cognitive_set(next_cognitive_set_id(state))
  state["working_cognitive_set_exchange_count"] = 0
  reset_cognitive_gap_tracking(state)


def task_key(task: dict[str, Any]) -> str:
  title = ensure_str(task.get("task_title")).strip()
  if title:
    return title.lower()
  return json.dumps(task, sort_keys=True, ensure_ascii=True)


def merge_tasks(state: dict[str, Any], tasks: Any, *, replace: bool = False) -> None:
  if not isinstance(tasks, list):
    raise ReconstructionError("tasks payload must be a JSON list.")
  normalized: list[dict[str, Any]] = []
  for item in tasks:
    normalized.append(require_dict(item, "task"))
  if replace:
    state["tasks"] = deepcopy(normalized)
    return

  by_key = {task_key(task): deepcopy(task) for task in state["tasks"]}
  order = [task_key(task) for task in state["tasks"]]
  for task in normalized:
    key = task_key(task)
    if key not in by_key:
      order.append(key)
    by_key[key] = deepcopy(task)
  state["tasks"] = [by_key[key] for key in order]


def minimal_homework_task() -> list[dict[str, Any]]:
  return [
    {
      "task_title": "Brief Observation Note",
      "task_description": "Once per day, write one short note about a problematic moment and one thought you noticed.",
      "rationale": "Minimal-effort task when resistance to homework remains high.",
      "expected_outcomes": "Maintain continuity without overload.",
    }
  ]


def current_technique(state: dict[str, Any]) -> dict[str, Any] | None:
  idx = int(state["current_technique_index"])
  if idx < 0 or idx >= len(state["selected_techniques"]):
    return None
  return state["selected_techniques"][idx]


def selected_technique_from_payload(payload: dict[str, Any], label: str) -> dict[str, Any]:
  name = ensure_str(payload.get("name"))
  if not name:
    raise ReconstructionError(f"{label}.name must be non-empty.")
  return {
    "name": name,
    "target_distortion": ensure_str(payload.get("target_distortion")),
    "rationale": ensure_str(payload.get("rationale")),
  }


def complete_current_technique(state: dict[str, Any]) -> None:
  state["techniques_completed"] += 1
  state["current_technique_index"] += 1
  state["current_technique_plan"] = ""
  state["current_technique_expected_outcome"] = ""
  state["current_technique_current_progress"] = ""
  state["technique_turn_count"] = 0


def set_stage_from_message(state: dict[str, Any], message: dict[str, Any]) -> None:
  stage = ensure_str(message.get("stage"))
  substage = ensure_str(message.get("substage"))
  if not stage or not substage:
    raise ReconstructionError("latest-format messages must include stage and substage.")
  if stage not in DEFAULT_SUBSTAGE_BY_STAGE:
    raise ReconstructionError(f"unknown stage {stage!r}.")
  state["current_stage"] = stage
  state["current_substage"] = substage


def increment_exchange_count(state: dict[str, Any], stage: str) -> int:
  history = state["exchange_counts"].setdefault(stage, [0])
  if not history:
    history.append(0)
  history[-1] = int(history[-1]) + 1
  return history[-1]


def apply_patient_message_effects(state: dict[str, Any], message: dict[str, Any]) -> None:
  stage = ensure_str(message.get("stage"))
  if not stage:
    return
  increment_exchange_count(state, stage)
  if stage == "id_distortions":
    state["working_cognitive_set_exchange_count"] = int(state["working_cognitive_set_exchange_count"]) + 1
  if stage == "cbt_technique" and state["current_technique_plan"]:
    state["technique_turn_count"] = int(state["technique_turn_count"]) + 1


def compact_message(message: dict[str, Any]) -> dict[str, Any]:
  return {
    "utterance_index": message["utterance_index"],
    "role": message["role"],
    "stage": message.get("stage"),
    "substage": message.get("substage"),
    "content": message.get("content", ""),
  }


def last_role_limited_messages(messages_so_far: list[dict[str, Any]]) -> list[dict[str, Any]]:
  therapist = [msg for msg in messages_so_far if msg.get("role") == "therapist"][-5:]
  patient = [msg for msg in messages_so_far if msg.get("role") == "patient"][-5:]
  selected = therapist + patient
  selected.sort(key=lambda msg: int(msg.get("utterance_index", 0)))
  return [compact_message(msg) for msg in selected]


def apply_trace_updates(
  state: dict[str, Any],
  trace_record: dict[str, Any],
  therapist_message: dict[str, Any],
) -> list[str]:
  warnings: list[str] = []
  pending_technique_complete = False
  traces = require_list(trace_record.get("traces"), "reasoning_traces.traces")

  for trace in traces:
    if not isinstance(trace, dict):
      warnings.append("trace entry is not an object")
      continue
    step = ensure_str(trace.get("step"))
    try:
      parsed = parse_trace_output(trace)
      if isinstance(parsed, dict) and ensure_str(parsed.get("reasoning")):
        state["last_reasoning"] = ensure_str(parsed.get("reasoning"))

      if step == "opening_info_update":
        payload = require_dict(parsed, step)
        mood = ensure_str(payload.get("mood", state["opening_patient_mood"]))
        problem = ensure_str(payload.get("problem", state["opening_patient_problem"]))
        if mood:
          state["opening_patient_mood"] = mood
        if problem:
          state["opening_patient_problem"] = problem

      elif step == "resistance_prediction":
        payload = require_dict(parsed, step)
        if "new_p" in payload:
          try:
            state["predicted_resistance"] = float(payload["new_p"])
          except (TypeError, ValueError) as exc:
            raise ReconstructionError("resistance_prediction.new_p must be numeric.") from exc

      elif step == "opening_readiness_check":
        payload = require_dict(parsed, step)
        state["last_reasoning"] = ensure_str(payload.get("reasoning", state["last_reasoning"]))

      elif step in {"tib_avoidance_new_cognitive_set"}:
        shelve_current_cognitive_set_and_begin_new(state)

      elif step in {"address_tib_enter", "address_tib_replaced"}:
        payload = require_dict(parsed, step)
        state["address_tib_name"] = ensure_str(payload.get("tib_name"))

      elif step == "address_tib_exit":
        state["address_tib_name"] = ""

      elif step == "initial_cognitive_model_bootstrap":
        payload = require_dict(parsed, step)
        merge_extracted_items(state, payload.get("items", {}))
        state["last_reasoning"] = ensure_str(payload.get("reasoning", state["last_reasoning"]))

      elif step in {"cognitive_gap_response_review", "cognitive_refinement_response_review"}:
        payload = require_dict(parsed, step)
        answered = bool(payload.get("answered_satisfactorily"))
        if answered:
          merge_extracted_items(state, payload.get("identified_items", {}))
          state["last_reasoning"] = ensure_str(payload.get("reasoning", state["last_reasoning"]))
          if step == "cognitive_refinement_response_review":
            reset_cognitive_gap_tracking(state)
          elif not state["active_cognitive_target"] or is_cognitive_target_filled(state, state["active_cognitive_target"]):
            reset_cognitive_gap_tracking(state)

      elif step == "cognitive_gap_best_guess":
        payload = require_dict(parsed, step)
        identified = payload.get("identified_items", payload)
        if state["active_cognitive_target"]:
          merge_extracted_items(state, identified)
        if isinstance(identified, dict) and ensure_str(identified.get("reasoning")):
          state["last_reasoning"] = ensure_str(identified.get("reasoning"))
        reset_cognitive_gap_tracking(state)

      elif step == "cognitive_gap_planning":
        payload = require_dict(parsed, step)
        state["cognitive_gap_plan"] = ensure_str(payload.get("plan"))
        if not state["cognitive_gap_plan"]:
          raise ReconstructionError("cognitive_gap_planning.plan must be non-empty.")
        state["active_cognitive_target"] = infer_gap_planning_target(state)
        state["active_cognitive_question"] = ensure_str(therapist_message.get("content"))
        state["active_cognitive_follow_up_attempts"] = 1

      elif step == "cognitive_refinement_planning":
        payload = require_dict(parsed, step)
        target = ensure_str(payload.get("target"))
        if target not in {"auto_thoughts", "emotions", "behaviors"}:
          raise ReconstructionError(f"{step}.target must be auto_thoughts, emotions, or behaviors.")
        state["cognitive_gap_plan"] = ensure_str(payload.get("plan"))
        if not state["cognitive_gap_plan"]:
          raise ReconstructionError("cognitive_refinement_planning.plan must be non-empty.")
        state["active_cognitive_target"] = target
        state["active_cognitive_question"] = ensure_str(therapist_message.get("content"))
        state["active_cognitive_follow_up_attempts"] = 1

      elif step == "core_belief_probability":
        payload = require_dict(parsed, step)
        probs = normalize_probabilities(payload, CORE_BELIEF_KEYS, step)
        state["working_cognitive_set"]["core_belief_probabilities"] = probs
        state["working_cognitive_set"]["core_belief"] = best_key(probs)

      elif step == "distortion_probability":
        payload = require_dict(parsed, step)
        probs = normalize_probabilities(payload, DISTORTION_KEYS, step)
        state["working_cognitive_set"]["distortion_probabilities"] = probs
        state["working_cognitive_set"]["distortion"] = best_key(probs)
        commit_working_cognitive_set_if_complete(state)

      elif step == "technique_selection":
        payload = require_dict(parsed, step)
        raw_selected = require_list(payload.get("selected_techniques"), "technique_selection.selected_techniques")
        state["selected_techniques"] = [
          selected_technique_from_payload(require_dict(item, "selected_techniques item"), step)
          for item in raw_selected[:3]
        ]

      elif step == "additional_technique_selection":
        payload = require_dict(parsed, step)
        technique = selected_technique_from_payload(payload, step)
        if any(isinstance(t, dict) and t.get("step") == "tib_avoidance_new_technique" for t in traces):
          idx = int(state["current_technique_index"])
          if idx < len(state["selected_techniques"]):
            state["selected_techniques"][idx] = technique
          else:
            state["selected_techniques"].append(technique)
            state["current_technique_index"] = len(state["selected_techniques"]) - 1
          state["current_technique_plan"] = ""
          state["current_technique_expected_outcome"] = ""
          state["current_technique_current_progress"] = ""
          state["technique_turn_count"] = 0
        else:
          state["selected_techniques"].append(technique)

      elif step == "technique_completion_check":
        payload = require_dict(parsed, step)
        pending_technique_complete = bool(payload.get("completed"))
        state["last_reasoning"] = ensure_str(payload.get("reasoning", state["last_reasoning"]))

      elif step == "technique_auto_complete":
        pending_technique_complete = True

      elif step == "technique_evaluation":
        state["technique_evaluations"].append(deepcopy(parsed))
        if pending_technique_complete:
          complete_current_technique(state)
          pending_technique_complete = False

      elif step == "mi_evaluation":
        state["technique_evaluations"].append(
          {
            "type": "mi_evaluation",
            "evaluation": deepcopy(parsed),
          }
        )

      elif step == "technique_fit_check":
        payload = require_dict(parsed, step)
        if not bool(payload.get("fits")):
          idx = int(state["current_technique_index"])
          current = current_technique(state) or {}
          replacement = {
            "name": ensure_str(payload.get("alternative_technique")),
            "target_distortion": ensure_str(current.get("target_distortion")),
            "rationale": ensure_str(payload.get("reasoning")),
          }
          # if not replacement["name"]:
          #   raise ReconstructionError("technique_fit_check.alternative_technique must be non-empty when fits=false.")
          if idx < len(state["selected_techniques"]):
            state["selected_techniques"][idx] = replacement
          else:
            state["selected_techniques"].append(replacement)

      elif step == "technique_plan":
        payload = require_dict(parsed, step)
        plan = ensure_str(payload.get("plan_description")) or ensure_str(payload.get("detailed_plan_description"))
        if not plan:
          raise ReconstructionError("technique_plan must include plan_description.")
        state["current_technique_plan"] = plan
        expected = ensure_str(payload.get("expected_outcome"))
        if expected:
          state["current_technique_expected_outcome"] = expected
        progress = ensure_str(payload.get("current_progress"))
        if progress:
          state["current_technique_current_progress"] = progress

      elif step == "homework_task_proposal":
        payload = require_dict(parsed, step)
        merge_tasks(state, payload.get("tasks"), replace=True)
        state["homework_proposal_attempts"] = int(state["homework_proposal_attempts"]) + 1
        state["homework_waiting_for_patient"] = True

      elif step == "homework_agreement_check":
        payload = require_dict(parsed, step)
        state["last_reasoning"] = ensure_str(payload.get("reasoning", state["last_reasoning"]))
        if bool(payload.get("agreement_reached")) and isinstance(payload.get("agreed_tasks"), list):
          merge_tasks(state, payload.get("agreed_tasks"), replace=False)
        state["homework_waiting_for_patient"] = False

      elif step == "homework_fallback":
        merge_tasks(state, minimal_homework_task(), replace=True)
        state["homework_fallback_to_apply"] = False
        state["homework_fallback_done"] = True

    except ReconstructionError as exc:
      warnings.append(f"{step or '<missing step>'}: {exc}")

  if pending_technique_complete:
    warnings.append("technique_completion_check completed=true without a following technique_evaluation trace")

  if (
    state["current_stage"] == "id_distortions"
    and ensure_str(therapist_message.get("stage")) == "id_distortions"
    and state["active_cognitive_target"]
    and state["active_cognitive_question"] != ensure_str(therapist_message.get("content"))
    and any(
      isinstance(trace, dict)
      and ensure_str(trace.get("step")) in {"cognitive_gap_response_review", "cognitive_refinement_response_review"}
      and isinstance(parse_trace_output(trace), dict)
      and not bool(parse_trace_output(trace).get("answered_satisfactorily"))
      for trace in traces
    )
  ):
    state["active_cognitive_question"] = ensure_str(therapist_message.get("content"))
    state["active_cognitive_follow_up_attempts"] = int(state["active_cognitive_follow_up_attempts"]) + 1

  return warnings


def validate_latest_message(message: Any) -> dict[str, Any]:
  msg = require_dict(message, "message")
  role = ensure_str(msg.get("role"))
  content = ensure_str(msg.get("content"))
  if role not in {"therapist", "patient"}:
    raise ReconstructionError(f"message role must be therapist or patient, got {role!r}.")
  if not content:
    raise ReconstructionError("message content must be non-empty.")
  if "utterance_index" not in msg:
    raise ReconstructionError("latest-format message is missing utterance_index.")
  try:
    idx = int(msg["utterance_index"])
  except (TypeError, ValueError) as exc:
    raise ReconstructionError("message utterance_index must be an integer.") from exc
  if idx <= 0:
    raise ReconstructionError("message utterance_index must be positive.")
  if not ensure_str(msg.get("stage")) or not ensure_str(msg.get("substage")):
    raise ReconstructionError("latest-format message is missing stage/substage.")
  normalized = dict(msg)
  normalized["utterance_index"] = idx
  normalized["role"] = role
  normalized["content"] = content
  return normalized


def trace_index(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
  result: dict[tuple[int, str], dict[str, Any]] = {}
  traces = require_list(payload.get("reasoning_traces"), "reasoning_traces")
  for item in traces:
    record = require_dict(item, "reasoning_trace")
    try:
      idx = int(record.get("utterance_index"))
    except (TypeError, ValueError) as exc:
      raise ReconstructionError("reasoning_trace.utterance_index must be an integer.") from exc
    speaker = ensure_str(record.get("speaker"))
    if not speaker:
      raise ReconstructionError("reasoning_trace.speaker must be non-empty.")
    result[(idx, speaker)] = record
  return result


def reconstruct_session(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
  payload = require_dict(payload, "session")
  raw_messages = require_list(payload.get("messages"), "messages")
  messages = [validate_latest_message(message) for message in raw_messages]
  traces_by_turn = trace_index(payload)

  state = initial_state()
  messages_so_far: list[dict[str, Any]] = []
  reconstructed_turns: list[dict[str, Any]] = []
  warnings: list[str] = []
  skipped_turns: list[int] = []

  for message in messages:
    role = message["role"]
    idx = int(message["utterance_index"])

    if role == "therapist":
      trace_record = traces_by_turn.get((idx, "therapist"))
      if trace_record is None:
        warnings.append(f"{path.name}: skipped therapist utterance {idx}: missing associated therapist reasoning trace.")
        skipped_turns.append(idx)
        messages_so_far.append(message)
        try:
          set_stage_from_message(state, message)
        except ReconstructionError as exc:
          warnings.append(f"{path.name}: therapist utterance {idx}: {exc}")
        continue

      context_before = curated_context(state)
      turn_warnings = apply_trace_updates(state, trace_record, message)
      if turn_warnings:
        warnings.extend(f"{path.name}: skipped therapist utterance {idx}: {warning}" for warning in turn_warnings)
        skipped_turns.append(idx)
      else:
        reconstructed_turns.append(
          {
            "utterance_index": idx,
            "stage": message.get("stage"),
            "substage": message.get("substage"),
            "therapist_utterance": message["content"],
            "last_utterances": last_role_limited_messages(messages_so_far),
            "reasoning_traces": deepcopy(trace_record.get("traces", [])),
            "context_before_therapist_turn": context_before,
          }
        )

      set_stage_from_message(state, message)
      messages_so_far.append(message)
      continue

    messages_so_far.append(message)
    apply_patient_message_effects(state, message)

  return {
    "source_file": path.name,
    "reconstruction_mode": "deterministic_forward_from_traces",
    "patient_profile": deepcopy(payload.get("patient_profile")),
    "turns": reconstructed_turns,
    "skipped_turns": skipped_turns,
    "warnings": warnings,
  }


def output_path_for(input_file: Path, output_dir: Path) -> Path:
  return output_dir / input_file.name


def reconstruct_directory(input_dir: Path, output_dir: Path) -> int:
  if not input_dir.is_dir():
    raise ReconstructionError(f"--input must be a directory: {input_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
  if not input_files:
    print(f"No JSON files found directly under {input_dir}", file=sys.stderr)
    return 0

  total_warnings = 0
  for input_file in input_files:
    out_file = output_path_for(input_file, output_dir)
    if input_file.resolve() == out_file.resolve():
      raise ReconstructionError(f"Refusing to overwrite input file in place: {input_file}")
    try:
      reconstruction = reconstruct_session(input_file)
    except (json.JSONDecodeError, OSError, ReconstructionError) as exc:
      total_warnings += 1
      print(f"WARNING {input_file.name}: skipped file: {exc}", file=sys.stderr)
      continue

    with out_file.open("w", encoding="utf-8") as f:
      json.dump(reconstruction, f, ensure_ascii=False, indent=2)
      f.write("\n")

    for warning in reconstruction["warnings"]:
      print(f"WARNING {warning}", file=sys.stderr)
    total_warnings += len(reconstruction["warnings"])
    print(f"Wrote {out_file} ({len(reconstruction['turns'])} therapist turns)", file=sys.stderr)
  return total_warnings


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Reconstruct curated context before each therapist utterance from simulation JSON outputs."
  )
  parser.add_argument("--input", required=True, help="Directory containing session JSON files.")
  parser.add_argument("--output", required=True, help="Directory where reconstructed JSON files will be written.")
  args = parser.parse_args()

  try:
    reconstruct_directory(Path(args.input), Path(args.output))
  except ReconstructionError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
