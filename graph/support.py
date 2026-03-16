from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from agents.interfaces import LLMFactory, StructuredOutputError, clamp, coerce_float, ensure_str, ensure_str_list
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent

from .state import CORE_BELIEF_KEYS, DEFAULT_SUBSTAGE_BY_STAGE, DISTORTION_KEYS, SessionState, Stage, SubStage


class ConversationProcessingError(RuntimeError):
  pass


class GraphSupportMixin:
  cfg: Any
  profile: dict
  therapist: TherapistAgent
  patient: PatientAgent
  cbt_techniques: dict[str, str]

  def _project_root(self) -> Path:
    return Path(__file__).resolve().parent.parent

  def _init_agents(self) -> None:
    therapist_llm = LLMFactory.create(self.cfg.therapist.type, self.cfg.therapist.model_id)
    patient_llm = LLMFactory.create(self.cfg.patient.type, self.cfg.patient.model_id)
    self.therapist = TherapistAgent(therapist_llm, self.profile)
    self.patient = PatientAgent(patient_llm, self.profile)

  def _load_techniques(self) -> dict[str, str]:
    techniques_path = self._project_root() / "data" / "knowledge" / "cactus_techniques.json"
    if not techniques_path.exists():
      raise ConversationProcessingError(f"Missing techniques file: {techniques_path}")
    try:
      with open(techniques_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load techniques from {techniques_path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ConversationProcessingError(f"Techniques file must contain a JSON object: {techniques_path}")
    return {
      ensure_str(name): ensure_str(description)
      for name, description in data.items()
      if ensure_str(name)
    }

  def _load_distortions(self) -> dict:
    distortions_path = self._project_root() / "data" / "knowledge" / "cognitive_distortions.json"
    if not distortions_path.exists():
      raise ConversationProcessingError(f"Missing cognitive distortions file: {distortions_path}")
    try:
      with open(distortions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load cognitive distortions from {distortions_path}: {exc}") from exc
    return {ensure_str(name): value for name, value in data.items() if ensure_str(name)}

  def initial_state(self, max_turns: int, min_turns: int = 0) -> SessionState:
    return self._ensure_state_defaults(
      {
        "max_turns": int(max_turns),
        "min_turns": int(min_turns),
      }
    )

  def resume_state(self, saved_state: SessionState, max_turns: int, min_turns: int = 0) -> SessionState:
    resumed: SessionState = deepcopy(saved_state)
    resumed["session_ended"] = False
    resumed["session_ended_reason"] = ""
    resumed["conversation_error"] = ""
    resumed["final_patient_pending_end"] = False
    if isinstance(resumed.get("reasoning_traces"), list):
      resumed["reasoning_traces"] = [
        trace for trace in resumed["reasoning_traces"] if not self._is_error_trace(trace)
      ]
    resumed["max_turns"] = int(max_turns)
    resumed["min_turns"] = int(min_turns)
    return self._ensure_state_defaults(resumed)

  def _initial_resistance(self) -> float:
    attitude = str(self.profile.get("attitude", "neutral")).lower()
    mapping = {"negative": 0.75, "neutral": 0.55, "positive": 0.35}
    return mapping.get(attitude, 0.55)

  def _is_error_trace(self, trace: Any) -> bool:
    if not isinstance(trace, dict):
      return False
    if trace.get("speaker") != "system":
      return False
    steps = trace.get("traces")
    if not isinstance(steps, list):
      return False
    return any(isinstance(step, dict) and step.get("step") == "conversation_error" for step in steps)

  def _ensure_state_defaults(self, state: SessionState) -> SessionState:
    s: SessionState = dict(state)
    stage = ensure_str(s.get("current_stage"), "opening")
    if stage not in DEFAULT_SUBSTAGE_BY_STAGE:
      stage = "opening"
    s["current_stage"] = stage
    s["current_substage"] = ensure_str(
      s.get("current_substage"),
      DEFAULT_SUBSTAGE_BY_STAGE[stage],
    )

    s.setdefault("messages", [])

    base_resistance = self._initial_resistance()
    s["resistance"] = coerce_float(s.get("resistance"), base_resistance, 0.0, 1.0)
    s["predicted_resistance"] = coerce_float(s.get("predicted_resistance"), 0.55, 0.0, 1.0)
    s.setdefault("patient_profile", deepcopy(self.profile))

    s.setdefault("opening_patient_mood", "")
    s.setdefault("opening_patient_problem", "")

    s.setdefault("cognitive_model_sets", [])
    s.setdefault("working_cognitive_set", self._new_cognitive_set(len(s["cognitive_model_sets"]) + 1))
    s.setdefault("working_cognitive_summary", "")
    s.setdefault("cognitive_gap_plan", "")
    s.setdefault("active_cognitive_target", "")
    s.setdefault("active_cognitive_question", "")
    s.setdefault("active_cognitive_follow_up_attempts", 0)

    s.setdefault("selected_techniques", [])
    s.setdefault("current_technique_index", 0)
    s.setdefault("current_technique_plan", "")
    s.setdefault("current_technique_expected_outcome", "")
    s.setdefault("current_technique_current_progress", "")
    s.setdefault("technique_evaluations", [])
    s.setdefault("techniques_completed", 0)
    s.setdefault("technique_turn_count", 0)
    s.setdefault("technique_slot_replaced", {})
    s.setdefault("is_completed_reasoning", "")

    s.setdefault("mi_active", False)
    s.setdefault("mi_context", [])
    s.setdefault("cbt_flow_saved_state", "")
    s.setdefault("cbt_resume_stage", "id_distortions")
    s.setdefault("mi_topics_tried", [])
    s.setdefault("mi_techniques_used", [])
    s.setdefault("meta_conversation_turns", 0)
    s.setdefault("mi_technique_turns", 0)
    s.setdefault("current_mi_topic", "")
    s.setdefault("last_mi_predicted_resistance", s["predicted_resistance"])

    s.setdefault("proposed_tasks", [])
    s.setdefault("homework_proposal_attempts", 0)
    s.setdefault("homework_waiting_for_patient", False)
    s.setdefault("homework_fallback_done", False)
    s.setdefault("agreed_tasks", [])
    s.setdefault("hw_reasoning", "")
    s.setdefault("homework_fallback_to_apply", False)

    s.setdefault("session_summary", "")
    s.setdefault("next_session_plan", "")
    s.setdefault("patient_feedback", "")
    s.setdefault("closing_plan_refinements", 0)

    s.setdefault("reasoning_traces", [])
    s["utterance_index"] = int(s.get("utterance_index", 0))
    s.setdefault("exchange_counts", {})
    s.setdefault("active_exchange_stage", "")

    s.setdefault("final_patient_pending_end", False)
    s.setdefault("session_ended", False)
    s.setdefault("session_ended_reason", "")
    s.setdefault("conversation_error", "")
    s.setdefault("opening_fallback_triggered", False)

    s["max_turns"] = int(s.get("max_turns", getattr(self.cfg.env, "max_turns", 80)))
    s["min_turns"] = int(s.get("min_turns", getattr(self.cfg.env, "min_turns", 0)))
    return s

  def _set_stage(self, state: SessionState, stage: Stage, substage: Optional[SubStage] = None) -> None:
    state["current_stage"] = stage
    state["current_substage"] = substage or DEFAULT_SUBSTAGE_BY_STAGE[stage]

  def _set_substage(self, state: SessionState, substage: SubStage) -> None:
    state["current_substage"] = substage

  def _format_conversation(self, messages: list[dict]) -> str:
    if not messages:
      return "(conversation just started)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

  def _build_session_context(self, state: SessionState) -> str:
    summary = {
      "predicted_resistance": state["predicted_resistance"],
      "opening_patient_mood": state["opening_patient_mood"],
      "opening_patient_problem": state["opening_patient_problem"],
      "cognitive_model_sets": state["cognitive_model_sets"],
      "working_cognitive_summary": state["working_cognitive_summary"],
      "cognitive_gap_plan": state["cognitive_gap_plan"],
      "active_cognitive_target": state["active_cognitive_target"],
      "active_cognitive_question": state["active_cognitive_question"],
      "active_cognitive_follow_up_attempts": state["active_cognitive_follow_up_attempts"],
      "selected_techniques": state["selected_techniques"],
      "techniques_completed": state["techniques_completed"],
      "technique_evaluations": state["technique_evaluations"][-3:],
      "mi_active": state["mi_active"],
      "mi_topics_tried": state["mi_topics_tried"],
      "mi_techniques_used": state["mi_techniques_used"][-5:],
      "cbt_resume_stage": state["cbt_resume_stage"],
      "proposed_tasks": state["proposed_tasks"],
      "agreed_tasks": state["agreed_tasks"],
      "exchange_counts": state["exchange_counts"],
    }
    return json.dumps(summary, ensure_ascii=True)

  def _append_trace(self, state: SessionState, utterance_index: int, speaker: str, traces: list[dict]) -> None:
    state["reasoning_traces"].append(
      {
        "utterance_index": utterance_index,
        "speaker": speaker,
        "traces": traces,
      }
    )

  def _append_message(self, state: SessionState, role: str, content: str) -> int:
    state["utterance_index"] += 1
    state["messages"].append({"role": role, "content": ensure_str(content)})
    return state["utterance_index"]

  def _current_band(self, value: float) -> str:
    if value >= 0.75:
      return "precontemplation"
    if value >= 0.50:
      return "contemplation"
    return "action"

  def _new_cognitive_set(self, set_id: int) -> dict:
    return {
      "cognitive_model_set_id": set_id,
      "situation": "",
      "automatic_thoughts": [],
      "emotions": [],
      "behaviors": [],
      "core_belief": "",
      "core_belief_probabilities": {k: 0.0 for k in CORE_BELIEF_KEYS},
      "distortion": "",
      "distortion_probabilities": {k: 0.0 for k in DISTORTION_KEYS},
    }

  def _normalize_probabilities(self, payload: dict, keys: list[str]) -> dict:
    if not isinstance(payload, dict):
      raise ConversationProcessingError("Probability payload must be a JSON object.")
    probs = {}
    for key in keys:
      if key not in payload:
        raise ConversationProcessingError(f"Probability payload is missing required key '{key}'.")
      raw_value = payload[key]
      try:
        parsed = float(raw_value)
      except (TypeError, ValueError) as exc:
        raise ConversationProcessingError(f"Probability '{key}' must be numeric, got {raw_value!r}.") from exc
      if parsed < 0.0 or parsed > 1.0:
        raise ConversationProcessingError(f"Probability '{key}' must be in [0, 1], got {parsed}.")
      probs[key] = parsed
    total = sum(probs.values())
    if total <= 0:
      raise ConversationProcessingError("Probability payload sums to 0 and cannot be normalized.")
    return {k: v / total for k, v in probs.items()}

  def _best_key(self, probs: dict) -> str:
    if not isinstance(probs, dict) or not probs:
      raise ConversationProcessingError("Cannot choose a best key from an empty probability map.")
    return max(probs.items(), key=lambda item: item[1])[0]

  def _require_dict(self, payload: Any, step_name: str) -> dict:
    if not isinstance(payload, dict):
      raise ConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON object.")
    return payload

  def _require_list(self, payload: Any, step_name: str) -> list:
    if not isinstance(payload, list):
      raise ConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON array.")
    return payload

  def _require_bool(self, payload: dict, key: str, step_name: str) -> bool:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required boolean field '{key}'.")
    value = payload[key]
    if isinstance(value, bool):
      return value
    if isinstance(value, str):
      lowered = value.strip().lower()
      if lowered in {"true", "false"}:
        return lowered == "true"
    raise ConversationProcessingError(f"{step_name}.{key} must be a boolean, got {value!r}.")

  def _require_bounded_float(self, payload: dict, key: str, step_name: str, low: float, high: float) -> float:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required numeric field '{key}'.")
    value = payload[key]
    try:
      parsed = float(value)
    except (TypeError, ValueError) as exc:
      raise ConversationProcessingError(f"{step_name}.{key} must be numeric, got {value!r}.") from exc
    if parsed < low or parsed > high:
      raise ConversationProcessingError(f"{step_name}.{key} must be in [{low}, {high}], got {parsed}.")
    return parsed

  def _require_non_empty_string(self, payload: dict, key: str, step_name: str) -> str:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required field '{key}'.")
    value = ensure_str(payload.get(key))
    if not value:
      raise ConversationProcessingError(f"{step_name}.{key} must be a non-empty string.")
    return value

  def _require_string_list(self, payload: dict, key: str, step_name: str, min_items: int = 1) -> list[str]:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required list field '{key}'.")
    values = ensure_str_list(payload.get(key))
    if len(values) < min_items:
      raise ConversationProcessingError(f"{step_name}.{key} must contain at least {min_items} item(s).")
    return values

  def _end_session_with_error(self, state: SessionState, traces: list[dict], error: Exception) -> None:
    message = ensure_str(str(error), "Unknown conversation processing error")
    state["conversation_error"] = message
    state["session_ended"] = True
    state["final_patient_pending_end"] = False
    if not state["session_ended_reason"]:
      state["session_ended_reason"] = "conversation_error"
    state["reasoning_traces"].append(
      {
        "utterance_index": state["utterance_index"],
        "speaker": "system",
        "traces": [
          {
            "step": "conversation_error",
            "input": state["current_stage"],
            "output": message,
          }
        ],
      }
    )

  def _trace_structured_step(
    self,
    traces: list[dict],
    step_name: str,
    instruction: str,
    parsed: Any,
  ) -> None:
    traces.append(
      {
        "step": step_name,
        "input": instruction,
        "output": json.dumps(parsed, ensure_ascii=True),
      }
    )

  def _run_therapist_structured(
    self,
    state: SessionState,
    traces: list[dict],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
  ) -> Any:
    try:
      parsed, _meta = self.therapist.generate_structured(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
        expected_type=expected_type,
      )
      print(f"    <DEBUG>{parsed}</DEBUG>") #devlog
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    self._trace_structured_step(traces, step_name, instruction, parsed)
    return parsed

  def _run_patient_structured(
    self,
    state: SessionState,
    traces: list[dict],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
  ) -> Any:
    try:
      parsed, _meta = self.patient.generate_structured(
        conversation=self._format_conversation(state["messages"]),
        resistance_level=state["resistance"],
        current_stage=state["current_stage"],
        instruction=instruction,
        expected_type=expected_type,
      )
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    self._trace_structured_step(traces, step_name, instruction, parsed)
    return parsed

  def _extract_last_patient_message(self, state: SessionState) -> str:
    for msg in reversed(state["messages"]):
      if msg["role"] == "patient":
        return msg["content"]
    return ""

  def _run_resistance_prediction(self, state: SessionState, traces: list[dict]) -> None:
    old_r = state["predicted_resistance"]
    instruction = (
      f"Given the conversation, last patient response, and previous predicted resistance R (R={old_r:.2f}), "
      "predict the change in patient resistance R. "
      "Predict the change in patient's resistance level for the next turn as a float between [-0.10, +0.15], "
      "where negative values indicate a decrease in resistance (more openness) and positive values indicate an increase in resistance (more pushback). "
      "The prediction must reflect how the patient responded to the therapist's last utterance. "
      "Additionally, provide your reasoning for this update. "
      'Respond JSON: {"r_change": 0.0, "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "resistance_prediction", instruction)
    payload = self._require_dict(parsed, "resistance_prediction")
    r_change = self._require_bounded_float(payload, "r_change", "resistance_prediction", -0.10, 0.15)
    state["predicted_resistance"] = clamp(state["predicted_resistance"] + r_change, 0.0, 1.0)

  def _maybe_force_closing(self, state: SessionState) -> None:
    if state["session_ended"] or state["final_patient_pending_end"]:
      return
    if state["current_stage"] in {"anticipated_closing", "forced_closing"}:
      return
    threshold = max(0, state["max_turns"] - 2)
    if state["utterance_index"] >= threshold:
      self._set_stage(state, "forced_closing")
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "max_turns_forced_closing"
