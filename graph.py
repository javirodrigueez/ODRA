from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from omegaconf import OmegaConf

from agents.interfaces import LLMFactory, StructuredOutputError, clamp, coerce_bool, coerce_float, ensure_str, ensure_str_list
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent

Stage = Literal[
  "opening",
  "mi_topic_exploration",
  "mi_techniques",
  "meta_conversation",
  "id_distortions",
  "cbt_technique",
  "homework",
  "closing",
  "anticipated_closing",
  "forced_closing",
]

SubStage = Literal[
  "open_initial",
  "open_mood",
  "open_problem"
]

CBT_STAGES = {"id_distortions", "cbt_technique", "homework"}
MI_STAGES = {"mi_topic_exploration", "mi_techniques", "meta_conversation"}
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
MI_TOPICS = ["Health", "Economy", "Interpersonal Relationships", "Law", "Education"]
MI_TECHNIQUES = [
  "Open Question",
  "Complex Reflection",
  "Simple Reflection",
  "Affirm",
  "Support",
  "Emphasize Control",
  "Reframe",
  "Inform",
]


class SessionState(TypedDict, total=False):
  current_stage: Stage
  current_substage: SubStage
  messages: list[dict]

  patient_profile: dict
  resistance: float
  predicted_resistance: float

  opening_patient_mood: str
  opening_patient_problem: str

  cognitive_model_sets: list[dict]
  current_situation_index: int
  current_extraction_target: str
  working_cognitive_set: dict

  selected_techniques: list[dict]
  current_technique_index: int
  current_technique_plan: str
  current_technique_steps: list[str]
  current_technique_expected_outcomes: list[str]
  technique_evaluations: list[dict]
  techniques_completed: int
  technique_turn_count: int
  technique_slot_replaced: dict

  mi_active: bool
  mi_context: list[dict]
  cbt_flow_saved_state: str
  cbt_resume_stage: str
  mi_topics_tried: list[str]
  mi_techniques_used: list[str]
  meta_conversation_turns: int
  mi_technique_turns: int
  current_mi_topic: str
  last_mi_predicted_resistance: float

  proposed_tasks: list[dict]
  homework_proposal_attempts: int
  homework_waiting_for_patient: bool
  homework_fallback_done: bool
  agreed_tasks: list[dict]

  session_summary: str
  next_session_plan: str
  patient_feedback: str
  closing_phase: str
  closing_plan_refinements: int

  reasoning_traces: list[dict]
  utterance_index: int
  exchange_counts: dict
  active_exchange_stage: str

  final_patient_pending_end: bool
  session_ended: bool
  session_ended_reason: str
  conversation_error: str
  opening_fallback_triggered: bool

  max_turns: int
  min_turns: int


class ConversationProcessingError(RuntimeError):
  pass


class TherapyGraphBuilder:
  def __init__(self, config_path: str, profile: dict):
    self.cfg = OmegaConf.load(config_path)
    self.profile = profile
    self._init_agents()
    self.cbt_techniques = self._load_techniques()

  def _init_agents(self):
    therapist_llm = LLMFactory.create(self.cfg.therapist.type, self.cfg.therapist.model_id)
    patient_llm = LLMFactory.create(self.cfg.patient.type, self.cfg.patient.model_id)
    self.therapist = TherapistAgent(therapist_llm, self.profile)
    self.patient = PatientAgent(patient_llm, self.profile)

  def _load_techniques(self) -> list[str]:
    techniques_path = Path(__file__).parent / "data" / "knowledge" / "cactus_techniques.json"
    if not techniques_path.exists():
      raise ConversationProcessingError(f"Missing techniques file: {techniques_path}")
    try:
      with open(techniques_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load techniques from {techniques_path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ConversationProcessingError(f"Techniques payload in {techniques_path} must be a JSON object.")
    techniques = [ensure_str(name) for name in data.keys() if ensure_str(name)]
    if not techniques:
      raise ConversationProcessingError(f"Techniques payload in {techniques_path} does not contain any usable technique names.")
    return techniques

  def initial_state(self, max_turns: int, min_turns: int = 0) -> SessionState:
    return self._ensure_state_defaults(
      {
        "max_turns": int(max_turns),
        "min_turns": int(min_turns),
      }
    )

  def _initial_resistance(self) -> float:
    attitude = str(self.profile.get("attitude", "neutral")).lower()
    mapping = {"negative": 0.75, "neutral": 0.55, "positive": 0.35}
    return mapping.get(attitude)

  def _ensure_state_defaults(self, state: SessionState) -> SessionState:
    s: SessionState = dict(state)
    s.setdefault("current_stage", "opening")
    s.setdefault("current_substage", "open_initial")
    s.setdefault("messages", [])

    base_resistance = self._initial_resistance()
    s["resistance"] = coerce_float(s.get("resistance"), base_resistance, 0.0, 1.0)
    s["predicted_resistance"] = coerce_float(s.get("predicted_resistance"), 0.55, 0.0, 1.0)
    s.setdefault("patient_profile", deepcopy(self.profile))

    s.setdefault("opening_patient_mood", "")
    s.setdefault("opening_patient_problem", "")

    s.setdefault("cognitive_model_sets", [])
    s.setdefault("current_situation_index", 0)
    s.setdefault("current_extraction_target", "situation")
    s.setdefault("working_cognitive_set", self._new_cognitive_set(len(s["cognitive_model_sets"]) + 1))

    s.setdefault("selected_techniques", [])
    s.setdefault("current_technique_index", 0)
    s.setdefault("current_technique_plan", "")
    s.setdefault("current_technique_steps", [])
    s.setdefault("current_technique_expected_outcomes", [])
    s.setdefault("technique_evaluations", [])
    s.setdefault("techniques_completed", 0)
    s.setdefault("technique_turn_count", 0)
    s.setdefault("technique_slot_replaced", {})

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

    s.setdefault("session_summary", "")
    s.setdefault("next_session_plan", "")
    s.setdefault("patient_feedback", "")
    s.setdefault("closing_phase", "summary")
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
      "selected_techniques": state["selected_techniques"],
      #"current_technique_index": state["current_technique_index"],
      #"current_technique_plan": state["current_technique_plan"],
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

  def _is_normal_cbt_stage(self, stage: str) -> bool:
    return stage in CBT_STAGES

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
    meta: dict,
  ) -> None:
    traces.append(
      {
        "step": step_name,
        "input": instruction,
        "output": json.dumps(
          {"parsed": parsed, "meta": meta},
          ensure_ascii=True,
        ),
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
      parsed, meta = self.therapist.generate_structured(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
        expected_type=expected_type,
      )
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    self._trace_structured_step(traces, step_name, instruction, parsed, meta)
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
      parsed, meta = self.patient.generate_structured(
        conversation=self._format_conversation(state["messages"]),
        resistance_level=state["resistance"],
        current_stage=state["current_stage"],
        instruction=instruction,
        expected_type=expected_type,
      )
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    self._trace_structured_step(traces, step_name, instruction, parsed, meta)
    return parsed

  def _enter_mi(self, state: SessionState, target_stage: str, resume_stage: Optional[str], reason: str) -> None:
    if resume_stage:
      state["cbt_resume_stage"] = resume_stage
    state["mi_active"] = True
    if not state["cbt_flow_saved_state"] and resume_stage:
      state["cbt_flow_saved_state"] = (
        f"CBT interrupted at stage={resume_stage}, "
        f"techniques_completed={state['techniques_completed']}, "
        f"working_set_id={state['working_cognitive_set'].get('cognitive_model_set_id')}"
      )
    state["current_stage"] = target_stage
    state["mi_context"].append({"role": "system_note", "content": reason})
    state["mi_technique_turns"] = 0 if target_stage == "mi_techniques" else state["mi_technique_turns"]
    state["meta_conversation_turns"] = 0 if target_stage == "meta_conversation" else state["meta_conversation_turns"]

  def _evaluate_mi_and_resume(self, state: SessionState, traces: list[dict], note: str = "") -> None:
    instruction = (
      "Using mi_context, create an MI intervention evaluation. Respond with JSON: "
      '{"reason_for_mi":"...", "conclusions":"...", "patient_state_change":"..."}'
    )
    mi_eval = self._require_dict(
      self._run_therapist_structured(state, traces, "mi_evaluation", instruction),
      "mi_evaluation",
    )
    if not ensure_str(mi_eval.get("reason_for_mi")):
      mi_eval["reason_for_mi"] = note or "resistance management"
    state["technique_evaluations"].append(
      {
        "type": "mi_evaluation",
        "evaluation": mi_eval,
        "resume_stage": state["cbt_resume_stage"],
      }
    )
    state["mi_active"] = False
    state["mi_technique_turns"] = 0
    state["meta_conversation_turns"] = 0
    state["last_mi_predicted_resistance"] = state["predicted_resistance"]
    state["current_stage"] = state["cbt_resume_stage"] or "id_distortions"

  def _extract_last_patient_message(self, state: SessionState) -> str:
    for msg in reversed(state["messages"]):
      if msg["role"] == "patient":
        return msg["content"]
    return ""

  def _select_initial_techniques(self, state: SessionState, traces: list[dict]) -> None:
    instruction = (
      "Given the conversation context, cognitive model sets, and identified distortions, choose up to 3 CBT techniques. "
      "Respond JSON: {\"selected_techniques\": [{\"name\":\"...\", \"target_distortion\":\"...\", \"rationale\":\"...\"}], \"reasoning\":\"...\"}"
    )
    parsed = self._run_therapist_structured(state, traces, "technique_selection", instruction)
    payload = self._require_dict(parsed, "technique_selection")
    chosen = payload.get("selected_techniques")
    if not isinstance(chosen, list):
      raise ConversationProcessingError("technique_selection.selected_techniques must be a list.")
    selected: list[dict] = []
    for item in chosen:
      if not isinstance(item, dict):
        raise ConversationProcessingError("technique_selection.selected_techniques items must be JSON objects.")
      selected.append(
        {
          "name": self._require_non_empty_string(item, "name", "technique_selection.selected_techniques"),
          "target_distortion": ensure_str(item.get("target_distortion")),
          "rationale": ensure_str(item.get("rationale")),
        }
      )
    if not selected:
      raise ConversationProcessingError("technique_selection.selected_techniques must contain at least one technique.")
    state["selected_techniques"] = selected[:3]

  def _select_additional_technique(self, state: SessionState, traces: list[dict]) -> dict:
    instruction = (
      "Choose one additional CBT technique that fits the current session after previous evaluations. "
      "Respond JSON: {\"name\":\"...\", \"target_distortion\":\"...\", \"rationale\":\"...\"}"
    )
    parsed = self._run_therapist_structured(state, traces, "additional_technique_selection", instruction)
    payload = self._require_dict(parsed, "additional_technique_selection")
    return {
      "name": self._require_non_empty_string(payload, "name", "additional_technique_selection"),
      "target_distortion": ensure_str(payload.get("target_distortion")),
      "rationale": ensure_str(payload.get("rationale")),
    }

  # TODO: IN PATIENT IT GIVES R CHANGE, NOT ABSOLUTE R. WHAT APPROACH TO FOLLOW?
  def _run_resistance_prediction(self, state: SessionState, traces: list[dict]) -> None:
    old_r = state["predicted_resistance"]
    instruction = (
      f"Given the conversation, last patient response, and previous predicted resistance R (R={old_r:.2f}). predict the change in patient resistance R. "
      "Predict the change in patient's resistance level for the next turn as a float between [-0.10, +0.15], where negative values indicate a decrease in resistance (more openness) and positive values indicate an increase in resistance (more pushback). "
      "The prediction must reflect how how the patient responded to the therapist's last utterance. "
      "Additionally, provide your reasoning for this update. "
      'Respond JSON: {"r_change": 0.0, "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "resistance_prediction", instruction)
    #print(f"  <think_ther>{parsed}</think_ther>") #devlog
    payload = self._require_dict(parsed, "resistance_prediction")
    r_change = self._require_bounded_float(
      payload,
      "r_change",
      "resistance_prediction",
      -0.10,
      0.15,
    )
    state["predicted_resistance"] = clamp(state["predicted_resistance"] + r_change, 0.0, 1.0)
    traces.append(
      {
        "step": "resistance_prediction",
        "input": f"old_r={old_r:.2f}",
        "output": f"predicted_r={state['predicted_resistance']:.2f}, reasoning={payload.get('reasoning', '')}",
      }
    )

  def _maybe_force_closing(self, state: SessionState) -> None:
    if state["session_ended"] or state["final_patient_pending_end"]:
      return
    if state["current_stage"] in {"anticipated_closing", "forced_closing"}:
      return
    threshold = max(0, state["max_turns"] - 2)
    if state["utterance_index"] >= threshold:
      state["current_stage"] = "forced_closing"
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "max_turns_forced_closing"

  def _handle_pre_stage_routing(self, state: SessionState, traces: list[dict]) -> None:
    stage = state["current_stage"]
    band = self._current_band(state["predicted_resistance"])

    if stage == "opening":
      return

    if stage in CBT_STAGES:
      if band == "contemplation":
        self._enter_mi(state, "mi_techniques", stage, "CBT interrupted due to contemplation-level resistance")
      elif band == "precontemplation":
        self._enter_mi(state, "mi_topic_exploration", stage, "CBT interrupted due to precontemplation resistance")
      return

    if stage == "mi_techniques":
      if band == "precontemplation":
        state["current_stage"] = "mi_topic_exploration"
        return
      if band == "action":
        self._evaluate_mi_and_resume(state, traces, note="MI reached action band")
        return
      if state["mi_technique_turns"] >= 10:
        traces.append(
          {
            "step": "mi_techniques_escape",
            "input": "MI techniques reached max 10 exchanges without full resolution.",
            "output": "Proceeding with CBT and noting unresolved resistance.",
          }
        )
        self._evaluate_mi_and_resume(state, traces, note="MI reached max-turn escape condition")
      return

    if stage == "mi_topic_exploration":
      if band in {"contemplation", "action"}:
        state["current_stage"] = "mi_techniques"
      return

    if stage == "meta_conversation":
      if band in {"contemplation", "action"}:
        state["current_stage"] = "mi_techniques"
      return

  def _check_readiness(self, state: SessionState, traces: list[dict]) -> dict:

    # Update mood/problem if needed
    instruction = (
      f"Based on the last patient response, the current saved mood='{state['opening_patient_mood']}' and problem='{state['opening_patient_problem']}'. Can patient's mood/problem be updated based on the new information? If so, provide the updated mood and/or problem. If not, respond with the current values.",
      "Respond JSON: {'mood': '...', 'problem': '...'}"
    )
    update = self._run_therapist_structured(state, traces, "opening_info_update", instruction)
    #print(f"  <think_ther>{update}</think_ther>") #devlog
    update_payload = self._require_dict(update, "opening_info_update")
    updated_mood = ensure_str(update_payload.get("mood", state["opening_patient_mood"]))
    updated_problem = ensure_str(update_payload.get("problem", state["opening_patient_problem"]))
    if updated_mood and updated_mood != state["opening_patient_mood"]:
      state["opening_patient_mood"] = updated_mood
    if updated_problem and updated_problem != state["opening_patient_problem"]:
      state["opening_patient_problem"] = updated_problem

    # Check readiness to move on
    instruction = (
      f"Based on the conversation so far, current saved mood='{state['opening_patient_mood']}' and problem='{state['opening_patient_problem']}', is there enough information to proceed beyond the opening stage? ",
      "Consider if the patient's mood and problem are sufficiently understood to move to CBT. "
      "If you are not sure about patient's mood or specific problem, respond with ready=False and specify the specific issue in the 'issue' field. If both fields are issues, begin exploring patient mood. Include a reasoning field to explain your decision. "
      "The specific problem should be a specific (e.g. the relationship with my family is degraded because drugs consumption), not just a general area (e.g. stress, anxiety)."
      'Respond JSON: {"ready": true/false, "issue": "mood/problem", "mood": "...", "problem": "...", "reasoning":"..."}'
    )
    readiness = self._run_therapist_structured(state, traces, "opening_readiness_check", instruction)
    print(f"  <think_ther>{readiness}</think_ther>") #devlog
    payload = self._require_dict(readiness, "opening_readiness_check")
    ready = self._require_bool(payload, "ready", "opening_readiness_check")
    issue = ensure_str(payload.get("issue", ""))
    state['current_substage'] = "open_problem" if issue == "problem" else "open_mood" if issue == "mood" else state['current_substage']
    return ready

  def _handle_opening(self, state: SessionState, traces: list[dict]) -> str:
    opening_exchanges = int(state["exchange_counts"].get("opening", 0))

    # First utterance
    if opening_exchanges == 0:
      instruction = (
        "This is the first utterance of the therapy session. Begin the session with an open and empathetic greeting. It should be brief (e.g. 'Hello! Glad to meet you.', 'Hi! How are you doing?', etc.)"
      )
      return self.therapist.generate_utterance(
        conversation=[],
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )
    
    # Check if ready to move on
    ready = self._check_readiness(state, traces)
    current_substage = state["current_substage"]

    # In case patient doesn't cooperates, after 20 exchanges force Topic Exploration to find engagement
    if opening_exchanges >= 20 and not ready:
      raise ConversationProcessingError("exchanges > 20. just trying...")
      state["opening_fallback_triggered"] = True
      state["cbt_resume_stage"] = "id_distortions"
      self._enter_mi(state, "mi_topic_exploration", "id_distortions", "Opening fallback after 20 exchanges")
      return self._handle_stage_dispatch(state, traces)

    # Continue gathering information
    if not ready:
      if current_substage == "open_mood":
        instruction = (
          "Based on the conversation so far, ask the patient a question to understand their current mood. This is the mood check of a standard CBT session."
        )
      elif current_substage == "open_problem":
        instruction = (
          "Based on the conversation so far, ask the patient a question to clarify the specific problem they want to work on. This should be a specific issue (e.g. the relationship with my family is degraded because drugs consumption), not just a general area (e.g. stress, anxiety)."
        )
      return self.therapist.generate_utterance(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )

    # Ready to move on
    if ready:
      raise ConversationProcessingError('Just trying...')
      band = self._current_band(state["predicted_resistance"])
      if band == "action":
        state["current_stage"] = "id_distortions"
      elif band == "contemplation":
        state["cbt_resume_stage"] = "id_distortions"
        self._enter_mi(state, "mi_techniques", "id_distortions", "Opening completed with contemplation")
      else:
        state["cbt_resume_stage"] = "id_distortions"
        self._enter_mi(state, "mi_topic_exploration", "id_distortions", "Opening completed with precontemplation")
      return self._handle_stage_dispatch(state, traces)

  def _merge_extracted_items(self, state: SessionState, extracted: dict) -> None:
    working = state["working_cognitive_set"]
    situation = ensure_str_list(extracted.get("situation"))
    thoughts = ensure_str_list(extracted.get("auto_thoughts"))
    emotions = ensure_str_list(extracted.get("emotions"))
    behaviors = ensure_str_list(extracted.get("behaviors"))

    if not working["situation"] and situation:
      working["situation"] = situation

    for key, values in [("automatic_thoughts", thoughts), ("emotions", emotions), ("behaviors", behaviors)]:
      existing = list(working.get(key, []))
      for item in values:
        if item not in existing:
          existing.append(item)
      working[key] = existing

  def _update_extraction_target(self, state: SessionState) -> None:
    working = state["working_cognitive_set"]
    if not working.get("situation"):
      state["current_extraction_target"] = "situation"
      return
    if not working.get("automatic_thoughts"):
      state["current_extraction_target"] = "auto_thoughts"
      return
    if not working.get("emotions"):
      state["current_extraction_target"] = "emotions"
      return
    if not working.get("behaviors"):
      state["current_extraction_target"] = "behaviors"
      return
    state["current_extraction_target"] = "core_belief"

  def _working_set_base_complete(self, state: SessionState) -> bool:
    working = state["working_cognitive_set"]
    return bool(
      working.get("situation")
      and working.get("automatic_thoughts")
      and working.get("emotions")
      and working.get("behaviors")
    )

  def _working_set_full_complete(self, state: SessionState) -> bool:
    working = state["working_cognitive_set"]
    return self._working_set_base_complete(state) and bool(working.get("core_belief")) and bool(working.get("distortion"))

  def _handle_id_distortions(self, state: SessionState, traces: list[dict]) -> str:
    last_patient = self._extract_last_patient_message(state)
    if last_patient:
      extraction_instruction = (
        "Based on the patient's last utterance, extract Beck cognitive-model items. "
        "If no items can be reliably extracted, respond with empty lists. "
        'Respond JSON: {"situation": <string>, "auto_thoughts":[], "emotions":[], "behaviors":[]}'
      )
      extracted = self._run_therapist_structured(
        state,
        traces,
        "cognitive_item_extraction",
        extraction_instruction,
      )
      payload = self._require_dict(extracted, "cognitive_item_extraction")
      self._merge_extracted_items(state, payload)

    if self._working_set_base_complete(state) and not state["working_cognitive_set"].get("core_belief"):
      belief_instruction = (
        "Given the current cognitive set (situation, thoughts, emotions, behaviors), assign probabilities to core belief categories. "
        'Respond JSON: {"helplessness": <float>, "unlovability": <float>, "worthlessness": <float>}. '
      )
      belief_probs = self._run_therapist_structured(
        state,
        traces,
        "core_belief_probability",
        belief_instruction,
      )
      #normalized_beliefs = self._normalize_probabilities(belief_probs, CORE_BELIEF_KEYS)
      #state["working_cognitive_set"]["core_belief_probabilities"] = normalized_beliefs
      #state["working_cognitive_set"]["core_belief"] = self._best_key(normalized_beliefs)
      state["working_cognitive_set"]["core_belief_probabilities"] = belief_probs
      state["working_cognitive_set"]["core_belief"] = self._best_key(belief_probs)

    if self._working_set_base_complete(state) and state["working_cognitive_set"].get("core_belief") and not state["working_cognitive_set"].get("distortion"):
      distortion_instruction = (
        "Given the cognitive set including core belief, assign probabilities to distortions. "
        "Respond JSON with keys: "
        + ", ".join(DISTORTION_KEYS)
      )
      distortion_parsed = self._run_therapist_structured(
        state,
        traces,
        "distortion_probability",
        distortion_instruction,
      )
      distortion_probs = self._normalize_probabilities(distortion_parsed, DISTORTION_KEYS)
      state["working_cognitive_set"]["distortion_probabilities"] = distortion_probs
      state["working_cognitive_set"]["distortion"] = self._best_key(distortion_probs)

    if self._working_set_full_complete(state):
      completed_set = deepcopy(state["working_cognitive_set"])
      state["cognitive_model_sets"].append(completed_set)
      state["current_situation_index"] = len(state["cognitive_model_sets"]) - 1
      state["working_cognitive_set"] = self._new_cognitive_set(len(state["cognitive_model_sets"]) + 1)
      state["current_extraction_target"] = "situation"

    self._update_extraction_target(state)
    completed_count = len(state["cognitive_model_sets"])
    id_exchanges = int(state["exchange_counts"].get("id_distortions", 0))
    enough_sets = completed_count >= 3 or (id_exchanges >= 20 and completed_count >= 2)

    if enough_sets:
      if id_exchanges >= 20 and completed_count == 2:
        traces.append(
          {
            "step": "id_distortions_fallback",
            "input": "20 exchanges reached with only 2 complete cognitive sets.",
            "output": "Proceeding with hard floor of 2 complete sets.",
          }
        )
      need_more_instruction = (
        "Based on collected cognitive model sets, does the therapist need more information before technique application? "
        'Respond JSON: {"needs_more": true/false, "reasoning":"..."}'
      )
      need_more = self._run_therapist_structured(
        state,
        traces,
        "id_distortions_completion_check",
        need_more_instruction,
      )
      needs_more = self._require_bool(
        self._require_dict(need_more, "id_distortions_completion_check"),
        "needs_more",
        "id_distortions_completion_check",
      )
      if not needs_more:
        state["current_stage"] = "cbt_technique"
        return self._handle_stage_dispatch(state, traces)

    target = state["current_extraction_target"]
    situation = state["working_cognitive_set"].get("situation") or "the current problem situation"
    if target == "situation":
      return "Can you walk me through a specific recent situation where this problem became intense for you?"
    if target == "auto_thoughts":
      return f"In that situation ({situation}), what thoughts went through your mind right away?"
    if target == "emotions":
      return f"When that happened, what emotions did you notice most strongly, and how intense were they?"
    if target == "behaviors":
      return f"After those thoughts and feelings, what did you do or avoid doing?"
    return "Let’s explore another similar situation so we can understand this pattern more clearly."

  def _current_technique(self, state: SessionState) -> Optional[dict]:
    idx = state["current_technique_index"]
    if idx < 0:
      return None
    if idx >= len(state["selected_techniques"]):
      return None
    return state["selected_techniques"][idx]

  def _evaluate_technique_completion(self, state: SessionState, traces: list[dict]) -> bool:
    if not state["current_technique_plan"]:
      return False
    if state["technique_turn_count"] >= 12:
      traces.append(
        {
          "step": "technique_auto_complete",
          "input": "Technique reached 12 exchange turns without completion.",
          "output": "Automatically marking technique as complete.",
        }
      )
      return True
    instruction = (
      "Based on the technique plan and conversation, has current technique completed all steps? "
      "(Minimum 4 exchanges required). Respond JSON: {\"completed\": true/false, \"reasoning\":\"...\"}"
    )
    parsed = self._run_therapist_structured(state, traces, "technique_completion_check", instruction)
    completed = self._require_bool(
      self._require_dict(parsed, "technique_completion_check"),
      "completed",
      "technique_completion_check",
    )
    return completed and state["technique_turn_count"] >= 4

  def _evaluate_technique(self, state: SessionState, traces: list[dict], technique_name: str) -> None:
    instruction = (
      "Evaluate the completed technique. Respond JSON with fields: "
      "{\"patient_resistance\":\"...\", \"results\":\"...\", \"expected_outcomes_fulfilled\":true/false, "
      "\"problems\":\"...\", \"follow_up\":\"...\"}"
    )
    evaluation = self._require_dict(
      self._run_therapist_structured(state, traces, "technique_evaluation", instruction),
      "technique_evaluation",
    )
    state["technique_evaluations"].append(
      {
        "technique": technique_name,
        "evaluation": evaluation,
        "turns": state["technique_turn_count"],
      }
    )

  def _should_apply_more_techniques(self, state: SessionState, traces: list[dict]) -> bool:
    instruction = (
      "Given conversation, cognitive sets, distortions and technique evaluations, should we apply an additional technique today? "
      'Respond JSON: {"apply_more": true/false, "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "apply_more_techniques_check", instruction)
    return self._require_bool(
      self._require_dict(parsed, "apply_more_techniques_check"),
      "apply_more",
      "apply_more_techniques_check",
    )

  def _handle_cbt_technique(self, state: SessionState, traces: list[dict]) -> str:
    if not state["selected_techniques"]:
      self._select_initial_techniques(state, traces)

    current = self._current_technique(state)
    if current and state["current_technique_plan"] and state["technique_turn_count"] > 0:
      if self._evaluate_technique_completion(state, traces):
        self._evaluate_technique(state, traces, current["name"])
        state["techniques_completed"] += 1
        state["current_technique_index"] += 1
        state["current_technique_plan"] = ""
        state["current_technique_steps"] = []
        state["current_technique_expected_outcomes"] = []
        state["technique_turn_count"] = 0

    if state["techniques_completed"] >= 5:
      state["current_stage"] = "homework"
      return self._handle_stage_dispatch(state, traces)

    if state["techniques_completed"] >= 3 and not state["current_technique_plan"]:
      apply_more = False
      if state["techniques_completed"] in {3, 4}:
        apply_more = self._should_apply_more_techniques(state, traces)
      if not apply_more:
        state["current_stage"] = "homework"
        return self._handle_stage_dispatch(state, traces)
      if state["current_technique_index"] >= len(state["selected_techniques"]) and len(state["selected_techniques"]) < 5:
        state["selected_techniques"].append(self._select_additional_technique(state, traces))

    if state["current_technique_index"] >= len(state["selected_techniques"]):
      state["selected_techniques"].append(self._select_additional_technique(state, traces))

    current = self._current_technique(state)
    if current is None:
      raise ConversationProcessingError("Current CBT technique could not be resolved.")

    slot_idx = state["current_technique_index"]
    if not state["current_technique_plan"] and state["techniques_completed"] > 0:
      fit_instruction = (
        f'The next planned technique is "{current["name"]}". Does it fit now? '
        'Respond JSON: {"fits": true/false, "reasoning":"...", "alternative_technique":"<name or null>", "alternative_reasoning":"..."}'
      )
      fit_parsed = self._require_dict(
        self._run_therapist_structured(state, traces, "technique_fit_check", fit_instruction),
        "technique_fit_check",
      )
      fits = self._require_bool(fit_parsed, "fits", "technique_fit_check")
      alt_name = ensure_str(fit_parsed.get("alternative_technique"))
      already_replaced = bool(state["technique_slot_replaced"].get(str(slot_idx), False))
      if not fits and alt_name and not already_replaced:
        current = {
          "name": alt_name,
          "target_distortion": current.get("target_distortion", ""),
          "rationale": ensure_str(fit_parsed.get("alternative_reasoning", current.get("rationale", ""))),
        }
        state["selected_techniques"][slot_idx] = current
        state["technique_slot_replaced"][str(slot_idx)] = True

    if not state["current_technique_plan"]:
      plan_instruction = (
        f'Create a technique plan for "{current["name"]}" for this patient. '
        'Respond JSON: {"technique_plan":"...", "steps":["..."], "expected_outcomes":["..."]}'
      )
      plan_parsed = self._require_dict(
        self._run_therapist_structured(state, traces, "technique_plan", plan_instruction),
        "technique_plan",
      )
      state["current_technique_plan"] = self._require_non_empty_string(plan_parsed, "technique_plan", "technique_plan")
      state["current_technique_steps"] = self._require_string_list(plan_parsed, "steps", "technique_plan")
      state["current_technique_expected_outcomes"] = self._require_string_list(
        plan_parsed,
        "expected_outcomes",
        "technique_plan",
      )

    return (
      f'Let’s work with "{current["name"]}" now. '
      f'{state["current_technique_plan"]} '
      "I’ll guide this step by step and we’ll see what changes for you."
    )

  def _propose_homework_tasks(self, state: SessionState, traces: list[dict]) -> None:
    instruction = (
      "Propose 1-2 homework tasks based on cognitive sets, distortions, technique evaluations and context. "
      "Respond JSON: {\"tasks\": [{\"task_title\":\"...\", \"task_description\":\"...\", \"rationale\":\"...\", \"expected_outcomes\":\"...\"}], \"reasoning\":\"...\"}"
    )
    parsed = self._run_therapist_structured(state, traces, "homework_task_proposal", instruction)
    payload = self._require_dict(parsed, "homework_task_proposal")
    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list):
      raise ConversationProcessingError("homework_task_proposal.tasks must be a list.")
    tasks = []
    for item in tasks_raw:
      if not isinstance(item, dict):
        raise ConversationProcessingError("homework_task_proposal.tasks items must be JSON objects.")
      tasks.append(
        {
          "task_title": self._require_non_empty_string(item, "task_title", "homework_task_proposal.tasks"),
          "task_description": self._require_non_empty_string(item, "task_description", "homework_task_proposal.tasks"),
          "rationale": ensure_str(item.get("rationale")),
          "expected_outcomes": ensure_str(item.get("expected_outcomes")),
        }
      )
    if not tasks:
      raise ConversationProcessingError("homework_task_proposal.tasks must contain at least one task.")
    state["proposed_tasks"] = tasks[:2]
    state["homework_proposal_attempts"] += 1
    state["homework_waiting_for_patient"] = True

  def _check_homework_agreement(self, state: SessionState, traces: list[dict]) -> bool:
    instruction = (
      "Has agreement been reached for at least one homework task? "
      'Respond JSON: {"agreement_reached": true/false, "agreed_tasks": [...], "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "homework_agreement_check", instruction)
    payload = self._require_dict(parsed, "homework_agreement_check")
    agreed = self._require_bool(payload, "agreement_reached", "homework_agreement_check")
    agreed_tasks_raw = payload.get("agreed_tasks")
    if agreed and isinstance(agreed_tasks_raw, list):
      normalized = []
      for item in agreed_tasks_raw:
        if isinstance(item, dict):
          title = ensure_str(item.get("task_title"))
          if title:
            normalized.append(item)
      if normalized:
        state["agreed_tasks"] = normalized
      else:
        raise ConversationProcessingError("homework_agreement_check.agreed_tasks must contain at least one agreed task.")
    return agreed

  def _minimal_homework_task(self) -> list[dict]:
    return [
      {
        "task_title": "Brief Observation Note",
        "task_description": "Once per day, write one short note about a stressful moment and one thought you noticed.",
        "rationale": "Minimal-effort task when resistance to homework remains high.",
        "expected_outcomes": "Maintain continuity without overload.",
      }
    ]

  def _handle_homework(self, state: SessionState, traces: list[dict]) -> str:
    if state["agreed_tasks"] and not state["homework_waiting_for_patient"]:
      state["current_stage"] = "closing"
      return self._handle_stage_dispatch(state, traces)

    if state["homework_waiting_for_patient"]:
      agreed = self._check_homework_agreement(state, traces)
      state["homework_waiting_for_patient"] = False
      if agreed:
        state["current_stage"] = "closing"
        return self._handle_stage_dispatch(state, traces)
      if state["homework_proposal_attempts"] >= 3:
        state["proposed_tasks"] = self._minimal_homework_task()
        state["agreed_tasks"] = state["proposed_tasks"]
        state["homework_fallback_done"] = True
        traces.append(
          {
            "step": "homework_fallback",
            "input": "3 homework proposal rounds exhausted without agreement.",
            "output": "Using simplified minimal-task suggestion and proceeding.",
          }
        )
        return "We can keep it very light: would you try one brief daily observation note this week, as a gentle experiment?"

    if state["homework_fallback_done"] and state["agreed_tasks"]:
      state["current_stage"] = "closing"
      return self._handle_stage_dispatch(state, traces)

    self._propose_homework_tasks(state, traces)
    task_lines = [f'- {t["task_title"]}: {t["task_description"]}' for t in state["proposed_tasks"]]
    task_text = "\n".join(task_lines)
    return (
      "Based on today’s work, I’d like to suggest 1-2 homework options:\n"
      f"{task_text}\n"
      "Which of these feels realistic for you to do before our next session?"
    )

  def _generate_closing_summary(self, state: SessionState, traces: list[dict]) -> str:
    instruction = (
      "Create a brief session summary using opening mood/problem, cognitive sets, techniques, and homework. "
      'Respond JSON: {"session_summary":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "closing_summary_generation", instruction)
    payload = self._require_dict(parsed, "closing_summary_generation")
    return self._require_non_empty_string(payload, "session_summary", "closing_summary_generation")

  def _generate_next_session_plan(self, state: SessionState, traces: list[dict]) -> str:
    instruction = (
      "Create a concise plan for next session based on today's outcomes and homework. "
      'Respond JSON: {"next_session_plan":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "next_session_plan_generation", instruction)
    payload = self._require_dict(parsed, "next_session_plan_generation")
    return self._require_non_empty_string(payload, "next_session_plan", "next_session_plan_generation")

  def _handle_closing(self, state: SessionState, traces: list[dict]) -> str:
    phase = state["closing_phase"]
    if phase == "summary":
      if not state["session_summary"]:
        state["session_summary"] = self._generate_closing_summary(state, traces)
      state["closing_phase"] = "plan_discussion"
      return f"{state['session_summary']} Before we finish, does that summary fit your experience today?"

    if phase == "plan_discussion":
      if not state["next_session_plan"]:
        state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      state["closing_phase"] = "await_plan_response"
      return f"For next time, my proposed plan is: {state['next_session_plan']} How does that sound to you?"

    if phase == "await_plan_response":
      agreement_instruction = (
        "Has the patient agreed with the next session plan? "
        'Respond JSON: {"agreed": true/false, "reasoning":"..."}'
      )
      agreement = self._run_therapist_structured(
        state,
        traces,
        "closing_plan_agreement_check",
        agreement_instruction,
      )
      agreed = self._require_bool(
        self._require_dict(agreement, "closing_plan_agreement_check"),
        "agreed",
        "closing_plan_agreement_check",
      )

      if agreed or state["closing_plan_refinements"] >= 3:
        state["closing_phase"] = "await_feedback_response"
        return "Before we end, what feedback do you have about today’s session?"

      state["closing_plan_refinements"] += 1
      state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      return f"Thanks for that feedback. Let me adjust the plan: {state['next_session_plan']} Would this version work better?"

    if phase == "await_feedback_response":
      feedback_instruction = (
        "Did the patient request something the therapist should improve next session? "
        'Respond JSON: {"improvement_requested": true/false, "improvement_description":"...", "improvement_plan":"..."}'
      )
      feedback = self._run_therapist_structured(
        state,
        traces,
        "closing_feedback_check",
        feedback_instruction,
      )
      feedback_payload = self._require_dict(feedback, "closing_feedback_check")
      requested = self._require_bool(feedback_payload, "improvement_requested", "closing_feedback_check")
      description = ensure_str(feedback_payload.get("improvement_description"))
      plan = ensure_str(feedback_payload.get("improvement_plan"))
      state["patient_feedback"] = description

      if requested:
        state["closing_phase"] = "final_close_pending"
        return (
          "Thank you for that feedback. "
          f"I hear that {description or 'you want us to adjust the process'}. "
          f"Next session I’ll improve by {plan or 'adapting pacing and focus to your preference'}."
        )

      state["closing_phase"] = "completed"
      state["final_patient_pending_end"] = True
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "normal_closing_completed"
      return "Thank you for your work today. We’ll pick this up next time and build on what you practiced this week."

    if phase == "final_close_pending":
      state["closing_phase"] = "completed"
      state["final_patient_pending_end"] = True
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "normal_closing_completed"
      return "I appreciate your openness today. Let’s pause here, and we’ll continue from this point next session."

    state["final_patient_pending_end"] = True
    if not state["session_ended_reason"]:
      state["session_ended_reason"] = "normal_closing_completed"
    return "We can stop here for today. Thank you for showing up and doing this work."

  def _handle_mi_topic_exploration(self, state: SessionState, traces: list[dict]) -> str:
    previous_pred = state["last_mi_predicted_resistance"]
    current_pred = state["predicted_resistance"]
    need_new_topic = (
      not state["current_mi_topic"] or current_pred > previous_pred + 0.01
    )
    if need_new_topic:
      available = [t for t in MI_TOPICS if t not in state["mi_topics_tried"]]
      if not available:
        available = MI_TOPICS
      topic_instruction = (
        f"Select the best MI topic to explore next from: {', '.join(available)}. "
        'Respond JSON: {"topic":"...", "reasoning":"..."}'
      )
      topic = self._run_therapist_structured(state, traces, "mi_topic_selection", topic_instruction)
      topic_payload = self._require_dict(topic, "mi_topic_selection")
      chosen = self._require_non_empty_string(topic_payload, "topic", "mi_topic_selection")
      if chosen not in MI_TOPICS:
        raise ConversationProcessingError(f"mi_topic_selection.topic must be one of {MI_TOPICS}, got {chosen!r}.")
      if chosen != state.get("current_mi_topic"):
        state["current_mi_topic"] = chosen
        if chosen not in state["mi_topics_tried"]:
          state["mi_topics_tried"].append(chosen)

    state["last_mi_predicted_resistance"] = current_pred
    if len(state["mi_topics_tried"]) >= 3 and self._current_band(state["predicted_resistance"]) == "precontemplation":
      state["current_stage"] = "meta_conversation"
      state["meta_conversation_turns"] = 0
      return self._handle_stage_dispatch(state, traces)

    topic = state["current_mi_topic"] or "Health"
    return (
      f"I’d like to look at one area that might matter to you: {topic}. "
      "How do you see this current struggle affecting that part of your life?"
    )

  def _handle_mi_techniques(self, state: SessionState, traces: list[dict]) -> str:
    band = self._current_band(state["predicted_resistance"])
    if band == "precontemplation":
      state["current_stage"] = "mi_topic_exploration"
      return self._handle_stage_dispatch(state, traces)
    if band == "action":
      self._evaluate_mi_and_resume(state, traces, note="MI techniques reached action")
      return self._handle_stage_dispatch(state, traces)

    instruction = (
      f"Choose one MI communication technique from {', '.join(MI_TECHNIQUES)} for the next utterance. "
      'Respond JSON: {"technique":"...", "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "mi_technique_selection", instruction)
    payload = self._require_dict(parsed, "mi_technique_selection")
    technique = self._require_non_empty_string(payload, "technique", "mi_technique_selection")
    if technique not in MI_TECHNIQUES:
      raise ConversationProcessingError(f"mi_technique_selection.technique must be one of {MI_TECHNIQUES}, got {technique!r}.")
    state["mi_techniques_used"].append(technique)
    return (
      f"I want to stay with you in this. Using a {technique} approach: "
      "what feels most difficult about making even a small change right now?"
    )

  def _handle_meta_conversation(self, state: SessionState, traces: list[dict]) -> str:
    turns = state["meta_conversation_turns"]
    if turns >= 8 and self._current_band(state["predicted_resistance"]) == "precontemplation":
      state["current_stage"] = "anticipated_closing"
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "anticipated_closing"
      return self._handle_stage_dispatch(state, traces)

    if turns == 0:
      return (
        "It seems this process has felt difficult, and I appreciate you staying here. "
        "What would make therapy feel safer or more useful for you right now?"
      )
    if turns == 1:
      return "What specific parts of this process feel hardest or least helpful from your perspective?"
    if turns == 2:
      return (
        "Thank you for naming that. "
        "Would it help if we adjusted the pace and focused only on one small, practical point today?"
      )
    return "I hear you. What would feel like a manageable next step that still respects your boundaries?"

  def _handle_anticipated_closing(self, state: SessionState, traces: list[dict]) -> str:
    state["final_patient_pending_end"] = True
    if not state["session_ended_reason"]:
      state["session_ended_reason"] = "anticipated_closing"
    traces.append(
      {
        "step": "anticipated_closing",
        "input": "Meta-conversation reached max turns with persistent precontemplation.",
        "output": "Ending session with warm open-door close.",
      }
    )
    return "Thank you for attending today. Even if this wasn’t the right moment to go deeper, you’re welcome to return whenever you feel ready."

  def _handle_forced_closing(self, state: SessionState, traces: list[dict]) -> str:
    state["final_patient_pending_end"] = True
    state["session_ended_reason"] = "max_turns_forced_closing"
    traces.append(
      {
        "step": "max_turns_forced_closing",
        "input": f"Reached max_turns threshold ({state['max_turns']}).",
        "output": "Triggering brief forced closing exchange.",
      }
    )
    return "We need to pause here for today. Thank you for your time, and we can continue from this point next session."

  def _handle_stage_dispatch(self, state: SessionState, traces: list[dict]) -> str:
    stage = state["current_stage"]
    if stage == "opening":
      return self._handle_opening(state, traces)
    if stage == "id_distortions":
      return self._handle_id_distortions(state, traces)
    if stage == "cbt_technique":
      return self._handle_cbt_technique(state, traces)
    if stage == "homework":
      return self._handle_homework(state, traces)
    if stage == "closing":
      return self._handle_closing(state, traces)
    if stage == "mi_topic_exploration":
      return self._handle_mi_topic_exploration(state, traces)
    if stage == "mi_techniques":
      return self._handle_mi_techniques(state, traces)
    if stage == "meta_conversation":
      return self._handle_meta_conversation(state, traces)
    if stage == "anticipated_closing":
      return self._handle_anticipated_closing(state, traces)
    if stage == "forced_closing":
      return self._handle_forced_closing(state, traces)
    state["current_stage"] = "opening"
    return self._handle_opening(state, traces)

  def therapist_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    first_exchange = len(s["messages"]) == 0
    if s["session_ended"]:
      return s

    traces: list[dict] = []
    try:
      if not first_exchange:
        self._maybe_force_closing(s)
        self._run_resistance_prediction(s, traces)
        self._handle_pre_stage_routing(s, traces)
      utterance = self._handle_stage_dispatch(s, traces)

      s["active_exchange_stage"] = s["current_stage"]
      idx = self._append_message(s, "therapist", utterance)
      if s["current_stage"] in MI_STAGES or s["current_stage"] in {"anticipated_closing"}:
        s["mi_context"].append({"role": "therapist", "content": utterance})
      self._append_trace(s, idx, "therapist", traces)
    except ConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s

  def _patient_utterance_instruction(self, state: SessionState) -> str:
    stage = state["current_stage"]
    if state["final_patient_pending_end"]:
      return "Provide a brief closing response."
    if stage == "opening":
      return "Respond naturally to the therapist's opening and share only what feels comfortable."
    if stage == "id_distortions":
      return "Answer the therapist's question based on your real thoughts, emotions, and behavior."
    if stage == "cbt_technique":
      return "Respond to the CBT exercise with realistic engagement for your current resistance."
    if stage == "homework":
      return "Respond honestly about whether the proposed homework feels feasible."
    if stage in {"mi_topic_exploration", "mi_techniques", "meta_conversation"}:
      return "Respond to the motivational discussion in line with your current openness."
    if stage in {"closing", "anticipated_closing", "forced_closing"}:
      return "Respond briefly to close the session."
    return "Respond naturally."

  def patient_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    old_r = s["resistance"]

    if s["session_ended"]:
      return s

    traces: list[dict] = []
    try:
      update_instruction = (
        f"Given the conversation and current resistance {s['resistance']:.2f} to engage in treatment. "
        "Predict the change in patient's resistance level for the next turn as a float between [-0.10, +0.15], where negative values indicate a decrease in resistance (more openness) and positive values indicate an increase in resistance (more pushback). "
        "The prediction must reflect how the last therapist message helped to address patient's problems. "
        "Additionally, provide your reasoning for this update. "
        'Respond JSON: {"r_change": 0.0, "reasoning":"..."}'
      )
      update = self._run_patient_structured(
        s,
        traces,
        "resistance_update",
        update_instruction,
      )
      #print(f"  <thinkPat>{update}</thinkPat>") #devlog

      payload = self._require_dict(update, "resistance_update")
      r_change = self._require_bounded_float(payload, "r_change", "resistance_update", -0.10, 0.15)
      s["resistance"] = clamp(s["resistance"] + r_change, 0.0, 1.0)
      traces.append(
        {
          "step": "resistance_update",
          "input": f"old={old_r:.2f}",
          "output": f"new_r={s['resistance']:.2f}, reasoning={payload.get('reasoning', '')}",
        }
      )

      utterance = self.patient.generate_utterance(
        conversation=self._format_conversation(s["messages"]),
        resistance_level=s["resistance"],
        current_stage=s["current_stage"],
        instruction=self._patient_utterance_instruction(s),
      )

      if s["final_patient_pending_end"]:
        utterance = f"{utterance} [/END]".strip()

      idx = self._append_message(s, "patient", utterance)
      active_stage = s.get("active_exchange_stage") or s["current_stage"]
      s["exchange_counts"][active_stage] = int(s["exchange_counts"].get(active_stage, 0)) + 1
      if active_stage == "cbt_technique" and s["current_technique_plan"]:
        s["technique_turn_count"] += 1
      if active_stage == "mi_techniques":
        s["mi_technique_turns"] += 1
      if active_stage == "meta_conversation":
        s["meta_conversation_turns"] += 1

      if s["current_stage"] in MI_STAGES or s["current_stage"] in {"anticipated_closing"}:
        s["mi_context"].append({"role": "patient", "content": utterance})

      if s["final_patient_pending_end"] and "[/END]" in utterance:
        s["session_ended"] = True
        if not s["session_ended_reason"]:
          s["session_ended_reason"] = "normal_closing_completed"

      self._append_trace(s, idx, "patient", traces)
    except ConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s

  def route_after_patient(self, state: SessionState) -> str:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return "end"
    return "continue"

  def create_chain(self):
    workflow = StateGraph(SessionState)
    workflow.add_node("therapist_turn", self.therapist_turn)
    workflow.add_node("patient_turn", self.patient_turn)
    workflow.add_edge(START, "therapist_turn")
    workflow.add_edge("therapist_turn", "patient_turn")
    workflow.add_conditional_edges(
      "patient_turn",
      self.route_after_patient,
      {"continue": "therapist_turn", "end": END},
    )
    return workflow.compile()

  def createChain(self):
    return self.create_chain()
