from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, cast

from agents.interfaces import (
  LLMFactory,
  StructuredOutputError,
  ensure_str,
  ensure_str_list,
  extract_generation_config,
  load_prompt,
  render_prompt,
)
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent

from .resistance import (
  HomeostaticResistanceUpdate,
  TherapistResistanceUpdate,
  apply_homeostatic_resistance_update,
  apply_bayesian_predicted_resistance_update,
  normalize_predicted_resistance_state_fields,
  normalize_resistance_state_fields,
)
from .state import ALL_SUBSTAGES, CORE_BELIEF_KEYS, DEFAULT_SUBSTAGE_BY_STAGE, DISTORTION_KEYS, MIN_COGNITIVE_SET_EXCHANGES, CognitiveModelSet, ConversationMessage, ConversationRole, SessionState, Stage, SubStage


class ConversationProcessingError(RuntimeError):
  pass


class GraphSupportMixin:
  cfg: Any = None
  profile: dict[str, Any] = {}
  therapist: Any = None
  patient: Any = None
  cbt_techniques: dict[str, str] = {}
  mi_techniques: dict[str, str] = {}

  def _project_root(self) -> Path:
    return Path(__file__).resolve().parent.parent

  def _init_agents(self) -> None:
    therapist_llm = LLMFactory.create_from_config(self.cfg.therapist)
    patient_llm = LLMFactory.create_from_config(self.cfg.patient)
    self.therapist = TherapistAgent(
      therapist_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.therapist),
    )
    self.patient = PatientAgent(
      patient_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.patient),
    )

  def _load_stage_prompt(self, stage: str, name: str, **values: Any) -> str:
    filename = f"{stage}/{name}.txt"
    if values:
      return render_prompt(filename, **values)
    return load_prompt(filename)

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
  
  def _load_mi_techniques(self) -> dict[str, str]:
    techniques_path = self._project_root() / "data" / "knowledge" / "mi_techniques.json"
    if not techniques_path.exists():
      raise ConversationProcessingError(f"Missing MI techniques file: {techniques_path}")
    try:
      with open(techniques_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load MI techniques from {techniques_path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ConversationProcessingError(f"MI Techniques file must contain a JSON object: {techniques_path}")
    return {
      ensure_str(name): ensure_str(description)
      for name, description in data.items()
      if ensure_str(name)
    }

  def _load_distortions(self) -> dict[str, Any]:
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

  def resume_state(self, saved_state: dict[str, Any] | SessionState, max_turns: int, min_turns: int = 0) -> SessionState:
    resumed = cast(SessionState, cast(object, deepcopy(saved_state)))
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

  def _is_error_trace(self, trace: Any) -> bool:
    if not isinstance(trace, dict):
      return False
    if trace.get("speaker") != "system":
      return False
    steps = trace.get("traces")
    if not isinstance(steps, list):
      return False
    return any(isinstance(step, dict) and step.get("step") == "conversation_error" for step in steps)

  def _ensure_state_defaults(self, state: dict[str, Any] | SessionState) -> SessionState:
    s = cast(SessionState, cast(object, dict(state)))
    raw_stage = ensure_str(s.get("current_stage"), "opening")
    stage: Stage = cast(Stage, raw_stage) if raw_stage in DEFAULT_SUBSTAGE_BY_STAGE else "opening"
    s["current_stage"] = stage
    raw_substage = ensure_str(
      s.get("current_substage"),
      DEFAULT_SUBSTAGE_BY_STAGE[stage],
    )
    if raw_substage not in ALL_SUBSTAGES:
      raw_substage = DEFAULT_SUBSTAGE_BY_STAGE[stage]
    s["current_substage"] = cast(SubStage, raw_substage)

    raw_messages = s.get("messages")
    s["messages"] = self._normalize_messages(raw_messages, stage, cast(SubStage, raw_substage))

    s = cast(SessionState, cast(object, normalize_resistance_state_fields(dict(s), deepcopy(self.profile))))
    s = cast(SessionState, cast(object, normalize_predicted_resistance_state_fields(dict(s))))

    s.setdefault("opening_patient_mood", "")
    s.setdefault("opening_patient_problem", "")
    s.setdefault("last_reasoning", "")
    legacy_is_complete_reasoning = s.get("is_completed_reasoning")    
    if legacy_is_complete_reasoning and not s.get("last_reasoning"):
      s["last_reasoning"] = ensure_str(legacy_is_complete_reasoning)

    s.setdefault("cognitive_model_sets", [])
    s.setdefault("working_cognitive_set", self._new_cognitive_set(len(s["cognitive_model_sets"]) + 1))
    s.setdefault("shelved_cognitive_sets", [])
    s.setdefault("cognitive_gap_plan", "")
    s.setdefault("active_cognitive_target", "")
    s.setdefault("active_cognitive_question", "")
    s.setdefault("active_cognitive_follow_up_attempts", 0)
    s.setdefault("working_cognitive_set_exchange_count", 0)

    s.setdefault("selected_techniques", [])
    s.setdefault("current_technique_index", 0)
    s.setdefault("current_technique_plan", "")
    s.setdefault("current_technique_expected_outcome", "")
    s.setdefault("current_technique_current_progress", "")
    s.setdefault("technique_evaluations", [])
    s.setdefault("techniques_completed", 0)
    s.setdefault("technique_turn_count", 0)
    s.setdefault("technique_slot_replaced", {})

    s.setdefault("mi_active", False)
    raw_mi_context = s.get("mi_context")
    s["mi_context"] = self._normalize_messages(raw_mi_context, stage, cast(SubStage, raw_substage))
    s.setdefault("cbt_flow_saved_state", "")
    s.setdefault("cbt_resume_stage", "id_distortions")
    s.setdefault("mi_topics_tried", [])
    s.setdefault("mi_techniques_used", [])
    s.setdefault("mi_techniques_paused_until_precontemplation", False)
    s.setdefault("current_mi_topic", "")
    s.setdefault("last_mi_predicted_resistance", s["predicted_resistance"])
    s.setdefault("mi_end_session_demand_redirect_used", False)
    s.setdefault("patient_demand_text", "")
    s.setdefault("patient_demand_plan", "")
    s.setdefault("patient_demand_reasoning", "")
    raw_demand_resume_stage = ensure_str(s.get("patient_demand_resume_stage"))
    s["patient_demand_resume_stage"] = cast(Stage, raw_demand_resume_stage) if raw_demand_resume_stage in DEFAULT_SUBSTAGE_BY_STAGE else ""
    raw_demand_resume_substage = ensure_str(s.get("patient_demand_resume_substage"))
    if raw_demand_resume_substage not in ALL_SUBSTAGES:
      raw_demand_resume_substage = ""
    s["patient_demand_resume_substage"] = cast(SubStage, raw_demand_resume_substage) if raw_demand_resume_substage else ""
    s.setdefault("patient_demand_end_session_attempt_active", False)

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
    max_message_index = 0
    if s["messages"]:
      max_message_index = max(int(msg.get("utterance_index", 0)) for msg in s["messages"])
    max_trace_index = 0
    if s["reasoning_traces"]:
      max_trace_index = max(int(trace["utterance_index"]) for trace in s["reasoning_traces"])
    s["utterance_index"] = max(int(s.get("utterance_index", 0)), max_message_index, max_trace_index)
    raw_exchange_counts = s.get("exchange_counts")
    normalized_exchange_counts: dict[str, list[int]] = {}
    if isinstance(raw_exchange_counts, dict):
      for key, value in raw_exchange_counts.items():
        stage_name = ensure_str(key)
        if not stage_name:
          continue
        if isinstance(value, list):
          normalized_exchange_counts[stage_name] = [int(item) for item in value]
        elif value is not None:
          normalized_exchange_counts[stage_name] = [int(value)]
    s["exchange_counts"] = normalized_exchange_counts
    
    # devlog. remove when create new conversations
    legacy_metaconv_turns = s.get("meta_conversation_turns")
    if legacy_metaconv_turns is not None and "meta_conversation" not in s["exchange_counts"]:
      s["exchange_counts"]["meta_conversation"] = [int(legacy_metaconv_turns)]
    legacy_mi_technique_turns = s.get("mi_technique_turns")
    if legacy_mi_technique_turns is not None and "mi_techniques" not in s["exchange_counts"]:
      s["exchange_counts"]["mi_techniques"] = [int(legacy_mi_technique_turns)]
    s.setdefault("active_exchange_stage", "")
    s.setdefault("active_exchange_substage", "")
    # devlog end

    s.setdefault("high_resistance_redirect_triggered", False)
    s.setdefault("high_resistance_level", "")
    s.setdefault("high_resistance_source_stage", "")
    s.setdefault("high_resistance_source_substage", "")
    s.setdefault("high_resistance_reasoning", "")
    
    s.setdefault("final_patient_pending_end", False)
    s.setdefault("session_ended", False)
    s.setdefault("session_ended_reason", "")
    s.setdefault("conversation_error", "")
    s.setdefault("opening_fallback_triggered", False)

    s["max_turns"] = int(s.get("max_turns", getattr(self.cfg.env, "max_turns", 80)))
    s["min_turns"] = int(s.get("min_turns", getattr(self.cfg.env, "min_turns", 0)))
    self._ensure_stage_exchange_counter(s, stage)
    return s

  def _set_stage(self, state: SessionState, stage: Stage, substage: Optional[SubStage] = None) -> None:
    previous_stage = ensure_str(state.get("current_stage"))
    if previous_stage != stage:
      self._start_stage_exchange_counter(state, stage)
    else:
      self._ensure_stage_exchange_counter(state, stage)
    state["current_stage"] = stage
    state["current_substage"] = substage or DEFAULT_SUBSTAGE_BY_STAGE[stage]

  def _set_substage(self, state: SessionState, substage: SubStage) -> None:
    state["current_substage"] = substage

  def _handle_stage_dispatch(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    raise NotImplementedError

  def _handle_closing(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    raise NotImplementedError

  def _enter_mi(self, state: SessionState, target_stage: Stage, resume_stage: Stage | None, reason: str) -> None:
    raise NotImplementedError

  def _format_conversation(self, messages: list[ConversationMessage]) -> str:
    if not messages:
      return "(not utterances yet)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

  def _format_last_utterances(self, messages: list[ConversationMessage], limit: int = 10) -> str:
    if not messages:
      return "(no utterances yet)"
    recent = messages[-limit:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

  def _build_stage_user_context(self, state: SessionState) -> str:
    stage = state["current_stage"]
    context: dict[str, Any] = {
      "current_stage": stage,
      "predicted_resistance": state["predicted_resistance"],
      "stage_exchange_count": self._current_stage_exchange_count(state, stage),
      "stage_exchange_history": list(state["exchange_counts"].get(stage, [])),
      "high_resistance_redirect_triggered": state["high_resistance_redirect_triggered"],
      "high_resistance_level": state["high_resistance_level"],
      "high_resistance_source_stage": state["high_resistance_source_stage"],
      "high_resistance_source_substage": state["high_resistance_source_substage"],
      "high_resistance_reasoning": state["high_resistance_reasoning"],
      "patient_demand_text": state["patient_demand_text"],
      "patient_demand_plan": state["patient_demand_plan"],
      "patient_demand_reasoning": state["patient_demand_reasoning"],
      "patient_demand_resume_stage": state["patient_demand_resume_stage"],
      "patient_demand_resume_substage": state["patient_demand_resume_substage"],
      "patient_demand_end_session_attempt_active": state["patient_demand_end_session_attempt_active"],
      "mi_end_session_demand_redirect_used": state["mi_end_session_demand_redirect_used"],
      "last_patient_response": state["messages"][-1]["content"] if state["messages"] else "(no utterances yes)",
    }

    if stage == "opening":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "opening_fallback_triggered": state["opening_fallback_triggered"],
        }
      )
    elif stage == "id_distortions":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "cognitive_model_sets": state["cognitive_model_sets"],
          "shelved_cognitive_sets": state["shelved_cognitive_sets"],
          "working_cognitive_set": state["working_cognitive_set"],
          "active_cognitive_target": state["active_cognitive_target"],
          "active_cognitive_question": state["active_cognitive_question"],
          "active_cognitive_follow_up_attempts": state["active_cognitive_follow_up_attempts"],
          "cognitive_gap_plan": state["cognitive_gap_plan"],
          "working_cognitive_set_exchange_count": state["working_cognitive_set_exchange_count"],
          "working_cognitive_set_min_exchanges": MIN_COGNITIVE_SET_EXCHANGES,
        }
      )
    elif stage == "cbt_technique":
      current_selected_technique = None
      current_technique_index = int(state["current_technique_index"])
      if 0 <= current_technique_index < len(state["selected_techniques"]):
        current_selected_technique = state["selected_techniques"][current_technique_index]
      context.update(
        {
          "cognitive_model_sets": state["cognitive_model_sets"],
          "selected_techniques": state["selected_techniques"],
          "current_technique_index": current_technique_index,
          "current_selected_technique": current_selected_technique,
          "current_technique_plan": state["current_technique_plan"],
          "current_technique_expected_outcome": state["current_technique_expected_outcome"],
          "current_technique_current_progress": state["current_technique_current_progress"],
          "technique_turn_count": state["technique_turn_count"],
          "techniques_completed": state["techniques_completed"],
          "technique_evaluations": state["technique_evaluations"][-3:],
        }
      )
    elif stage == "homework":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "cognitive_model_sets": state["cognitive_model_sets"],
          "technique_evaluations": state["technique_evaluations"][-3:],
          "proposed_tasks": state["proposed_tasks"],
          "agreed_tasks": state["agreed_tasks"],
          "homework_proposal_attempts": state["homework_proposal_attempts"],
          "homework_waiting_for_patient": state["homework_waiting_for_patient"],
          "homework_fallback_done": state["homework_fallback_done"],
          "homework_fallback_to_apply": state["homework_fallback_to_apply"],
        }
      )
    elif stage == "closing":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "cognitive_model_sets": state["cognitive_model_sets"],
          "technique_evaluations": state["technique_evaluations"],
          "selected_techniques": state["selected_techniques"],
          "session_summary": state["session_summary"],
          "next_session_plan": state["next_session_plan"],
          "agreed_tasks": state["agreed_tasks"],
          "patient_feedback": state["patient_feedback"],
          "closing_plan_refinements": state["closing_plan_refinements"],
        }
      )
    elif stage == "mi_topic_exploration":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "mi_topics_tried": state["mi_topics_tried"],
          "mi_context": state["mi_context"],
          "last_mi_predicted_resistance": state["last_mi_predicted_resistance"],
          "cbt_resume_stage": state["cbt_resume_stage"],
        }
      )
    elif stage == "mi_techniques":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "mi_techniques_used": state["mi_techniques_used"][-5:],
          "mi_context": state["mi_context"],
          "last_mi_predicted_resistance": state["last_mi_predicted_resistance"],
          "cbt_resume_stage": state["cbt_resume_stage"],
        }
      )
    elif stage == "meta_conversation":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "mi_topics_tried": state["mi_topics_tried"],
          "mi_techniques_used": state["mi_techniques_used"][-5:],
          "mi_context": state["mi_context"],
          "cbt_resume_stage": state["cbt_resume_stage"],
        }
      )
    elif stage == "patient_demand":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "cognitive_model_sets": state["cognitive_model_sets"],
          "selected_techniques": state["selected_techniques"],
          "current_technique_index": state["current_technique_index"],
          "patient_demand_turn_count": self._current_stage_exchange_count(state, "patient_demand"),
          "patient_demand_turn_cap": 10,
        }
      )
    elif stage in {"anticipated_closing", "forced_closing"}:
      context.update(
        {
          "session_ended_reason": state["session_ended_reason"],
          "cbt_resume_stage": state["cbt_resume_stage"],
        }
      )

    return json.dumps(context, ensure_ascii=True)

  def _instruction_with_previous_reasoning(self, state: SessionState, instruction: str) -> str:
    text = ensure_str(instruction)
    previous_reasoning = ensure_str(state.get("last_reasoning"))
    if not previous_reasoning:
      return text

    suffix = self._load_stage_prompt(
      "shared",
      "previous_reasoning_suffix",
      previous_reasoning=previous_reasoning,
      last_utterance=state["messages"][-2]["content"] if state["messages"] else "(no messages yet)",
    )
    return f"{text} {suffix}"

  def _instruction_with_high_resistance_redirect(self, state: SessionState, instruction: str) -> str:
    text = ensure_str(instruction)
    if not state.get("high_resistance_redirect_triggered"):
      return text

    level = ensure_str(state.get("high_resistance_level"), "partial")
    source_stage = ensure_str(state.get("high_resistance_source_stage"), "unknown stage")
    suffix = (
      " Before moving forward, briefly acknowledge the patient's wish to stop, step back, or change focus "
      f"in a natural therapist voice. This was a high-resistance reroute with level='{level}' from {source_stage}. "
      "Then continue with the current stage goal smoothly, without sounding abrupt, defensive, or scripted."
    )
    return f"{text} {suffix}"

  def _generate_therapist_utterance(self, state: SessionState, instruction: str) -> str:
    enriched_instruction = self._instruction_with_high_resistance_redirect(state, instruction)
    return self.therapist.generate_utterance(
      full_conversation=self._format_conversation(state["messages"]),
      last_utterances=self._format_last_utterances(state["messages"]),
      last_reasoning=state.get("last_reasoning", ""),
      session_context=self._build_stage_user_context(state),
      instruction=enriched_instruction,
    )

  def _append_trace(self, state: SessionState, utterance_index: int, speaker: str, traces: list[dict[str, Any]]) -> None:
    state["reasoning_traces"].append(
      {
        "utterance_index": utterance_index,
        "speaker": speaker,
        "traces": traces,
      }
    )

  def _append_message(self, state: SessionState, role: ConversationRole, content: str) -> int:
    return self._append_message_with_metadata(state, role, content, state["current_stage"], state["current_substage"])

  def _append_message_with_metadata(
    self,
    state: SessionState,
    role: ConversationRole,
    content: str,
    stage: Stage,
    substage: SubStage,
  ) -> int:
    resistance = state["predicted_resistance"] if role == "therapist" else state["resistance"]
    state["utterance_index"] += 1
    state["messages"].append(
      {
        "utterance_index": state["utterance_index"],
        "role": role,
        "content": ensure_str(content),
        "stage": stage,
        "substage": substage,
        "resistance": resistance,
      }
    )
    return state["utterance_index"]

  def _append_mi_context_message(
    self,
    state: SessionState,
    role: ConversationRole,
    content: str,
    utterance_index: int,
    stage: Stage,
    substage: SubStage,
  ) -> None:
    state["mi_context"].append(
      {
        "utterance_index": utterance_index,
        "role": role,
        "content": ensure_str(content),
        "stage": stage,
        "substage": substage,
      }
    )

  def _normalize_messages(self, raw_messages: Any, default_stage: Stage, default_substage: SubStage) -> list[ConversationMessage]:
    if not isinstance(raw_messages, list):
      return []
    normalized: list[ConversationMessage] = []
    next_index = 1
    for item in raw_messages:
      if not isinstance(item, dict):
        continue
      role = ensure_str(item.get("role"))
      content = ensure_str(item.get("content"))
      if not role or not content:
        continue
      raw_index = item.get("utterance_index")
      try:
        utterance_index = int(raw_index) if raw_index is not None else next_index
      except (TypeError, ValueError):
        utterance_index = next_index
      if utterance_index <= 0:
        utterance_index = next_index

      raw_stage = ensure_str(item.get("stage"), default_stage)
      message_stage: Stage = cast(Stage, raw_stage) if raw_stage in DEFAULT_SUBSTAGE_BY_STAGE else default_stage
      raw_substage = ensure_str(item.get("substage"), default_substage)
      if raw_substage not in ALL_SUBSTAGES:
        raw_substage = default_substage

      normalized.append(
        {
          "utterance_index": utterance_index,
          "role": cast(ConversationRole, role),
          "content": content,
          "stage": message_stage,
          "substage": cast(SubStage, raw_substage),
        }
      )
      next_index = max(next_index + 1, utterance_index + 1)
    return normalized

  def _ensure_stage_exchange_counter(self, state: SessionState, stage: Stage) -> None:
    history = state["exchange_counts"].setdefault(stage, [])
    if not history:
      history.append(0)

  def _start_stage_exchange_counter(self, state: SessionState, stage: Stage) -> None:
    history = state["exchange_counts"].setdefault(stage, [])
    history.append(0)

  def _current_stage_exchange_count(self, state: SessionState, stage: Stage) -> int:
    history = state["exchange_counts"].get(stage, [])
    if not history:
      return 0
    return int(history[-1])

  def _increment_stage_exchange_count(self, state: SessionState, stage: Stage) -> int:
    self._ensure_stage_exchange_counter(state, stage)
    history = state["exchange_counts"][stage]
    history[-1] = int(history[-1]) + 1
    return history[-1]

  def _mark_high_resistance_redirect(
    self,
    state: SessionState,
    level: str,
    source_stage: Stage,
    source_substage: SubStage,
    reasoning: str,
  ) -> None:
    state["high_resistance_redirect_triggered"] = True
    state["high_resistance_level"] = level
    state["high_resistance_source_stage"] = source_stage
    state["high_resistance_source_substage"] = source_substage
    state["high_resistance_reasoning"] = reasoning

  def _clear_high_resistance_redirect(self, state: SessionState) -> None:
    state["high_resistance_redirect_triggered"] = False
    state["high_resistance_level"] = ""
    state["high_resistance_source_stage"] = ""
    state["high_resistance_source_substage"] = ""
    state["high_resistance_reasoning"] = ""

  def _clear_patient_demand_state(self, state: SessionState, clear_resume_cursor: bool = True) -> None:
    state["patient_demand_text"] = ""
    state["patient_demand_plan"] = ""
    state["patient_demand_reasoning"] = ""
    state["patient_demand_end_session_attempt_active"] = False
    if clear_resume_cursor:
      state["patient_demand_resume_stage"] = ""
      state["patient_demand_resume_substage"] = ""

  def _next_cognitive_set_id(self, state: SessionState) -> int:
    known_ids = [
      int(state["working_cognitive_set"].get("cognitive_model_set_id", 0)),
      *[int(set_.get("cognitive_model_set_id", 0)) for set_ in state["cognitive_model_sets"]],
      *[int(set_.get("cognitive_model_set_id", 0)) for set_ in state["shelved_cognitive_sets"]],
    ]
    return max(known_ids, default=0) + 1

  def _current_band(self, value: float | None) -> Literal["precontemplation", "contemplation", "action"]:
    if value is None:
      return "contemplation"
    if value >= 0.75:
      return "precontemplation"
    if value >= 0.50:
      return "contemplation"
    return "action"

  def _new_cognitive_set(self, set_id: int) -> CognitiveModelSet:
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

  def _normalize_probabilities(self, payload: dict[str, Any], keys: list[str]) -> dict[str, float]:
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

  def _best_key(self, probs: dict[str, float]) -> str:
    if not isinstance(probs, dict) or not probs:
      raise ConversationProcessingError("Cannot choose a best key from an empty probability map.")
    return max(probs.items(), key=lambda item: item[1])[0]

  def _require_dict(self, payload: Any, step_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
      raise ConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON object.")
    return payload

  def _require_list(self, payload: Any, step_name: str) -> list[Any]:
    if not isinstance(payload, list):
      raise ConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON array.")
    return payload

  def _require_bool(self, payload: dict[str, Any], key: str, step_name: str) -> bool:
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

  def _require_bounded_float(self, payload: dict[str, Any], key: str, step_name: str, low: float, high: float) -> float:
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

  def _require_non_empty_string(self, payload: dict[str, Any], key: str, step_name: str) -> str:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required field '{key}'.")
    value = ensure_str(payload.get(key))
    if not value:
      raise ConversationProcessingError(f"{step_name}.{key} must be a non-empty string.")
    return value

  def _require_string_list(self, payload: dict[str, Any], key: str, step_name: str, min_items: int = 1) -> list[str]:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required list field '{key}'.")
    values = ensure_str_list(payload.get(key))
    if len(values) < min_items:
      raise ConversationProcessingError(f"{step_name}.{key} must contain at least {min_items} item(s).")
    return values

  def _end_session_with_error(self, state: SessionState, traces: list[dict[str, Any]], error: Exception) -> None:
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
    traces: list[dict[str, Any]],
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

  def _set_last_trace_output(self, traces: list[dict[str, Any]], step_name: str, payload: dict[str, Any]) -> None:
    for trace in reversed(traces):
      if trace.get("step") == step_name:
        trace["output"] = json.dumps(payload, ensure_ascii=True)
        return

  def _run_therapist_structured(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
    temperature: float | None = None,
  ) -> Any:
    try:
      parsed, _meta = self.therapist.generate_structured(
        full_conversation=self._format_conversation(state["messages"]),
        last_utterances=self._format_last_utterances(state["messages"]),
        last_reasoning=state["last_reasoning"],
        session_context=self._build_stage_user_context(state),
        instruction=instruction,
        expected_type=expected_type,
        temperature=temperature,
      )
      print(f"    <DEBUG>{parsed}</DEBUG>") #devlog
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    if isinstance(parsed, dict):
      reasoning = ensure_str(parsed.get("reasoning"))
      if reasoning:
        state["last_reasoning"] = reasoning
    self._trace_structured_step(traces, step_name, instruction, parsed)
    return parsed

  def _run_patient_structured(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
  ) -> Any:
    try:
      parsed, _meta = self.patient.generate_structured(
        full_conversation=self._format_conversation(state["messages"]),
        last_utterances=self._format_last_utterances(state["messages"]),
        resistance_level=state["resistance"],
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

  def _run_resistance_prediction(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    previous_p = state["predicted_resistance"]
    instruction = self._load_stage_prompt(
      "shared",
      "resistance_prediction",
      old_r="none yet" if previous_p is None else f"{previous_p:.2f}",
    )
    parsed = self._run_therapist_structured(state, traces, "resistance_prediction", instruction)
    payload = self._require_dict(parsed, "resistance_prediction")
    p_target = self._require_bounded_float(payload, "p_target", "resistance_prediction", 0.0, 1.0)
    signal_clarity = self._require_bounded_float(payload, "signal_clarity", "resistance_prediction", 0.0, 1.0)
    prediction_update = self._apply_bayesian_predicted_resistance_update(previous_p, p_target, signal_clarity)
    state["predicted_resistance"] = prediction_update["new_p"]
    trace_payload = dict(payload)
    trace_payload.update(prediction_update)
    self._set_last_trace_output(traces, "resistance_prediction", trace_payload)

  def _apply_bayesian_predicted_resistance_update(
    self,
    previous_p: float | None,
    p_target: float,
    signal_clarity: float,
  ) -> TherapistResistanceUpdate:
    return apply_bayesian_predicted_resistance_update(previous_p, p_target, signal_clarity)

  def _apply_homeostatic_resistance_update(
    self,
    previous_r: float,
    delta_llm: float,
    baseline: float,
    alpha: float,
  ) -> HomeostaticResistanceUpdate:
    return apply_homeostatic_resistance_update(previous_r, delta_llm, baseline, alpha)

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
