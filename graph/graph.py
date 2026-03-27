from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]
from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]

from agents.interfaces import ensure_str

from .closing import ClosingStageMixin
from .cognitive import CognitiveStageMixin
from .demand import PatientDemandStageMixin
from .homework import HomeworkStageMixin
from .mi import MIStageMixin
from .opening import OpeningStageMixin
from .state import DEFAULT_SUBSTAGE_BY_STAGE, MI_STAGES, SessionState, Stage, SubStage
from .support import ConversationProcessingError
from .techniques import TechniqueStageMixin


class TherapyGraphBuilder(
  OpeningStageMixin,
  MIStageMixin,
  PatientDemandStageMixin,
  CognitiveStageMixin,
  TechniqueStageMixin,
  HomeworkStageMixin,
  ClosingStageMixin,
):
  MI_END_SESSION_DEMAND_REDIRECT_STAGES = {"mi_topic_exploration", "mi_techniques", "meta_conversation"}

  def __init__(self, config_path: str, profile: dict[str, Any]):
    self.cfg = OmegaConf.load(config_path)
    self.profile = profile
    self._init_agents()
    self.cbt_techniques = self._load_techniques()
    self.mi_techniques = self._load_mi_techniques()

  def _handle_stage_dispatch(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
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
    if stage == "patient_demand":
      return self._handle_patient_demand(state, traces)
    if stage == "anticipated_closing":
      return self._handle_anticipated_closing(state, traces)
    if stage == "forced_closing":
      return self._handle_forced_closing(state, traces) #TODO: review it
    self._set_stage(state, "opening")
    return self._handle_opening(state, traces)

  def _route_high_resistance(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    level: str,
    source_stage: str,
    source_substage: str,
  ) -> None:
    if level == "yes":
      if source_stage in {"opening", "id_distortions", "cbt_technique", "homework"}:
        resume_stage = "id_distortions" if source_stage == "opening" else source_stage
        self._enter_mi(
          state,
          "mi_topic_exploration",
          resume_stage,
          f"High-resistance request to stop detected during {source_stage}.",
        )
        return
      if source_stage in self.MI_END_SESSION_DEMAND_REDIRECT_STAGES:
        state["mi_active"] = False
        if state["mi_end_session_demand_redirect_used"]:
          self._set_stage(state, "closing")
          return
        self._set_stage(state, "closing", "closing_fallback")
        return
      if source_stage == "closing":
        self._set_stage(state, "closing", "closing_fallback")
        return
      return

    if level != "partial":
      return

    if source_stage == "opening":
      if source_substage == "open_mood":
        self._set_substage(state, "open_problem")
      else:
        self._set_stage(state, "id_distortions")
      return

    if source_stage == "id_distortions":
      self._shelve_current_cognitive_set_and_begin_new(state)
      traces.append(
        {
          "step": "high_resistance_new_cognitive_set",
          "input": "Patient pushed back against the current cognitive topic.",
          "output": "Shelved the current working set and began a new one.",
        }
      )
      return

    if source_stage == "cbt_technique":
      self._replace_current_technique_due_to_resistance(state, traces)
      traces.append(
        {
          "step": "high_resistance_new_technique",
          "input": "Patient pushed back against the current CBT technique.",
          "output": "Selected a different CBT technique and reset the active plan.",
        }
      )
      return

    if source_stage == "homework":
      state["homework_waiting_for_patient"] = False
      state["homework_fallback_to_apply"] = False
      state["homework_fallback_done"] = False
      return

    if source_stage == "closing":
      self._set_substage(state, "closing_fallback")
      return

    if source_stage == "mi_topic_exploration":
      state["current_mi_topic"] = ""
      self._set_substage(state, "mi_choose_topic")
      return

    if source_stage == "mi_techniques":
      self._set_substage(state, "mi_choose_technique")

  def _high_resistance_route_source(self, state: SessionState) -> tuple[Stage, SubStage]:
    if state["current_stage"] == "patient_demand" and state["patient_demand_resume_stage"]:
      source_stage = state["patient_demand_resume_stage"]
      source_substage = state["patient_demand_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[source_stage]
      return source_stage, source_substage
    return state["current_stage"], state["current_substage"]

  def _maybe_handle_high_resistance(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    if state["session_ended"] or state["final_patient_pending_end"]:
      return "none"
    if state["current_stage"] in {"anticipated_closing", "forced_closing"}:
      return "none"

    instruction = self._load_stage_prompt("shared", "high_resistance_check")
    parsed = self._run_therapist_structured(state, traces, "high_resistance_check", instruction)
    payload = self._require_dict(parsed, "high_resistance_check")
    has_demand = self._require_bool(payload, "has_demand", "high_resistance_check")
    demand_kind = ensure_str(payload.get("demand_kind")).strip().lower()
    if demand_kind not in {"none", "end_session", "item_move", "other"}:
      raise ConversationProcessingError(
        "high_resistance_check.demand_kind must be one of none, end_session, item_move, or other, "
        f"got {demand_kind!r}."
      )
    demand = ensure_str(payload.get("demand")).strip()
    reasoning = ensure_str(payload.get("reasoning"))
    normalized_payload = dict(payload)
    normalized_payload["has_demand"] = has_demand
    normalized_payload["demand_kind"] = demand_kind
    normalized_payload["demand"] = demand or None
    self._set_last_trace_output(traces, "high_resistance_check", normalized_payload)

    if not has_demand:
      if demand_kind != "none":
        raise ConversationProcessingError(
          "high_resistance_check.has_demand=false must be paired with demand_kind='none'."
        )
      return "none"

    if demand_kind == "none":
      raise ConversationProcessingError(
        "high_resistance_check.has_demand=true must not be paired with demand_kind='none'."
      )
    if not demand:
      raise ConversationProcessingError("high_resistance_check.demand must be a non-empty string when has_demand=true.")

    source_stage, source_substage = self._high_resistance_route_source(state)
    if (
      demand_kind == "end_session"
      and source_stage in self.MI_END_SESSION_DEMAND_REDIRECT_STAGES
      and not state["mi_end_session_demand_redirect_used"]
    ):
      self._mark_high_resistance_redirect(state, "yes", source_stage, source_substage, reasoning)
      if state["current_stage"] == "patient_demand":
        self._replace_active_patient_demand(state, traces, demand, reasoning)
        state["patient_demand_end_session_attempt_active"] = True
        state["mi_end_session_demand_redirect_used"] = True
      else:
        self._enter_patient_demand(
          state,
          traces,
          demand,
          reasoning,
          source_stage,
          source_substage,
          end_session_attempt_active=True,
        )
      return demand_kind
    if demand_kind in {"end_session", "item_move"}:
      level = "yes" if demand_kind == "end_session" else "partial"
      if state["current_stage"] == "patient_demand":
        self._clear_patient_demand_state(state, clear_resume_cursor=True)
        self._set_stage(state, source_stage, source_substage)
      self._mark_high_resistance_redirect(state, level, source_stage, source_substage, reasoning)
      self._route_high_resistance(state, traces, level, source_stage, source_substage)
      return demand_kind

    if state["current_stage"] == "patient_demand":
      self._replace_active_patient_demand(state, traces, demand, reasoning)
    else:
      self._enter_patient_demand(state, traces, demand, reasoning, source_stage, source_substage)
    return "other"

  def therapist_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    first_exchange = len(s["messages"]) == 0
    if s["session_ended"]:
      return s

    traces: list[dict[str, Any]] = []
    try:
      self._clear_high_resistance_redirect(s)
      if not first_exchange:
        self._maybe_force_closing(s)
        high_resistance_level = self._maybe_handle_high_resistance(s, traces)
        self._run_resistance_prediction(s, traces)
        if high_resistance_level == "none":
          self._handle_pre_stage_routing(s, traces)
      utterance = self._handle_stage_dispatch(s, traces)

      s["active_exchange_stage"] = s["current_stage"]
      s["active_exchange_substage"] = s["current_substage"]
      idx = self._append_message_with_metadata(
        s,
        "therapist",
        utterance,
        s["active_exchange_stage"],
        s["active_exchange_substage"],
      )
      if s["current_stage"] in MI_STAGES or s["current_stage"] in {"anticipated_closing"}:
        self._append_mi_context_message(
          s,
          "therapist",
          utterance,
          idx,
          s["active_exchange_stage"],
          s["active_exchange_substage"],
        )
      self._append_trace(s, idx, "therapist", traces)
      self._clear_high_resistance_redirect(s)
    except ConversationProcessingError as exc:
      self._clear_high_resistance_redirect(s)
      self._end_session_with_error(s, traces, exc)
    return s

  def _patient_utterance_instruction(self, state: SessionState) -> str:
    stage = state["current_stage"]
    substage = state["current_substage"]
    if state["final_patient_pending_end"]:
      return "Provide a brief closing response."
    if stage == "opening":
      return "Respond to the therapist's opening and share only what feels comfortable."
    if stage == "id_distortions":
      return "Answer the therapist's question based on your real thoughts, emotions, and behavior."
    if stage == "cbt_technique":
      return "Respond to the CBT exercise with realistic engagement for your current resistance level."
    if stage == "homework":
      return "Respond honestly about whether the proposed homework feels feasible."
    if stage == "patient_demand":
      return "Respond to the therapist's attempt to address your request, including whether it feels sufficiently helpful."
    if stage in {"mi_topic_exploration", "mi_techniques", "meta_conversation"}:
      return "Respond to the motivational discussion in line with your current resistance level."
    if stage in {"anticipated_closing", "forced_closing"} or (stage == "closing" and substage == 'closing_finalize_feedback'):
      return "Respond briefly to close the session."
    if stage == "closing":
      return "Respond to the therapist's closing prompts without closing the conversation."
    return "Respond naturally."

  def patient_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s

    traces: list[dict[str, Any]] = []
    try:
      update_instruction = self._load_stage_prompt(
        "shared",
        "patient_resistance_update",
        resistance=f"{s['resistance']:.2f}",
      )
      update = self._run_patient_structured(
        s,
        traces,
        "resistance_update",
        update_instruction,
      )
      payload = self._require_dict(update, "resistance_update")
      delta_llm = self._require_bounded_float(payload, "r_change", "resistance_update", -0.10, 0.15)
      resistance_update = self._apply_homeostatic_resistance_update(
        previous_r=s["resistance"],
        delta_llm=delta_llm,
        baseline=s["resistance_baseline"],
        alpha=s["resistance_recovery_rate"],
      )
      s["resistance"] = resistance_update["new_r"]
      trace_payload = dict(payload)
      trace_payload.update(resistance_update)
      self._set_last_trace_output(traces, "resistance_update", trace_payload)

      utterance = self.patient.generate_utterance(
        full_conversation=self._format_conversation(s["messages"]),
        last_utterances=self._format_last_utterances(s["messages"]),
        resistance_level=s["resistance"],
        instruction=self._patient_utterance_instruction(s),
      )

      if s["final_patient_pending_end"]:
        utterance = f"{utterance} [/END]".strip()

      active_stage = s.get("active_exchange_stage") or s["current_stage"]
      active_substage = s.get("active_exchange_substage") or s["current_substage"]
      idx = self._append_message_with_metadata(
        s,
        "patient",
        utterance,
        active_stage,
        active_substage,
      )
      self._increment_stage_exchange_count(s, active_stage)
      if active_stage == "id_distortions":
        s["working_cognitive_set_exchange_count"] = int(s.get("working_cognitive_set_exchange_count", 0)) + 1
      if active_stage == "cbt_technique" and s["current_technique_plan"]:
        s["technique_turn_count"] += 1

      if s["current_stage"] in MI_STAGES or s["current_stage"] in {"anticipated_closing"}:
        self._append_mi_context_message(
          s,
          "patient",
          utterance,
          idx,
          active_stage,
          active_substage,
        )

      if s["final_patient_pending_end"] and "[/END]" in utterance:
        s["session_ended"] = True
        if not s["session_ended_reason"]:
          s["session_ended_reason"] = "normal_closing_completed"

      self._append_trace(s, idx, "patient", traces)
    except ConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s

  def route_after_patient(self, state: SessionState) -> Literal["continue", "end"]:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return "end"
    return "continue"

  def create_chain(self) -> Any:
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

  def createChain(self) -> Any:
    return self.create_chain()
