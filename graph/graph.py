from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]
from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]

from agents.interfaces import clamp

from .closing import ClosingStageMixin
from .cognitive import CognitiveStageMixin
from .homework import HomeworkStageMixin
from .mi import MIStageMixin
from .opening import OpeningStageMixin
from .state import MI_STAGES, SessionState
from .support import ConversationProcessingError
from .techniques import TechniqueStageMixin


class TherapyGraphBuilder(
  OpeningStageMixin,
  MIStageMixin,
  CognitiveStageMixin,
  TechniqueStageMixin,
  HomeworkStageMixin,
  ClosingStageMixin,
):
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
    if stage == "anticipated_closing":
      return self._handle_anticipated_closing(state, traces)
    if stage == "forced_closing":
      return self._handle_forced_closing(state, traces) #TODO: review it
    self._set_stage(state, "opening")
    return self._handle_opening(state, traces)

  def therapist_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    first_exchange = len(s["messages"]) == 0
    if s["session_ended"]:
      return s

    traces: list[dict[str, Any]] = []
    try:
      if not first_exchange:
        self._maybe_force_closing(s)
        utterance = self._maybe_handle_ending_needed(s, traces)
        if utterance is None:
          self._run_resistance_prediction(s, traces)
          self._handle_pre_stage_routing(s, traces)
          utterance = self._handle_stage_dispatch(s, traces)
      else:
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
    substage = state["current_substage"]
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
    if stage in {"anticipated_closing", "forced_closing"} or (stage == "closing" and substage == 'closing_finalize_feedback'):
      return "Respond briefly to close the session."
    if stage == "closing":
      return "Respond naturally to the therapist's closing prompts without closing the conversation."
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
      r_change = self._require_bounded_float(payload, "r_change", "resistance_update", -0.10, 0.15)
      s["resistance"] = clamp(s["resistance"] + r_change, 0.0, 1.0)

      utterance = self.patient.generate_utterance(
        full_conversation=self._format_conversation(s["messages"]),
        last_utterances=self._format_last_utterances(s["messages"]),
        resistance_level=s["resistance"],
        instruction=self._patient_utterance_instruction(s),
      )

      if s["final_patient_pending_end"]:
        utterance = f"{utterance} [/END]".strip()

      idx = self._append_message(s, "patient", utterance)
      active_stage = s.get("active_exchange_stage") or s["current_stage"]
      s["exchange_counts"][active_stage] = int(s["exchange_counts"].get(active_stage, 0)) + 1
      if active_stage == "id_distortions":
        s["working_cognitive_set_exchange_count"] = int(s.get("working_cognitive_set_exchange_count", 0)) + 1
      if active_stage == "cbt_technique" and s["current_technique_plan"]:
        s["technique_turn_count"] += 1

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
