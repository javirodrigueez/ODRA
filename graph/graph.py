from __future__ import annotations

import json
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]

from agents.interfaces import StructuredOutputError, ensure_str
from runtime_config import load_runtime_config

from .address_tib import AddressTIBStageMixin
from .audit_tokens import record_large_prompt_request
from .closing import ClosingStageMixin
from .cognitive import CognitiveStageMixin
from .homework import HomeworkStageMixin
from .mi import MIStageMixin
from .opening import OpeningStageMixin
from .resistance import (
  PatientAction,
  PatientActionDistribution,
  action_definition_for_action,
  action_prior_for_resistance,
  constrain_action_distribution_for_utterance,
  fallback_action_distribution_for_resistance,
  merge_action_distributions,
  normalize_patient_action,
  sample_patient_action,
  should_select_patient_action,
  validate_action_distribution,
)
from .state import DEFAULT_SUBSTAGE_BY_STAGE, MI_STAGES, SessionState, Stage, SubStage, TIBRouteBucket
from .support import ConversationProcessingError
from .techniques import TechniqueStageMixin


class TherapyGraphBuilder(
  OpeningStageMixin,
  MIStageMixin,
  AddressTIBStageMixin,
  CognitiveStageMixin,
  TechniqueStageMixin,
  HomeworkStageMixin,
  ClosingStageMixin,
):
  MI_END_SESSION_TIB_REDIRECT_STAGES = {"mi_techniques_with_te", "mi_techniques", "meta_conversation"}

  def __init__(self, config_path: str, profile: dict[str, Any], *, disable_tib: bool = False):
    self.cfg = load_runtime_config(config_path)
    self.profile = profile
    self.disable_tib = bool(disable_tib)
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
    if stage == "mi_techniques_with_te":
      return self._handle_mi_techniques_with_te(state, traces)
    if stage == "mi_techniques":
      return self._handle_mi_techniques(state, traces)
    if stage == "meta_conversation":
      return self._handle_meta_conversation(state, traces)
    if stage == "address_tib":
      return self._handle_address_tib(state, traces)
    if stage == "anticipated_closing":
      return self._handle_anticipated_closing(state, traces)
    if stage == "forced_closing":
      return self._handle_forced_closing(state, traces) #TODO: review it
    self._set_stage(state, "opening")
    return self._handle_opening(state, traces)

  def _route_tib_redirect(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    level: str,
    source_stage: Stage,
    source_substage: SubStage,
  ) -> None:
    if level == "yes":
      if source_stage in {"opening", "id_distortions", "cbt_technique", "homework"}:
        resume_stage: Stage = "id_distortions" if source_stage == "opening" else source_stage
        self._enter_mi(
          state,
          "meta_conversation",
          resume_stage,
          DEFAULT_SUBSTAGE_BY_STAGE[resume_stage],
          f"End-session signal detected during {source_stage}.",
        )
        return
      if source_stage in self.MI_END_SESSION_TIB_REDIRECT_STAGES:
        state["mi_active"] = False
        if state["mi_end_session_tib_redirect_used"]:
          self._set_stage(state, "closing", "closing_fallback")
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
          "step": "tib_avoidance_new_cognitive_set",
          "input": "Avoidance TIB detected against the current cognitive topic.",
          "output": "Shelved the current working set and began a new one.",
        }
      )
      return

    if source_stage == "cbt_technique":
      self._replace_current_technique_due_to_resistance(state, traces)
      traces.append(
        {
          "step": "tib_avoidance_new_technique",
          "input": "Avoidance TIB detected against the current CBT technique.",
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

    if source_stage == "mi_techniques_with_te":
      self._reset_mi_topic_exploration_state(state)
      self._reset_patient_topic_exploration_state(state)
      self._set_substage(state, "mi_choose_topic")
      return

    if source_stage == "mi_techniques":
      self._set_substage(state, "mi_choose_technique")

  def _tib_route_source(self, state: SessionState) -> tuple[Stage, SubStage]:
    if state["current_stage"] == "address_tib" and state["address_tib_resume_stage"]:
      source_stage = state["address_tib_resume_stage"]
      source_substage = state["address_tib_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[source_stage]
      return source_stage, source_substage
    return state["current_stage"], state["current_substage"]

  def _route_bucket_for_tib_name(self, tib_name: str) -> TIBRouteBucket:
    normalized_tib_name = ensure_str(tib_name).strip().lower()
    if not normalized_tib_name or normalized_tib_name == "none":
      return "none"
    if normalized_tib_name == "end_session":
      return "end_session"
    if "avoidance" in normalized_tib_name:
      return "avoidance"
    return "other"

  def _maybe_handle_tib(self, state: SessionState, traces: list[dict[str, Any]]) -> TIBRouteBucket:
    if self._tib_detection_disabled():
      return "none"
    if state["session_ended"] or state["final_patient_pending_end"]:
      return "none"
    if state["current_stage"] in {"anticipated_closing", "forced_closing"}:
      return "none"

    instruction = self._load_stage_prompt("shared", "tib_check")
    parsed = self._run_therapist_structured(state, traces, "tib_check", instruction)
    payload = self._require_dict(parsed, "tib_check")
    has_tib = self._require_bool(payload, "has_tib", "tib_check")
    tib_name = ensure_str(payload.get("tib_name")).strip()
    normalized_tib_name = tib_name.lower() if tib_name else "none"
    route_bucket = self._route_bucket_for_tib_name(normalized_tib_name)
    tib_description = ensure_str(payload.get("tib_description")).strip()
    reasoning = ensure_str(payload.get("reasoning"))
    normalized_payload = dict(payload)
    normalized_payload["has_tib"] = has_tib
    normalized_payload["tib_name"] = normalized_tib_name
    normalized_payload["tib_description"] = tib_description or None
    normalized_payload["route_bucket"] = route_bucket
    self._set_last_trace_output(traces, "tib_check", normalized_payload)

    if not has_tib:
      if normalized_tib_name != "none":
        raise ConversationProcessingError(
          "tib_check.has_tib=false must be paired with tib_name='none'."
        )
      self._clear_completed_address_tib_memory(state)
      self._clear_inline_tib_ack(state)
      return "none"

    if route_bucket == "none":
      raise ConversationProcessingError(
        "tib_check.has_tib=true must not be paired with tib_name='none'."
      )
    if not tib_description:
      raise ConversationProcessingError("tib_check.tib_description must be a non-empty string when has_tib=true.")

    source_stage, source_substage = self._tib_route_source(state)
    remembered_tib_name = ensure_str(state["last_completed_address_tib_name"]).strip().lower()
    if (
      route_bucket == "other"
      and state["last_completed_address_tib_no_reroute_active"]
      and state["last_completed_address_tib_route_bucket"] == "other"
      and normalized_tib_name == remembered_tib_name
    ):
      self._set_inline_tib_ack(state, normalized_tib_name, tib_description)
      traces.append(
        {
          "step": "address_tib_repeat_inline_suppressed",
          "input": tib_description,
          "output": json.dumps(
            {
              "tib_name": normalized_tib_name,
              "current_stage": state["current_stage"],
              "current_substage": state["current_substage"],
              "remembered_tib_name": state["last_completed_address_tib_name"],
              "remembered_tib_no_reroute_active": state["last_completed_address_tib_no_reroute_active"],
            },
            ensure_ascii=True,
          ),
        }
      )
      return "none"

    if route_bucket == "end_session":
      if state["current_stage"] == "meta_conversation":
        self._mark_tib_redirect(state, "yes", source_stage, source_substage, reasoning)
        self._set_stage(state, "closing", "closing_fallback")
        traces.append(
          {
            "step": "meta_conversation_end_session_repeat",
            "input": tib_description,
            "output": "Repeated end-session demand during meta-conversation. Routing directly to closing fallback.",
          }
        )
        return route_bucket

      if state["current_stage"] == "address_tib":
        self._clear_address_tib_state(state, clear_resume_cursor=True)
        self._clear_inline_tib_ack(state)
        self._set_stage(state, source_stage, source_substage)

      self._mark_tib_redirect(state, "yes", source_stage, source_substage, reasoning)
      self._enter_mi(
        state,
        "meta_conversation",
        source_stage,
        source_substage,
        f"Explicit request to end the session: {tib_description}",
      )
      state["mi_end_session_tib_redirect_used"] = True
      traces.append(
        {
          "step": "meta_conversation_end_session_redirect",
          "input": tib_description,
          "output": f"Routing to meta-conversation before resuming {source_stage}.",
        }
      )
      return route_bucket
    if route_bucket == "avoidance":
      if state["current_stage"] == "address_tib":
        self._clear_address_tib_state(state, clear_resume_cursor=True)
        self._clear_inline_tib_ack(state)
        self._set_stage(state, source_stage, source_substage)
      self._mark_tib_redirect(state, "partial", source_stage, source_substage, reasoning)
      self._route_tib_redirect(state, traces, "partial", source_stage, source_substage)
      return route_bucket

    if state["current_stage"] == "address_tib":
      self._replace_active_address_tib(state, traces, route_bucket, normalized_tib_name, tib_description, reasoning)
    else:
      self._enter_address_tib(
        state,
        traces,
        route_bucket,
        normalized_tib_name,
        tib_description,
        reasoning,
        source_stage,
        source_substage,
      )
    return route_bucket

  def therapist_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    first_exchange = len(s["messages"]) == 0
    if s["session_ended"]:
      return s

    traces: list[dict[str, Any]] = []
    try:
      self._clear_tib_redirect(s)
      if self._tib_detection_disabled() and s["current_stage"] == "address_tib":
        raise ConversationProcessingError(
          "TIB detection is disabled, but the session is already in address_tib. "
          "Set env.disable_tib=false or restart from a non-address_tib state."
        )
      if not first_exchange:
        self._maybe_force_closing(s)
        completed_address_tib = self._maybe_complete_finished_address_tib(s, traces)
        tib_route_bucket = "none"
        if not completed_address_tib and not self._tib_detection_disabled():
          tib_route_bucket = self._maybe_handle_tib(s, traces)
        if self._resistance_enabled():
          self._run_resistance_prediction(s, traces)
        if tib_route_bucket == "none" and self._resistance_enabled():
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
      self._clear_tib_redirect(s)
    except ConversationProcessingError as exc:
      self._clear_tib_redirect(s)
      self._end_session_with_error(s, traces, exc)
    return s

  def _patient_utterance_instruction(self, state: SessionState) -> str:
    stage = state["current_stage"]
    substage = state["current_substage"]
    if state["final_patient_pending_end"]:
      return "Provide a brief closing response."
    if not self._resistance_enabled():
      if stage == "opening":
        return "Respond to the therapist's utterance, sharing only what feels comfortable."
      if stage == "id_distortions":
        return "Respond to the therapist's utterance. When you are comfortable, share your real thoughts, emotions, and behavior."
      if stage == "cbt_technique":
        return "Respond to the therapist's utterance."
      if stage == "homework":
        return "Respond honestly about whether the proposed homework feels feasible."
      if stage == "address_tib":
        return "Respond to the therapist's attempt to address your concern."
      if stage in {"mi_techniques_with_te", "mi_techniques", "meta_conversation"}:
        return "Respond to the motivational discussion naturally."
      if stage in {"anticipated_closing", "forced_closing"} or (stage == "closing" and substage == 'closing_finalize_feedback'):
        return "Respond briefly to close the session."
      if stage == "closing":
        return "Respond to the therapist's closing prompts without closing the conversation."
      return "Respond naturally."
    if stage == "opening":
      return "Respond to the therapist's utterance, sharing only what feels comfortable with respect to your resistance level."
    if stage == "id_distortions":
      return "Respond to the therapist's question according to your resistance level. When you are comfortable, share your real thoughts, emotions, and behavior."
    if stage == "cbt_technique":
      return "Respond to the CBT exercise with realistic engagement for your current resistance level."
    if stage == "homework":
      return "Respond honestly about whether the proposed homework feels feasible depending on your current resistance level."
    if stage == "address_tib":
      return "Respond to the therapist's attempt to address your concern according to your resistance level."
    if stage in {"mi_techniques_with_te", "mi_techniques", "meta_conversation"}:
      return "Respond to the motivational discussion in line with your current resistance level."
    if stage in {"anticipated_closing", "forced_closing"} or (stage == "closing" and substage == 'closing_finalize_feedback'):
      return "Respond briefly to close the session."
    if stage == "closing":
      return "Respond to the therapist's closing prompts without closing the conversation."
    return "Respond naturally."

  def _last_therapist_utterance(self, state: SessionState) -> str:
    for message in reversed(state["messages"]):
      if message["role"] == "therapist":
        return message["content"]
    return "(no therapist utterance yet)"

  def _patient_reasoning_for_prompt(self, reasoning: str) -> str:
    text = ensure_str(reasoning)
    if text:
      return text
    return "No explicit resistance-update reasoning was generated. Infer the impact from the last therapist utterance."

  def _reset_patient_action_state(self, state: SessionState) -> None:
    state["last_patient_action"] = ""
    state["last_patient_action_prior_distribution"] = cast(PatientActionDistribution, {})
    state["last_patient_action_context_distribution"] = cast(PatientActionDistribution, {})
    state["last_patient_action_distribution"] = cast(PatientActionDistribution, {})

  def _build_patient_utterance_core_instruction(
    self,
    state: SessionState,
    resistance_update_reasoning: str,
    selected_action: PatientAction | None = None,
    *,
    prompt_stage: str = "shared",
  ) -> str:
    if not self._resistance_enabled():
      return self._load_stage_prompt(
        prompt_stage,
        "patient_utterance_core_weak",
        patient_utterance_instruction=self._patient_utterance_instruction(state),
      )
    action = normalize_patient_action(selected_action)
    action_definition = action_definition_for_action(action)
    return self._load_stage_prompt(
      prompt_stage,
      "patient_utterance_core",
      patient_utterance_instruction=self._patient_utterance_instruction(state),
      resistance_update_reasoning=self._patient_reasoning_for_prompt(resistance_update_reasoning),
      selected_action=action or "No action override",
      selected_action_definition=action_definition,
    )

  def _build_patient_resistance_update_instruction(
    self,
    state: SessionState,
    *,
    prompt_stage: str = "shared",
  ) -> str:
    base_values = {
      "resistance": f"{state['resistance']:.2f}",
      "presenting_problem": ensure_str(state["patient_profile"].get("presenting_problem")),
      "reason_for_seeking_counseling": ensure_str(state["patient_profile"].get("reason_seeking_counseling")),
      "last_utterance": self._last_therapist_utterance(state),
    }
    if prompt_stage != "mi_techniques_with_te":
      return self._load_stage_prompt(
        prompt_stage,
        "patient_resistance_update",
        **base_values,
      )
    return self._load_stage_prompt(
      prompt_stage,
      "patient_resistance_update",
      **base_values,
      perceived_topic=ensure_str(state.get("last_patient_perceived_topic")) or "(unknown)",
      motivation_topic=ensure_str(state["patient_profile"].get("motivation_topic")) or "(unknown)",
      engagement_level=(
        str(state["last_patient_engagement_level"])
        if state.get("last_patient_engagement_level") is not None
        else "(unknown)"
      ),
    )

  def _build_patient_action_distribution_instruction(
    self,
    state: SessionState,
    resistance_value: float,
    resistance_update_reasoning: str,
  ) -> str:
    return self._load_stage_prompt(
      "shared",
      "patient_action_distribution",
      patient_utterance_instruction=self._patient_utterance_instruction(state),
      resistance=f"{resistance_value:.2f}",
      resistance_update_reasoning=self._patient_reasoning_for_prompt(resistance_update_reasoning),
    )

  def _select_patient_action(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    resistance_value: float,
    resistance_update_reasoning: str,
  ) -> PatientAction:
    instruction = self._build_patient_action_distribution_instruction(state, resistance_value, resistance_update_reasoning)
    record_large_prompt_request(
      state,
      self._build_patient_request_messages(state, instruction, structured=True),
      "patient",
      "patient_action_selection",
    )

    context_distribution: PatientActionDistribution | None = None
    fallback_reason = ""
    try:
      parsed, _meta = self.patient.generate_structured(
        full_conversation=self._full_conversation_for_prompt(state),
        last_utterances=self._format_last_utterances(state["messages"]),
        resistance_level=state["resistance"],
        instruction=instruction,
        expected_type=dict,
      )
      context_distribution = validate_action_distribution(parsed)
      if context_distribution is None:
        fallback_reason = "invalid_action_distribution_payload"
    except StructuredOutputError as exc:
      fallback_reason = f"structured_generation_failed: {exc}"

    if context_distribution is None:
      context_distribution = fallback_action_distribution_for_resistance(
        state["resistance"],
        state["utterance_index"],
      )
    else:
      context_distribution = constrain_action_distribution_for_utterance(
        context_distribution,
        state["utterance_index"],
      )

    prior_distribution = action_prior_for_resistance(
      state["resistance"],
      state["utterance_index"],
    )
    merged_distribution = merge_action_distributions(prior_distribution, context_distribution)
    merged_distribution = constrain_action_distribution_for_utterance(
      merged_distribution,
      state["utterance_index"],
      normalize=True,
    )
    selected_action = sample_patient_action(
      merged_distribution,
      utterance_index=state["utterance_index"],
    )

    state["last_patient_action"] = selected_action
    state["last_patient_action_prior_distribution"] = {
      action: float(weight) for action, weight in prior_distribution.items()
    }
    state["last_patient_action_context_distribution"] = {
      action: float(weight) for action, weight in context_distribution.items()
    }
    state["last_patient_action_distribution"] = {
      action: float(weight) for action, weight in merged_distribution.items()
    }

    trace_payload: dict[str, Any] = {
      "prior_distribution": prior_distribution,
      "context_distribution": context_distribution,
      "action_distribution": merged_distribution,
      "selected_action": selected_action,
    }
    if fallback_reason:
      trace_payload["fallback_reason"] = fallback_reason
    traces.append(
      {
        "step": "patient_action_selection",
        "input": instruction,
        "output": json.dumps(trace_payload, ensure_ascii=True),
      }
    )
    return selected_action

  def patient_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s

    traces: list[dict[str, Any]] = []
    try:
      resistance_prompt_stage = "shared"
      resistance_engagement_block: str | None = None
      utterance_prompt_stage = "shared"
      utterance_engagement_block: str | None = None
      if s["current_stage"] == "mi_techniques_with_te":
        topic_engagement = self._run_patient_topic_exploration_engagement(s, traces)
        resistance_prompt_stage = "mi_techniques_with_te"
        resistance_engagement_block = topic_engagement["engagement_analysis"]
        utterance_prompt_stage = "mi_techniques_with_te"
        utterance_engagement_block = topic_engagement["engagement_instruction"]

      selected_action: PatientAction | None = None
      if self._resistance_enabled():
        update_instruction = self._build_patient_resistance_update_instruction(
          s,
          prompt_stage=resistance_prompt_stage,
        )
        update = self._run_patient_structured(
          s,
          traces,
          "resistance_update",
          update_instruction,
          engagement_block=resistance_engagement_block,
        )
        payload = self._require_dict(update, "resistance_update")
        delta_llm = self._require_bounded_float(payload, "r_change", "resistance_update", -0.10, 0.15)
        print(f"    <DEBUG>Predicted resistance change (delta_llm): {delta_llm:.4f}</DEBUG>") #devlog
        resistance_update = self._apply_homeostatic_resistance_update(
          previous_r=s["resistance"],
          delta_llm=delta_llm,
          baseline=s["resistance_baseline"],
          alpha=s["resistance_recovery_rate"],
        )
        s["resistance"] = resistance_update["new_r"]
        s["last_patient_resistance_update_reasoning"] = ensure_str(payload.get("reasoning"))
        print(f"    <DEBUG>Resistance reasoning: {s['last_patient_resistance_update_reasoning']}</DEBUG>") #devlog
        trace_payload = dict(payload)
        trace_payload.update(resistance_update)
        self._set_last_trace_output(traces, "resistance_update", trace_payload)

        if should_select_patient_action(s["final_patient_pending_end"]):
          selected_action = self._select_patient_action(
            s,
            traces,
            resistance_update["new_r"],
            s["last_patient_resistance_update_reasoning"],
          )
        else:
          self._reset_patient_action_state(s)
      else:
        self._reset_patient_action_state(s)
        s["last_patient_resistance_update_reasoning"] = ""

      if self._resistance_enabled():
        print(f"    <DEBUG>Selected patient action: {selected_action}</DEBUG>") #devlog
      utterance_instruction = self._build_patient_utterance_core_instruction(
        s,
        s["last_patient_resistance_update_reasoning"],
        selected_action=selected_action,
        prompt_stage=utterance_prompt_stage,
      )
      record_large_prompt_request(
        s,
        self._build_patient_request_messages(
          s,
          utterance_instruction,
          structured=False,
          engagement_block=utterance_engagement_block,
        ),
        "patient",
        "patient_utterance",
      )
      utterance = self.patient.generate_utterance(
        full_conversation=self._full_conversation_for_prompt(s),
        last_utterances=self._format_last_utterances(s["messages"]),
        resistance_level=s["resistance"],
        instruction=utterance_instruction,
        engagement_block=utterance_engagement_block,
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
