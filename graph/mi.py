from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .state import CBT_STAGES, DEFAULT_SUBSTAGE_BY_STAGE, SessionState, Stage, SubStage
from .support import ConversationProcessingError
from .topic_exploration import TopicExplorationMixin


class MIStageMixin(TopicExplorationMixin):
  META_CONVERSATION_TURN_CAP = 5

  def _parse_selected_mi_techniques(
    self,
    payload: dict[str, Any],
    techniques_catalog: dict[str, str],
  ) -> list[str]:
    raw_selected = payload.get("techniques")
    if isinstance(raw_selected, list):
      selected = [ensure_str(item) for item in raw_selected]
    else:
      selected = [self._require_non_empty_string(payload, "technique", "mi_technique_selection")]

    deduped: list[str] = []
    for technique in selected:
      if not technique:
        continue
      if technique not in techniques_catalog:
        raise ConversationProcessingError(
          f"mi_technique_selection.techniques must contain only catalog techniques, got {technique!r}."
        )
      if technique not in deduped:
        deduped.append(technique)

    if not deduped:
      raise ConversationProcessingError("mi_technique_selection.techniques must contain at least one technique.")
    if len(deduped) > 2:
      raise ConversationProcessingError("mi_technique_selection.techniques must contain at most two techniques.")
    return deduped

  def _format_mi_technique_blend(self, techniques_catalog: dict[str, str], techniques: list[str]) -> str:
    lines = [f"{technique}:{techniques_catalog[technique]}" for technique in techniques if technique in techniques_catalog]
    if not lines:
      raise ConversationProcessingError("Cannot raise an MI technique utterance without a valid selected technique.")
    return "\n".join(lines)

  def _reactivate_mi_techniques(self, state: SessionState, traces: list[dict[str, Any]], reason: str) -> None:
    if not state["mi_techniques_paused_until_precontemplation"]:
      return
    state["mi_techniques_paused_until_precontemplation"] = False
    traces.append(
      {
        "step": "mi_techniques_reactivated",
        "input": "MI techniques were paused after a max-turn contemplation escape.",
        "output": reason,
      }
    )

  def _enter_mi(
    self,
    state: SessionState,
    target_stage: Stage,
    resume_stage: Stage | None,
    resume_substage: SubStage | None,
    reason: str,
  ) -> None:
    if resume_stage:
      state["cbt_resume_stage"] = resume_stage
      state["cbt_resume_substage"] = resume_substage or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
    if target_stage == "mi_topic_exploration":
      self._reset_mi_topic_exploration_state(state)
    state["mi_active"] = True
    if not state["cbt_flow_saved_state"] and resume_stage:
      state["cbt_flow_saved_state"] = (
        f"CBT interrupted at stage={resume_stage}, "
        f"techniques_completed={state['techniques_completed']}, "
        f"working_set_id={state['working_cognitive_set'].get('cognitive_model_set_id')}"
      )
    self._set_stage(state, target_stage)
    self._append_mi_context_message(
      state,
      "system_note",
      reason,
      state["utterance_index"],
      state["current_stage"],
      state["current_substage"],
    )

  def _evaluate_mi_and_resume(self, state: SessionState, traces: list[dict[str, Any]], note: str = "") -> None:
    instruction = self._load_stage_prompt("mi_techniques", "evaluation")
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
    state["last_mi_predicted_resistance"] = state["predicted_resistance"]
    self._set_stage(state, "mi_techniques", "mi_transition_to_cbt")

  def _handle_pre_stage_routing(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    stage = state["current_stage"]
    band = self._current_band(state["predicted_resistance"], state)

    if stage == "opening":
      return

    if stage == "mi_techniques" and state["current_substage"] == "mi_transition_to_cbt":
      resume_stage = state["cbt_resume_stage"] or "id_distortions"
      resume_substage = state["cbt_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
      self._set_stage(state, resume_stage, resume_substage)
      return

    if stage in CBT_STAGES:
      if band == "contemplation":
        if not state["mi_techniques_paused_until_precontemplation"]:
          self._enter_mi(
            state,
            "mi_techniques",
            stage,
            state["current_substage"],
            "CBT interrupted due to contemplation-level resistance",
          )
      elif band == "precontemplation":
        self._reactivate_mi_techniques(
          state,
          traces,
          "Predicted resistance returned to precontemplation, so MI techniques can be used again.",
        )
        self._enter_mi(
          state,
          "mi_topic_exploration",
          stage,
          state["current_substage"],
          "CBT interrupted due to precontemplation resistance",
        )
      return

    if stage == "mi_techniques":
      if band == "precontemplation":
        self._set_stage(state, "mi_topic_exploration")
        return
      if band == "action":
        self._evaluate_mi_and_resume(state, traces, note="MI reached action band")
        return
      if self._current_stage_exchange_count(state, "mi_techniques") >= 10:
        traces.append(
          {
            "step": "mi_techniques_escape",
            "input": "MI techniques reached max 10 exchanges without full resolution.",
            "output": "Proceeding with CBT and noting unresolved resistance.",
          }
        )
        state["mi_techniques_paused_until_precontemplation"] = True
        self._evaluate_mi_and_resume(state, traces, note="MI reached max-turn escape condition")
      return

    if stage == "mi_topic_exploration" and band in {"contemplation", "action"}:
      self._set_stage(state, "mi_techniques")
      return

  def _handle_mi_techniques(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    if state["current_substage"] == "mi_transition_to_cbt":
      resume_stage = state["cbt_resume_stage"] or "id_distortions"
      instruction = self._load_stage_prompt(
        "mi_techniques",
        "transition_to_cbt",
        resume_stage=resume_stage,
      )
      return self._generate_therapist_utterance(state, instruction)

    band = self._current_band(state["predicted_resistance"], state)
    if band == "precontemplation":
      self._set_stage(state, "mi_topic_exploration")
      return self._handle_stage_dispatch(state, traces)
    if band == "action":
      self._evaluate_mi_and_resume(state, traces, note="MI techniques reached action")
      return self._handle_stage_dispatch(state, traces)

    self._set_substage(state, "mi_choose_technique")
    current_substage = state["current_substage"]

    # Choose the next MI communication technique.
    if current_substage == "mi_choose_technique":
      techniques_catalog = self.mi_techniques
      formatted_catalog = "\n".join(
        f'- {name}: {description}'
        for name, description in techniques_catalog.items()
      )
      instruction = self._load_stage_prompt(
        "mi_techniques",
        "technique_selection",
        formatted_catalog=formatted_catalog,
      )
      parsed = self._run_therapist_structured(state, traces, "mi_technique_selection", instruction)
      payload = self._require_dict(parsed, "mi_technique_selection")
      selected_techniques = self._parse_selected_mi_techniques(payload, techniques_catalog)
      state["current_mi_technique_selection"] = selected_techniques
      state["mi_techniques_used"].extend(selected_techniques)
      self._set_substage(state, "mi_raise_technique")
      current_substage = state["current_substage"]

    # Deliver the selected MI technique as the next therapist utterance.
    if current_substage == "mi_raise_technique":
      techniques_catalog = self.mi_techniques
      selected_techniques = state["current_mi_technique_selection"] or state["mi_techniques_used"][-1:]
      technique_blend = self._format_mi_technique_blend(techniques_catalog, selected_techniques)
      instruction = self._load_stage_prompt(
        "mi_techniques",
        "raise_technique_utterance",
        technique_blend=technique_blend,
      )
      utterance = self._generate_therapist_utterance(state, instruction)
      state["current_mi_technique_selection"] = []
      return utterance

    raise ConversationProcessingError(f"Unexpected MI technique substage: {current_substage}")

  def _meta_substage_for_turns(self, turns: int) -> SubStage:
    if turns == 0:
      return "mi_meta_open"
    if turns >= self.META_CONVERSATION_TURN_CAP - 1:
      return "mi_meta_close"
    # if turns % 2 == 1:
    #   return "mi_meta_probe"
    # return "mi_meta_adjust"
    return "mi_meta_probe"

  def _check_meta_conversation_readiness(self, state: SessionState, traces: list[dict[str, Any]]) -> bool:
    instruction = self._load_stage_prompt("meta_conversation", "readiness_check")
    readiness = self._run_therapist_structured(
      state,
      traces,
      "meta_conversation_readiness_check",
      instruction,
    )
    payload = self._require_dict(readiness, "meta_conversation_readiness_check")
    ready = self._require_bool(payload, "ready", "meta_conversation_readiness_check")
    return ready

  def _handle_meta_conversation_legacy(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    turns = self._current_stage_exchange_count(state, "meta_conversation")
    if turns >= 8 and self._current_band(state["predicted_resistance"], state) == "precontemplation":
      self._set_stage(state, "anticipated_closing")
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "anticipated_closing"
      return self._handle_stage_dispatch(state, traces)

    #self._set_substage(state, self._meta_substage_for_turns(turns))
    current_substage = state["current_substage"] 
    #print(f"<META DEBUG> turns={turns}, substage={current_substage}, predicted_resistance={state['predicted_resistance']:.2f} ({self._current_band(state['predicted_resistance'])}) </META DEBUG>")

    # Open the meta-conversation by acknowledging the difficulty.
    if current_substage == "mi_meta_open":
      state["current_substage"] = "mi_meta_probe"
      instruction = self._load_stage_prompt("meta_conversation", "open_utterance")
      return self._generate_therapist_utterance(state, instruction)

    # Probe the process for specific points of friction.
    if current_substage == "mi_meta_probe":
      state["current_substage"] = "mi_meta_close"
      instruction = self._load_stage_prompt("meta_conversation", "probe_utterance")
      return self._generate_therapist_utterance(state, instruction)

    # Offer a concrete adjustment after the patient has named a barrier.
    # if current_substage == "mi_meta_adjust":
    #   instruction = self._load_stage_prompt("meta_conversation", "adjust_utterance")
    #   return self._generate_therapist_utterance(state, instruction)

    # Close meta-conversation and session
    if current_substage == "mi_meta_close":
      state["final_patient_pending_end"] = True
      instruction = self._load_stage_prompt("meta_conversation", "close_utterance")
      return self._generate_therapist_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected meta-conversation substage: {current_substage}")

  def _handle_meta_conversation(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    turns = self._current_stage_exchange_count(state, "meta_conversation")

    if turns == 0:
      self._set_substage(state, "mi_meta_open")
      instruction = self._load_stage_prompt("meta_conversation", "open_utterance")
      return self._generate_therapist_utterance(state, instruction)

    # ready_to_resume = self._check_meta_conversation_readiness(state, traces)
    # if ready_to_resume:
    #   resume_stage = state["cbt_resume_stage"] or "id_distortions"
    #   traces.append(
    #     {
    #       "step": "meta_conversation_resume",
    #       "input": f"Meta-conversation readiness check after {turns} completed exchange(s).",
    #       "output": f"Patient appears ready to resume CBT flow at {resume_stage}.",
    #     }
    #   )
    #   self._set_stage(state, resume_stage)
    #   return self._handle_stage_dispatch(state, traces)

    current_substage = self._meta_substage_for_turns(turns)
    self._set_substage(state, current_substage)

    if current_substage == "mi_meta_close":
      traces.append(
        {
          "step": "meta_conversation_turn_cap",
          "input": f"Meta-conversation remained unresolved after {turns} completed exchange(s).",
          "output": "Closing the conversation with the meta-conversation close utterance.",
        }
      )
      state["final_patient_pending_end"] = True
      instruction = self._load_stage_prompt("meta_conversation", "close_utterance")
      return self._generate_therapist_utterance(state, instruction)

    prompt_name = "probe_utterance" if current_substage == "mi_meta_probe" else "adjust_utterance"
    instruction = self._load_stage_prompt("meta_conversation", prompt_name)
    return self._generate_therapist_utterance(state, instruction)

  def _handle_anticipated_closing(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
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
    return self._load_stage_prompt("anticipated_closing", "final_utterance")

  def _handle_forced_closing(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    state["final_patient_pending_end"] = True
    state["session_ended_reason"] = "max_turns_forced_closing"
    traces.append(
      {
        "step": "max_turns_forced_closing",
        "input": f"Reached max_turns threshold ({state['max_turns']}).",
        "output": "Triggering brief forced closing exchange.",
      }
    )
    return self._load_stage_prompt("forced_closing", "final_utterance")
