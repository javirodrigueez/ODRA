from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .patient_topic_exploration import PatientTopicExplorationMixin
from .state import CBT_STAGES, DEFAULT_SUBSTAGE_BY_STAGE, SessionState, Stage, SubStage
from .support import ConversationProcessingError
from .topic_exploration import TopicExplorationMixin


class MIStageMixin(PatientTopicExplorationMixin, TopicExplorationMixin):
  META_CONVERSATION_TURN_CAP = 5
  PRECONTEMPLATION_ENTRY_STREAK = 5
  CAMI_RECOVERY_STREAK = 5
  LEGACY_MI_TECHNIQUE_TURN_CAP = 10

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

  def _is_resistance_handling_stage(self, stage: Stage) -> bool:
    return stage in {"mi_techniques", "mi_techniques_with_te"}

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
    state["preconsecutive_turns"] = 0
    state["cami_recovery_turns"] = 0
    state["current_mi_technique_selection"] = []
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

  def _evaluate_mi_and_resume(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    *,
    note: str = "",
    prompt_stage: str = "mi_techniques",
    transition_stage: Stage = "mi_techniques",
  ) -> None:
    instruction = self._load_stage_prompt(prompt_stage, "evaluation")
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
    state["preconsecutive_turns"] = 0
    state["cami_recovery_turns"] = 0
    state["last_mi_predicted_resistance"] = state["predicted_resistance"]
    self._set_stage(state, transition_stage, "mi_transition_to_cbt")

  def _update_preconsecutive_turns(self, state: SessionState, band: str) -> None:
    if self._is_resistance_handling_stage(state["current_stage"]) or state["current_stage"] == "meta_conversation":
      state["preconsecutive_turns"] = 0
      return
    if band == "precontemplation":
      state["preconsecutive_turns"] += 1
    else:
      state["preconsecutive_turns"] = 0

  def _maybe_enter_resistance_handling_from_cbt(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    reason: str,
  ) -> bool:
    if state["current_stage"] not in CBT_STAGES:
      return False
    if state["preconsecutive_turns"] < self.PRECONTEMPLATION_ENTRY_STREAK:
      return False

    mode = self._resistance_handling_mode()
    if mode == "none":
      return False

    preconsecutive_turns = state["preconsecutive_turns"]
    target_stage: Stage = "mi_techniques" if mode == "mi_techniques" else "mi_techniques_with_te"
    self._enter_mi(
      state,
      target_stage,
      state["current_stage"],
      state["current_substage"],
      reason,
    )
    traces.append(
      {
        "step": "resistance_handling_entry",
        "input": f"preconsecutive_turns={preconsecutive_turns}, mode={mode}",
        "output": f"Entering {target_stage} after {self.PRECONTEMPLATION_ENTRY_STREAK} consecutive precontemplation turns.",
      }
    )
    return True

  def _select_mi_techniques(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    *,
    prompt_stage: str,
    step_name: str,
  ) -> list[str]:
    techniques_catalog = self.mi_techniques
    formatted_catalog = "\n".join(
      f"- {name}: {description}"
      for name, description in techniques_catalog.items()
    )
    instruction = self._load_stage_prompt(
      prompt_stage,
      "technique_selection",
      formatted_catalog=formatted_catalog,
    )
    parsed = self._run_therapist_structured(state, traces, step_name, instruction)
    payload = self._require_dict(parsed, step_name)
    selected_techniques = self._parse_selected_mi_techniques(payload, techniques_catalog)
    state["current_mi_technique_selection"] = selected_techniques
    state["mi_techniques_used"].extend(selected_techniques)
    return selected_techniques

  def _handle_pre_stage_routing(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    stage = state["current_stage"]
    band = self._current_band(state["predicted_resistance"], state)
    self._update_preconsecutive_turns(state, band)

    if stage in {"mi_techniques", "mi_techniques_with_te"} and state["long_session_mi_route_to_homework_pending"]:
      state["long_session_mi_route_to_homework_pending"] = False
      state["mi_active"] = False
      state["current_mi_technique_selection"] = []
      self._set_stage(state, "homework")
      traces.append(
        {
          "step": "long_session_mi_homework_handoff",
          "input": f"Reached long-session threshold at utterance {state['utterance_index']}.",
          "output": "Routing directly from MI to homework after the timeout-style MI exit utterance.",
        }
      )
      return

    if stage in {"mi_techniques", "mi_techniques_with_te"} and self._long_session_threshold_reached(state):
      return

    if stage == "opening":
      return

    if self._is_resistance_handling_stage(stage) and state["current_substage"] == "mi_transition_to_cbt":
      resume_stage = state["cbt_resume_stage"] or "id_distortions"
      resume_substage = state["cbt_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
      self._set_stage(state, resume_stage, resume_substage)
      return

    if stage in CBT_STAGES:
      self._maybe_enter_resistance_handling_from_cbt(
        state,
        traces,
        f"CBT interrupted after {self.PRECONTEMPLATION_ENTRY_STREAK} consecutive precontemplation turns.",
      )
      return

    if stage == "mi_techniques":
      if band == "action":
        self._evaluate_mi_and_resume(state, traces, note="MI reached action band")
        return
      if self._current_stage_exchange_count(state, "mi_techniques") >= self.LEGACY_MI_TECHNIQUE_TURN_CAP:
        traces.append(
          {
            "step": "mi_techniques_escape",
            "input": f"MI techniques reached max {self.LEGACY_MI_TECHNIQUE_TURN_CAP} exchanges without full resolution.",
            "output": "Proceeding with CBT and noting unresolved resistance.",
          }
        )
        state["mi_techniques_paused_until_precontemplation"] = True
        self._evaluate_mi_and_resume(state, traces, note="MI reached max-turn escape condition")
      return

    if stage == "mi_techniques_with_te":
      if band == "precontemplation":
        state["cami_recovery_turns"] = 0
      else:
        state["cami_recovery_turns"] += 1
      if state["cami_recovery_turns"] >= self.CAMI_RECOVERY_STREAK:
        traces.append(
          {
            "step": "cami_recovery_exit",
            "input": f"CAMI recovery streak reached {state['cami_recovery_turns']} turn(s) below 0.66.",
            "output": "Transitioning from CAMI resistance handling back to CBT.",
          }
        )
        self._evaluate_mi_and_resume(
          state,
          traces,
          note="CAMI reached sustained recovery condition",
          prompt_stage="mi_techniques_with_te",
          transition_stage="mi_techniques_with_te",
        )
      return

  def _handle_mi_techniques(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    long_session_exit = self._long_session_threshold_reached(state)
    if long_session_exit:
      state["long_session_timeout_instruction_pending"] = True
      state["long_session_mi_route_to_homework_pending"] = True
      self._set_substage(state, "mi_choose_technique")
      traces.append(
        {
          "step": "long_session_mi_exit",
          "input": f"Reached long-session threshold ({state['long_session_turn_threshold']}) during legacy MI.",
          "output": "Generating one timeout-style MI exit utterance, then handing off to homework.",
        }
      )

    if state["current_substage"] == "mi_transition_to_cbt":
      resume_stage = state["cbt_resume_stage"] or "id_distortions"
      instruction = self._load_stage_prompt(
        "mi_techniques",
        "transition_to_cbt",
        resume_stage=resume_stage,
      )
      return self._generate_therapist_utterance(state, instruction)

    band = self._current_band(state["predicted_resistance"], state)
    if band == "action" and not long_session_exit:
      self._evaluate_mi_and_resume(state, traces, note="MI techniques reached action")
      return self._handle_stage_dispatch(state, traces)

    self._set_substage(state, "mi_choose_technique")
    current_substage = state["current_substage"]

    # Choose the next MI communication technique.
    if current_substage == "mi_choose_technique":
      self._select_mi_techniques(
        state,
        traces,
        prompt_stage="mi_techniques",
        step_name="mi_technique_selection",
      )
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

  def _handle_mi_techniques_with_te(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    if self._long_session_threshold_reached(state):
      state["long_session_timeout_instruction_pending"] = True
      state["long_session_mi_route_to_homework_pending"] = True
      self._set_substage(state, "mi_choose_topic")
      traces.append(
        {
          "step": "long_session_cami_exit",
          "input": f"Reached long-session threshold ({state['long_session_turn_threshold']}) during CAMI.",
          "output": "Generating one timeout-style CAMI exit utterance, then handing off to homework.",
        }
      )

    if state["current_substage"] == "mi_transition_to_cbt":
      resume_stage = state["cbt_resume_stage"] or "id_distortions"
      instruction = self._load_stage_prompt(
        "mi_techniques_with_te",
        "transition_to_cbt",
        resume_stage=resume_stage,
      )
      return self._generate_therapist_utterance(state, instruction)

    self._set_substage(state, "mi_choose_topic")
    self._select_next_mi_topic(state, traces)
    if self._mi_topic_exploration_exhausted(state) and self._current_band(state["predicted_resistance"], state) == "precontemplation":
      self._set_stage(state, "meta_conversation")
      return self._handle_stage_dispatch(state, traces)

    self._set_substage(state, "mi_choose_technique")
    selected_techniques = self._select_mi_techniques(
      state,
      traces,
      prompt_stage="mi_techniques_with_te",
      step_name="mi_cami_technique_selection",
    )

    topic = state["current_mi_topic"]
    if not topic:
      raise ConversationProcessingError("CAMI resistance handling selected an empty topic.")
    technique_blend = self._format_mi_technique_blend(self.mi_techniques, selected_techniques)
    self._set_substage(state, "mi_raise_cami_utterance")
    instruction = self._load_stage_prompt(
      "mi_techniques_with_te",
      "raise_utterance",
      topic=topic,
      topic_path=" -> ".join(state["mi_topic_stack"]),
      topic_focus=self._topic_focus_hint(topic),
      technique_blend=technique_blend,
    )
    utterance = self._generate_therapist_utterance(state, instruction)
    state["current_mi_technique_selection"] = []
    return utterance

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
    state["session_ended_reason"] = "max_turns_forced_closing"
    state["final_patient_pending_end"] = True
    self._set_stage(state, "closing", "closing_fallback")
    return self._handle_closing(state, traces)
