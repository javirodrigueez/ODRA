from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .support import ConversationProcessingError, GraphSupportMixin
from .state import CBT_STAGES, MI_TOPICS, SessionState, Stage, SubStage


class MIStageMixin(GraphSupportMixin):
  def _maybe_handle_ending_needed(self, state: SessionState, traces: list[dict[str, Any]]) -> str | None:
    if state["final_patient_pending_end"]:
      return None
    if state["current_stage"] not in {"mi_topic_exploration", "mi_techniques", "meta_conversation"}:
      return None

    instruction = self._load_stage_prompt(
      "shared",
      "ending_needed_check",
    )
    parsed = self._run_therapist_structured(state, traces, "ending_needed_check", instruction)
    payload = self._require_dict(parsed, "ending_needed_check")
    if not self._require_bool(payload, "ending_needed", "ending_needed_check"):
      return None

    state["mi_active"] = False
    state["final_patient_pending_end"] = True
    state["session_ended_reason"] = "patient_requested_ending"
    self._set_stage(state, "meta_conversation", "mi_meta_close")
    traces.append(
      {
        "step": "ending_needed_close",
        "input": state["messages"][-1]["content"] if state["messages"] else "",
        "output": "Patient clearly requested to end the conversation; generating empathic closing utterance.",
      }
    )
    instruction = self._load_stage_prompt("meta_conversation", "close_utterance")
    return self._generate_therapist_utterance(state, instruction)

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

  def _enter_mi(self, state: SessionState, target_stage: Stage, resume_stage: Stage | None, reason: str) -> None:
    if resume_stage:
      state["cbt_resume_stage"] = resume_stage
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
    band = self._current_band(state["predicted_resistance"])

    if stage == "opening":
      return

    if stage == "mi_techniques" and state["current_substage"] == "mi_transition_to_cbt":
      self._set_stage(state, state["cbt_resume_stage"] or "id_distortions")
      return

    if stage in CBT_STAGES:
      if band == "contemplation":
        if not state["mi_techniques_paused_until_precontemplation"]:
          self._enter_mi(state, "mi_techniques", stage, "CBT interrupted due to contemplation-level resistance")
      elif band == "precontemplation":
        self._reactivate_mi_techniques(
          state,
          traces,
          "Predicted resistance returned to precontemplation, so MI techniques can be used again.",
        )
        self._enter_mi(state, "mi_topic_exploration", stage, "CBT interrupted due to precontemplation resistance")
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

    if stage == "meta_conversation" and band in {"contemplation", "action"}:
      self._set_stage(state, "mi_techniques")

  def _handle_mi_topic_exploration(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    previous_pred = state["last_mi_predicted_resistance"]
    current_pred = state["predicted_resistance"]
    need_new_topic = not state["current_mi_topic"] or current_pred > previous_pred + 0.02 # 0.02 is a small tolerance to avoid overreacting to minor prediction fluctuations.
    if need_new_topic:
      self._set_substage(state, "mi_choose_topic")

    current_substage = state["current_substage"]

    # Select the next MI topic when resistance has not improved.
    if current_substage == "mi_choose_topic":
      available = [topic for topic in MI_TOPICS if topic not in state["mi_topics_tried"]]
      if not available:
        available = MI_TOPICS
      topic_instruction = self._load_stage_prompt(
        "mi_topic_exploration",
        "topic_selection",
        available_topics=", ".join(available),
      )
      topic = self._run_therapist_structured(state, traces, "mi_topic_selection", topic_instruction)
      topic_payload = self._require_dict(topic, "mi_topic_selection")
      chosen = self._require_non_empty_string(topic_payload, "topic", "mi_topic_selection")

      state["current_mi_topic"] = chosen
      state["mi_topics_tried"].append(chosen)
      state["last_mi_predicted_resistance"] = current_pred
      self._set_substage(state, "mi_raise_topic")
      current_substage = state["current_substage"]

    # Escalate to meta-conversation after repeated topic attempts without movement.
    if len(state["mi_topics_tried"]) >= len(MI_TOPICS) and self._current_band(state["predicted_resistance"]) == "precontemplation":
      self._set_stage(state, "meta_conversation")
      return self._handle_stage_dispatch(state, traces)

    # Ask the topic-exploration question.
    if current_substage == "mi_raise_topic":
      topic = state["current_mi_topic"]
      state["last_mi_predicted_resistance"] = current_pred
      instruction = self._load_stage_prompt(
        "mi_topic_exploration",
        "raise_topic_utterance",
        topic=topic,
      )
      return self._generate_therapist_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected MI topic substage: {current_substage}")

  def _handle_mi_techniques(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    if state["current_substage"] == "mi_transition_to_cbt":
      resume_stage = state["cbt_resume_stage"] or "id_distortions"
      instruction = self._load_stage_prompt(
        "mi_techniques",
        "transition_to_cbt",
        resume_stage=resume_stage,
      )
      return self._generate_therapist_utterance(state, instruction)

    band = self._current_band(state["predicted_resistance"])
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
      technique = self._require_non_empty_string(payload, "technique", "mi_technique_selection")
      state["mi_techniques_used"].append(technique)
      self._set_substage(state, "mi_raise_technique")
      current_substage = state["current_substage"]

    # Deliver the selected MI technique as the next therapist utterance.
    if current_substage == "mi_raise_technique":
      technique = state["mi_techniques_used"][-1]
      instruction = self._load_stage_prompt(
        "mi_techniques",
        "raise_technique_utterance",
        technique=technique,
      )
      return self._generate_therapist_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected MI technique substage: {current_substage}")

  def _meta_substage_for_turns(self, turns: int) -> SubStage:
    if turns == 0:
      return "mi_meta_open"
    if turns == 1:
      return "mi_meta_probe"
    if turns == 2:
      return "mi_meta_adjust"
    return "mi_meta_close"

  def _handle_meta_conversation(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    turns = self._current_stage_exchange_count(state, "meta_conversation")
    if turns >= 8 and self._current_band(state["predicted_resistance"]) == "precontemplation":
      self._set_stage(state, "anticipated_closing")
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "anticipated_closing"
      return self._handle_stage_dispatch(state, traces)

    self._set_substage(state, self._meta_substage_for_turns(turns))
    current_substage = state["current_substage"]
    print(f"<META DEBUG> turns={turns}, substage={current_substage}, predicted_resistance={state['predicted_resistance']:.2f} ({self._current_band(state['predicted_resistance'])}) </META DEBUG>")

    # Open the meta-conversation by acknowledging the difficulty.
    if current_substage == "mi_meta_open":
      instruction = self._load_stage_prompt("meta_conversation", "open_utterance")
      return self._generate_therapist_utterance(state, instruction)

    # Probe the process for specific points of friction.
    if current_substage == "mi_meta_probe":
      instruction = self._load_stage_prompt("meta_conversation", "probe_utterance")
      return self._generate_therapist_utterance(state, instruction)

    # Offer a concrete adjustment after the patient has named a barrier.
    if current_substage == "mi_meta_adjust":
      instruction = self._load_stage_prompt("meta_conversation", "adjust_utterance")
      return self._generate_therapist_utterance(state, instruction)

    # Close meta-conversation and session
    if current_substage == "mi_meta_close":
      state["final_patient_pending_end"] = True
      instruction = self._load_stage_prompt("meta_conversation", "close_utterance")
      return self._generate_therapist_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected meta-conversation substage: {current_substage}")

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
