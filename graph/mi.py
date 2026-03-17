from __future__ import annotations

from agents.interfaces import ensure_str

from .support import ConversationProcessingError
from .state import CBT_STAGES, MI_TOPICS, SessionState


class MIStageMixin:
  def _enter_mi(self, state: SessionState, target_stage: str, resume_stage: str | None, reason: str) -> None:
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
    state["mi_context"].append({"role": "system_note", "content": reason})
    if target_stage == "mi_techniques":
      state["mi_technique_turns"] = 0
    if target_stage == "meta_conversation":
      state["meta_conversation_turns"] = 0

  def _evaluate_mi_and_resume(self, state: SessionState, traces: list[dict], note: str = "") -> None:
    instruction = (
      "Using mi_context, create an MI intervention evaluation. "
      'Respond with JSON: {"reason_for_mi":"...", "conclusions":"...", "patient_state_change":"..."}'
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
    self._set_stage(state, state["cbt_resume_stage"] or "id_distortions")

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
        self._set_stage(state, "mi_topic_exploration")
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

    if stage == "mi_topic_exploration" and band in {"contemplation", "action"}:
      self._set_stage(state, "mi_techniques")
      return

    if stage == "meta_conversation" and band in {"contemplation", "action"}:
      self._set_stage(state, "mi_techniques")

  def _handle_mi_topic_exploration(self, state: SessionState, traces: list[dict]) -> str:
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
      topic_instruction = (
        f"Select the best MI topic to explore next from: {', '.join(available)}. "
        'Respond JSON: {"topic":"...", "reasoning":"..."}'
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
    if len(state["mi_topics_tried"]) >= 3 and self._current_band(state["predicted_resistance"]) == "precontemplation":
      self._set_stage(state, "meta_conversation")
      return self._handle_stage_dispatch(state, traces)

    # Ask the topic-exploration question.
    if current_substage == "mi_raise_topic":
      topic = state["current_mi_topic"]
      state["last_mi_predicted_resistance"] = current_pred
      instruction = (
        f'You are doing motivational interviewing topic exploration. Focus on the topic "{topic}". '
        "Use the conversation and session context to explore how the patient's current struggle affects this topic area, without sounding templated. ",
        "This utterance should be designed to elicit the patient's own motivations for change related to this topic and to gently challenge their resistance. "
      )
      return self._generate_therapist_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected MI topic substage: {current_substage}")

  def _handle_mi_techniques(self, state: SessionState, traces: list[dict]) -> str:
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
      instruction = (
        f"Choose one MI communication technique from the following ones. "
        f"{formatted_catalog} "
        "Select the technique that best fits the patient's current resistance, mood, and conversation. "
        'Respond JSON: {"technique":"...", "reasoning":"..."}'
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
      instruction = (
        f'You are doing motivational interviewing using the technique "{technique}". '
        "Generate a therapist utterance that applies this technique to the patient's current resistance and problems, using the conversation and session context. "
      )
      return self._generate_therapist_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected MI technique substage: {current_substage}")

  def _meta_substage_for_turns(self, turns: int) -> str:
    if turns == 0:
      return "mi_meta_open"
    if turns == 1:
      return "mi_meta_probe"
    if turns == 2:
      return "mi_meta_adjust"
    return "mi_meta_close"

  def _handle_meta_conversation(self, state: SessionState, traces: list[dict]) -> str:  #TODO: LLM dynamism
    turns = state["meta_conversation_turns"]
    if turns >= 8 and self._current_band(state["predicted_resistance"]) == "precontemplation":
      self._set_stage(state, "anticipated_closing")
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "anticipated_closing"
      return self._handle_stage_dispatch(state, traces)

    self._set_substage(state, self._meta_substage_for_turns(turns))
    current_substage = state["current_substage"]

    # Open the meta-conversation by acknowledging the difficulty.
    if current_substage == "mi_meta_open":
      return (
        "It seems this process has felt difficult, and I appreciate you staying here. "
        "What would make therapy feel safer or more useful for you right now?"
      )

    # Probe the process for specific points of friction.
    if current_substage == "mi_meta_probe":
      return "What specific parts of this process feel hardest or least helpful from your perspective?"

    # Offer a concrete adjustment after the patient has named a barrier.
    if current_substage == "mi_meta_adjust":
      return (
        "Thank you for naming that. "
        "Would it help if we adjusted the pace and focused only on one small, practical point today?"
      )

    # Keep the meta-conversation focused on a manageable next step.
    if current_substage == "mi_meta_close":
      return "I hear you. What would feel like a manageable next step that still respects your boundaries?"

    raise ConversationProcessingError(f"Unexpected meta-conversation substage: {current_substage}")

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
    return (
      "Thank you for attending today. Even if this wasn’t the right moment to go deeper, "
      "you’re welcome to return whenever you feel ready."
    )

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
