from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .state import SessionState
from .support import ConversationProcessingError, GraphSupportMixin


class ClosingStageMixin(GraphSupportMixin):
  def _generate_closing_summary(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    instruction = self._load_stage_prompt("closing", "summary_generation")
    parsed = self._run_therapist_structured(state, traces, "closing_summary_generation", instruction)
    payload = self._require_dict(parsed, "closing_summary_generation")
    return self._require_non_empty_string(payload, "session_summary", "closing_summary_generation")

  def _generate_next_session_plan(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    instruction = self._load_stage_prompt("closing", "next_session_plan_generation")
    parsed = self._run_therapist_structured(state, traces, "next_session_plan_generation", instruction)
    payload = self._require_dict(parsed, "next_session_plan_generation")
    return self._require_non_empty_string(payload, "next_session_plan", "next_session_plan_generation")

  def _generate_closing_utterance(self, state: SessionState, instruction: str) -> str:
    return self._generate_therapist_utterance(state, instruction)

  def _handle_closing(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    current_substage = state["current_substage"]

    # Share the session summary and invite the patient's confirmation.
    if current_substage == "closing_share_summary":
      if not state["session_summary"]:
        state["session_summary"] = self._generate_closing_summary(state, traces)
      self._set_substage(state, "closing_share_plan")
      instruction = self._load_stage_prompt("closing", "share_summary_utterance")
      return self._generate_closing_utterance(state, instruction)

    # Share the proposed next-session plan.
    if current_substage == "closing_share_plan":
      if not state["next_session_plan"]:
        state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      self._set_substage(state, "closing_review_plan")
      instruction = self._load_stage_prompt("closing", "share_plan_utterance")
      return self._generate_closing_utterance(state, instruction)

    # Review the patient's response to the proposed plan.
    if current_substage == "closing_review_plan":
      agreement_instruction = self._load_stage_prompt("closing", "plan_agreement_check")
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
      state["last_reasoning"] = ensure_str(agreement.get("reasoning", ""), "closing_plan_agreement_check")
      if agreed or state["closing_plan_refinements"] >= 10:
        self._set_substage(state, "closing_review_feedback")
        instruction = self._load_stage_prompt("closing", "request_feedback_utterance")
        return self._generate_closing_utterance(state, instruction)
      state["closing_plan_refinements"] += 1
      state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      instruction = self._load_stage_prompt("closing", "revised_plan_utterance")
      return self._generate_closing_utterance(state, instruction)

    # Acknowledge the patient's feedback and show commitment for next session.
    if current_substage == "closing_review_feedback":
      feedback_instruction = self._load_stage_prompt("closing", "feedback_check")
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
      self._set_substage(state, "closing_finalize_feedback")
      current_substage = state["current_substage"]

    # After acknowledging feedback, deliver the final closing turn.
    if current_substage == "closing_finalize_feedback":
      state["final_patient_pending_end"] = True
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "normal_closing_completed"
      instruction = self._load_stage_prompt("closing", "finalize_feedback_utterance")
      return self._generate_closing_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected closing substage: {current_substage}")
