from __future__ import annotations

from agents.interfaces import ensure_str

from .state import SessionState
from .support import ConversationProcessingError


class ClosingStageMixin:
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

  def _generate_closing_utterance(self, state: SessionState, instruction: str) -> str:
    return self._generate_therapist_utterance(state, instruction)

  def _handle_closing(self, state: SessionState, traces: list[dict]) -> str:
    current_substage = state["current_substage"]

    # Share the session summary and invite the patient's confirmation.
    if current_substage == "closing_share_summary":
      if not state["session_summary"]:
        state["session_summary"] = self._generate_closing_summary(state, traces)
      state["closing_phase"] = "summary"
      self._set_substage(state, "closing_share_plan")
      instruction = (
        "Share the session summary with the patient and invite them to say whether it fits their experience today. "
        f"Session summary to convey: {state['session_summary']}. "
        "Generate only the therapist's next utterance."
      )
      return self._generate_closing_utterance(state, instruction)

    # Share the proposed next-session plan.
    if current_substage == "closing_share_plan":
      if not state["next_session_plan"]:
        state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      state["closing_phase"] = "plan_discussion"
      self._set_substage(state, "closing_review_plan")
      instruction = (
        "Present the proposed next-session plan to the patient and ask for their reaction in a natural way. "
        f"Next-session plan to convey: {state['next_session_plan']}. "
        "Generate only the therapist's next utterance."
      )
      return self._generate_closing_utterance(state, instruction)

    # Review the patient's response to the proposed plan.
    if current_substage == "closing_review_plan":
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
      if agreed or state["closing_plan_refinements"] >= 10:
        state["closing_phase"] = "await_feedback_response"
        self._set_substage(state, "closing_review_feedback")
        instruction = (
          "Proactively ask the patient for feedback about today's session. "
        )
        return self._generate_closing_utterance(state, instruction)
      state["closing_plan_refinements"] += 1
      state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      instruction = (
        "Acknowledge the patient's concern about the next-session plan, present the revised plan, and ask whether it fits better. "
        f"Revised next-session plan to convey: {state['next_session_plan']}. "
        "Generate only the therapist's next utterance."
      )
      return self._generate_closing_utterance(state, instruction)

    # Acknowledge the patient's feedback and show commitment for next session.
    if current_substage == "closing_review_feedback":
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
      state["closing_phase"] = "final_close_pending"
      self._set_substage(state, "closing_finalize_feedback")
      current_substage = state["current_substage"]

    # After acknowledging feedback, deliver the final closing turn.
    if current_substage == "closing_finalize_feedback":
      state["final_patient_pending_end"] = True
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "normal_closing_completed"
      instruction = (
        "Close the session naturally after acknowledging the patient's feedback, with a warm ending that fits the conversation and references continuing the work next time. "
      )
      return self._generate_closing_utterance(state, instruction)

    raise ConversationProcessingError(f"Unexpected closing substage: {current_substage}")
