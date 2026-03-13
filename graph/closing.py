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

  def _handle_closing(self, state: SessionState, traces: list[dict]) -> str:
    current_substage = state["current_substage"]

    # Share the session summary and invite the patient's confirmation.
    if current_substage == "closing_share_summary":
      if not state["session_summary"]:
        state["session_summary"] = self._generate_closing_summary(state, traces)
      state["closing_phase"] = "summary"
      self._set_substage(state, "closing_share_plan")
      return f"{state['session_summary']} Before we finish, does that summary fit your experience today?"

    # Share the proposed next-session plan.
    if current_substage == "closing_share_plan":
      if not state["next_session_plan"]:
        state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      state["closing_phase"] = "plan_discussion"
      self._set_substage(state, "closing_review_plan")
      return f"For next time, my proposed plan is: {state['next_session_plan']} How does that sound to you?"

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
      if agreed or state["closing_plan_refinements"] >= 3:
        state["closing_phase"] = "await_feedback_response"
        self._set_substage(state, "closing_review_feedback")
        return "Before we end, what feedback do you have about today’s session?"
      state["closing_plan_refinements"] += 1
      state["next_session_plan"] = self._generate_next_session_plan(state, traces)
      return f"Thanks for that feedback. Let me adjust the plan: {state['next_session_plan']} Would this version work better?"

    # Review the patient's feedback about the session.
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
      if requested:
        state["closing_phase"] = "final_close_pending"
        self._set_substage(state, "closing_finalize_feedback")
        return (
          "Thank you for that feedback. "
          f"I hear that {description or 'you want us to adjust the process'}. "
          f"Next session I’ll improve by {plan or 'adapting pacing and focus to your preference'}."
        )
      state["closing_phase"] = "completed"
      self._set_substage(state, "closing_end_session")
      return self._handle_closing(state, traces)

    # Provide the final response after specific improvement feedback.
    if current_substage == "closing_finalize_feedback":
      state["closing_phase"] = "completed"
      state["final_patient_pending_end"] = True
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "normal_closing_completed"
      self._set_substage(state, "closing_end_session")
      return "I appreciate your openness today. Let’s pause here, and we’ll continue from this point next session."

    # End the session with the standard closing utterance.
    if current_substage == "closing_end_session":
      state["final_patient_pending_end"] = True
      if not state["session_ended_reason"]:
        state["session_ended_reason"] = "normal_closing_completed"
      return "Thank you for your work today. We’ll pick this up next time and build on what you practiced this week."

    raise ConversationProcessingError(f"Unexpected closing substage: {current_substage}")
