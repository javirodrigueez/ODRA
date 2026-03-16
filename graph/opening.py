from __future__ import annotations

from agents.interfaces import ensure_str

from .support import ConversationProcessingError
from .state import SessionState


class OpeningStageMixin:
  def _check_readiness(self, state: SessionState, traces: list[dict]) -> bool:
    instruction = (
      f"Based on the last patient response, the current saved mood='{state['opening_patient_mood']}' "
      f"and problem='{state['opening_patient_problem']}', can the patient's mood/problem be updated based on the new information? "
      "If so, provide the updated mood and/or problem. If not, respond with the current values. "
      'Respond JSON: {"mood":"...", "problem":"..."}'
    )
    update = self._run_therapist_structured(state, traces, "opening_info_update", instruction)
    update_payload = self._require_dict(update, "opening_info_update")
    updated_mood = ensure_str(update_payload.get("mood", state["opening_patient_mood"]))
    updated_problem = ensure_str(update_payload.get("problem", state["opening_patient_problem"]))
    if updated_mood and updated_mood != state["opening_patient_mood"]:
      state["opening_patient_mood"] = updated_mood
    if updated_problem and updated_problem != state["opening_patient_problem"]:
      state["opening_patient_problem"] = updated_problem

    instruction = (
      f"Based on the conversation so far, current saved mood='{state['opening_patient_mood']}' "
      f"and problem='{state['opening_patient_problem']}', is there enough information to proceed beyond the opening stage? "
      "Consider if the patient's mood and problem are sufficiently understood to move to CBT. "
      "If you are not sure about the patient's mood or specific problem, respond with ready=false and specify the issue in the 'issue' field. "
      "If both fields are issues, begin exploring the patient's mood. Include a reasoning field to explain your decision. "
      "The specific problem should be concrete, for example: the relationship with my family is degraded because of drug consumption, "
      "not just a general area like stress or anxiety. "
      'Respond JSON: {"ready": true/false, "issue": "mood/problem", "mood": "...", "problem": "...", "reasoning":"..."}'
    )
    readiness = self._run_therapist_structured(state, traces, "opening_readiness_check", instruction)
    payload = self._require_dict(readiness, "opening_readiness_check")
    ready = self._require_bool(payload, "ready", "opening_readiness_check")
    issue = ensure_str(payload.get("issue", ""))
    if issue == "problem":
      state["current_substage"] = "open_problem"
    elif issue == "mood":
      state["current_substage"] = "open_mood"
    return ready

  def _handle_opening(self, state: SessionState, traces: list[dict]) -> str:
    opening_exchanges = int(state["exchange_counts"].get("opening", 0))
    if opening_exchanges == 0:
      state["current_substage"] = "open_initial"

    current_substage = state["current_substage"]

    # Initial greeting
    if current_substage == "open_initial":
      instruction = (
        "This is the first utterance of the therapy session. Begin the session with an open and empathetic greeting. "
        "It should be brief, for example: Hello, glad to meet you, or Hi, how are you doing?"
      )
      state["current_substage"] = ""
      return self.therapist.generate_utterance(
        conversation=[],
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )

    ready = self._check_readiness(state, traces)
    current_substage = state["current_substage"]
    force_mi = opening_exchanges >= 10 and not ready

    # Patient doesn't cooperates, force MI
    if force_mi:
      state["opening_fallback_triggered"] = True
      state["cbt_resume_stage"] = "id_distortions"
      self._enter_mi(state, "mi_topic_exploration", "id_distortions", "Opening fallback after 10 exchanges")
      return self._handle_stage_dispatch(state, traces)

    # Not ready to continue to id_distortions
    if not ready:
      if current_substage == "open_mood":
        instruction = (
          "Based on the conversation so far, ask the patient a question to understand their current mood. "
          "This is the mood check of a standard CBT session."
        )
      elif current_substage == "open_problem":
        instruction = (
          "Based on the conversation so far, ask the patient a question to clarify the specific problem they want to work on. "
          "This should be a specific issue, not just a general area."
        )

      return self.therapist.generate_utterance(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )

    # Ready to move on to id_distortions
    if ready:
      band = self._current_band(state["predicted_resistance"])
      if band == "action":
        self._set_stage(state, "id_distortions")
      elif band == "contemplation":
        state["cbt_resume_stage"] = "id_distortions"
        self._enter_mi(state, "mi_techniques", "id_distortions", "Opening completed with contemplation")
      else:
        state["cbt_resume_stage"] = "id_distortions"
        self._enter_mi(state, "mi_topic_exploration", "id_distortions", "Opening completed with precontemplation")
      return self._handle_stage_dispatch(state, traces)

    raise ConversationProcessingError("Opening stage reached an unexpected state.")
