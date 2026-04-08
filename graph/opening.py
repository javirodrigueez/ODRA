from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .support import ConversationProcessingError, GraphSupportMixin
from .state import SessionState


class OpeningStageMixin(GraphSupportMixin):
  def _check_readiness(self, state: SessionState, traces: list[dict[str, Any]]) -> bool:
    instruction = self._load_stage_prompt(
      "opening",
      "info_update"
    )
    update = self._run_therapist_structured(state, traces, "opening_info_update", instruction)
    update_payload = self._require_dict(update, "opening_info_update")
    updated_mood = ensure_str(update_payload.get("mood", state["opening_patient_mood"]))
    updated_problem = ensure_str(update_payload.get("problem", state["opening_patient_problem"]))
    if updated_mood and updated_mood != state["opening_patient_mood"]:
      state["opening_patient_mood"] = updated_mood
    if updated_problem and updated_problem != state["opening_patient_problem"]:
      state["opening_patient_problem"] = updated_problem

    instruction = self._load_stage_prompt(
      "opening",
      "readiness_check"
    )
    readiness = self._run_therapist_structured(state, traces, "opening_readiness_check", instruction)
    payload = self._require_dict(readiness, "opening_readiness_check")
    ready = self._require_bool(payload, "ready", "opening_readiness_check")
    state["last_reasoning"] = ensure_str(payload.get("reasoning", ""))
    issue = ensure_str(payload.get("issue", ""))
    if issue == "problem":
      state["current_substage"] = "open_problem"
    elif issue == "mood":
      state["current_substage"] = "open_mood"
    return ready

  def _handle_opening(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    opening_exchanges = self._current_stage_exchange_count(state, "opening")
    if opening_exchanges == 0:
      self._set_substage(state, "open_initial")

    current_substage = state["current_substage"]

    # Initial greeting
    if current_substage == "open_initial":
      instruction = self._load_stage_prompt("opening", "initial_greeting")
      self._set_substage(state, "open_mood")
      return self._generate_therapist_utterance(state, instruction)

    ready = self._check_readiness(state, traces)
    current_substage = state["current_substage"]
    force_mi = opening_exchanges >= 10 and not ready

    # Patient doesn't cooperates, force MI
    if force_mi:
      state["opening_fallback_triggered"] = True
      state["cbt_resume_stage"] = "id_distortions"
      self._enter_mi(
        state,
        "mi_topic_exploration",
        "id_distortions",
        None,
        "Opening fallback after 10 exchanges",
      )
      return self._handle_stage_dispatch(state, traces)

    # Not ready to continue to id_distortions
    if not ready:
      if current_substage == "open_mood":
        instruction = self._load_stage_prompt("opening", "mood_question")
      else:
        instruction = self._load_stage_prompt("opening", "problem_question")

      return self._generate_therapist_utterance(state, instruction)

    # Ready to move on to id_distortions
    if ready:
      band = self._current_band(state["predicted_resistance"], state)
      if band == "action":
        self._set_stage(state, "id_distortions")
      elif band == "contemplation":
        state["cbt_resume_stage"] = "id_distortions"
        self._enter_mi(
          state,
          "mi_techniques",
          "id_distortions",
          None,
          "Opening completed with contemplation",
        )
      else:
        state["cbt_resume_stage"] = "id_distortions"
        self._enter_mi(
          state,
          "mi_topic_exploration",
          "id_distortions",
          None,
          "Opening completed with precontemplation",
        )
      return self._handle_stage_dispatch(state, traces)

    raise ConversationProcessingError("Opening stage reached an unexpected state.")
