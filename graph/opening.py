from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .support import ConversationProcessingError, GraphSupportMixin
from .state import SessionState


class OpeningStageMixin(GraphSupportMixin):
  OPENING_MAX_EXCHANGES = 5

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

  def _route_after_opening(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    *,
    reason: str,
  ) -> str:
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
        reason,
      )
    else:
      state["cbt_resume_stage"] = "id_distortions"
      self._enter_mi(
        state,
        "mi_topic_exploration",
        "id_distortions",
        None,
        reason,
      )
    return self._handle_stage_dispatch(state, traces)

  def _handle_opening(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    opening_exchanges = self._current_stage_exchange_count(state, "opening")
    opening_total_exchanges = self._total_stage_exchange_count(state, "opening")
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
    reached_opening_cap = opening_total_exchanges >= self.OPENING_MAX_EXCHANGES and not ready

    # Opening cap reached. Move on using the same downstream routing as a natural opening completion.
    if reached_opening_cap:
      state["opening_fallback_triggered"] = True
      return self._route_after_opening(
        state,
        traces,
        reason=f"Opening fallback after {self.OPENING_MAX_EXCHANGES} exchanges",
      )

    # Not ready to continue to id_distortions
    if not ready:
      if current_substage == "open_mood":
        instruction = self._load_stage_prompt("opening", "mood_question")
      else:
        instruction = self._load_stage_prompt("opening", "problem_question")

      return self._generate_therapist_utterance(state, instruction)

    # Ready to move on to id_distortions
    if ready:
      return self._route_after_opening(
        state,
        traces,
        reason="Opening completed",
      )

    raise ConversationProcessingError("Opening stage reached an unexpected state.")
