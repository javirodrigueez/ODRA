from __future__ import annotations

from typing import Any

from agents.interfaces import ensure_str

from .state import HomeworkTask, SessionState
from .support import ConversationProcessingError, GraphSupportMixin


class HomeworkStageMixin(GraphSupportMixin):
  def _propose_homework_tasks(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    instruction = self._load_stage_prompt("homework", "task_proposal")
    parsed = self._run_therapist_structured(state, traces, "homework_task_proposal", instruction)
    payload = self._require_dict(parsed, "homework_task_proposal")
    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list):
      raise ConversationProcessingError("homework_task_proposal.tasks must be a list.")
    tasks: list[HomeworkTask] = []
    for item in tasks_raw:
      item_payload = self._require_dict(item, "homework_task_proposal.tasks")
      tasks.append(
        {
          "task_title": self._require_non_empty_string(item_payload, "task_title", "homework_task_proposal.tasks"),
          "task_description": self._require_non_empty_string(item_payload, "task_description", "homework_task_proposal.tasks"),
          "rationale": ensure_str(item_payload.get("rationale")),
          "expected_outcomes": ensure_str(item_payload.get("expected_outcomes")),
        }
      )
    if not tasks:
      raise ConversationProcessingError("homework_task_proposal.tasks must contain at least one task.")
    state["proposed_tasks"] = tasks[:2]
    state["homework_proposal_attempts"] += 1
    state["homework_waiting_for_patient"] = True

  def _check_homework_agreement(self, state: SessionState, traces: list[dict[str, Any]]) -> tuple[bool, list[HomeworkTask]]:
    instruction = self._load_stage_prompt("homework", "agreement_check")
    parsed = self._run_therapist_structured(state, traces, "homework_agreement_check", instruction)
    payload = self._require_dict(parsed, "homework_agreement_check")

    agreed = self._require_bool(payload, "agreement_reached", "homework_agreement_check")
    agreed_tasks_raw = payload.get("agreed_tasks")
    state["last_reasoning"] = ensure_str(payload.get("reasoning", ""), "homework_agreement_check")
    agreed_tasks: list[HomeworkTask] = []

    if agreed and isinstance(agreed_tasks_raw, list):
      for item in agreed_tasks_raw:
        item_payload = self._require_dict(item, "homework_agreement_check.agreed_tasks")
        agreed_tasks.append(
          {
            "task_title": self._require_non_empty_string(item_payload, "task_title", "homework_agreement_check.agreed_tasks"),
            "task_description": self._require_non_empty_string(item_payload, "task_description", "homework_agreement_check.agreed_tasks"),
            "rationale": ensure_str(item_payload.get("rationale")),
            "expected_outcomes": ensure_str(item_payload.get("expected_outcomes")),
          }
        )
      state["agreed_tasks"] = agreed_tasks
    return agreed, agreed_tasks

  def _minimal_homework_task(self) -> list[HomeworkTask]:
    return [
      {
        "task_title": "Brief Observation Note",
        "task_description": "Once per day, write one short note about a problematic moment and one thought you noticed.",
        "rationale": "Minimal-effort task when resistance to homework remains high.",
        "expected_outcomes": "Maintain continuity without overload.",
      }
    ]

  def _handle_homework(self, state: SessionState, traces: list[dict[str, Any]]) -> str:

    # Determine branch
    if state["homework_waiting_for_patient"]:
      self._set_substage(state, "homework_review_response")
    elif state["homework_fallback_to_apply"]:
      self._set_substage(state, "homework_apply_fallback")
    elif state["homework_fallback_done"]:
      self._set_substage(state, "homework_end_fallback")
    else:
      self._set_substage(state, "homework_propose_tasks")
    current_substage = state["current_substage"]

    # Review the patient's response to the proposed homework.
    if current_substage == "homework_review_response":
      agreed, _ = self._check_homework_agreement(state, traces)
      state["homework_waiting_for_patient"] = False
      if agreed:
        self._set_stage(state, "closing")
        return self._handle_stage_dispatch(state, traces)
      if state["homework_proposal_attempts"] >= 5:
        state["homework_fallback_to_apply"] = True
        self._set_substage(state, "homework_apply_fallback")
      else:
        self._set_substage(state, "homework_propose_tasks")
      current_substage = state["current_substage"]

    # Apply the lightweight homework fallback after repeated failed proposals.
    if current_substage == "homework_apply_fallback":
      state["proposed_tasks"] = self._minimal_homework_task()
      state["agreed_tasks"] = state["proposed_tasks"]
      state["homework_fallback_to_apply"] = False
      state["homework_fallback_done"] = True
      traces.append(
        {
          "step": "homework_fallback",
          "input": "5 homework proposal rounds exhausted without agreement.",
          "output": "Using simplified minimal-task suggestion and proceeding.",
        }
      )

      instruction = self._load_stage_prompt(
        "homework",
        "fallback_task_proposal_utterance",
      )

      utterance = self._generate_therapist_utterance(state, instruction)

      return utterance

    # Propose a fresh homework set.
    if current_substage == "homework_propose_tasks":
      self._propose_homework_tasks(state, traces)
      instruction = self._load_stage_prompt("homework", "propose_tasks_utterance")
      utterance = self._generate_therapist_utterance(state, instruction)

      state["homework_waiting_for_patient"] = True

      return utterance

    if current_substage == "homework_end_fallback":
      instruction = self._load_stage_prompt("homework", "end_fallback_utterance")
      utterance = self._generate_therapist_utterance(state, instruction)
      state["current_substage"] = "homework_transition_to_closing"

      return utterance

    if current_substage == "homework_transition_to_closing":
      self._set_stage(state, "closing")
      return self._handle_stage_dispatch(state, traces)

    raise ConversationProcessingError(f"Unexpected homework substage: {current_substage}")
