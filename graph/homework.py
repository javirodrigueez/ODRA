from __future__ import annotations

from .state import SessionState
from .support import ConversationProcessingError


class HomeworkStageMixin:
  def _propose_homework_tasks(self, state: SessionState, traces: list[dict]) -> None:
    instruction = (
      "Propose 1-2 homework tasks based on cognitive sets, distortions, technique evaluations and context. "
      'Respond JSON: {"tasks": [{"task_title":"...", "task_description":"...", "rationale":"...", "expected_outcomes":"..."}], "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "homework_task_proposal", instruction)
    payload = self._require_dict(parsed, "homework_task_proposal")
    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list):
      raise ConversationProcessingError("homework_task_proposal.tasks must be a list.")
    tasks = []
    for item in tasks_raw:
      if not isinstance(item, dict):
        raise ConversationProcessingError("homework_task_proposal.tasks items must be JSON objects.")
      tasks.append(
        {
          "task_title": self._require_non_empty_string(item, "task_title", "homework_task_proposal.tasks"),
          "task_description": self._require_non_empty_string(item, "task_description", "homework_task_proposal.tasks"),
          "rationale": item.get("rationale", ""),
          "expected_outcomes": item.get("expected_outcomes", ""),
        }
      )
    if not tasks:
      raise ConversationProcessingError("homework_task_proposal.tasks must contain at least one task.")
    state["proposed_tasks"] = tasks[:2]
    state["homework_proposal_attempts"] += 1
    state["homework_waiting_for_patient"] = True

  def _check_homework_agreement(self, state: SessionState, traces: list[dict]) -> bool:
    instruction = (
      "Has agreement been reached for at least one homework task? "
      'Respond JSON: {"agreement_reached": true/false, "agreed_tasks": [...], "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "homework_agreement_check", instruction)
    payload = self._require_dict(parsed, "homework_agreement_check")
    agreed = self._require_bool(payload, "agreement_reached", "homework_agreement_check")
    agreed_tasks_raw = payload.get("agreed_tasks")
    if agreed and isinstance(agreed_tasks_raw, list):
      normalized = []
      for item in agreed_tasks_raw:
        if isinstance(item, dict) and item.get("task_title"):
          normalized.append(item)
      if normalized:
        state["agreed_tasks"] = normalized
      else:
        raise ConversationProcessingError("homework_agreement_check.agreed_tasks must contain at least one agreed task.")
    return agreed

  def _minimal_homework_task(self) -> list[dict]:
    return [
      {
        "task_title": "Brief Observation Note",
        "task_description": "Once per day, write one short note about a stressful moment and one thought you noticed.",
        "rationale": "Minimal-effort task when resistance to homework remains high.",
        "expected_outcomes": "Maintain continuity without overload.",
      }
    ]

  def _handle_homework(self, state: SessionState, traces: list[dict]) -> str:
    if state["agreed_tasks"] and not state["homework_waiting_for_patient"]:
      self._set_stage(state, "closing")
      return self._handle_stage_dispatch(state, traces)

    if state["homework_waiting_for_patient"]:
      self._set_substage(state, "homework_review_response")
    elif state["homework_fallback_done"]:
      self._set_substage(state, "homework_apply_fallback")
    else:
      self._set_substage(state, "homework_propose_tasks")

    current_substage = state["current_substage"]

    # Review the patient's response to the proposed homework.
    if current_substage == "homework_review_response":
      agreed = self._check_homework_agreement(state, traces)
      state["homework_waiting_for_patient"] = False
      if agreed:
        self._set_stage(state, "closing")
        return self._handle_stage_dispatch(state, traces)
      if state["homework_proposal_attempts"] >= 3:
        self._set_substage(state, "homework_apply_fallback")
        current_substage = state["current_substage"]
      else:
        self._set_substage(state, "homework_propose_tasks")
        current_substage = state["current_substage"]

    # Apply the lightweight homework fallback after repeated failed proposals.
    if current_substage == "homework_apply_fallback":
      state["proposed_tasks"] = self._minimal_homework_task()
      state["agreed_tasks"] = state["proposed_tasks"]
      state["homework_fallback_done"] = True
      traces.append(
        {
          "step": "homework_fallback",
          "input": "3 homework proposal rounds exhausted without agreement.",
          "output": "Using simplified minimal-task suggestion and proceeding.",
        }
      )
      return "We can keep it very light: would you try one brief daily observation note this week, as a gentle experiment?"

    # Propose a fresh homework set.
    if current_substage == "homework_propose_tasks":
      self._propose_homework_tasks(state, traces)
      task_lines = [f'- {task["task_title"]}: {task["task_description"]}' for task in state["proposed_tasks"]]
      task_text = "\n".join(task_lines)
      return (
        "Based on today’s work, I’d like to suggest 1-2 homework options:\n"
        f"{task_text}\n"
        "Which of these feels realistic for you to do before our next session?"
      )

    raise ConversationProcessingError(f"Unexpected homework substage: {current_substage}")
