from __future__ import annotations

from agents.interfaces import ensure_str

from .state import SessionState
from .support import ConversationProcessingError


class HomeworkStageMixin:
  def _propose_homework_tasks(self, state: SessionState, traces: list[dict]) -> None:
    # Format cognitive sets
    format_csets_str = "\n".join([f'Situation: {cset["situation"]}\n  Automatic Thought: {cset["automatic_thoughts"]}\n  Emotion: {cset["emotions"]}\n  Behavior: {cset["behaviors"]}\n  Core Belief: {cset["core_belief"]}\n  Distortions: {", ".join(cset["distortion"])}' for cset in state["cognitive_model_sets"]])
    # Format technique evaluations
    format_evals_str = "\n".join([f'Technique: {e["technique"]}\n  Evaluation: {e["evaluation"]}' for e in state["technique_evaluations"]])
    instruction = (
      "Propose 1-2 homework tasks based on cognitive sets, distortions, technique evaluations, reasoning about last patient response (if any), and conversation. "
      f"Cognitive sets with distortions: {format_csets_str}. "
      f"Technique evaluations: {format_evals_str}. "
      f"Last reasoning: {state['hw_reasoning']}. "
      'Respond JSON: {"tasks": [{"task_title":"...", "task_description":"...", "rationale":"...", "expected_outcomes":"..."}], "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "homework_task_proposal", instruction)
    payload = self._require_dict(parsed, "homework_task_proposal")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
      raise ConversationProcessingError("homework_task_proposal.tasks must be a list.")
    # tasks = []
    # for item in tasks_raw:
    #   if not isinstance(item, dict):
    #     raise ConversationProcessingError("homework_task_proposal.tasks items must be JSON objects.")
    #   tasks.append(
    #     {
    #       "task_title": self._require_non_empty_string(item, "task_title", "homework_task_proposal.tasks"),
    #       "task_description": self._require_non_empty_string(item, "task_description", "homework_task_proposal.tasks"),
    #       "rationale": item.get("rationale", ""),
    #       "expected_outcomes": item.get("expected_outcomes", ""),
    #     }
    #   )
    # if not tasks:
    #   raise ConversationProcessingError("homework_task_proposal.tasks must contain at least one task.")
    state["proposed_tasks"] = tasks[:2]
    state["homework_proposal_attempts"] += 1
    state["homework_waiting_for_patient"] = True

  def _check_homework_agreement(self, state: SessionState, traces: list[dict]):
    instruction = (
      "Has agreement been reached for at least one homework task? If the patient shows hesitation but does not flat out reject, consider that agreement. ",
      'Respond JSON: {"agreement_reached": true/false, "agreed_tasks": ["<task_title_1>", "<task_title_n>", ...], "reasoning":"..."}'
    )
    parsed = self._run_therapist_structured(state, traces, "homework_agreement_check", instruction)
    payload = self._require_dict(parsed, "homework_agreement_check")

    agreed = self._require_bool(payload, "agreement_reached", "homework_agreement_check")
    agreed_tasks_raw = payload.get("agreed_tasks")
    state["last_reasoning"] = ensure_str(payload.get("reasoning", ""), "homework_agreement_check")

    if agreed and isinstance(agreed_tasks_raw, list):
      state["agreed_tasks"] = agreed_tasks_raw
    return agreed, agreed_tasks_raw

  def _minimal_homework_task(self) -> list[dict]:
    return [
      {
        "task_title": "Brief Observation Note",
        "task_description": "Once per day, write one short note about a problematic moment and one thought you noticed.",
        "rationale": "Minimal-effort task when resistance to homework remains high.",
        "expected_outcomes": "Maintain continuity without overload.",
      }
    ]

  def _handle_homework(self, state: SessionState, traces: list[dict]) -> str:

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

    # Apply the lightweight homework fallback after repeated failed proposals.
    if current_substage == "homework_apply_fallback":
      state["proposed_tasks"] = self._minimal_homework_task()
      state["agreed_tasks"] = state["proposed_tasks"]
      state["homework_fallback_done"] = True
      traces.append(
        {
          "step": "homework_fallback",
          "input": "5 homework proposal rounds exhausted without agreement.",
          "output": "Using simplified minimal-task suggestion and proceeding.",
        }
      )

      format_task_str = f'Title: {state["proposed_tasks"][0]["task_title"]}\nDescription: {state["proposed_tasks"][0]["task_description"]}'
      instruction = (
        "Generate a therapist utterance proposing the simplified homework task to the patient, based on the task and conversation context."
        f"Proposed task: {format_task_str}"
      )

      utterance =  self.therapist.generate_utterance(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )

      return utterance

    # Propose a fresh homework set.
    if current_substage == "homework_propose_tasks":
      self._propose_homework_tasks(state, traces)
      format_tasks_str = "\n".join([f'Title: {task["task_title"]}\nDescription: {task["task_description"]}' for task in state["proposed_tasks"]])
      instruction = (
        "Generate a therapist utterance proposing the homework tasks to the patient, based on the proposed tasks, reasoning about last patient response, and conversation context. ",
        f"The proposed tasks are: {format_tasks_str}. ",
        f"The last reasoning was: {state['hw_reasoning']}. ",
        "Previously in the convesation the therapist could have already proposed homework tasks, and the patient could have accepted or rejected them. Use this history to inform your proposal. If patient has already agreed, mention briefly the task to avoid repetition. If not, propose the task putting focus on emphasizing its relevance and benefits."
      )
      utterance = self.therapist.generate_utterance(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )

      state["homework_waiting_for_patient"] = True

      return utterance

    if current_substage == "homework_end_fallback":
      instruction = (
        "Independently of homework agreement, generate a therapist utterance that acknowledges the patient's stance and smoothly transitions toward closing, based on conversation context.",
        "If homework agreement was not reached, remember the importance of engaging with the treatment to make real progress."
      )
      utterance = self.therapist.generate_utterance(
        conversation=self._format_conversation(state["messages"]),
        current_stage=state["current_stage"],
        session_context=self._build_session_context(state),
        instruction=instruction,
      )

      return utterance

    raise ConversationProcessingError(f"Unexpected homework substage: {current_substage}")
