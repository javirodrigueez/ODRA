from __future__ import annotations

from typing import Any, Optional

from agents.interfaces import ensure_str

from .state import SelectedTechnique, SessionState
from .support import ConversationProcessingError, GraphSupportMixin


class TechniqueStageMixin(GraphSupportMixin):
  def _choose_additional_technique(self, state: SessionState, traces: list[dict[str, Any]]) -> SelectedTechnique:
    techniques_catalog = self.cbt_techniques
    formatted_catalog = "\n".join(
      f'- {name}: {description}'
      for name, description in techniques_catalog.items()
    )

    instruction = self._load_stage_prompt(
      "cbt_technique",
      "additional_technique_selection",
      formatted_catalog=formatted_catalog,
    )
    parsed = self._run_therapist_structured(state, traces, "additional_technique_selection", instruction)
    payload = self._require_dict(parsed, "additional_technique_selection")
    return {
      "name": self._require_non_empty_string(payload, "name", "additional_technique_selection"),
      "target_distortion": ensure_str(payload.get("target_distortion")),
      "rationale": ensure_str(payload.get("rationale")),
    }

  def _select_initial_techniques(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    techniques_catalog = self.cbt_techniques
    formatted_catalog = "\n".join(
      f'- {name}: {description}'
      for name, description in techniques_catalog.items()
    )

    # Generate the initial set of techniques from the approved catalog only.
    instruction = self._load_stage_prompt(
      "cbt_technique",
      "technique_selection",
      formatted_catalog=formatted_catalog,
    )
    parsed = self._run_therapist_structured(state, traces, "technique_selection", instruction)
    payload = self._require_dict(parsed, "technique_selection")
    chosen = payload.get("selected_techniques")
    chosen = self._require_list(chosen, "technique_selection.selected_techniques")
    selected: list[SelectedTechnique] = []
    for item in chosen:
      parsed_item = self._require_dict(item, "technique_selection.selected_techniques")
      selected.append(
        {
          "name": self._require_non_empty_string(parsed_item, "name", "technique_selection.selected_techniques"),
          "target_distortion": ensure_str(parsed_item.get("target_distortion")),
          "rationale": ensure_str(parsed_item.get("rationale")),
        }
      )
    state["selected_techniques"] = selected[:3]

    # DON'T UNCOMMENT: this is for strict catalog enforcement for initial technique selection.
    # selected: list[dict] = []
    # for item in chosen:
    #   parsed_item = self._require_dict(item, "technique_selection.selected_techniques")
    #   name = self._require_non_empty_string(parsed_item, "name", "technique_selection.selected_techniques")
    #   if name not in techniques_catalog:
    #     raise ConversationProcessingError(
    #       f"technique_selection.selected_techniques.name must be one of the catalog techniques, got {name!r}."
    #     )
    #   selected.append(
    #     {
    #       "name": name,
    #       "target_distortion": ensure_str(parsed_item.get("target_distortion")),
    #       "rationale": ensure_str(parsed_item.get("rationale")),
    #     }
    #   )
    # if not selected:
    #   raise ConversationProcessingError("technique_selection.selected_techniques must contain at least one technique.")
    # state["selected_techniques"] = selected[:3]

  def _select_additional_technique(self, state: SessionState, traces: list[dict[str, Any]]) -> SelectedTechnique:
    technique = self._choose_additional_technique(state, traces)
    state["selected_techniques"].append(technique)
    return technique

  def _replace_current_technique_due_to_resistance(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    replacement = self._choose_additional_technique(state, traces)
    idx = int(state["current_technique_index"])
    if idx < len(state["selected_techniques"]):
      state["selected_techniques"][idx] = replacement
    else:
      state["selected_techniques"].append(replacement)
      state["current_technique_index"] = len(state["selected_techniques"]) - 1
    state["current_technique_plan"] = ""
    state["current_technique_expected_outcome"] = ""
    state["current_technique_current_progress"] = ""
    state["technique_turn_count"] = 0

  def _current_technique(self, state: SessionState) -> Optional[SelectedTechnique]:
    idx = state["current_technique_index"]
    if idx < 0 or idx >= len(state["selected_techniques"]):
      return None
    return state["selected_techniques"][idx]

  def _evaluate_technique_completion(self, state: SessionState, traces: list[dict[str, Any]]) -> bool:
    if not state["current_technique_plan"]:
      return False
    current = self._current_technique(state)
    if current is None:
      raise ConversationProcessingError("Current CBT technique could not be resolved.")
    max_technique_turns = state["technique_turn_count"] >= 15
    
    if max_technique_turns:
      traces.append(
        {
          "step": "technique_auto_complete",
          "input": "Technique reached 15 exchange turns without completion.",
          "output": "Automatically marking technique as complete.",
        }
      )
      return True
    
    instruction = self._load_stage_prompt( #TODO: Check if now evaluating current technique dont mix info with previous technique
      "cbt_technique",
      "completion_check",
    )
    parsed = self._run_therapist_structured(state, traces, "technique_completion_check", instruction)
    parsed_payload = self._require_dict(parsed, "technique_completion_check")
    completed = self._require_bool(
      parsed_payload,
      "completed",
      "technique_completion_check",
    )
    state["last_reasoning"] = self._require_non_empty_string(
      parsed_payload,
      "reasoning",
      "technique_completion_check",
    )

    #completed_with_min_turns = completed and state["technique_turn_count"] >= 4

    return completed

  def _evaluate_technique(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    instruction = self._load_stage_prompt("cbt_technique", "technique_evaluation")
    parsed = self._run_therapist_structured(
      state,
      traces,
      "technique_evaluation",
      instruction,
    )
    payload = self._require_dict(parsed, "technique_evaluation")
    evaluation = self._require_non_empty_string(
      payload,
      "evaluation",
      "technique_evaluation",
    )
    state["technique_evaluations"].append(evaluation)

  def _should_apply_more_techniques(self, state: SessionState, traces: list[dict[str, Any]]) -> bool:
    instruction = self._load_stage_prompt("cbt_technique", "apply_more_check")
    parsed = self._run_therapist_structured(state, traces, "apply_more_techniques_check", instruction)
    return self._require_bool(
      self._require_dict(parsed, "apply_more_techniques_check"),
      "apply_more",
      "apply_more_techniques_check",
    )

  def _complete_current_technique(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    current = self._current_technique(state)
    if current is None:
      raise ConversationProcessingError("Current CBT technique could not be resolved.")
    self._evaluate_technique(state, traces)
    state["techniques_completed"] += 1
    state["current_technique_index"] += 1
    state["current_technique_plan"] = ""
    state["current_technique_expected_outcome"] = ""
    state["current_technique_current_progress"] = ""
    state["technique_turn_count"] = 0

  def _handle_cbt_technique(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    if not state["selected_techniques"] and self._long_session_threshold_reached(state):
      self._set_stage(state, "homework")
      return self._handle_stage_dispatch(state, traces)

    if not state["selected_techniques"]:
      self._set_substage(state, "cbt_select_techniques")
    elif state["current_technique_plan"] and state["technique_turn_count"] > 0:
      self._set_substage(state, "cbt_review_current")
    elif not state["current_technique_plan"]:
      self._set_substage(state, "cbt_decide_next")

    current_substage = state["current_substage"]

    # Select the initial technique set for the session.
    if current_substage == "cbt_select_techniques":
      self._select_initial_techniques(state, traces)
      self._set_substage(state, "cbt_decide_next")
      current_substage = state["current_substage"]

    # Review whether the in-progress technique has been completed.
    if current_substage == "cbt_review_current":
      if self._evaluate_technique_completion(state, traces):
        self._complete_current_technique(state, traces)
        self._set_substage(state, "cbt_decide_next")
        current_substage = state["current_substage"]
      else:
        self._set_substage(state, "cbt_plan_current")
        current_substage = state["current_substage"]

    # Decide whether to continue with techniques or move to homework.
    if current_substage == "cbt_decide_next":
      if self._long_session_threshold_reached(state) and not state["current_technique_plan"]:
        self._set_stage(state, "homework")
        return self._handle_stage_dispatch(state, traces)

      max_techniques = state["techniques_completed"] >= 5
      enough_techniques = state["techniques_completed"] >= 3 and not state["current_technique_plan"]
      are_techniques_completed = state["current_technique_index"] >= len(state["selected_techniques"])

      # End conditions
      if max_techniques:
        self._set_stage(state, "homework")
        return self._handle_stage_dispatch(state, traces)
      if enough_techniques:
        apply_more = self._should_apply_more_techniques(state, traces)
        if not apply_more:
          self._set_stage(state, "homework")
          return self._handle_stage_dispatch(state, traces)
      
      # All techniques completed case
      if are_techniques_completed:
        self._select_additional_technique(state, traces)
        self._set_substage(state, "cbt_plan_current")
        current_substage = state["current_substage"]  
      else:
        self._set_substage(state, "cbt_validate_fit")
        current_substage = state["current_substage"]

    current = self._current_technique(state)
    if current is None:
      raise ConversationProcessingError("Current CBT technique could not be resolved.")

    # Validate that the next queued technique still fits the session.
    if current_substage == "cbt_validate_fit":
      slot_idx = state["current_technique_index"]
      is_first_technique = state["techniques_completed"] == 0
      is_technique_check = not state["current_technique_plan"] and not is_first_technique

      if is_technique_check:
        fit_instruction = self._load_stage_prompt(
          "cbt_technique",
          "fit_check",
        )
        fit_parsed = self._require_dict(
          self._run_therapist_structured(state, traces, "technique_fit_check", fit_instruction),
          "technique_fit_check",
        )
        fits = self._require_bool(fit_parsed, "fits", "technique_fit_check")
        alt_name = ensure_str(fit_parsed.get("alternative_technique"))
        #already_replaced = bool(state["technique_slot_replaced"].get(str(slot_idx), False))
        if not fits:
          replacement: SelectedTechnique = {
            "name": alt_name,
            "target_distortion": current.get("target_distortion"),
            "rationale": ensure_str(fit_parsed.get("reasoning")),
          }
          current = replacement
          state["selected_techniques"][slot_idx] = replacement
          #state["technique_slot_replaced"][str(slot_idx)] = True
          
      self._set_substage(state, "cbt_plan_current")
      current_substage = state["current_substage"]

    # Build the detailed plan for the current technique.
    if current_substage == "cbt_plan_current":
      current = self._current_technique(state)
      if current is None:
        raise ConversationProcessingError("Current CBT technique could not be resolved.")
      if not state["current_technique_plan"]:
        plan_instruction = self._load_stage_prompt(
          "cbt_technique",
          "plan_initial",
        )
        plan_parsed = self._require_dict(
          self._run_therapist_structured(state, traces, "technique_plan", plan_instruction),
          "technique_plan",
        )
        state["current_technique_plan"] = self._require_non_empty_string(
          plan_parsed,
          "plan_description",
          "technique_plan",
        )
        state["current_technique_expected_outcome"] = self._require_non_empty_string(
          plan_parsed,
          "expected_outcome",
          "technique_plan",
        )
        state["current_technique_current_progress"] = self._require_non_empty_string(
          plan_parsed,
          "current_progress",
          "technique_plan",
        )
      else:
        plan_instruction = self._load_stage_prompt(
          "cbt_technique",
          "plan_update",
        )
        plan_parsed = self._require_dict(
          self._run_therapist_structured(state, traces, "technique_plan", plan_instruction),
          "technique_plan",
        )
        state["current_technique_plan"] = self._require_non_empty_string(
          plan_parsed,
          "plan_description",
          "technique_plan",
        )
        state["current_technique_current_progress"] = self._require_non_empty_string(
          plan_parsed,
          "current_progress",
          "technique_plan",
        )
      self._set_substage(state, "cbt_deliver_current")
      current_substage = state["current_substage"]

    # Deliver the technique to the patient.
    if current_substage == "cbt_deliver_current":
      current = self._current_technique(state)
      if current is None:
        raise ConversationProcessingError("Current CBT technique could not be resolved.")
      instruction = self._load_stage_prompt(
        "cbt_technique",
        "deliver_current",
      )
      utterance = self._generate_therapist_utterance(state, instruction)
      state["current_substage"] = "cbt_review_current"
      current_substage = state["current_substage"]
      return utterance

    raise ConversationProcessingError(f"Unexpected CBT technique substage: {current_substage}")
