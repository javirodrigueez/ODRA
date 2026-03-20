from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from agents.interfaces import ensure_str, ensure_str_list

from .state import CORE_BELIEF_KEYS, DISTORTION_KEYS, SessionState
from .support import ConversationProcessingError, GraphSupportMixin


class CognitiveStageMixin(GraphSupportMixin):
  def _merge_extracted_items(self, state: SessionState, extracted: dict[str, Any]) -> None:
    working = state["working_cognitive_set"]
    situation = ensure_str(extracted.get("situation"))
    thoughts = ensure_str_list(extracted.get("auto_thoughts"))
    emotions = ensure_str_list(extracted.get("emotions"))
    behaviors = ensure_str_list(extracted.get("behaviors"))

    if not working["situation"] and situation:
      working["situation"] = situation

    for key, values in [("automatic_thoughts", thoughts), ("emotions", emotions), ("behaviors", behaviors)]:
      existing = list(working.get(key, []))
      for item in values:
        if item not in existing:
          existing.append(item)
      working[key] = existing

  def _working_set_base_complete(self, state: SessionState) -> bool:
    working = state["working_cognitive_set"]
    return bool(
      working.get("situation")
      and working.get("automatic_thoughts")
      and working.get("emotions")
      and working.get("behaviors")
    )

  def _working_set_full_complete(self, state: SessionState) -> bool:
    working = state["working_cognitive_set"]
    return self._working_set_base_complete(state) and bool(working.get("core_belief")) and bool(working.get("distortion"))

  def _working_set_has_content(self, state: SessionState) -> bool:
    working = state["working_cognitive_set"]
    return bool(
      working.get("situation")
      or working.get("automatic_thoughts")
      or working.get("emotions")
      or working.get("behaviors")
    )

  def _ordered_cognitive_gap_targets(self, state: SessionState) -> tuple[list[str], list[str]]:
    working = state["working_cognitive_set"]
    targets: list[str] = []
    filled: list[str] = []
    if not working.get("situation"):
      targets.append("situation")
    else:
      filled.append(f"situation: {working.get('situation')}")
    if not working.get("automatic_thoughts"):
      targets.append("auto_thoughts")
    else:
      filled.append(f"automatic_thoughts: {', '.join(working.get('automatic_thoughts', []))}")
    if not working.get("emotions"):
      targets.append("emotions")
    else:
      filled.append(f"emotions: {', '.join(working.get('emotions', []))}")
    if not working.get("behaviors"):
      targets.append("behaviors")
    else:
      filled.append(f"behaviors: {', '.join(working.get('behaviors', []))}")
    return targets, filled

  def _is_cognitive_target_filled(self, state: SessionState, target: str) -> bool:
    working = state["working_cognitive_set"]
    if target == "situation":
      return bool(working.get("situation"))
    if target == "auto_thoughts":
      return bool(working.get("automatic_thoughts"))
    if target == "emotions":
      return bool(working.get("emotions"))
    if target == "behaviors":
      return bool(working.get("behaviors"))
    return False

  def _reset_cognitive_gap_tracking(self, state: SessionState) -> None:
    state["cognitive_gap_plan"] = ""
    state["active_cognitive_target"] = ""
    state["active_cognitive_question"] = ""
    state["active_cognitive_follow_up_attempts"] = 0

  def _apply_cognitive_target_payload(self, state: SessionState, target: str, payload: Any) -> bool:
    source = payload if isinstance(payload, dict) else {}
    extracted: dict[str, Any] = {}
    if target == "situation":
      value = ensure_str(source.get("situation"))
      if value:
        extracted["situation"] = value
    elif target == "auto_thoughts":
      values = ensure_str_list(source.get("auto_thoughts"))
      if values:
        extracted["auto_thoughts"] = values
    elif target == "emotions":
      values = ensure_str_list(source.get("emotions"))
      if values:
        extracted["emotions"] = values
    elif target == "behaviors":
      values = ensure_str_list(source.get("behaviors"))
      if values:
        extracted["behaviors"] = values
    if extracted:
      self._merge_extracted_items(state, extracted)
    return self._is_cognitive_target_filled(state, target)

  def _plan_cognitive_gaps(self, state: SessionState, traces: list[dict[str, Any]]) -> tuple[str, str]:
    # Id. unfilled targets in order
    ordered_targets, filled = self._ordered_cognitive_gap_targets(state)
    if not ordered_targets:
      return "", ""
    is_discover_situation = ordered_targets and ordered_targets[0] == "situation"
    id_situations = ". ".join([set_.get("situation") for set_ in state["cognitive_model_sets"] if set_.get("situation")])

    # Put first active target
    preferred_target = state["active_cognitive_target"]
    if preferred_target in ordered_targets:
      ordered_targets = [preferred_target] + [target for target in ordered_targets if target != preferred_target]

    # Plan instruction, special when situation must be identified
    if is_discover_situation:
      instruction = self._load_stage_prompt(
        "id_distortions",
        "plan_new_situation",
        identified_situations=id_situations,
      )
    else:
      instruction = self._load_stage_prompt(
        "id_distortions",
        "plan_missing_items",
        filled_items=", ".join(filled),
        remaining_targets=ordered_targets,
      )

    # Plan generation
    plan = self._generate_therapist_utterance(state, instruction)
    traces.append({"step": "cognitive_gap_planning", "input": instruction, "output": ensure_str(plan)})
    return plan, ordered_targets[0]

  def _generate_cognitive_follow_up_utterance(self, state: SessionState, plan: str, target: str) -> str:
    instruction = self._load_stage_prompt(
      "id_distortions",
      "gap_utterance_follow_up",
      active_target=target,
      current_plan=plan,
    )
    utterance = self._generate_therapist_utterance(state, instruction)
    return ensure_str(utterance)

  def _generate_cognitive_gap_utterance(self, state: SessionState, plan: str, target: str) -> str:
    if not plan:
      return ""
    instruction = self._load_stage_prompt(
      "id_distortions",
      "gap_utterance",
      current_target=target,
      therapist_plan=plan,
    )
    utterance = self._generate_therapist_utterance(state, instruction)
    return ensure_str(utterance)

  def _generate_cognitive_gap_best_guess(self, state: SessionState, traces: list[dict[str, Any]], target: str) -> Any:
    instruction = self._load_stage_prompt(
      "id_distortions",
      "gap_best_guess",
      active_target=target,
    )
    guessed = self._run_therapist_structured(state, traces, "cognitive_gap_best_guess", instruction)
    payload = self._require_dict(guessed, "cognitive_gap_best_guess")
    return payload.get("identified_items", {})

  def _planning_and_asking(self, state: SessionState, traces: list[dict[str, Any]]) -> Optional[str]:
    active_target = state["active_cognitive_target"]
    active_question = state["active_cognitive_question"]
    target_completed = bool(active_target) and self._is_cognitive_target_filled(state, active_target)


    if target_completed:
      self._reset_cognitive_gap_tracking(state)
      active_target = ""
      active_question = ""

    target_in_progress = bool(active_target) and bool(active_question)
    if target_in_progress:
      follow_ups = state["active_cognitive_follow_up_attempts"]
      instruction = self._load_stage_prompt(
        "id_distortions",
        "gap_response_review",
        active_target=active_target,
        current_question=active_question,
        follow_up_attempts=follow_ups,
      )
      reviewed = self._run_therapist_structured(state, traces, "cognitive_gap_response_review", instruction)
      payload = self._require_dict(reviewed, "cognitive_gap_response_review")
      correct_answer = self._require_bool(payload, "answered_satisfactorily", "cognitive_gap_response_review")

      if correct_answer:
        identified = payload.get("identified_items", {})
        self._merge_extracted_items(state, identified)
        if self._is_cognitive_target_filled(state, active_target):
          self._reset_cognitive_gap_tracking(state)
          active_target = ""
          active_question = ""
      elif follow_ups > 5:
        best_guess = self._generate_cognitive_gap_best_guess(state, traces, active_target)
        self._apply_cognitive_target_payload(state, active_target, best_guess)
        self._reset_cognitive_gap_tracking(state)
        active_target = ""
        active_question = ""
      else:
        follow_up_question = self._generate_cognitive_gap_utterance(
          state,
          ensure_str(state.get("last_reasoning")),
          active_target,
        )
        state["active_cognitive_question"] = follow_up_question
        active_question = follow_up_question

    if self._working_set_base_complete(state):
      self._reset_cognitive_gap_tracking(state)
      return None

    next_question = state["active_cognitive_question"]; next_target = state["active_cognitive_target"]
    target_or_set_completed = not next_target and not next_question

    if target_or_set_completed:
      plan, next_target = self._plan_cognitive_gaps(state, traces)
      state["cognitive_gap_plan"] = plan
      next_question = self._generate_cognitive_gap_utterance(state, plan, next_target)
      if next_target != state["active_cognitive_target"]:
        state["active_cognitive_target"] = next_target
        state["active_cognitive_follow_up_attempts"] = 0
      state["active_cognitive_question"] = next_question

    state["active_cognitive_follow_up_attempts"] = int(state["active_cognitive_follow_up_attempts"]) + 1
    return next_question

  def _bootstrap_first_cognitive_model(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    instruction = self._load_stage_prompt("id_distortions", "bootstrap_first_cognitive_model")
    parsed = self._run_therapist_structured(state, traces, "initial_cognitive_model_bootstrap", instruction)
    payload = self._require_dict(parsed, "initial_cognitive_model_bootstrap")
    filled_items = payload.get("items", {})
    if not isinstance(filled_items, dict):
      raise ConversationProcessingError("initial_cognitive_model_bootstrap.items must be a JSON object.")
    self._merge_extracted_items(state, filled_items)
    state["last_reasoning"] = ensure_str(payload.get("reasoning"))

  def _assign_core_belief(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    belief_instruction = self._load_stage_prompt("id_distortions", "core_belief_probability")
    belief_probs = self._run_therapist_structured(state, traces, "core_belief_probability", belief_instruction)
    normalized = self._normalize_probabilities(
      self._require_dict(belief_probs, "core_belief_probability"),
      CORE_BELIEF_KEYS,
    )
    state["working_cognitive_set"]["core_belief_probabilities"] = normalized
    state["working_cognitive_set"]["core_belief"] = self._best_key(normalized)

  def _assign_distortion(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    working = state["working_cognitive_set"]
    format_working_cognitive_set = (
      f"situation: {working.get('situation', '')}, "
      f"automatic_thoughts: {', '.join(working.get('automatic_thoughts', []))}, "
      f"emotions: {', '.join(working.get('emotions', []))}, "
      f"behaviors: {', '.join(working.get('behaviors', []))}, "
      f"core_belief: {working.get('core_belief', '')}"
    )
    distortions_dict = self._load_distortions()
    format_distortions = ", ".join([f"{key} ({desc})" for key, desc in distortions_dict.items()])
    distortion_instruction = self._load_stage_prompt(
      "id_distortions",
      "distortion_probability",
      working_cognitive_set=format_working_cognitive_set,
      distortions=format_distortions,
      distortion_keys=", ".join(DISTORTION_KEYS),
    )
    distortion_probs = self._run_therapist_structured(state, traces, "distortion_probability", distortion_instruction)
    normalized = self._normalize_probabilities(
      self._require_dict(distortion_probs, "distortion_probability"),
      DISTORTION_KEYS,
    )
    state["working_cognitive_set"]["distortion_probabilities"] = normalized
    state["working_cognitive_set"]["distortion"] = self._best_key(normalized)

  def _check_distortion_completion(self, state: SessionState, traces: list[dict[str, Any]], id_exchanges: int) -> bool:
    completed_count = len(state["cognitive_model_sets"])
    max_sets = completed_count >= 5
    enough_sets = completed_count >= 3 or (id_exchanges >= 30 and completed_count >= 2)

    # Conditions
    if max_sets:
      return True
    if not enough_sets:
      return False
    
    # Check if more sets are needed
    need_more_instruction = self._load_stage_prompt("id_distortions", "completion_check")
    need_more = self._run_therapist_structured(
      state,
      traces,
      "id_distortions_completion_check",
      need_more_instruction,
    )
    needs_more = self._require_bool(
      self._require_dict(need_more, "id_distortions_completion_check"),
      "needs_more",
      "id_distortions_completion_check",
    )
    return not needs_more

  def _handle_id_distortions(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    id_exchanges = int(state["exchange_counts"].get("id_distortions", 0))
    first_utterance = id_exchanges == 0 and not state["cognitive_model_sets"] and not self._working_set_has_content(state)

    # Bootstrap only the very first cognitive set from opening material.
    if first_utterance:
      self._set_substage(state, "cognitive_bootstrap")
    current_substage = state["current_substage"]

    # Bootstrap the first cognitive model from the opening material.
    if current_substage == "cognitive_bootstrap":
      self._bootstrap_first_cognitive_model(state, traces)
      self._set_substage(state, "cognitive_review_active_question")
      current_substage = state["current_substage"]

    # Review the active gap question and keep collecting the base cognitive set.
    if current_substage == "cognitive_review_active_question":
      next_gap_question = self._planning_and_asking(
        state,
        traces,
      )
      if next_gap_question:
        return next_gap_question
      self._set_substage(state, "cognitive_assign_core_belief")
      current_substage = state["current_substage"]

    # Assign the core belief once the base cognitive set is complete.
    if current_substage == "cognitive_assign_core_belief":
      if not self._working_set_base_complete(state):
        self._set_substage(state, "cognitive_review_active_question")
        return self._handle_id_distortions(state, traces)
      if not state["working_cognitive_set"].get("core_belief"):
        self._assign_core_belief(state, traces)
      self._set_substage(state, "cognitive_assign_distortion")
      current_substage = state["current_substage"]

    # Assign the distortion after the core belief is available.
    if current_substage == "cognitive_assign_distortion":
      if not state["working_cognitive_set"].get("core_belief"):
        self._set_substage(state, "cognitive_assign_core_belief")
        return self._handle_id_distortions(state, traces)
      if not state["working_cognitive_set"].get("distortion"):
        self._assign_distortion(state, traces)
      self._set_substage(state, "cognitive_commit_set")
      current_substage = state["current_substage"]

    # Commit the completed cognitive set and reset the working area.
    if current_substage == "cognitive_commit_set":
      if self._working_set_full_complete(state):
        completed_set = deepcopy(state["working_cognitive_set"])
        state["cognitive_model_sets"].append(completed_set)
        state["working_cognitive_set"] = self._new_cognitive_set(len(state["cognitive_model_sets"]) + 1)
        self._reset_cognitive_gap_tracking(state)
      self._set_substage(state, "cognitive_check_completion")
      current_substage = state["current_substage"]

    # Decide whether distortion identification is complete for this session.
    if current_substage == "cognitive_check_completion":
      if self._check_distortion_completion(state, traces, id_exchanges):
        self._set_stage(state, "cbt_technique")
        return self._handle_stage_dispatch(state, traces)
      self._set_substage(state, "cognitive_review_active_question")
      return self._handle_id_distortions(state, traces)
    
    # Fallback
    # fallback_plan_question = self._planning_and_asking(state, traces)
    # if fallback_plan_question:
    #   return fallback_plan_question

    raise ConversationProcessingError(f"Unexpected cognitive substage: {current_substage}")
