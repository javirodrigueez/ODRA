from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, cast


from agents.interfaces import (
  LLMFactory,
  StructuredOutputError,
  ensure_str,
  ensure_str_list,
  extract_generation_config,
  load_prompt,
  render_prompt,
  ChatMessage
)
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent

from .audit_tokens import count_text_tokens, record_large_prompt_request
from .resistance import (
  DEFAULT_PROMPT_VARIANT,
  HomeostaticResistanceUpdate,
  PatientActionDistribution,
  TherapistResistanceUpdate,
  apply_homeostatic_resistance_update,
  apply_bayesian_predicted_resistance_update,
  apply_predicted_resistance_delta_clip,
  normalize_predicted_resistance_state_fields,
  resolve_prompt_ablation_settings,
  normalize_resistance_state_fields,
)
from .state import (
  ALL_SUBSTAGES,
  CORE_BELIEF_KEYS,
  DEFAULT_SUBSTAGE_BY_STAGE,
  DISTORTION_KEYS,
  HomeworkTask,
  MI_STAGES,
  MIN_COGNITIVE_SET_EXCHANGES,
  CognitiveModelSet,
  ConversationMessage,
  ConversationRole,
  SelectedTechnique,
  SessionState,
  Stage,
  SubStage,
  TIBRouteBucket,
  normalize_stage_name,
  normalize_substage_name,
)


class ConversationProcessingError(RuntimeError):
  pass


FULL_CONVERSATION_TOKEN_LIMIT = 500
FULL_CONVERSATION_SUMMARY_WORD_LIMIT = 300


class GraphSupportMixin:
  cfg: Any = None
  profile: dict[str, Any] = {}
  therapist: Any = None
  patient: Any = None
  cbt_techniques: dict[str, str] = {}
  mi_techniques: dict[str, str] = {}
  disable_tib: bool = False

  def _project_root(self) -> Path:
    return Path(__file__).resolve().parent.parent

  def _env_value(self, name: str, default: Any) -> Any:
    env = getattr(self.cfg, "env", {})
    if isinstance(env, dict):
      return env.get(name, default)
    return getattr(env, name, default)

  def _env_bool(self, name: str, default: bool) -> bool:
    value = self._env_value(name, default)
    if value is None:
      return default
    if isinstance(value, bool):
      return value
    if isinstance(value, str):
      normalized = value.strip().lower()
      if normalized in {"true", "1", "yes", "y", "on"}:
        return True
      if normalized in {"false", "0", "no", "n", "off"}:
        return False
    if isinstance(value, (int, float)):
      return bool(value)
    return default

  def _env_float(self, name: str, default: float) -> float:
    try:
      return float(self._env_value(name, default))
    except (TypeError, ValueError):
      return default

  def _patient_resistance_update_bounds(self) -> tuple[float, float]:
    min_delta = self._env_float("patient_resistance_update_min_delta", -0.10)
    max_delta = self._env_float("patient_resistance_update_max_delta", 0.15)
    if min_delta > max_delta:
      min_delta, max_delta = max_delta, min_delta
    return min_delta, max_delta

  def _format_signed_delta(self, value: float) -> str:
    return f"{value:+.2f}"

  def _resistance_enabled(self) -> bool:
    return self._env_bool("resistance", True)

  def _tib_detection_disabled(self) -> bool:
    return bool(getattr(self, "disable_tib", False))

  def _patient_prompt_ablation(self) -> dict[str, str]:
    patient_cfg = getattr(self.cfg, "patient", {})
    if isinstance(patient_cfg, dict):
      return resolve_prompt_ablation_settings(
        patient_cfg.get("prompt_variant", DEFAULT_PROMPT_VARIANT),
      )
    return resolve_prompt_ablation_settings(
      getattr(patient_cfg, "prompt_variant", DEFAULT_PROMPT_VARIANT),
    )

  def _config_bool(self, block: Any, name: str, default: bool) -> bool:
    value = block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)
    if value is None:
      return default
    if isinstance(value, bool):
      return value
    if isinstance(value, str):
      normalized = value.strip().lower()
      if normalized in {"true", "1", "yes", "y", "on"}:
        return True
      if normalized in {"false", "0", "no", "n", "off"}:
        return False
    if isinstance(value, (int, float)):
      return bool(value)
    return default

  def _init_agents(self) -> None:
    therapist_llm = LLMFactory.create_from_config(self.cfg.therapist)
    patient_llm = LLMFactory.create_from_config(self.cfg.patient)
    prompt_ablation = self._patient_prompt_ablation()
    self.therapist = TherapistAgent(
      therapist_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.therapist),
      user_instruction=self._config_bool(self.cfg.therapist, "user_instruction", False),
      length_restriction=self._config_bool(self.cfg.therapist, "length_restriction", False),
    )
    self.patient = PatientAgent(
      patient_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.patient),
      resistance_enabled=self._resistance_enabled(),
      prompt_variant=prompt_ablation["prompt_variant"],
    )

  def _load_stage_prompt(self, stage: str, name: str, **values: Any) -> str:
    if stage == "mi_topic_exploration":
      stage = "mi_techniques_with_te"
    filename = f"{stage}/{name}.txt"
    if values:
      return render_prompt(filename, **values)
    return load_prompt(filename)

  def _load_techniques(self) -> dict[str, str]:
    techniques_path = self._project_root() / "data" / "knowledge" / "cactus_techniques.json"
    if not techniques_path.exists():
      raise ConversationProcessingError(f"Missing techniques file: {techniques_path}")
    try:
      with open(techniques_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load techniques from {techniques_path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ConversationProcessingError(f"Techniques file must contain a JSON object: {techniques_path}")
    return {
      ensure_str(name): ensure_str(description)
      for name, description in data.items()
      if ensure_str(name)
    }
  
  def _load_mi_techniques(self) -> dict[str, str]:
    techniques_path = self._project_root() / "data" / "knowledge" / "cami_mi_techniques.json"
    if not techniques_path.exists():
      raise ConversationProcessingError(f"Missing MI techniques file: {techniques_path}")
    try:
      with open(techniques_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load MI techniques from {techniques_path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ConversationProcessingError(f"MI Techniques file must contain a JSON object: {techniques_path}")
    return {
      ensure_str(name): ensure_str(description)
      for name, description in data.items()
      if ensure_str(name)
    }

  def _load_distortions(self) -> dict[str, Any]:
    distortions_path = self._project_root() / "data" / "knowledge" / "cognitive_distortions.json"
    if not distortions_path.exists():
      raise ConversationProcessingError(f"Missing cognitive distortions file: {distortions_path}")
    try:
      with open(distortions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception as exc:
      raise ConversationProcessingError(f"Failed to load cognitive distortions from {distortions_path}: {exc}") from exc
    return {ensure_str(name): value for name, value in data.items() if ensure_str(name)}

  def initial_state(self, max_turns: int, min_turns: int = 0) -> SessionState:
    return self._ensure_state_defaults(
      {
        "max_turns": int(max_turns),
        "min_turns": int(min_turns),
        "long_session_turn_threshold": int(
          self._env_value("long_session_turn_threshold", 70)
        ),
      }
    )

  def resume_state(self, saved_state: dict[str, Any] | SessionState, max_turns: int, min_turns: int = 0) -> SessionState:
    resumed = cast(SessionState, cast(object, deepcopy(saved_state)))
    resumed["session_ended"] = False
    resumed["session_ended_reason"] = ""
    resumed["conversation_error"] = ""
    resumed["final_patient_pending_end"] = False
    if isinstance(resumed.get("reasoning_traces"), list):
      resumed["reasoning_traces"] = [
        trace for trace in resumed["reasoning_traces"] if not self._is_error_trace(trace)
      ]
    if isinstance(resumed.get("messages"), list) and resumed["messages"]:
      last_message = resumed["messages"][-1]
      if isinstance(last_message, dict) and last_message.get("role") == "therapist":
        self._rewind_last_resumed_therapist_turn(resumed)
    preconsecutive_turns, mi_recovery_turns, cami_recovery_turns = self._infer_resume_streak_counters(
      resumed.get("messages"),
      resumed.get("reasoning_traces"),
    )
    resumed["preconsecutive_turns"] = preconsecutive_turns
    resumed["mi_recovery_turns"] = mi_recovery_turns if resumed.get("current_stage") == "mi_techniques" else 0
    resumed["cami_recovery_turns"] = (
      cami_recovery_turns if resumed.get("current_stage") == "mi_techniques_with_te" else 0
    )
    resumed["max_turns"] = int(max_turns)
    resumed["min_turns"] = int(min_turns)
    resumed["long_session_turn_threshold"] = int(
      resumed.get("long_session_turn_threshold", self._env_value("long_session_turn_threshold", 70))
    )
    return self._ensure_state_defaults(resumed)

  def _rewind_last_resumed_therapist_turn(self, state: SessionState) -> None:
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages:
      return

    deleted_message = messages.pop()
    deleted_index = self._coerce_utterance_index(deleted_message)
    last_message_index = self._conversation_rollup_latest_index(messages) if messages else 0

    reasoning_traces = state.get("reasoning_traces")
    if isinstance(reasoning_traces, list):
      self._remove_trace_for_utterance(reasoning_traces, deleted_index, "therapist")
      state["reasoning_traces"] = [
        trace
        for trace in reasoning_traces
        if self._coerce_utterance_index(trace) <= last_message_index
      ]

    if isinstance(state.get("prompt_audit_records"), list):
      state["prompt_audit_records"] = [
        record
        for record in state["prompt_audit_records"]
        if not (
          ensure_str(record.get("speaker")).strip().lower() == "therapist"
          and self._coerce_utterance_index(record) == last_message_index
        )
      ]

    self._prune_resumed_mi_context(state, last_message_index)

    if int(state.get("conversation_rollup_until_utterance_index", 0)) >= deleted_index > 0:
      state["conversation_rollup_summary"] = ""
      state["conversation_rollup_until_utterance_index"] = 0

    if messages:
      last_message = messages[-1]
      restored_stage = normalize_stage_name(last_message.get("stage"))
      stage: Stage = cast(Stage, restored_stage) if restored_stage else "opening"
      restored_substage = normalize_substage_name(last_message.get("substage"))
      substage = cast(SubStage, restored_substage) if restored_substage else DEFAULT_SUBSTAGE_BY_STAGE[stage]
      state["current_stage"] = stage
      state["current_substage"] = substage
      state["active_exchange_stage"] = stage
      state["active_exchange_substage"] = substage
    else:
      state["current_stage"] = "opening"
      state["current_substage"] = "open_initial"
      state["active_exchange_stage"] = ""
      state["active_exchange_substage"] = ""
      state["opening_patient_mood"] = ""
      state["opening_patient_problem"] = ""
      state["opening_fallback_triggered"] = False

    inferred_predicted_found, inferred_predicted = self._infer_latest_predicted_resistance_from_traces(
      state.get("reasoning_traces")
    )
    if inferred_predicted_found:
      state["predicted_resistance"] = inferred_predicted
    elif not messages:
      state["predicted_resistance"] = None

    inferred_reasoning = self._infer_last_reasoning_from_traces(state.get("reasoning_traces"))
    if inferred_reasoning is not None:
      state["last_reasoning"] = inferred_reasoning
    elif not messages:
      state["last_reasoning"] = ""

    inferred_opening_context = self._infer_opening_context_from_traces(state.get("reasoning_traces"))
    if inferred_opening_context["mood"] is not None:
      state["opening_patient_mood"] = inferred_opening_context["mood"]
    if inferred_opening_context["problem"] is not None:
      state["opening_patient_problem"] = inferred_opening_context["problem"]

    if state["current_stage"] == "opening":
      state["opening_fallback_triggered"] = False

    state["exchange_counts"] = self._rebuild_exchange_counts_from_messages(messages)
    preconsecutive_turns, mi_recovery_turns, cami_recovery_turns = self._infer_resume_streak_counters(
      messages,
      state.get("reasoning_traces"),
    )
    state["preconsecutive_turns"] = preconsecutive_turns
    state["mi_recovery_turns"] = mi_recovery_turns if state["current_stage"] == "mi_techniques" else 0
    state["cami_recovery_turns"] = cami_recovery_turns if state["current_stage"] == "mi_techniques_with_te" else 0

    if state["current_stage"] == "address_tib":
      self._clear_address_tib_state(state, clear_resume_cursor=True)
      self._restore_active_address_tib_state_from_traces(state)
      self._clear_completed_address_tib_memory(state)
      self._clear_resume_context(state)
    else:
      self._clear_address_tib_state(state, clear_resume_cursor=True)

    if state["current_stage"] in MI_STAGES:
      state["mi_active"] = True
    else:
      state["mi_active"] = False
      state["current_mi_technique_selection"] = []

    state["utterance_index"] = last_message_index

  def _coerce_utterance_index(self, item: Any) -> int:
    if not isinstance(item, dict):
      return 0
    try:
      return int(item.get("utterance_index", 0))
    except (TypeError, ValueError):
      return 0

  def _trace_steps(self, trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict):
      return []
    steps = trace.get("traces")
    if not isinstance(steps, list):
      return []
    return [step for step in steps if isinstance(step, dict)]

  def _trace_has_step(self, trace: Any, step_name: str) -> bool:
    return any(ensure_str(step.get("step")) == step_name for step in self._trace_steps(trace))

  def _trace_step_payload(self, trace: Any, step_name: str) -> dict[str, Any]:
    for step in reversed(self._trace_steps(trace)):
      if ensure_str(step.get("step")) != step_name:
        continue
      raw_output = step.get("output")
      if isinstance(raw_output, dict):
        return raw_output
      text_output = ensure_str(raw_output).strip()
      if not text_output:
        return {}
      try:
        payload = json.loads(text_output)
      except json.JSONDecodeError:
        return {}
      return payload if isinstance(payload, dict) else {}
    return {}

  def _remove_trace_for_utterance(self, traces: list[Any], utterance_index: int, speaker: str) -> None:
    normalized_speaker = ensure_str(speaker).strip().lower()
    for idx in range(len(traces) - 1, -1, -1):
      trace = traces[idx]
      if ensure_str(trace.get("speaker")).strip().lower() != normalized_speaker:
        continue
      if self._coerce_utterance_index(trace) != utterance_index:
        continue
      traces.pop(idx)
      return

  def _prune_resumed_mi_context(self, state: SessionState, last_message_index: int) -> None:
    raw_mi_context = state.get("mi_context")
    if not isinstance(raw_mi_context, list):
      return

    known_message_pairs = {
      (self._coerce_utterance_index(message), ensure_str(message.get("role")).strip().lower())
      for message in state.get("messages", [])
      if isinstance(message, dict)
    }
    pruned: list[ConversationMessage] = []
    for entry in raw_mi_context:
      if not isinstance(entry, dict):
        continue
      utterance_index = self._coerce_utterance_index(entry)
      role = ensure_str(entry.get("role")).strip().lower()
      if utterance_index > last_message_index:
        continue
      if role == "system_note":
        if utterance_index >= last_message_index:
          continue
        pruned.append(cast(ConversationMessage, entry))
        continue
      if (utterance_index, role) not in known_message_pairs:
        continue
      pruned.append(cast(ConversationMessage, entry))
    state["mi_context"] = pruned

  def _infer_latest_predicted_resistance_from_traces(self, traces: Any) -> tuple[bool, float | None]:
    if not self._resistance_enabled():
      return False, None
    if not isinstance(traces, list):
      return False, None
    for trace in reversed(traces):
      if ensure_str(trace.get("speaker")).strip().lower() != "therapist":
        continue
      payload = self._trace_step_payload(trace, "resistance_prediction")
      for key in ("new_p", "p_target", "previous_p"):
        if key not in payload:
          continue
        value = payload.get(key)
        if value is None:
          return True, None
        try:
          return True, float(value)
        except (TypeError, ValueError):
          continue
    return False, None

  def _infer_last_reasoning_from_traces(self, traces: Any) -> str | None:
    if not isinstance(traces, list):
      return None
    for trace in reversed(traces):
      if ensure_str(trace.get("speaker")).strip().lower() != "therapist":
        continue
      for step in reversed(self._trace_steps(trace)):
        step_name = ensure_str(step.get("step"))
        if not self._resistance_enabled() and step_name == "resistance_prediction":
          continue
        payload = self._trace_step_payload({"traces": [step]}, step_name)
        reasoning = ensure_str(payload.get("reasoning")).strip()
        if reasoning:
          return reasoning
    return None

  def _infer_opening_context_from_traces(self, traces: Any) -> dict[str, str | None]:
    latest_mood: str | None = None
    latest_problem: str | None = None
    if not isinstance(traces, list):
      return {"mood": latest_mood, "problem": latest_problem}
    for trace in traces:
      if ensure_str(trace.get("speaker")).strip().lower() != "therapist":
        continue
      payload = self._trace_step_payload(trace, "opening_info_update")
      if not payload:
        continue
      mood = ensure_str(payload.get("mood")).strip()
      problem = ensure_str(payload.get("problem")).strip()
      if mood:
        latest_mood = mood
      if problem:
        latest_problem = problem
    return {"mood": latest_mood, "problem": latest_problem}

  def _rebuild_exchange_counts_from_messages(self, messages: Any) -> dict[str, list[int]]:
    if not isinstance(messages, list):
      return {}
    rebuilt: dict[str, list[int]] = {}
    previous_patient_stage: Stage | Literal[""] = ""
    for message in messages:
      if not isinstance(message, dict):
        continue
      if ensure_str(message.get("role")).strip().lower() != "patient":
        continue
      raw_stage = normalize_stage_name(message.get("stage"))
      if not raw_stage:
        continue
      stage = cast(Stage, raw_stage)
      history = rebuilt.setdefault(stage, [])
      if not history or previous_patient_stage != stage:
        history.append(0)
      history[-1] = int(history[-1]) + 1
      previous_patient_stage = stage
    return rebuilt

  def _trace_tib_route_bucket(self, trace: Any) -> str:
    payload = self._trace_step_payload(trace, "tib_check")
    route_bucket = ensure_str(payload.get("route_bucket")).strip().lower()
    if route_bucket in {"none", "end_session", "avoidance", "other"}:
      return route_bucket
    return ""

  def _trace_predicted_band(self, trace: Any) -> str | None:
    payload = self._trace_step_payload(trace, "resistance_prediction")
    value = payload.get("new_p", payload.get("p_target"))
    try:
      return self._current_band(float(value))
    except (TypeError, ValueError):
      return None

  def _infer_resume_streak_counters(self, messages: Any, traces: Any) -> tuple[int, int, int]:
    if not self._resistance_enabled():
      return 0, 0, 0
    if not isinstance(messages, list) or not isinstance(traces, list):
      return 0, 0, 0

    therapist_traces = {
      self._coerce_utterance_index(trace): trace
      for trace in traces
      if ensure_str(trace.get("speaker")).strip().lower() == "therapist"
    }

    preconsecutive_turns = 0
    mi_recovery_turns = 0
    cami_recovery_turns = 0
    for idx, message in enumerate(messages):
      if not isinstance(message, dict):
        continue
      if ensure_str(message.get("role")).strip().lower() != "therapist":
        continue
      if idx == 0 or not isinstance(messages[idx - 1], dict):
        continue

      previous_message = messages[idx - 1]
      if ensure_str(previous_message.get("role")).strip().lower() != "patient":
        continue

      trace = therapist_traces.get(self._coerce_utterance_index(message))
      if trace is None:
        continue

      route_bucket = self._trace_tib_route_bucket(trace)
      inline_tib_suppressed = self._trace_has_step(trace, "address_tib_repeat_inline_suppressed")
      if route_bucket and route_bucket != "none" and not inline_tib_suppressed:
        if self._trace_has_step(trace, "resistance_handling_entry") or self._trace_has_step(trace, "mi_evaluation"):
          preconsecutive_turns = 0
          mi_recovery_turns = 0
          cami_recovery_turns = 0
        continue

      band = self._trace_predicted_band(trace)
      if band is None:
        continue

      raw_stage_before = normalize_stage_name(previous_message.get("stage"))
      stage_before = cast(Stage, raw_stage_before) if raw_stage_before else "opening"
      if stage_before in MI_STAGES:
        preconsecutive_turns = 0
      elif band == "precontemplation":
        preconsecutive_turns += 1
      else:
        preconsecutive_turns = 0

      if stage_before == "mi_techniques":
        if band == "precontemplation":
          mi_recovery_turns = 0
        else:
          mi_recovery_turns += 1
      else:
        mi_recovery_turns = 0

      if stage_before == "mi_techniques_with_te":
        if band == "precontemplation":
          cami_recovery_turns = 0
        else:
          cami_recovery_turns += 1
      else:
        cami_recovery_turns = 0

      if self._trace_has_step(trace, "resistance_handling_entry") or self._trace_has_step(trace, "mi_evaluation"):
        preconsecutive_turns = 0
        mi_recovery_turns = 0
        cami_recovery_turns = 0

    return preconsecutive_turns, mi_recovery_turns, cami_recovery_turns

  def _restore_active_address_tib_state_from_traces(self, state: SessionState) -> None:
    latest_enter: dict[str, Any] = {}
    latest_tib_check: dict[str, Any] = {}
    latest_plan: dict[str, Any] = {}
    traces = state.get("reasoning_traces")
    if isinstance(traces, list):
      for trace in reversed(traces):
        if ensure_str(trace.get("speaker")).strip().lower() != "therapist":
          continue
        if not latest_enter:
          latest_enter = self._trace_step_payload(trace, "address_tib_enter")
        if not latest_tib_check:
          latest_tib_check = self._trace_step_payload(trace, "tib_check")
        if not latest_plan:
          latest_plan = self._trace_step_payload(trace, "address_tib_plan_generation")
        if latest_enter and latest_tib_check and latest_plan:
          break

    route_bucket = ensure_str((latest_enter or latest_tib_check).get("route_bucket")).strip().lower()
    if route_bucket not in {"none", "end_session", "avoidance", "other"}:
      route_bucket = "other" if latest_enter or latest_tib_check else "none"
    state["address_tib_route_bucket"] = cast(TIBRouteBucket, route_bucket)
    state["address_tib_name"] = ensure_str(
      latest_enter.get("tib_name") or latest_tib_check.get("tib_name") or state.get("address_tib_name")
    ).strip()
    state["address_tib_description"] = ensure_str(
      latest_tib_check.get("tib_description") or state.get("address_tib_description")
    ).strip()
    state["address_tib_reasoning"] = ensure_str(
      latest_enter.get("reasoning") or latest_tib_check.get("reasoning") or state.get("address_tib_reasoning")
    ).strip()
    state["address_tib_plan"] = ensure_str(latest_plan.get("tib_plan") or state.get("address_tib_plan")).strip()

    raw_resume_stage = normalize_stage_name(latest_enter.get("resume_stage") or state.get("address_tib_resume_stage"))
    state["address_tib_resume_stage"] = cast(Stage, raw_resume_stage) if raw_resume_stage else ""
    raw_resume_substage = normalize_substage_name(
      latest_enter.get("resume_substage") or state.get("address_tib_resume_substage")
    )
    state["address_tib_resume_substage"] = cast(SubStage, raw_resume_substage) if raw_resume_substage else ""

  def _is_error_trace(self, trace: Any) -> bool:
    if not isinstance(trace, dict):
      return False
    if trace.get("speaker") != "system":
      return False
    steps = trace.get("traces")
    if not isinstance(steps, list):
      return False
    return any(isinstance(step, dict) and step.get("step") == "conversation_error" for step in steps)

  def _ensure_state_defaults(self, state: dict[str, Any] | SessionState) -> SessionState:
    s = cast(SessionState, cast(object, dict(state)))
    raw_stage = normalize_stage_name(s.get("current_stage"))
    stage: Stage = cast(Stage, raw_stage) if raw_stage else "opening"
    s["current_stage"] = stage
    raw_substage = normalize_substage_name(s.get("current_substage"))
    s["current_substage"] = cast(SubStage, raw_substage) if raw_substage else DEFAULT_SUBSTAGE_BY_STAGE[stage]

    raw_messages = s.get("messages")
    s["messages"] = self._normalize_messages(raw_messages, stage, s["current_substage"])

    s = cast(SessionState, cast(object, normalize_resistance_state_fields(
      dict(s),
      deepcopy(self.profile),
      moderate_recovery_rate=self._env_float("moderate_emotional_recovery_rate", 0.15),
    )))
    s = cast(SessionState, cast(object, normalize_predicted_resistance_state_fields(dict(s))))
    s["prompt_ablation"] = self._patient_prompt_ablation()
    s["disable_tib"] = self._tib_detection_disabled()

    s.setdefault("last_patient_resistance_update_reasoning", "")
    s.setdefault("last_patient_action", "")
    s.setdefault("last_patient_action_prior_distribution", cast(PatientActionDistribution, {}))
    s.setdefault("last_patient_action_context_distribution", cast(PatientActionDistribution, {}))
    s.setdefault("last_patient_action_distribution", cast(PatientActionDistribution, {}))
    s.setdefault("last_patient_perceived_topic", "")
    raw_topic_distance = s.get("last_patient_topic_distance")
    if raw_topic_distance is None:
      s["last_patient_topic_distance"] = None
    else:
      try:
        s["last_patient_topic_distance"] = int(raw_topic_distance)
      except (TypeError, ValueError):
        s["last_patient_topic_distance"] = None
    raw_engagement_level = s.get("last_patient_engagement_level")
    if raw_engagement_level is None:
      s["last_patient_engagement_level"] = None
    else:
      try:
        parsed_engagement_level = int(raw_engagement_level)
      except (TypeError, ValueError):
        parsed_engagement_level = 0
      s["last_patient_engagement_level"] = parsed_engagement_level if 1 <= parsed_engagement_level <= 4 else None
    s.setdefault("last_patient_engagement_analysis", "")
    s.setdefault("last_patient_engagement_instruction", "")
    s.setdefault("opening_patient_mood", "")
    s.setdefault("opening_patient_problem", "")
    s.setdefault("last_reasoning", "")
    s.setdefault("conversation_rollup_summary", "")
    s.setdefault("conversation_rollup_until_utterance_index", 0)
    legacy_is_complete_reasoning = s.get("is_completed_reasoning")    
    if legacy_is_complete_reasoning and not s.get("last_reasoning"):
      s["last_reasoning"] = ensure_str(legacy_is_complete_reasoning)

    s.setdefault("cognitive_model_sets", [])
    s.setdefault("working_cognitive_set", self._new_cognitive_set(len(s["cognitive_model_sets"]) + 1))
    s.setdefault("shelved_cognitive_sets", [])
    s.setdefault("cognitive_gap_plan", "")
    s.setdefault("active_cognitive_target", "")
    s.setdefault("active_cognitive_question", "")
    s.setdefault("active_cognitive_follow_up_attempts", 0)
    s.setdefault("working_cognitive_set_exchange_count", 0)

    s.setdefault("selected_techniques", [])
    s.setdefault("current_technique_index", 0)
    s.setdefault("current_technique_plan", "")
    s.setdefault("current_technique_expected_outcome", "")
    s.setdefault("current_technique_current_progress", "")
    s.setdefault("technique_evaluations", [])
    s.setdefault("techniques_completed", 0)
    s.setdefault("technique_turn_count", 0)
    s.setdefault("technique_slot_replaced", {})

    s.setdefault("mi_active", False)
    raw_mi_context = s.get("mi_context")
    s["mi_context"] = self._normalize_messages(raw_mi_context, stage, s["current_substage"])
    s.setdefault("cbt_flow_saved_state", "")
    raw_cbt_resume_stage = normalize_stage_name(s.get("cbt_resume_stage"))
    s["cbt_resume_stage"] = cast(Stage, raw_cbt_resume_stage) if raw_cbt_resume_stage else "id_distortions"
    raw_cbt_resume_substage = normalize_substage_name(s.get("cbt_resume_substage"))
    s["cbt_resume_substage"] = cast(SubStage, raw_cbt_resume_substage) if raw_cbt_resume_substage else ""
    raw_mi_topic_stack = s.get("mi_topic_stack")
    if isinstance(raw_mi_topic_stack, list):
      s["mi_topic_stack"] = [item for item in ensure_str_list(raw_mi_topic_stack) if item]
    else:
      s["mi_topic_stack"] = []
    raw_mi_topic_probabilities = s.get("mi_topic_probabilities")
    normalized_probabilities: dict[str, float] = {}
    if isinstance(raw_mi_topic_probabilities, dict):
      for key, value in raw_mi_topic_probabilities.items():
        topic = ensure_str(key)
        if not topic:
          continue
        try:
          normalized_probabilities[topic] = float(value)
        except (TypeError, ValueError):
          continue
    s["mi_topic_probabilities"] = normalized_probabilities
    s.setdefault("mi_topics_tried", [])
    s.setdefault("mi_techniques_used", [])
    s.setdefault("current_mi_technique_selection", [])
    s.setdefault("mi_techniques_paused_until_precontemplation", False)
    s.setdefault("long_session_mi_route_to_homework_pending", False)
    s.setdefault("long_session_timeout_instruction_pending", False)
    s.setdefault("current_mi_topic", "")
    s.setdefault("last_mi_predicted_resistance", s["predicted_resistance"])
    s.setdefault("last_mi_topic_action", "")
    s.setdefault("preconsecutive_turns", 0)
    s.setdefault("mi_recovery_turns", 0)
    s.setdefault("cami_recovery_turns", 0)
    if "mi_end_session_tib_redirect_used" not in s and "mi_end_session_demand_redirect_used" in s:
      s["mi_end_session_tib_redirect_used"] = bool(s.get("mi_end_session_demand_redirect_used"))
    s.setdefault("mi_end_session_tib_redirect_used", False)

    legacy_address_tib_description = ensure_str(s.get("patient_demand_text"))
    legacy_address_tib_plan = ensure_str(s.get("patient_demand_plan"))
    legacy_address_tib_reasoning = ensure_str(s.get("patient_demand_reasoning"))
    if "address_tib_description" not in s and legacy_address_tib_description:
      s["address_tib_description"] = legacy_address_tib_description
    if "address_tib_plan" not in s and legacy_address_tib_plan:
      s["address_tib_plan"] = legacy_address_tib_plan
    if "address_tib_reasoning" not in s and legacy_address_tib_reasoning:
      s["address_tib_reasoning"] = legacy_address_tib_reasoning

    address_tib_name = ensure_str(s.get("address_tib_name"))
    if not address_tib_name and legacy_address_tib_description:
      address_tib_name = "legacy_patient_demand"
    s["address_tib_name"] = address_tib_name
    s["address_tib_description"] = ensure_str(s.get("address_tib_description"))
    s["address_tib_reasoning"] = ensure_str(s.get("address_tib_reasoning"))
    s["address_tib_plan"] = ensure_str(s.get("address_tib_plan"))

    raw_address_tib_bucket = ensure_str(s.get("address_tib_route_bucket")).strip().lower()
    if raw_address_tib_bucket not in {"none", "end_session", "avoidance", "other"}:
      if s["address_tib_description"] or s["current_stage"] == "address_tib":
        raw_address_tib_bucket = "other"
      else:
        raw_address_tib_bucket = "none"
    s["address_tib_route_bucket"] = cast(TIBRouteBucket, raw_address_tib_bucket)

    raw_address_tib_resume_stage = normalize_stage_name(
      s.get("address_tib_resume_stage", s.get("patient_demand_resume_stage"))
    )
    s["address_tib_resume_stage"] = cast(Stage, raw_address_tib_resume_stage) if raw_address_tib_resume_stage else ""
    raw_address_tib_resume_substage = normalize_substage_name(
      s.get("address_tib_resume_substage", s.get("patient_demand_resume_substage"))
    )
    s["address_tib_resume_substage"] = cast(SubStage, raw_address_tib_resume_substage) if raw_address_tib_resume_substage else ""
    s.setdefault("last_completed_address_tib_name", "")
    raw_last_completed_tib_bucket = ensure_str(s.get("last_completed_address_tib_route_bucket")).strip().lower()
    if raw_last_completed_tib_bucket not in {"none", "end_session", "avoidance", "other"}:
      raw_last_completed_tib_bucket = "none"
    s["last_completed_address_tib_route_bucket"] = cast(TIBRouteBucket, raw_last_completed_tib_bucket)
    s.setdefault("last_completed_address_tib_no_reroute_active", False)
    s.setdefault("inline_tib_ack_name", "")
    s.setdefault("inline_tib_ack_description", "")

    s.setdefault("resume_context_active", False)
    raw_resume_from_stage = normalize_stage_name(s.get("resume_context_from_stage"))
    s["resume_context_from_stage"] = cast(Stage, raw_resume_from_stage) if raw_resume_from_stage else ""
    s.setdefault("resume_context_reason", "")
    s.setdefault("resume_context_subject", "")

    s.setdefault("proposed_tasks", [])
    s.setdefault("homework_proposal_attempts", 0)
    s.setdefault("homework_waiting_for_patient", False)
    s.setdefault("homework_fallback_done", False)
    s.setdefault("agreed_tasks", [])
    s.setdefault("hw_reasoning", "")
    s.setdefault("homework_fallback_to_apply", False)

    s.setdefault("session_summary", "")
    s.setdefault("next_session_plan", "")
    s.setdefault("patient_feedback", "")
    s.setdefault("closing_plan_refinements", 0)

    s.setdefault("prompt_audit_records", [])
    s.setdefault("reasoning_traces", [])
    max_message_index = 0
    if s["messages"]:
      max_message_index = max(int(msg.get("utterance_index", 0)) for msg in s["messages"])
    max_trace_index = 0
    if s["reasoning_traces"]:
      max_trace_index = max(int(trace["utterance_index"]) for trace in s["reasoning_traces"])
    s["utterance_index"] = max(int(s.get("utterance_index", 0)), max_message_index, max_trace_index)
    raw_exchange_counts = s.get("exchange_counts")
    normalized_exchange_counts: dict[str, list[int]] = {}
    if isinstance(raw_exchange_counts, dict):
      for key, value in raw_exchange_counts.items():
        stage_name = normalize_stage_name(key)
        if not stage_name:
          continue
        if isinstance(value, list):
          normalized_exchange_counts[stage_name] = [int(item) for item in value]
        elif value is not None:
          normalized_exchange_counts[stage_name] = [int(value)]
    s["exchange_counts"] = normalized_exchange_counts
    
    # devlog. remove when create new conversations
    legacy_metaconv_turns = s.get("meta_conversation_turns")
    if legacy_metaconv_turns is not None and "meta_conversation" not in s["exchange_counts"]:
      s["exchange_counts"]["meta_conversation"] = [int(legacy_metaconv_turns)]
    legacy_mi_technique_turns = s.get("mi_technique_turns")
    if legacy_mi_technique_turns is not None and "mi_techniques" not in s["exchange_counts"]:
      s["exchange_counts"]["mi_techniques"] = [int(legacy_mi_technique_turns)]
    raw_active_exchange_stage = normalize_stage_name(s.get("active_exchange_stage"))
    s["active_exchange_stage"] = cast(Stage, raw_active_exchange_stage) if raw_active_exchange_stage else ""
    raw_active_exchange_substage = normalize_substage_name(s.get("active_exchange_substage"))
    s["active_exchange_substage"] = cast(SubStage, raw_active_exchange_substage) if raw_active_exchange_substage else ""
    # devlog end

    if "tib_redirect_triggered" not in s and "high_resistance_redirect_triggered" in s:
      s["tib_redirect_triggered"] = bool(s.get("high_resistance_redirect_triggered"))
    s.setdefault("tib_redirect_triggered", False)
    if not s.get("tib_redirect_level") and s.get("high_resistance_level"):
      s["tib_redirect_level"] = ensure_str(s.get("high_resistance_level"))
    s.setdefault("tib_redirect_level", "")
    raw_tib_redirect_source_stage = normalize_stage_name(
      s.get("tib_redirect_source_stage", s.get("high_resistance_source_stage"))
    )
    s["tib_redirect_source_stage"] = cast(Stage, raw_tib_redirect_source_stage) if raw_tib_redirect_source_stage else ""
    raw_tib_redirect_source_substage = normalize_substage_name(
      s.get("tib_redirect_source_substage", s.get("high_resistance_source_substage"))
    )
    s["tib_redirect_source_substage"] = (
      cast(SubStage, raw_tib_redirect_source_substage) if raw_tib_redirect_source_substage else ""
    )
    if not s.get("tib_redirect_reasoning") and s.get("high_resistance_reasoning"):
      s["tib_redirect_reasoning"] = ensure_str(s.get("high_resistance_reasoning"))
    s.setdefault("tib_redirect_reasoning", "")
    
    s.setdefault("final_patient_pending_end", False)
    s.setdefault("session_ended", False)
    s.setdefault("session_ended_reason", "")
    s.setdefault("conversation_error", "")
    s.setdefault("opening_fallback_triggered", False)

    s["max_turns"] = int(s.get("max_turns", self._env_value("max_turns", 80)))
    s["min_turns"] = int(s.get("min_turns", self._env_value("min_turns", 0)))
    s["long_session_turn_threshold"] = int(
      s.get("long_session_turn_threshold", self._env_value("long_session_turn_threshold", 70))
    )
    if not self._resistance_enabled():
      self._clear_resistance_state_fields(s)
    self._ensure_stage_exchange_counter(s, stage)
    return s

  def _clear_resistance_state_fields(self, state: SessionState) -> None:
    state["resistance_baseline"] = None
    state["resistance_recovery_rate"] = None
    state["resistance"] = None
    state["predicted_resistance"] = None
    state["last_mi_predicted_resistance"] = None
    state["last_patient_resistance_update_reasoning"] = ""
    state["preconsecutive_turns"] = 0
    state["mi_recovery_turns"] = 0
    state["cami_recovery_turns"] = 0
    state["last_patient_action"] = ""
    state["last_patient_action_prior_distribution"] = cast(PatientActionDistribution, {})
    state["last_patient_action_context_distribution"] = cast(PatientActionDistribution, {})
    state["last_patient_action_distribution"] = cast(PatientActionDistribution, {})
    for message in state["messages"]:
      message["resistance"] = None

  def _set_stage(self, state: SessionState, stage: Stage, substage: Optional[SubStage] = None) -> None:
    previous_stage = ensure_str(state.get("current_stage"))
    if previous_stage != stage:
      self._start_stage_exchange_counter(state, stage)
    else:
      self._ensure_stage_exchange_counter(state, stage)
    state["current_stage"] = stage
    state["current_substage"] = substage or DEFAULT_SUBSTAGE_BY_STAGE[stage]

  def _set_substage(self, state: SessionState, substage: SubStage) -> None:
    state["current_substage"] = substage

  def _handle_stage_dispatch(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    raise NotImplementedError

  def _handle_closing(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    raise NotImplementedError

  def _enter_mi(
    self,
    state: SessionState,
    target_stage: Stage,
    resume_stage: Stage | None,
    resume_substage: SubStage | None,
    reason: str,
  ) -> None:
    raise NotImplementedError

  def _format_conversation(self, messages: list[ConversationMessage]) -> str:
    if not messages:
      return "(not utterances yet)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

  def _format_last_utterances(self, messages: list[ConversationMessage], limit: int = 10) -> str:
    if not messages:
      return "(no utterances yet)"
    recent = messages[-limit:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

  def _build_therapist_request_messages(
    self,
    state: SessionState,
    instruction: str,
    *,
    structured: bool,
    full_conversation: str | None = None,
    last_reasoning: str | None = None,
    session_context: str | None = None,
  ) -> list[ChatMessage]:
    conversation_text = full_conversation if full_conversation is not None else self._current_prompt_conversation_text(state)
    resolved_last_reasoning = state.get("last_reasoning", "") if last_reasoning is None else last_reasoning
    resolved_session_context = self._build_stage_user_context(state) if session_context is None else session_context
    if structured:
      return self.therapist.build_structured_messages(
        full_conversation=conversation_text,
        last_utterances=self._format_last_utterances(state["messages"]),
        last_reasoning=resolved_last_reasoning,
        session_context=resolved_session_context,
        instruction=instruction,
      )
    return self.therapist.build_utterance_messages(
      full_conversation=conversation_text,
      last_utterances=self._format_last_utterances(state["messages"]),
      last_reasoning=resolved_last_reasoning,
      session_context=resolved_session_context,
      instruction=instruction,
    )

  def _build_patient_request_messages(
    self,
    state: SessionState,
    instruction: str,
    *,
    structured: bool,
    full_conversation: str | None = None,
    engagement_block: str | None = None,
  ) -> list[ChatMessage]:
    conversation_text = full_conversation if full_conversation is not None else self._current_prompt_conversation_text(state)
    if structured:
      return self.patient.build_structured_messages(
        full_conversation=conversation_text,
        last_utterances=self._format_last_utterances(state["messages"]),
        resistance_level=state["resistance"],
        instruction=instruction,
        engagement_block=engagement_block,
      )
    return self.patient.build_utterance_messages(
      full_conversation=conversation_text,
      last_utterances=self._format_last_utterances(state["messages"]),
      resistance_level=state["resistance"],
      instruction=instruction,
      engagement_block=engagement_block,
    )

  def _trim_words(self, text: str, limit: int) -> str:
    words = ensure_str(text).split()
    if len(words) <= limit:
      return ensure_str(text)
    return " ".join(words[:limit]).strip()

  def _conversation_rollup_latest_index(self, messages: list[ConversationMessage]) -> int:
    if not messages:
      return 0
    return max(int(message.get("utterance_index", 0)) for message in messages)

  def _recent_messages_since_rollup(self, state: SessionState) -> list[ConversationMessage]:
    cutoff = int(state.get("conversation_rollup_until_utterance_index", 0))
    return [
      message
      for message in state["messages"]
      if int(message.get("utterance_index", 0)) > cutoff
    ]

  def _compose_prompt_conversation(
    self,
    summary: str,
    recent_messages: list[ConversationMessage],
  ) -> str:
    summary_text = ensure_str(summary).strip()
    sections: list[str] = []
    if summary_text:
      sections.append(f"Summary of earlier conversation:\n{summary_text}")
    if recent_messages:
      recent_text = self._format_conversation(recent_messages)
      if summary_text:
        sections.append(f"Recent utterances since that summary:\n{recent_text}")
      else:
        sections.append(recent_text)
    if not sections:
      return "(no utterances yet)"
    return "\n".join(sections)

  def _current_prompt_conversation_text(self, state: SessionState) -> str:
    summary = ensure_str(state.get("conversation_rollup_summary"))
    if summary:
      return self._compose_prompt_conversation(summary, self._recent_messages_since_rollup(state))
    return self._format_conversation(state["messages"])

  def _summarize_prompt_conversation(self, state: SessionState, conversation_text: str) -> str:
    instruction = self._load_stage_prompt(
      "shared",
      "conversation_rollup_summary",
      word_limit=FULL_CONVERSATION_SUMMARY_WORD_LIMIT,
    )
    record_large_prompt_request(
      state,
      self._build_therapist_request_messages(
        state,
        instruction,
        structured=True,
        full_conversation=conversation_text,
        last_reasoning="",
        session_context=json.dumps(
          {
            "current_stage": state["current_stage"],
            "current_substage": state["current_substage"],
          },
          ensure_ascii=True,
        ),
      ),
      "therapist",
      "conversation_rollup_summary",
    )
    try:
      parsed, _meta = self.therapist.generate_structured(
        full_conversation=conversation_text,
        last_utterances=self._format_last_utterances(state["messages"]),
        last_reasoning="",
        session_context=json.dumps(
          {
            "current_stage": state["current_stage"],
            "current_substage": state["current_substage"],
          },
          ensure_ascii=True,
        ),
        instruction=instruction,
        expected_type=dict,
        temperature=0.2,
      )
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"conversation_rollup_summary failed: {exc}") from exc

    payload = self._require_dict(parsed, "conversation_rollup_summary")
    summary = self._require_non_empty_string(payload, "conversation_summary", "conversation_rollup_summary")
    return summary

  def _full_conversation_for_prompt(self, state: SessionState) -> str:
    current_text = self._current_prompt_conversation_text(state)
    if count_text_tokens(current_text) > FULL_CONVERSATION_TOKEN_LIMIT:
      state["conversation_rollup_summary"] = self._summarize_prompt_conversation(state, current_text)
      state["conversation_rollup_until_utterance_index"] = self._conversation_rollup_latest_index(state["messages"])
      current_text = self._current_prompt_conversation_text(state)
    return current_text

  # Format functions here

  def _join_compact_fields(self, fields: list[tuple[str, str]]) -> str:
    return ",".join(f"{key}={value}" for key, value in fields)

  def _format_cognitive_model_sets(self, sets: CognitiveModelSet | list[CognitiveModelSet]) -> str:
    entries = sets if isinstance(sets, list) else [sets]
    lines: list[str] = []
    for item in entries:
      if not item:
        continue
      situation = ensure_str(item.get("situation"))
      automatic_thoughts = " | ".join(ensure_str_list(item.get("automatic_thoughts")))
      emotions = " | ".join(ensure_str_list(item.get("emotions")))
      behaviors = " | ".join(ensure_str_list(item.get("behaviors")))
      core_belief = ensure_str(item.get("core_belief"))
      distortion = ensure_str(item.get("distortion"))
      if not any([situation, automatic_thoughts, emotions, behaviors, core_belief, distortion]):
        continue
      lines.append(
        self._join_compact_fields(
          [
            ("situation", situation),
            ("auto_thoughts", automatic_thoughts),
            ("emotions", emotions),
            ("behaviors", behaviors),
            ("core_belief", core_belief),
            ("distortion", distortion),
          ]
        )
      )
    return "\n".join(lines)

  def _format_selected_technique(self, techniques: SelectedTechnique | list[SelectedTechnique] | None) -> str:
    if not techniques:
      return ""
    entries = techniques if isinstance(techniques, list) else [techniques]
    lines: list[str] = []
    for item in entries:
      if not item:
        continue
      name = ensure_str(item.get("name"))
      target_distortion = ensure_str(item.get("target_distortion"))
      if not any([name, target_distortion]):
        continue
      lines.append(
        self._join_compact_fields(
          [
            ("name", name),
            ("target_distortion", target_distortion),
          ]
        )
      )
    return "\n".join(lines)

  def _format_tasks(self, tasks: list[HomeworkTask]) -> str:
    lines: list[str] = []
    for item in tasks:
      if not item:
        continue
      task_title = ensure_str(item.get("task_title"))
      task_description = ensure_str(item.get("task_description"))
      if not any([task_title, task_description]):
        continue
      lines.append(
        self._join_compact_fields(
          [
            ("task_title", task_title),
            ("task_description", task_description),
          ]
        )
      )
    return "\n".join(lines)

  def _build_stage_user_context(self, state: SessionState) -> str:
    stage = state["current_stage"]
    context: dict[str, Any] = {
      "current_stage": stage,
      #"stage_exchange_count": self._current_stage_exchange_count(state, stage),
      #"stage_exchange_history": list(state["exchange_counts"].get(stage, [])),
      # "tib_redirect_triggered": state["tib_redirect_triggered"],
      # "tib_redirect_level": state["tib_redirect_level"],
      # "tib_redirect_source_stage": state["tib_redirect_source_stage"],
      # "tib_redirect_source_substage": state["tib_redirect_source_substage"],
      # "tib_redirect_reasoning": state["tib_redirect_reasoning"],
      # "address_tib_route_bucket": state["address_tib_route_bucket"],
      # "address_tib_name": state["address_tib_name"],
      # "address_tib_description": state["address_tib_description"],
      # "address_tib_reasoning": state["address_tib_reasoning"],
      # "address_tib_plan": state["address_tib_plan"],
      # "address_tib_resume_stage": state["address_tib_resume_stage"],
      # "address_tib_resume_substage": state["address_tib_resume_substage"],
      # "mi_end_session_tib_redirect_used": state["mi_end_session_tib_redirect_used"],
      "last_patient_response": state["messages"][-1]["content"] if state["messages"] else "(no utterances yes)",
    }
    if self._resistance_enabled():
      context["predicted_resistance"] = state["predicted_resistance"]

    if stage == "opening":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "opening_fallback_triggered": state["opening_fallback_triggered"],
        }
      )
    elif stage == "id_distortions":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "cognitive_model_sets": self._format_cognitive_model_sets(state["cognitive_model_sets"]),
          "shelved_cognitive_sets": self._format_cognitive_model_sets(state["shelved_cognitive_sets"]),
          "working_cognitive_set": self._format_cognitive_model_sets(state["working_cognitive_set"]),
          "active_cognitive_target": state["active_cognitive_target"],
          "active_cognitive_question": state["active_cognitive_question"],
          "active_cognitive_follow_up_attempts": state["active_cognitive_follow_up_attempts"],
          "cognitive_gap_plan": state["cognitive_gap_plan"],
          #"working_cognitive_set_exchange_count": state["working_cognitive_set_exchange_count"],
          #"working_cognitive_set_min_exchanges": MIN_COGNITIVE_SET_EXCHANGES,
        }
      )
    elif stage == "cbt_technique":
      current_selected_technique = None
      current_technique_index = int(state["current_technique_index"])
      if 0 <= current_technique_index < len(state["selected_techniques"]):
        current_selected_technique = state["selected_techniques"][current_technique_index]
      context.update(
        {
          "cognitive_model_sets": self._format_cognitive_model_sets(state["cognitive_model_sets"]),
          "selected_techniques": self._format_selected_technique(state["selected_techniques"]),
          "current_technique_index": current_technique_index,
          "current_selected_technique": self._format_selected_technique(current_selected_technique),
          "current_technique_plan": state["current_technique_plan"],
          #"current_technique_expected_outcome": state["current_technique_expected_outcome"],
          "current_technique_current_progress": state["current_technique_current_progress"],
          #"technique_turn_count": state["technique_turn_count"],
          "techniques_completed": state["techniques_completed"],
          "technique_evaluations": state["technique_evaluations"][-3:],
        }
      )
    elif stage == "homework":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "cognitive_model_sets": self._format_cognitive_model_sets(state["cognitive_model_sets"]),
          "technique_evaluations": state["technique_evaluations"][-3:],
          "proposed_tasks": self._format_tasks(state["proposed_tasks"]),
          "agreed_tasks": self._format_tasks(state["agreed_tasks"]),
          "homework_proposal_attempts": state["homework_proposal_attempts"],
          #"homework_waiting_for_patient": state["homework_waiting_for_patient"],
          #"homework_fallback_done": state["homework_fallback_done"],
          #"homework_fallback_to_apply": state["homework_fallback_to_apply"],
        }
      )
    elif stage == "closing":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "cognitive_model_sets": self._format_cognitive_model_sets(state["cognitive_model_sets"]),
          "technique_evaluations": state["technique_evaluations"],
          "selected_techniques": self._format_selected_technique(state["selected_techniques"]),
          "session_summary": state["session_summary"],
          "next_session_plan": state["next_session_plan"],
          "agreed_tasks": self._format_tasks(state["agreed_tasks"]),
          "tib_redirect_triggered": state["tib_redirect_triggered"],
          "tib_redirect_level": state["tib_redirect_level"],
          "tib_redirect_reasoning": state["tib_redirect_reasoning"],
          "patient_feedback": state["patient_feedback"],
          #"closing_plan_refinements": state["closing_plan_refinements"],
        }
      )
    elif stage == "mi_techniques_with_te":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "conversation_rollup_summary": state["conversation_rollup_summary"],
          "current_mi_topic": state["current_mi_topic"],
          "mi_techniques_used": state["mi_techniques_used"][-5:],
          "cbt_resume_stage": state["cbt_resume_stage"],
          "cbt_resume_substage": state["cbt_resume_substage"],
        }
      )
    elif stage == "mi_techniques":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "mi_techniques_used": state["mi_techniques_used"][-5:],
          #"mi_context": state["mi_context"],
          #"last_mi_predicted_resistance": state["last_mi_predicted_resistance"],
          "cbt_resume_stage": state["cbt_resume_stage"],
          "cbt_resume_substage": state["cbt_resume_substage"],
        }
      )
    elif stage == "meta_conversation":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "mi_topics_tried": state["mi_topics_tried"],
          "mi_techniques_used": state["mi_techniques_used"][-5:],
          #"mi_context": state["mi_context"],
          "cbt_resume_stage": state["cbt_resume_stage"],
          "cbt_resume_substage": state["cbt_resume_substage"],
        }
      )
    elif stage == "address_tib":
      context.update(
        {
          "opening_patient_mood": state["opening_patient_mood"],
          "opening_patient_problem": state["opening_patient_problem"],
          "current_mi_topic": state["current_mi_topic"],
          "cognitive_model_sets": self._format_cognitive_model_sets(state["cognitive_model_sets"]),
          "selected_techniques": self._format_selected_technique(state["selected_techniques"]),
          "current_technique_index": state["current_technique_index"],
          "address_tib_route_bucket": state["address_tib_route_bucket"],
          "address_tib_name": state["address_tib_name"],
          "address_tib_description": state["address_tib_description"],
          "address_tib_reasoning": state["address_tib_reasoning"],
          "address_tib_plan": state["address_tib_plan"],
          #"address_tib_resume_stage": state["address_tib_resume_stage"],
          #"address_tib_resume_substage": state["address_tib_resume_substage"],
          #"address_tib_turn_count": self._current_stage_exchange_count(state, "address_tib"),
          #"address_tib_turn_cap": 5,
        }
      )
    elif stage in {"anticipated_closing", "forced_closing"}:
      context.update(
        {
          "session_ended_reason": state["session_ended_reason"],
          "cbt_resume_stage": state["cbt_resume_stage"],
        }
      )

    return json.dumps(context, ensure_ascii=True)

  def _instruction_with_previous_reasoning(self, state: SessionState, instruction: str) -> str:
    text = ensure_str(instruction)
    previous_reasoning = ensure_str(state.get("last_reasoning"))
    if not previous_reasoning:
      return text

    suffix = self._load_stage_prompt(
      "shared",
      "previous_reasoning_suffix",
      previous_reasoning=previous_reasoning,
      last_utterance=state["messages"][-2]["content"] if state["messages"] else "(no messages yet)",
    )
    return f"{text} {suffix}"

  def _instruction_with_tib_redirect(self, state: SessionState, instruction: str) -> str:
    text = ensure_str(instruction)
    if self._tib_detection_disabled():
      return text
    if not state.get("tib_redirect_triggered"):
      return text

    level = ensure_str(state.get("tib_redirect_level"), "partial")
    source_stage = ensure_str(state.get("tib_redirect_source_stage"), "unknown stage")
    suffix = (
      " Before moving forward, briefly acknowledge the patient's wish to stop, step back, or change focus "
      f"in a natural therapist voice. This was a TIB reroute from {source_stage}. "
      "Then continue with the current stage goal smoothly, without sounding abrupt, defensive, or scripted."
    )
    return f"{text} {suffix}"

  def _long_session_turn_threshold(self, state: SessionState) -> int:
    threshold = int(state.get("long_session_turn_threshold", self._env_value("long_session_turn_threshold", 70)))
    return max(0, threshold)

  def _long_session_threshold_reached(self, state: SessionState) -> bool:
    return int(state.get("utterance_index", 0)) >= self._long_session_turn_threshold(state)

  def _long_session_mi_timeout_applies_to_current_utterance(self, state: SessionState) -> bool:
    if state.get("long_session_timeout_instruction_pending"):
      return True
    return (
      state["current_stage"] == "closing"
      and state["current_substage"] == "closing_fallback"
      and ensure_str(state.get("session_ended_reason")) == "max_turns_forced_closing"
    )

  def _instruction_with_long_session_timeout(self, state: SessionState, instruction: str) -> tuple[str, bool]:
    text = ensure_str(instruction)
    if not self._long_session_mi_timeout_applies_to_current_utterance(state):
      return text, False

    stage = state["current_stage"]
    suffix = (
      " Briefly mention that time is running short for this session."
    )
    if stage in {"mi_techniques", "mi_techniques_with_te"}:
      suffix += (
        " Use this utterance to end the current motivational-interviewing segment and smoothly orient the patient"
        " toward the homework part of today's session rather than continuing this line of exploration."
      )
    else:
      suffix += (
        " Close for today rather than reopening the session or introducing new therapeutic work."
      )
    return f"{text}{suffix}", True

  def _resume_context_applies_to_current_utterance(self, state: SessionState) -> bool:
    if not state.get("resume_context_active"):
      return False
    stage = state["current_stage"]
    substage = state["current_substage"]
    if stage == "opening":
      return substage in {"open_mood", "open_problem"}
    if stage == "id_distortions":
      return substage in {"cognitive_review_active_question", "cognitive_refinement"}
    if stage == "cbt_technique":
      return substage == "cbt_deliver_current"
    if stage == "mi_techniques_with_te":
      return substage == "mi_raise_cami_utterance"
    if stage == "mi_techniques":
      return substage == "mi_raise_technique"
    return False

  def _instruction_with_resume_context(self, state: SessionState, instruction: str) -> tuple[str, bool]:
    text = ensure_str(instruction)
    if not self._resume_context_applies_to_current_utterance(state):
      return text, False

    from_stage = ensure_str(state.get("resume_context_from_stage"), "a previous stage")
    reason = ensure_str(state.get("resume_context_reason"))
    subject = ensure_str(state.get("resume_context_subject"))
    suffix = (
      f" Before addressing the instruction, briefly and naturally acknowledge that you are resuming after a detour from '{from_stage}'."
    )
    if subject:
      suffix += f" Use this detour context if helpful: {subject!r}."
    if reason == "max_turn_cap":
      suffix += " Frame the detour as addressed as far as possible for now."
    # suffix += (
    #   " Then continue directly with the current stage goal in the same utterance."
    #   " Do not produce a transition-only utterance and do not mention internal stage names."
    # )
    return f"{text}{suffix}", True

  def _instruction_with_inline_tib_ack(self, state: SessionState, instruction: str) -> tuple[str, bool]:
    text = ensure_str(instruction)
    if self._tib_detection_disabled():
      return text, False
    tib_name = ensure_str(state.get("inline_tib_ack_name")).strip()
    if not tib_name:
      return text, False

    tib_description = ensure_str(state.get("inline_tib_ack_description")).strip()
    suffix = (
      f" The patient's latest reply repeats the same therapy-interfering behavior ({tib_name!r}) that was already addressed."
    )
    if tib_description:
      suffix += f" It is showing up now as: {tib_description!r}."
    suffix += (
      " Do not reroute or start a separate TIB detour."
      " Address it inline very briefly in at most one sentence without naming the TIB name, connect it back to the current stage goal, and continue the current stage task in the same utterance."
    )
    if state.get("resume_context_active"):
      suffix += " If you are also resuming after a detour, combine that with this acknowledgment instead of creating separate transitions."
    return f"{text}{suffix}", True

  def _generate_therapist_utterance(self, state: SessionState, instruction: str) -> str:
    tib_adjusted_instruction = self._instruction_with_tib_redirect(state, instruction)
    resume_adjusted_instruction, used_resume_context = self._instruction_with_resume_context(state, tib_adjusted_instruction)
    inline_tib_adjusted_instruction, used_inline_tib_ack = self._instruction_with_inline_tib_ack(
      state,
      resume_adjusted_instruction,
    )
    enriched_instruction, used_long_session_timeout = self._instruction_with_long_session_timeout(
      state,
      inline_tib_adjusted_instruction,
    )
    record_large_prompt_request(
      state,
      self._build_therapist_request_messages(state, enriched_instruction, structured=False),
      "therapist",
      "therapist_utterance",
    )
    utterance = self.therapist.generate_utterance(
      full_conversation=self._full_conversation_for_prompt(state),
      last_utterances=self._format_last_utterances(state["messages"]),
      last_reasoning=state.get("last_reasoning", ""),
      session_context=self._build_stage_user_context(state),
      instruction=enriched_instruction,
    )
    if used_long_session_timeout and state.get("long_session_timeout_instruction_pending"):
      state["long_session_timeout_instruction_pending"] = False
    if used_inline_tib_ack or state.get("inline_tib_ack_name"):
      self._clear_inline_tib_ack(state)
    if used_resume_context or state.get("resume_context_active"):
      self._clear_resume_context(state)
    return utterance

  def _append_trace(self, state: SessionState, utterance_index: int, speaker: str, traces: list[dict[str, Any]]) -> None:
    state["reasoning_traces"].append(
      {
        "utterance_index": utterance_index,
        "speaker": speaker,
        "traces": traces,
      }
    )

  def _append_message(self, state: SessionState, role: ConversationRole, content: str) -> int:
    return self._append_message_with_metadata(state, role, content, state["current_stage"], state["current_substage"])

  def _append_message_with_metadata(
    self,
    state: SessionState,
    role: ConversationRole,
    content: str,
    stage: Stage,
    substage: SubStage,
  ) -> int:
    if self._resistance_enabled():
      resistance = state["predicted_resistance"] if role == "therapist" else state["resistance"]
    else:
      resistance = None
    state["utterance_index"] += 1
    state["messages"].append(
      {
        "utterance_index": state["utterance_index"],
        "role": role,
        "content": ensure_str(content),
        "stage": stage,
        "substage": substage,
        "resistance": resistance,
      }
    )
    return state["utterance_index"]

  def _append_mi_context_message(
    self,
    state: SessionState,
    role: ConversationRole,
    content: str,
    utterance_index: int,
    stage: Stage,
    substage: SubStage,
  ) -> None:
    state["mi_context"].append(
      {
        "utterance_index": utterance_index,
        "role": role,
        "content": ensure_str(content),
        "stage": stage,
        "substage": substage,
      }
    )

  def _normalize_messages(self, raw_messages: Any, default_stage: Stage, default_substage: SubStage) -> list[ConversationMessage]:
    if not isinstance(raw_messages, list):
      return []
    normalized: list[ConversationMessage] = []
    next_index = 1
    for item in raw_messages:
      if not isinstance(item, dict):
        continue
      role = ensure_str(item.get("role"))
      content = ensure_str(item.get("content"))
      if not role or not content:
        continue
      raw_index = item.get("utterance_index")
      try:
        utterance_index = int(raw_index) if raw_index is not None else next_index
      except (TypeError, ValueError):
        utterance_index = next_index
      if utterance_index <= 0:
        utterance_index = next_index

      raw_stage = normalize_stage_name(item.get("stage"))
      message_stage: Stage = cast(Stage, raw_stage) if raw_stage else default_stage
      raw_substage = normalize_substage_name(item.get("substage"))
      message_substage = cast(SubStage, raw_substage) if raw_substage else default_substage

      normalized.append(
        {
          "utterance_index": utterance_index,
          "role": cast(ConversationRole, role),
          "content": content,
          "stage": message_stage,
          "substage": message_substage,
        }
      )
      next_index = max(next_index + 1, utterance_index + 1)
    return normalized

  def _ensure_stage_exchange_counter(self, state: SessionState, stage: Stage) -> None:
    history = state["exchange_counts"].setdefault(stage, [])
    if not history:
      history.append(0)

  def _start_stage_exchange_counter(self, state: SessionState, stage: Stage) -> None:
    history = state["exchange_counts"].setdefault(stage, [])
    history.append(0)

  def _current_stage_exchange_count(self, state: SessionState, stage: Stage) -> int:
    history = state["exchange_counts"].get(stage, [])
    if not history:
      return 0
    return int(history[-1])

  def _total_stage_exchange_count(self, state: SessionState, stage: Stage) -> int:
    history = state["exchange_counts"].get(stage, [])
    if not history:
      return 0
    return sum(int(item) for item in history)

  def _increment_stage_exchange_count(self, state: SessionState, stage: Stage) -> int:
    self._ensure_stage_exchange_counter(state, stage)
    history = state["exchange_counts"][stage]
    history[-1] = int(history[-1]) + 1
    return history[-1]

  def _mark_tib_redirect(
    self,
    state: SessionState,
    level: str,
    source_stage: Stage,
    source_substage: SubStage,
    reasoning: str,
  ) -> None:
    state["tib_redirect_triggered"] = True
    state["tib_redirect_level"] = level
    state["tib_redirect_source_stage"] = source_stage
    state["tib_redirect_source_substage"] = source_substage
    state["tib_redirect_reasoning"] = reasoning

  def _clear_tib_redirect(self, state: SessionState) -> None:
    state["tib_redirect_triggered"] = False
    state["tib_redirect_level"] = ""
    state["tib_redirect_source_stage"] = ""
    state["tib_redirect_source_substage"] = ""
    state["tib_redirect_reasoning"] = ""

  def _set_resume_context(
    self,
    state: SessionState,
    from_stage: Stage,
    reason: str,
    subject: str,
  ) -> None:
    state["resume_context_active"] = True
    state["resume_context_from_stage"] = from_stage
    state["resume_context_reason"] = ensure_str(reason).strip()
    state["resume_context_subject"] = ensure_str(subject).strip()

  def _clear_resume_context(self, state: SessionState) -> None:
    state["resume_context_active"] = False
    state["resume_context_from_stage"] = ""
    state["resume_context_reason"] = ""
    state["resume_context_subject"] = ""

  def _remember_completed_address_tib(self, state: SessionState, route_bucket: TIBRouteBucket, tib_name: str) -> None:
    normalized_tib_name = ensure_str(tib_name).strip().lower()
    if route_bucket == "other" and normalized_tib_name and normalized_tib_name != "none":
      state["last_completed_address_tib_name"] = normalized_tib_name
      state["last_completed_address_tib_route_bucket"] = route_bucket
      state["last_completed_address_tib_no_reroute_active"] = True
      return
    self._clear_completed_address_tib_memory(state)

  def _clear_completed_address_tib_memory(self, state: SessionState) -> None:
    state["last_completed_address_tib_name"] = ""
    state["last_completed_address_tib_route_bucket"] = "none"
    state["last_completed_address_tib_no_reroute_active"] = False

  def _set_inline_tib_ack(self, state: SessionState, tib_name: str, tib_description: str) -> None:
    state["inline_tib_ack_name"] = ensure_str(tib_name).strip().lower()
    state["inline_tib_ack_description"] = ensure_str(tib_description).strip()

  def _clear_inline_tib_ack(self, state: SessionState) -> None:
    state["inline_tib_ack_name"] = ""
    state["inline_tib_ack_description"] = ""

  def _clear_address_tib_state(self, state: SessionState, clear_resume_cursor: bool = True) -> None:
    state["address_tib_route_bucket"] = "none"
    state["address_tib_name"] = ""
    state["address_tib_description"] = ""
    state["address_tib_reasoning"] = ""
    state["address_tib_plan"] = ""
    if clear_resume_cursor:
      state["address_tib_resume_stage"] = ""
      state["address_tib_resume_substage"] = ""

  def _next_cognitive_set_id(self, state: SessionState) -> int:
    known_ids = [
      int(state["working_cognitive_set"].get("cognitive_model_set_id", 0)),
      *[int(set_.get("cognitive_model_set_id", 0)) for set_ in state["cognitive_model_sets"]],
      *[int(set_.get("cognitive_model_set_id", 0)) for set_ in state["shelved_cognitive_sets"]],
    ]
    return max(known_ids, default=0) + 1

  def _resistance_handling_mode(self) -> Literal["none", "mi_techniques", "cami"]:
    if not self._resistance_enabled():
      return "none"
    raw_mode = ensure_str(self._env_value("resistance_handling", "mi_techniques"))
    normalized = raw_mode.strip().lower()
    if normalized in {"none", "mi_techniques", "cami"}:
      return cast(Literal["none", "mi_techniques", "cami"], normalized)
    return "mi_techniques"

  def _current_band(
    self,
    value: float | None,
    state: SessionState | None = None,
  ) -> Literal["precontemplation", "contemplation", "action"]:
    if value is None:
      return "contemplation"

    if value >= 0.66:
      return "precontemplation"
    if value > 0.33:
      return "contemplation"
    return "action"

  def _new_cognitive_set(self, set_id: int) -> CognitiveModelSet:
    return {
      "cognitive_model_set_id": set_id,
      "situation": "",
      "automatic_thoughts": [],
      "emotions": [],
      "behaviors": [],
      "core_belief": "",
      "core_belief_probabilities": {k: 0.0 for k in CORE_BELIEF_KEYS},
      "distortion": "",
      "distortion_probabilities": {k: 0.0 for k in DISTORTION_KEYS},
    }

  def _normalize_probabilities(self, payload: dict[str, Any], keys: list[str]) -> dict[str, float]:
    if not isinstance(payload, dict):
      raise ConversationProcessingError("Probability payload must be a JSON object.")
    probs = {}
    for key in keys:
      if key not in payload:
        raise ConversationProcessingError(f"Probability payload is missing required key '{key}'.")
      raw_value = payload[key]
      try:
        parsed = float(raw_value)
      except (TypeError, ValueError) as exc:
        raise ConversationProcessingError(f"Probability '{key}' must be numeric, got {raw_value!r}.") from exc
      if parsed < 0.0 or parsed > 1.0:
        raise ConversationProcessingError(f"Probability '{key}' must be in [0, 1], got {parsed}.")
      probs[key] = parsed
    total = sum(probs.values())
    if total <= 0:
      raise ConversationProcessingError("Probability payload sums to 0 and cannot be normalized.")
    return {k: v / total for k, v in probs.items()}

  def _best_key(self, probs: dict[str, float]) -> str:
    if not isinstance(probs, dict) or not probs:
      raise ConversationProcessingError("Cannot choose a best key from an empty probability map.")
    return max(probs.items(), key=lambda item: item[1])[0]

  def _require_dict(self, payload: Any, step_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
      raise ConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON object.")
    return payload

  def _require_list(self, payload: Any, step_name: str) -> list[Any]:
    if not isinstance(payload, list):
      raise ConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON array.")
    return payload

  def _require_bool(self, payload: dict[str, Any], key: str, step_name: str) -> bool:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required boolean field '{key}'.")
    value = payload[key]
    if isinstance(value, bool):
      return value
    if isinstance(value, str):
      lowered = value.strip().lower()
      if lowered in {"true", "false"}:
        return lowered == "true"
    raise ConversationProcessingError(f"{step_name}.{key} must be a boolean, got {value!r}.")

  def _require_bounded_float(
    self,
    payload: dict[str, Any],
    key: str,
    step_name: str,
    low: float,
    high: float,
    recover_sign: bool = False,
  ) -> float:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required numeric field '{key}'.")
    value = payload[key]
    try:
      parsed = float(value)
    except (TypeError, ValueError) as exc:
      raise ConversationProcessingError(f"{step_name}.{key} must be numeric, got {value!r}.") from exc
    if parsed < low or parsed > high:
      # Some providers emit a stray leading '-' on an otherwise valid value. Only safe to undo
      # when the field is non-negative by definition (low >= 0), never for signed fields.
      if recover_sign and low >= 0.0 and low <= abs(parsed) <= high:
        return abs(parsed)
      raise ConversationProcessingError(f"{step_name}.{key} must be in [{low}, {high}], got {parsed}.")
    return parsed

  def _require_non_empty_string(self, payload: dict[str, Any], key: str, step_name: str) -> str:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required field '{key}'.")
    value = ensure_str(payload.get(key))
    if not value:
      raise ConversationProcessingError(f"{step_name}.{key} must be a non-empty string.")
    return value

  def _require_string_list(self, payload: dict[str, Any], key: str, step_name: str, min_items: int = 1) -> list[str]:
    if key not in payload:
      raise ConversationProcessingError(f"{step_name} is missing required list field '{key}'.")
    values = ensure_str_list(payload.get(key))
    if len(values) < min_items:
      raise ConversationProcessingError(f"{step_name}.{key} must contain at least {min_items} item(s).")
    return values

  def _end_session_with_error(self, state: SessionState, traces: list[dict[str, Any]], error: Exception) -> None:
    message = ensure_str(str(error), "Unknown conversation processing error")
    state["conversation_error"] = message
    state["session_ended"] = True
    state["final_patient_pending_end"] = False
    if not state["session_ended_reason"]:
      state["session_ended_reason"] = "conversation_error"
    state["reasoning_traces"].append(
      {
        "utterance_index": state["utterance_index"],
        "speaker": "system",
        "traces": [
          {
            "step": "conversation_error",
            "input": state["current_stage"],
            "output": message,
          }
        ],
      }
    )

  def _trace_structured_step(
    self,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    parsed: Any,
  ) -> None:
    traces.append(
      {
        "step": step_name,
        "input": instruction,
        "output": json.dumps(parsed, ensure_ascii=True),
      }
    )

  def _set_last_trace_output(self, traces: list[dict[str, Any]], step_name: str, payload: dict[str, Any]) -> None:
    for trace in reversed(traces):
      if trace.get("step") == step_name:
        trace["output"] = json.dumps(payload, ensure_ascii=True)
        return

  def _run_therapist_structured(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
    temperature: float | None = None,
  ) -> Any:
    record_large_prompt_request(
      state,
      self._build_therapist_request_messages(state, instruction, structured=True),
      "therapist",
      step_name,
    )
    if step_name == "resistance_prediction":
      validator = self._validate_resistance_prediction
      max_retries = 4
    else:
      validator = None
      max_retries = 2
    try:
      parsed, _meta = self.therapist.generate_structured(
        full_conversation=self._full_conversation_for_prompt(state),
        last_utterances=self._format_last_utterances(state["messages"]),
        last_reasoning=state["last_reasoning"],
        session_context=self._build_stage_user_context(state),
        instruction=instruction,
        expected_type=expected_type,
        validator=validator,
        max_retries=max_retries,
        temperature=temperature,
      )
      #print(f"    <DEBUG>{parsed}</DEBUG>") #devlog
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    if isinstance(parsed, dict):
      reasoning = ensure_str(parsed.get("reasoning"))
      if reasoning:
        state["last_reasoning"] = reasoning
    self._trace_structured_step(traces, step_name, instruction, parsed)
    return parsed

  @staticmethod
  def _validate_resistance_prediction(payload: Any) -> str | None:
    if not isinstance(payload, dict):
      return "resistance_prediction must be a JSON object"
    for key in ("p_target", "signal_clarity"):
      try:
        value = float(payload[key])
      except (KeyError, TypeError, ValueError):
        return f"resistance_prediction.{key} must be a number in [0.0, 1.0]"
      # abs(): a stray leading '-' is recovered downstream, so it does not need a re-generation.
      if not 0.0 <= abs(value) <= 1.0:
        return f"resistance_prediction.{key} must be in [0.0, 1.0], got {value}"
    return None

  def _run_patient_structured(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
    engagement_block: str | None = None,
  ) -> Any:
    record_large_prompt_request(
      state,
      self._build_patient_request_messages(
        state,
        instruction,
        structured=True,
        engagement_block=engagement_block,
      ),
      "patient",
      step_name,
    )
    try:
      parsed, _meta = self.patient.generate_structured(
        full_conversation=self._full_conversation_for_prompt(state),
        last_utterances=self._format_last_utterances(state["messages"]),
        resistance_level=state["resistance"],
        instruction=instruction,
        engagement_block=engagement_block,
        expected_type=expected_type,
      )
    except StructuredOutputError as exc:
      raise ConversationProcessingError(f"{step_name} failed: {exc}") from exc
    self._trace_structured_step(traces, step_name, instruction, parsed)
    return parsed

  def _extract_last_patient_message(self, state: SessionState) -> str:
    for msg in reversed(state["messages"]):
      if msg["role"] == "patient":
        return msg["content"]
    return ""

  def _current_resistance_prediction_turn(self, state: SessionState) -> int:
    patient_turns = sum(
      1
      for message in state.get("messages", [])
      if isinstance(message, dict) and ensure_str(message.get("role")).strip().lower() == "patient"
    )
    completed_prediction_traces = sum(
      1
      for trace in state.get("reasoning_traces", [])
      if isinstance(trace, dict) and self._trace_has_step(trace, "resistance_prediction")
    )
    return max(1, patient_turns, completed_prediction_traces + 1)

  def _env_int(self, name: str, default: int) -> int:
    try:
      return int(self._env_value(name, default))
    except (TypeError, ValueError):
      return default

  def _run_resistance_prediction(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    if not self._resistance_enabled():
      state["predicted_resistance"] = None
      return
    previous_p = state["predicted_resistance"]
    instruction = self._load_stage_prompt(
      "shared",
      "resistance_prediction",
      current_utterance=self._extract_last_patient_message(state),
      old_r="none yet" if previous_p is None else f"{previous_p:.2f}",
    )
    parsed = self._run_therapist_structured(state, traces, "resistance_prediction", instruction)
    payload = self._require_dict(parsed, "resistance_prediction")
    p_target = self._require_bounded_float(payload, "p_target", "resistance_prediction", 0.0, 1.0, recover_sign=True)
    signal_clarity = self._require_bounded_float(
      payload, "signal_clarity", "resistance_prediction", 0.0, 1.0, recover_sign=True
    )
    sign_recovered = [
      key for key, value in (("p_target", p_target), ("signal_clarity", signal_clarity)) if float(payload[key]) != value
    ]
    if sign_recovered:
      print(f"    [resistance_prediction] recovered sign-corrupted field(s): {', '.join(sign_recovered)}")
    prediction_update = self._apply_bayesian_predicted_resistance_update(previous_p, p_target, signal_clarity)
    prediction_update = apply_predicted_resistance_delta_clip(
      prediction_update,
      previous_p,
      self._current_resistance_prediction_turn(state),
      initial_turns=self._env_int("resistance_prediction_clip_initial_turns", 5),
      initial_min_delta=self._env_float("resistance_prediction_clip_initial_min_delta", -0.2),
      initial_max_delta=self._env_float("resistance_prediction_clip_initial_max_delta", 0.3),
      later_min_delta=self._env_float("resistance_prediction_clip_later_min_delta", -0.1),
      later_max_delta=self._env_float("resistance_prediction_clip_later_max_delta", 0.15),
    )
    state["predicted_resistance"] = prediction_update["new_p"]
    trace_payload = dict(payload)
    trace_payload.update(prediction_update)
    if sign_recovered:
      trace_payload["sign_recovered"] = sign_recovered
    self._set_last_trace_output(traces, "resistance_prediction", trace_payload)

  def _apply_bayesian_predicted_resistance_update(
    self,
    previous_p: float | None,
    p_target: float,
    signal_clarity: float,
  ) -> TherapistResistanceUpdate:
    return apply_bayesian_predicted_resistance_update(previous_p, p_target, signal_clarity)

  def _apply_homeostatic_resistance_update(
    self,
    previous_r: float,
    delta_llm: float,
    baseline: float,
    alpha: float,
  ) -> HomeostaticResistanceUpdate:
    return apply_homeostatic_resistance_update(previous_r, delta_llm, baseline, alpha)

  def _maybe_force_closing(self, state: SessionState) -> None:
    if state["session_ended"] or state["final_patient_pending_end"]:
      return
    if state["current_stage"] in {"anticipated_closing", "forced_closing"}:
      return
    threshold = max(0, state["max_turns"] - 2)
    if state["utterance_index"] >= threshold:
      state["long_session_mi_route_to_homework_pending"] = False
      state["long_session_timeout_instruction_pending"] = False
      self._set_stage(state, "closing", "closing_fallback")
      state["final_patient_pending_end"] = True
      state["session_ended_reason"] = "max_turns_forced_closing"
