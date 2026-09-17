from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, cast

from openai import OpenAI  # pyright: ignore[reportMissingImports]

from agents.interfaces import (
  ChatMessage,
  LLMFactory,
  StructuredOutputError,
  ensure_str,
  extract_generation_config,
  render_prompt,
)
from agents.patient import PatientAgent
from agents.therapist import LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS, TherapistAgent
from runtime_config import load_runtime_config

from .resistance import (
  DEFAULT_PROMPT_VARIANT,
  PatientAction,
  PatientActionDistribution,
  action_definition_for_action,
  action_prior_for_resistance,
  apply_homeostatic_resistance_update,
  constrain_action_distribution_for_utterance,
  fallback_action_distribution_for_resistance,
  merge_action_distributions,
  normalize_patient_action,
  normalize_resistance_state_fields,
  resolve_prompt_ablation_settings,
  sample_patient_action,
  should_select_patient_action,
  validate_action_distribution,
)


AGENT_END_TAG = "[/END]"
AGENT_PATIENT_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "patient_system.txt"
CACTUS_PATIENT_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "cactus_system_patient.txt"
CACTUS_PATIENT_USER_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "cactus_utterance_patient.txt"
CACTUS_RESISTANCE_PATIENT_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "cactus_r_system_patient.txt"
CACTUS_RESISTANCE_PATIENT_USER_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "cactus_r_utterance_patient.txt"
THERAPIST_ROLE = "therapist"
PATIENT_ROLE = "patient"
CONVERSATION_ROLES = {THERAPIST_ROLE, PATIENT_ROLE}
MODERATOR_MODEL_ID = "deepseek/deepseek-v3.2"
MODERATOR_BASE_URL = "https://openrouter.ai/api/v1"
MODERATOR_MAX_ATTEMPTS = 2
MODERATOR_SYSTEM_PROMPT = "Answer only Yes or No."
MODERATOR_USER_PROMPT_TEMPLATE = (
  "Does this therapist utterance clearly close or end the therapy session? "
  "Answer Yes only for explicit goodbye, final wrap-up, or session-ending language. "
  "Answer No for ordinary summaries, homework, or next steps without a clear ending.\n\n"
  "Therapist utterance:\n{utterance}"
)


class AgentConversationProcessingError(RuntimeError):
  pass


def _trace_output_text(value: Any) -> str:
  if isinstance(value, str):
    return value.strip()
  if isinstance(value, dict):
    reasoning = ensure_str(value.get("reasoning"))
    if reasoning:
      return reasoning
    return json.dumps(value, ensure_ascii=True, sort_keys=True)
  if value is None:
    return ""
  return ensure_str(value)


def format_agent_trace_block(traces: list[dict[str, Any]]) -> str:
  rendered: list[str] = []
  for trace in traces:
    if not isinstance(trace, dict):
      continue
    text = _trace_output_text(trace.get("output"))
    if text:
      rendered.append(text)
  return "\n".join(rendered)


class AgentSimulationChain:
  def __init__(self, builder: AgentTherapyGraphBuilder):
    self.builder = builder

  def _run_next_turn_pair(self, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
    state = self.builder.therapist_turn(state)
    yield state
    if state.get("session_ended"):
      return
    state = self.builder.patient_turn(state)
    yield state

  def stream(self, initial_state: dict[str, Any], stream_mode: str = "values") -> Iterator[dict[str, Any]]:
    state = self.builder._ensure_state_defaults(initial_state)
    if not state.get("session_ended") and state.get("messages"):
      last_message = state["messages"][-1]
      if isinstance(last_message, dict) and last_message.get("role") == THERAPIST_ROLE:
        before_index = int(state.get("utterance_index", 0))
        state = self.builder.patient_turn(state)
        yield state
        if int(state.get("utterance_index", 0)) <= before_index and not state.get("session_ended"):
          self.builder._end_session_with_error(
            state,
            [],
            AgentConversationProcessingError("Agent simulation made no progress."),
          )
          yield state
    while not state.get("session_ended"):
      before_index = int(state.get("utterance_index", 0))
      for chunk in self._run_next_turn_pair(state):
        state = chunk
        yield state
      if int(state.get("utterance_index", 0)) <= before_index:
        self.builder._end_session_with_error(
          state,
          [],
          AgentConversationProcessingError("Agent simulation made no progress."),
        )
        yield state

  def invoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
    state = self.builder._ensure_state_defaults(initial_state)
    for chunk in self.stream(state, stream_mode="values"):
      state = chunk
    return state


class AgentTherapyGraphBuilder:
  def __init__(self, config_path: str, profile: dict[str, Any], *, cactus_patient: bool | None = None):
    self.cfg = load_runtime_config(config_path)
    self.profile = profile
    self.cactus_patient = self._resolve_cactus_patient(cactus_patient)
    self._init_agents()

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

  def _config_bool(self, block: Any, name: str, default: bool) -> bool:
    if isinstance(block, dict):
      value = block.get(name, default)
    else:
      value = getattr(block, name, default)
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

  def _resolve_cactus_patient(self, cactus_patient: bool | None) -> bool:
    if cactus_patient is not None:
      return bool(cactus_patient)
    return self._config_bool(getattr(self.cfg, "patient", {}), "cactus", False)

  def _cactus_patient_enabled(self) -> bool:
    return bool(getattr(self, "cactus_patient", False))

  def _patient_prompt_ablation(self) -> dict[str, str]:
    patient_cfg = getattr(self.cfg, "patient", {})
    if isinstance(patient_cfg, dict):
      settings = resolve_prompt_ablation_settings(
        patient_cfg.get("prompt_variant", DEFAULT_PROMPT_VARIANT),
      )
    else:
      settings = resolve_prompt_ablation_settings(
        getattr(patient_cfg, "prompt_variant", DEFAULT_PROMPT_VARIANT),
      )
    if self._cactus_patient_enabled():
      settings["cactus_patient"] = "true"
    return settings

  def _config_value(self, block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
      return block.get(name, default)
    return getattr(block, name, default)

  def _model_type(self, block: Any) -> str:
    return ensure_str(self._config_value(block, "type")).strip().lower()

  def _therapist_uses_local_chatml(self) -> bool:
    return self._model_type(getattr(self.cfg, "therapist", {})) == "local"

  def _patient_utterance_system_prompt_path(self) -> str | Path:
    if self._cactus_patient_enabled():
      if self._resistance_enabled():
        return CACTUS_RESISTANCE_PATIENT_SYSTEM_PROMPT
      return CACTUS_PATIENT_SYSTEM_PROMPT
    return self._config_value(self.cfg.patient, "system_prompt_path") or AGENT_PATIENT_SYSTEM_PROMPT

  def _patient_utterance_user_prompt_path(self) -> str | Path | None:
    if not self._cactus_patient_enabled():
      return None
    if self._resistance_enabled():
      return CACTUS_RESISTANCE_PATIENT_USER_PROMPT
    return CACTUS_PATIENT_USER_PROMPT

  def _init_agents(self) -> None:
    therapist_llm = LLMFactory.create_from_config(self.cfg.therapist)
    patient_llm = LLMFactory.create_from_config(self.cfg.patient)
    prompt_ablation = self._patient_prompt_ablation()
    self.therapist = TherapistAgent(
      therapist_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.therapist),
      agent_system_prompt_path=self._config_value(self.cfg.therapist, "system_prompt_path") or None,
      agent_ft_system_prompt_path=self._config_value(self.cfg.therapist, "ft_system_prompt_path") or None,
      user_instruction=self._config_bool(self.cfg.therapist, "user_instruction", False),
      length_restriction=self._config_bool(self.cfg.therapist, "length_restriction", False),
    )
    self.patient = PatientAgent(
      patient_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.patient),
      resistance_enabled=self._resistance_enabled(),
      prompt_variant=prompt_ablation["prompt_variant"],
      utterance_system_prompt_path=self._patient_utterance_system_prompt_path(),
      utterance_user_prompt_path=self._patient_utterance_user_prompt_path(),
    )

  def initial_state(self, max_turns: int, min_turns: int = 0) -> dict[str, Any]:
    return self._ensure_state_defaults(
      {
        "messages": [],
        "reasoning_traces": [],
        "patient_profile": deepcopy(self.profile),
        "utterance_index": 0,
        "max_turns": int(max_turns),
        "min_turns": int(min_turns),
      }
    )

  def resume_state(self, saved_state: dict[str, Any], max_turns: int, min_turns: int = 0) -> dict[str, Any]:
    resumed = deepcopy(saved_state)
    resumed["session_ended"] = False
    resumed["session_ended_reason"] = ""
    resumed["conversation_error"] = ""
    resumed["final_patient_pending_end"] = False
    resumed["max_turns"] = int(max_turns)
    resumed["min_turns"] = int(min_turns)
    return self._ensure_state_defaults(resumed)

  def create_chain(self) -> AgentSimulationChain:
    return AgentSimulationChain(self)

  def createChain(self) -> AgentSimulationChain:
    return self.create_chain()

  def _normalize_messages(self, raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
      return []
    normalized: list[dict[str, Any]] = []
    next_index = 1
    for item in raw_messages:
      if not isinstance(item, dict):
        continue
      role = ensure_str(item.get("role")).strip().lower()
      content = ensure_str(item.get("content"))
      if role not in CONVERSATION_ROLES or not content:
        continue
      try:
        utterance_index = int(item.get("utterance_index", next_index))
      except (TypeError, ValueError):
        utterance_index = next_index
      if utterance_index <= 0:
        utterance_index = next_index
      message: dict[str, Any] = {
        "utterance_index": utterance_index,
        "role": role,
        "content": content,
      }
      if role == PATIENT_ROLE and item.get("resistance") is not None:
        try:
          message["resistance"] = float(item["resistance"])
        except (TypeError, ValueError):
          pass
      normalized.append(message)
      next_index = max(next_index + 1, utterance_index + 1)
    return normalized

  def _ensure_state_defaults(self, state: Mapping[str, Any]) -> dict[str, Any]:
    s = dict(state)
    s["messages"] = self._normalize_messages(s.get("messages"))
    s["patient_profile"] = deepcopy(s.get("patient_profile") if isinstance(s.get("patient_profile"), dict) else self.profile)
    s["prompt_ablation"] = self._patient_prompt_ablation()

    if self._resistance_enabled():
      s = normalize_resistance_state_fields(
        s,
        deepcopy(self.profile),
        moderate_recovery_rate=self._env_float("moderate_emotional_recovery_rate", 0.15),
      )
    else:
      s["resistance_baseline"] = None
      s["resistance_recovery_rate"] = None
      s["resistance"] = None

    s.setdefault("last_patient_resistance_update_reasoning", "")
    s.setdefault("last_patient_action", "")
    s.setdefault("last_patient_action_prior_distribution", cast(PatientActionDistribution, {}))
    s.setdefault("last_patient_action_context_distribution", cast(PatientActionDistribution, {}))
    s.setdefault("last_patient_action_distribution", cast(PatientActionDistribution, {}))
    s.setdefault("reasoning_traces", [])
    if not isinstance(s["reasoning_traces"], list):
      s["reasoning_traces"] = []

    max_message_index = max((int(message.get("utterance_index", 0)) for message in s["messages"]), default=0)
    max_trace_index = max(
      (
        int(trace.get("utterance_index", 0))
        for trace in s["reasoning_traces"]
        if isinstance(trace, dict)
      ),
      default=0,
    )
    s["utterance_index"] = max(int(s.get("utterance_index", 0)), max_message_index, max_trace_index)
    s["max_turns"] = int(s.get("max_turns", self._env_value("max_turns", 80)))
    s["min_turns"] = int(s.get("min_turns", self._env_value("min_turns", 0)))
    s.setdefault("final_patient_pending_end", False)
    s.setdefault("session_ended", False)
    s.setdefault("session_ended_reason", "")
    s.setdefault("conversation_error", "")
    return s

  def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
    if not messages:
      return "(no utterances yet)"
    return "\n".join(f"{ensure_str(m.get('role')).capitalize()}: {ensure_str(m.get('content'))}" for m in messages)

  def _format_last_utterances(self, messages: list[dict[str, Any]], limit: int = 10) -> str:
    if not messages:
      return "(no utterances yet)"
    recent = messages[-limit:]
    return "\n".join(f"{ensure_str(m.get('role')).capitalize()}: {ensure_str(m.get('content'))}" for m in recent)

  def _trace_group_for_message(self, state: dict[str, Any], message: dict[str, Any]) -> dict[str, Any] | None:
    target_index = int(message.get("utterance_index", 0))
    target_role = ensure_str(message.get("role")).strip().lower()
    for trace in state.get("reasoning_traces", []):
      if not isinstance(trace, dict):
        continue
      if int(trace.get("utterance_index", 0)) != target_index:
        continue
      if ensure_str(trace.get("speaker")).strip().lower() != target_role:
        continue
      return trace
    return None

  def _therapist_reasoning_for_message(self, state: dict[str, Any], message: dict[str, Any]) -> str:
    trace_group = self._trace_group_for_message(state, message)
    if trace_group is None:
      return ""
    traces = trace_group.get("traces")
    if not isinstance(traces, list):
      return ""
    return format_agent_trace_block([trace for trace in traces if isinstance(trace, dict)])

  def _format_agent_context_message(self, state: dict[str, Any], message: dict[str, Any]) -> str:
    role = ensure_str(message.get("role")).strip().lower()
    content = ensure_str(message.get("content"))
    if role != THERAPIST_ROLE:
      return f"{role.title() if role else 'Unknown'}: {content}"
    reasoning = self._therapist_reasoning_for_message(state, message)
    if not reasoning:
      return f"Therapist: {content}"
    return f"Therapist: <think>{reasoning}</think> {content}"

  def _format_agent_prompt_context(self, state: dict[str, Any], limit: int = 10) -> str:
    messages = state.get("messages", [])
    if not messages:
      return "none"
    return "\n".join(
      self._format_agent_context_message(state, message)
      for message in messages[-limit:]
      if isinstance(message, dict)
    )

  def _local_therapist_chatml_content(self, state: dict[str, Any], message: dict[str, Any]) -> str:
    content = ensure_str(message.get("content"))
    if ensure_str(message.get("role")).strip().lower() != THERAPIST_ROLE:
      return content
    reasoning = self._therapist_reasoning_for_message(state, message)
    if not reasoning:
      return content
    return f"<think>\n{reasoning}\n</think>\n{content}"

  def _build_local_therapist_chatml_history(self, state: dict[str, Any], limit: int = 20) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    raw_messages = state.get("messages", [])
    if not isinstance(raw_messages, list):
      return history
    for message in raw_messages[-limit:]:
      if not isinstance(message, dict):
        continue
      role = ensure_str(message.get("role")).strip().lower()
      if role == THERAPIST_ROLE:
        history.append(
          {
            "role": "assistant",
            "content": self._local_therapist_chatml_content(state, message),
          }
        )
      elif role == PATIENT_ROLE:
        history.append(
          {
            "role": "user",
            "content": self._local_therapist_chatml_content(state, message),
          }
        )
    return history

  def _build_local_therapist_request_messages(self, state: dict[str, Any]) -> list[ChatMessage]:
    return self.therapist.build_agent_chatml_messages(
      self._build_local_therapist_chatml_history(state),
      use_ft_system_prompt=True,
    )

  def _last_therapist_utterance(self, state: dict[str, Any]) -> str:
    for message in reversed(state["messages"]):
      if message.get("role") == THERAPIST_ROLE:
        return ensure_str(message.get("content"))
    return "(no therapist utterance yet)"

  def _patient_utterance_instruction(self) -> str:
    if self._resistance_enabled():
      return "Respond naturally and in character according to your current resistance level."
    return "Respond naturally and in character."

  def _patient_reasoning_for_prompt(self, reasoning: str) -> str:
    text = ensure_str(reasoning)
    if text:
      return text
    return "No explicit resistance-update reasoning was generated. Infer the impact from the last therapist utterance."

  def _build_patient_resistance_update_instruction(self, state: dict[str, Any]) -> str:
    min_delta, max_delta = self._patient_resistance_update_bounds()
    return render_prompt(
      "shared/patient_resistance_update.txt",
      resistance=f"{float(state['resistance']):.2f}",
      presenting_problem=ensure_str(state["patient_profile"].get("presenting_problem")),
      reason_for_seeking_counseling=ensure_str(state["patient_profile"].get("reason_seeking_counseling")),
      last_utterance=self._last_therapist_utterance(state),
      patient_resistance_update_min_delta=self._format_signed_delta(min_delta),
      patient_resistance_update_max_delta=self._format_signed_delta(max_delta),
    )

  def _build_patient_action_distribution_instruction(
    self,
    state: dict[str, Any],
    resistance_value: float,
    resistance_update_reasoning: str,
  ) -> str:
    return render_prompt(
      "shared/patient_action_distribution.txt",
      patient_utterance_instruction=self._patient_utterance_instruction(),
      resistance=f"{resistance_value:.2f}",
      resistance_update_reasoning=self._patient_reasoning_for_prompt(resistance_update_reasoning),
    )

  def _build_patient_utterance_core_instruction(
    self,
    resistance_update_reasoning: str,
    selected_action: PatientAction | None = None,
  ) -> str:
    if not self._resistance_enabled():
      return render_prompt(
        "shared/patient_utterance_core_weak.txt",
        patient_utterance_instruction=self._patient_utterance_instruction(),
      )
    action = normalize_patient_action(selected_action)
    return render_prompt(
      "shared/patient_utterance_core.txt",
      patient_utterance_instruction=self._patient_utterance_instruction(),
      resistance_update_reasoning=self._patient_reasoning_for_prompt(resistance_update_reasoning),
      selected_action=action or "No action override",
      selected_action_definition=action_definition_for_action(action),
    )

  def _reset_patient_action_state(self, state: dict[str, Any]) -> None:
    state["last_patient_action"] = ""
    state["last_patient_action_prior_distribution"] = cast(PatientActionDistribution, {})
    state["last_patient_action_context_distribution"] = cast(PatientActionDistribution, {})
    state["last_patient_action_distribution"] = cast(PatientActionDistribution, {})

  def _require_dict(self, payload: Any, step_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
      raise AgentConversationProcessingError(f"{step_name} returned {type(payload).__name__}, expected a JSON object.")
    return payload

  def _require_bounded_float(self, payload: dict[str, Any], key: str, step_name: str, low: float, high: float) -> float:
    if key not in payload:
      raise AgentConversationProcessingError(f"{step_name} is missing required numeric field '{key}'.")
    try:
      parsed = float(payload[key])
    except (TypeError, ValueError) as exc:
      raise AgentConversationProcessingError(f"{step_name}.{key} must be numeric, got {payload[key]!r}.") from exc
    if parsed < low or parsed > high:
      raise AgentConversationProcessingError(f"{step_name}.{key} must be in [{low}, {high}], got {parsed}.")
    return parsed

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

  def _run_patient_structured(
    self,
    state: dict[str, Any],
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: type[Any] | tuple[type[Any], ...] | None = dict,
  ) -> Any:
    try:
      parsed, _meta = self.patient.generate_structured(
        full_conversation=None,
        last_utterances=self._format_last_utterances(state["messages"]),
        resistance_level=state["resistance"],
        instruction=instruction,
        expected_type=expected_type,
      )
    except StructuredOutputError as exc:
      raise AgentConversationProcessingError(f"{step_name} failed: {exc}") from exc
    self._trace_structured_step(traces, step_name, instruction, parsed)
    return parsed

  def _select_patient_action(
    self,
    state: dict[str, Any],
    traces: list[dict[str, Any]],
    resistance_value: float,
    resistance_update_reasoning: str,
  ) -> PatientAction:
    instruction = self._build_patient_action_distribution_instruction(
      state,
      resistance_value,
      resistance_update_reasoning,
    )
    context_distribution: PatientActionDistribution | None = None
    fallback_reason = ""
    try:
      parsed, _meta = self.patient.generate_structured(
        full_conversation=None,
        last_utterances=self._format_last_utterances(state["messages"]),
        resistance_level=state["resistance"],
        instruction=instruction,
        expected_type=dict,
      )
      context_distribution = validate_action_distribution(parsed)
      if context_distribution is None:
        fallback_reason = "invalid_action_distribution_payload"
    except StructuredOutputError as exc:
      fallback_reason = f"structured_generation_failed: {exc}"

    if context_distribution is None:
      context_distribution = fallback_action_distribution_for_resistance(
        state["resistance"],
        state["utterance_index"],
      )
    else:
      context_distribution = constrain_action_distribution_for_utterance(
        context_distribution,
        state["utterance_index"],
      )

    prior_distribution = action_prior_for_resistance(
      state["resistance"],
      state["utterance_index"],
    )
    merged_distribution = merge_action_distributions(prior_distribution, context_distribution)
    merged_distribution = constrain_action_distribution_for_utterance(
      merged_distribution,
      state["utterance_index"],
      normalize=True,
    )
    selected_action = sample_patient_action(
      merged_distribution,
      utterance_index=state["utterance_index"],
    )

    state["last_patient_action"] = selected_action
    state["last_patient_action_prior_distribution"] = {
      action: float(weight) for action, weight in prior_distribution.items()
    }
    state["last_patient_action_context_distribution"] = {
      action: float(weight) for action, weight in context_distribution.items()
    }
    state["last_patient_action_distribution"] = {
      action: float(weight) for action, weight in merged_distribution.items()
    }

    trace_payload: dict[str, Any] = {
      "prior_distribution": prior_distribution,
      "context_distribution": context_distribution,
      "action_distribution": merged_distribution,
      "selected_action": selected_action,
    }
    if fallback_reason:
      trace_payload["fallback_reason"] = fallback_reason
    traces.append(
      {
        "step": "patient_action_selection",
        "input": instruction,
        "output": json.dumps(trace_payload, ensure_ascii=True),
      }
    )
    return selected_action

  def _append_trace(self, state: dict[str, Any], utterance_index: int, speaker: str, traces: list[dict[str, Any]]) -> None:
    state["reasoning_traces"].append(
      {
        "utterance_index": utterance_index,
        "speaker": speaker,
        "traces": traces,
      }
    )

  def _append_message(self, state: dict[str, Any], role: Literal["therapist", "patient"], content: str) -> int:
    state["utterance_index"] += 1
    message: dict[str, Any] = {
      "utterance_index": state["utterance_index"],
      "role": role,
      "content": ensure_str(content),
    }
    if role == PATIENT_ROLE and self._resistance_enabled():
      message["resistance"] = state["resistance"]
    state["messages"].append(message)
    return state["utterance_index"]

  def _end_session_with_error(self, state: dict[str, Any], traces: list[dict[str, Any]], error: Exception) -> None:
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
            "input": "agent_simulation",
            "output": message,
          }
        ],
      }
    )

  def _max_turns_reached(self, state: dict[str, Any]) -> bool:
    return int(state.get("utterance_index", 0)) >= int(state.get("max_turns", 0))

  def _end_for_max_turns_if_needed(self, state: dict[str, Any]) -> bool:
    if not self._max_turns_reached(state):
      return False
    state["session_ended"] = True
    if not state["session_ended_reason"]:
      state["session_ended_reason"] = "max_turns"
    return True

  def _closing_moderator_client(self) -> Any:
    client = getattr(self, "_closing_moderator_client_instance", None)
    if client is None:
      client = OpenAI(
        api_key=os.getenv("OR_API_KEY") or "",
        base_url=MODERATOR_BASE_URL,
        timeout=60,
        max_retries=0,
      )
      self._closing_moderator_client_instance = client
    return client

  def _parse_closing_moderator_response(self, value: Any) -> bool | None:
    text = ensure_str(value).strip().strip("`").strip()
    text = text.removeprefix("```").removesuffix("```").strip()
    normalized = text.rstrip(".!").strip().lower()
    if normalized == "yes":
      return True
    if normalized == "no":
      return False
    return None

  def _closing_moderator_messages(self, therapist_utterance: str) -> list[ChatMessage]:
    return [
      {"role": "system", "content": MODERATOR_SYSTEM_PROMPT},
      {
        "role": "user",
        "content": MODERATOR_USER_PROMPT_TEMPLATE.format(utterance=therapist_utterance),
      },
    ]

  def _moderator_detects_closing(self, therapist_utterance: str) -> bool:
    messages = self._closing_moderator_messages(therapist_utterance)
    for _attempt in range(MODERATOR_MAX_ATTEMPTS):
      try:
        response = self._closing_moderator_client().chat.completions.create(
          model=MODERATOR_MODEL_ID,
          messages=messages,
          temperature=0,
          max_tokens=3,
          extra_body={
            "provider": {
              "order": ["deepseek", "atlas-cloud/fp8", "novita/fp8"],
              "allow_fallbacks": False,
            }
          },
        )
        decision = self._parse_closing_moderator_response(response.choices[0].message.content)
      except Exception:
        decision = None
      if decision is not None:
        return decision
    return False

  def therapist_turn(self, state: dict[str, Any]) -> dict[str, Any]:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s
    if self._end_for_max_turns_if_needed(s):
      return s

    traces: list[dict[str, Any]] = []
    try:
      first_utterance = not s["messages"]
      if first_utterance:
        context = "opening/initial_greeting.txt"
        if self._therapist_uses_local_chatml():
          utterance, reasoning, _raw = self.therapist.generate_agent_initial_utterance(
            use_ft_system_prompt=True,
            default_max_tokens=LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
          )
        else:
          utterance, reasoning, _raw = self.therapist.generate_agent_initial_utterance()
      elif self._therapist_uses_local_chatml():
        request_messages = self._build_local_therapist_request_messages(s)
        context = json.dumps(request_messages, ensure_ascii=True)
        utterance, reasoning, _raw = self.therapist.generate_agent_messages(
          request_messages,
          default_max_tokens=LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
        )
        #breakpoint()
      else:
        context = self._format_agent_prompt_context(s)
        utterance, reasoning, _raw = self.therapist.generate_agent_utterance(context)
      requested_end = AGENT_END_TAG in utterance
      #utterance = utterance.replace(AGENT_END_TAG, "").strip()
      if not utterance:
        raise AgentConversationProcessingError("Therapist generated an empty utterance.")
      traces.append(
        {
          "step": "agent_reasoning",
          "input": "",
          "output": reasoning,
        }
      )
      idx = self._append_message(s, THERAPIST_ROLE, utterance)
      self._append_trace(s, idx, THERAPIST_ROLE, traces)
      moderator_detected_end = self._moderator_detects_closing(utterance)
      #breakpoint()
      if requested_end:
        if self._max_turns_reached(s):
          s["session_ended"] = True
          s["session_ended_reason"] = "therapist_requested_end"
        else:
          s["final_patient_pending_end"] = True
        return s
      if moderator_detected_end:
        if self._max_turns_reached(s):
          s["session_ended"] = True
        else:
          s["final_patient_pending_end"] = True
        s["session_ended_reason"] = "moderator_detected_closing"
        return s
      self._end_for_max_turns_if_needed(s)
    except AgentConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s

  def patient_turn(self, state: dict[str, Any]) -> dict[str, Any]:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s
    if self._end_for_max_turns_if_needed(s):
      return s

    traces: list[dict[str, Any]] = []
    try:
      selected_action: PatientAction | None = None
      if self._resistance_enabled():
        update_instruction = self._build_patient_resistance_update_instruction(s)
        update = self._run_patient_structured(
          s,
          traces,
          "resistance_update",
          update_instruction,
          expected_type=dict,
        )
        payload = self._require_dict(update, "resistance_update")
        min_delta, max_delta = self._patient_resistance_update_bounds()
        delta_llm = self._require_bounded_float(payload, "r_change", "resistance_update", min_delta, max_delta)
        resistance_update = apply_homeostatic_resistance_update(
          previous_r=s["resistance"],
          delta_llm=delta_llm,
          baseline=s["resistance_baseline"],
          alpha=s["resistance_recovery_rate"],
        )
        s["resistance"] = resistance_update["new_r"]
        s["last_patient_resistance_update_reasoning"] = ensure_str(payload.get("reasoning"))
        trace_payload = dict(payload)
        trace_payload.update(resistance_update)
        traces[-1]["output"] = json.dumps(trace_payload, ensure_ascii=True)

        if should_select_patient_action(s["final_patient_pending_end"]):
          selected_action = self._select_patient_action(
            s,
            traces,
            resistance_update["new_r"],
            s["last_patient_resistance_update_reasoning"],
          )
        else:
          self._reset_patient_action_state(s)
      else:
        self._reset_patient_action_state(s)
        s["last_patient_resistance_update_reasoning"] = ""

      utterance_instruction = self._build_patient_utterance_core_instruction(
        s["last_patient_resistance_update_reasoning"],
        selected_action=selected_action,
      )
      utterance = self.patient.generate_utterance(
        full_conversation=None,
        last_utterances=self._format_last_utterances(s["messages"]),
        resistance_level=s["resistance"],
        instruction=utterance_instruction,
      )
      #breakpoint()
      if not utterance:
        raise AgentConversationProcessingError("Patient generated an empty utterance.")

      patient_requested_end = AGENT_END_TAG in utterance
      idx = self._append_message(s, PATIENT_ROLE, utterance)
      self._append_trace(s, idx, PATIENT_ROLE, traces)
      if patient_requested_end:
        s["session_ended"] = True
        s["session_ended_reason"] = "patient_requested_end"
        s["final_patient_pending_end"] = False
        return s
      if s["final_patient_pending_end"]:
        s["session_ended"] = True
        if not s["session_ended_reason"]:
          s["session_ended_reason"] = "therapist_requested_end"
        s["final_patient_pending_end"] = False
        return s
      self._end_for_max_turns_if_needed(s)
    except AgentConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s
