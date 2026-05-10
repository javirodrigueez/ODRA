from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from agents.interfaces import ChatMessage, ensure_str, parse_json_response
from agents.therapist import (
  AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
  LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
)

from .graph import TherapyGraphBuilder
from .state import SessionState
from .support import ConversationProcessingError, record_large_prompt_request


@dataclass
class ParsedReasoningTrace:
  step: str
  payload: Any
  raw: Any
  line_number: int


class AgentOrchestratorTraceError(ConversationProcessingError):
  pass


class AgentTherapyGraphWithOrchBuilder(TherapyGraphBuilder):
  """TherapyGraphBuilder variant driven by therapist-agent JSONL reasoning traces."""

  MAX_TRACE_REPAIR_ATTEMPTS = 3

  def _model_type(self, block: Any) -> str:
    if isinstance(block, dict):
      return ensure_str(block.get("type")).strip().lower()
    return ensure_str(getattr(block, "type", "")).strip().lower()

  def _therapist_uses_local_chatml(self) -> bool:
    return self._model_type(getattr(self.cfg, "therapist", {})) == "local"

  def _format_agent_history_message(self, message: dict[str, Any]) -> ChatMessage:
    role = ensure_str(message.get("role")).strip().lower()
    label = "Therapist" if role == "therapist" else "Patient"
    return {"role": "user", "content": f"{label}: {ensure_str(message.get('content'))}"}

  def _build_agent_orch_history(self, state: SessionState) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for message in state.get("messages", []):
      if not isinstance(message, dict):
        continue
      role = ensure_str(message.get("role")).strip().lower()
      if role not in {"therapist", "patient"}:
        continue
      history.append(self._format_agent_history_message(message))
    return history

  def _agent_generation_default_tokens(self) -> int:
    if self._therapist_uses_local_chatml():
      return LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS
    return AGENT_THERAPIST_DEFAULT_MAX_TOKENS

  def _parse_agent_reasoning_jsonl(self, reasoning: str) -> tuple[dict[str, list[ParsedReasoningTrace]], list[dict[str, Any]]]:
    traces_by_step: dict[str, list[ParsedReasoningTrace]] = {}
    parse_errors: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(ensure_str(reasoning).splitlines(), start=1):
      line = raw_line.strip()
      if not line or line in {"```", "```json", "```jsonl"}:
        continue
      try:
        parsed = json.loads(line)
      except json.JSONDecodeError as exc:
        parse_errors.append({"line_number": line_number, "line": line, "error": str(exc)})
        continue
      if not isinstance(parsed, dict) or len(parsed) != 1:
        parse_errors.append(
          {
            "line_number": line_number,
            "line": line,
            "error": "reasoning trace line must be a JSON object with exactly one top-level key",
          }
        )
        continue
      step, payload = next(iter(parsed.items()))
      step_name = ensure_str(step).strip()
      if not step_name:
        parse_errors.append({"line_number": line_number, "line": line, "error": "empty trace step name"})
        continue
      trace = ParsedReasoningTrace(step=step_name, payload=payload, raw=parsed, line_number=line_number)
      traces_by_step.setdefault(step_name, []).append(trace)
    return traces_by_step, parse_errors

  def _begin_agent_orch_turn(self, state: SessionState) -> None:
    history = self._build_agent_orch_history(state)
    request_messages = self.therapist.build_agent_chatml_messages(
      history,
      use_ft_system_prompt=True,
    )
    record_large_prompt_request(state, request_messages, "therapist", "agent_orch_therapist_turn")
    utterance, reasoning, raw = self.therapist.generate_agent_messages(
      request_messages,
      default_max_tokens=self._agent_generation_default_tokens(),
    )
    if not utterance:
      raise AgentOrchestratorTraceError("Therapist generated an empty utterance.")
    traces_by_step, parse_errors = self._parse_agent_reasoning_jsonl(reasoning)
    self._agent_orch_utterance = utterance
    self._agent_orch_raw_response = raw
    self._agent_orch_reasoning = reasoning
    self._agent_orch_traces_by_step = traces_by_step
    self._agent_orch_parse_errors = parse_errors
    self._agent_orch_consumed_steps: dict[str, int] = {}

  def _clear_agent_orch_turn(self) -> None:
    self._agent_orch_utterance = ""
    self._agent_orch_raw_response = ""
    self._agent_orch_reasoning = ""
    self._agent_orch_traces_by_step = {}
    self._agent_orch_parse_errors = []
    self._agent_orch_consumed_steps = {}

  def therapist_turn(self, state: SessionState) -> SessionState:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s
    try:
      self._begin_agent_orch_turn(s)
    except ConversationProcessingError as exc:
      self._end_session_with_error(s, [], exc)
      return s
    return super().therapist_turn(s)

  def _coerce_expected_payload(
    self,
    step_name: str,
    payload: Any,
    expected_type: Optional[type | tuple[type, ...]],
  ) -> Any:
    if expected_type is None or isinstance(payload, expected_type):
      return payload
    if expected_type is str and isinstance(payload, dict):
      for key in ("evaluation", "message", "text", "value"):
        value = payload.get(key)
        if isinstance(value, str):
          return value
      if len(payload) == 1:
        value = next(iter(payload.values()))
        if isinstance(value, str):
          return value
    expected_name = getattr(expected_type, "__name__", str(expected_type))
    raise AgentOrchestratorTraceError(
      f"{step_name} reasoning trace returned {type(payload).__name__}, expected {expected_name}."
    )

  def _next_cached_trace(self, step_name: str) -> ParsedReasoningTrace | None:
    traces = self._agent_orch_traces_by_step.get(step_name, [])
    consumed = self._agent_orch_consumed_steps.get(step_name, 0)
    if consumed >= len(traces):
      return None
    self._agent_orch_consumed_steps[step_name] = consumed + 1
    return traces[consumed]

  def _parse_repaired_trace(self, step_name: str, raw: str) -> Any:
    text = ensure_str(raw).strip()
    if "</think>" in text:
      text = text.split("</think>")[-1].strip()
    parsed, _meta = parse_json_response(text)
    if isinstance(parsed, dict) and len(parsed) == 1 and step_name in parsed:
      return parsed[step_name]
    if step_name == "technique_evaluation" and isinstance(parsed, str):
      return parsed
    if isinstance(parsed, dict):
      return parsed
    raise AgentOrchestratorTraceError(f"repair for {step_name} did not return a usable JSON object.")

  def _repair_trace_payload(
    self,
    state: SessionState,
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]],
    error: str,
    original: Any,
  ) -> tuple[Any, dict[str, Any]]:
    repair_errors: list[str] = []
    payload: Any = None
    for attempt in range(1, self.MAX_TRACE_REPAIR_ATTEMPTS + 1):
      messages: list[ChatMessage] = [
        {
          "role": "system",
          "content": (
            "You repair one CBT therapist reasoning trace. Return only valid JSON without markdown fences."
          ),
        },
        {
          "role": "user",
          "content": (
            f"Trace step name: {step_name}\n"
            f"Original trace or missing marker:\n{json.dumps(original, ensure_ascii=True)}\n\n"
            f"Problem: {error}\n\n"
            f"Structured instruction that this trace must satisfy:\n{instruction}\n\n"
            "Return exactly one JSON object. Prefer this shape:\n"
            f"{{\"{step_name}\": <corrected trace payload>}}\n"
            "For string-valued traces, the payload may be a string."
          ),
        },
      ]
      try:
        raw = self.therapist.llm.generate(
          messages,
          max_tokens=2048,
          temperature=0.0,
          strip_thinking=True,
        )
        payload = self._parse_repaired_trace(step_name, raw)
        payload = self._coerce_expected_payload(step_name, payload, expected_type)
        return payload, {
          "attempts": attempt,
          "original": original,
          "initial_error": error,
          "repair_errors": repair_errors,
        }
      except Exception as exc:  # noqa: BLE001 - repair errors are recorded for traceability.
        repair_errors.append(str(exc))
    raise AgentOrchestratorTraceError(
      f"{step_name} reasoning trace repair failed after {self.MAX_TRACE_REPAIR_ATTEMPTS} attempts: {repair_errors[-1] if repair_errors else error}"
    )

  def _consume_or_repair_trace(
    self,
    state: SessionState,
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]],
  ) -> tuple[Any, dict[str, Any] | None]:
    trace = self._next_cached_trace(step_name)
    if trace is None:
      original: Any = {"missing_step": step_name, "parse_errors": self._agent_orch_parse_errors}
      error = f"missing required reasoning trace {step_name!r}"
    else:
      original = trace.raw
      try:
        return self._coerce_expected_payload(step_name, trace.payload, expected_type), None
      except AgentOrchestratorTraceError as exc:
        error = str(exc)
    payload, repair_meta = self._repair_trace_payload(
      state,
      step_name,
      instruction,
      expected_type,
      error,
      original,
    )
    return payload, repair_meta

  def _trace_agent_orch_step(
    self,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    parsed: Any,
    repair_meta: dict[str, Any] | None,
  ) -> None:
    trace: dict[str, Any] = {
      "step": step_name,
      "input": instruction,
      "output": json.dumps(parsed, ensure_ascii=True),
      "source": "agent_reasoning",
    }
    if repair_meta is not None:
      trace["repair"] = repair_meta
    traces.append(trace)

  def _run_therapist_structured(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: Optional[type | tuple[type, ...]] = dict,
    temperature: float | None = None,
  ) -> Any:
    del temperature
    parsed, repair_meta = self._consume_or_repair_trace(state, step_name, instruction, expected_type)
    if isinstance(parsed, dict):
      reasoning = ensure_str(parsed.get("reasoning"))
      if reasoning:
        state["last_reasoning"] = reasoning
    self._trace_agent_orch_step(traces, step_name, instruction, parsed, repair_meta)
    return parsed

  def _generate_therapist_utterance(self, state: SessionState, instruction: str) -> str:
    tib_adjusted_instruction = self._instruction_with_tib_redirect(state, instruction)
    resume_adjusted_instruction, used_resume_context = self._instruction_with_resume_context(state, tib_adjusted_instruction)
    inline_tib_adjusted_instruction, used_inline_tib_ack = self._instruction_with_inline_tib_ack(
      state,
      resume_adjusted_instruction,
    )
    _enriched_instruction, used_long_session_timeout = self._instruction_with_long_session_timeout(
      state,
      inline_tib_adjusted_instruction,
    )
    if used_long_session_timeout and state.get("long_session_timeout_instruction_pending"):
      state["long_session_timeout_instruction_pending"] = False
    if used_inline_tib_ack or state.get("inline_tib_ack_name"):
      self._clear_inline_tib_ack(state)
    if used_resume_context or state.get("resume_context_active"):
      self._clear_resume_context(state)
    return ensure_str(getattr(self, "_agent_orch_utterance", ""))
