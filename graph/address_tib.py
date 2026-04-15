from __future__ import annotations

import json
from typing import Any

from agents.interfaces import ensure_str

from .state import DEFAULT_SUBSTAGE_BY_STAGE, SessionState, Stage, SubStage
from .support import ConversationProcessingError, GraphSupportMixin


class AddressTIBStageMixin(GraphSupportMixin):
  def _build_address_tib_resume_subject(
    self,
    tib_name: str,
    tib_description: str,
    completion_note: str,
  ) -> str:
    fields: list[str] = []
    normalized_tib_name = ensure_str(tib_name).strip()
    normalized_tib_description = ensure_str(tib_description).strip()
    normalized_completion_note = ensure_str(completion_note).strip()
    if normalized_tib_name:
      fields.append(f"TIB name: {normalized_tib_name}")
    if normalized_tib_description:
      fields.append(f"TIB description: {normalized_tib_description}")
    if normalized_completion_note:
      fields.append(f"Completion note: {normalized_completion_note}")
    return " ".join(fields)

  def _finish_address_tib(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    exit_reason: str,
    completion_reason: str,
  ) -> None:
    tib_name = ensure_str(state["address_tib_name"])
    tib_description = ensure_str(state["address_tib_description"])
    route_bucket = state["address_tib_route_bucket"]
    resume_stage = state["address_tib_resume_stage"] or "id_distortions"
    resume_substage = state["address_tib_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
    completion_note = completion_reason or "Addressed in the allotted single exchange turn."
    subject = self._build_address_tib_resume_subject(tib_name, tib_description, completion_note)
    band = self._current_band(state["predicted_resistance"], state)

    self._remember_completed_address_tib(state, route_bucket, tib_name)
    self._set_resume_context(state, "address_tib", exit_reason, subject)
    self._clear_address_tib_state(state, clear_resume_cursor=True)

    routed_stage = resume_stage
    routed_substage = resume_substage
    self._set_stage(state, resume_stage, resume_substage)
    self._maybe_enter_resistance_handling_from_cbt(
      state,
      traces,
      "Resuming after address_tib with persistent precontemplation resistance.",
    )
    routed_stage = state["current_stage"]
    routed_substage = state["current_substage"]
    if state["current_stage"] == resume_stage and resume_stage not in {"opening", "id_distortions", "cbt_technique"}:
      self._clear_resume_context(state)

    traces.append(
      {
        "step": "address_tib_exit",
        "input": tib_description,
        "output": json.dumps(
          {
            "tib_name": tib_name,
            "exit_reason": exit_reason,
            "completion_note": completion_note,
            "predicted_resistance": state["predicted_resistance"],
            "band": band,
            "resume_stage": resume_stage,
            "resume_substage": resume_substage,
            "routed_stage": routed_stage,
            "routed_substage": routed_substage,
            "last_completed_address_tib_name": state["last_completed_address_tib_name"],
            "last_completed_address_tib_no_reroute_active": state["last_completed_address_tib_no_reroute_active"],
            "resume_context_active": state["resume_context_active"],
            "resume_context_subject": state["resume_context_subject"],
          },
          ensure_ascii=True,
        ),
      }
    )

  def _route_after_address_tib(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    exit_reason: str,
    completion_reason: str,
  ) -> str:
    self._finish_address_tib(state, traces, exit_reason, completion_reason)
    return self._handle_stage_dispatch(state, traces)

  def _generate_address_tib_plan(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
  ) -> None:
    instruction = self._load_stage_prompt("address_tib", "plan_generation")
    parsed = self._run_therapist_structured(state, traces, "address_tib_plan_generation", instruction)
    payload = self._require_dict(parsed, "address_tib_plan_generation")
    plan = self._require_non_empty_string(payload, "tib_plan", "address_tib_plan_generation")
    state["address_tib_plan"] = plan
    self._set_last_trace_output(traces, "address_tib_plan_generation", {"tib_plan": plan})

  def _enter_address_tib(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    route_bucket: str,
    tib_name: str,
    tib_description: str,
    reasoning: str,
    resume_stage: Stage,
    resume_substage: SubStage,
  ) -> None:
    state["address_tib_route_bucket"] = route_bucket
    state["address_tib_name"] = tib_name
    state["address_tib_description"] = tib_description
    state["address_tib_reasoning"] = reasoning
    state["address_tib_plan"] = ""
    state["address_tib_resume_stage"] = resume_stage
    state["address_tib_resume_substage"] = resume_substage
    self._clear_inline_tib_ack(state)
    self._set_stage(state, "address_tib", "tib_plan")
    traces.append(
      {
        "step": "address_tib_enter",
        "input": tib_description,
        "output": json.dumps(
          {
            "route_bucket": route_bucket,
            "tib_name": tib_name,
            "resume_stage": resume_stage,
            "resume_substage": resume_substage,
            "reasoning": reasoning,
          },
          ensure_ascii=True,
        ),
      }
    )

  def _replace_active_address_tib(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    route_bucket: str,
    tib_name: str,
    tib_description: str,
    reasoning: str,
  ) -> None:
    state["address_tib_route_bucket"] = route_bucket
    state["address_tib_name"] = tib_name
    state["address_tib_description"] = tib_description
    state["address_tib_reasoning"] = reasoning
    state["address_tib_plan"] = ""
    self._clear_inline_tib_ack(state)
    self._set_substage(state, "tib_plan")
    traces.append(
      {
        "step": "address_tib_replaced",
        "input": tib_description,
        "output": json.dumps(
          {
            "route_bucket": route_bucket,
            "tib_name": tib_name,
            "resume_stage": state["address_tib_resume_stage"],
            "resume_substage": state["address_tib_resume_substage"],
            "reasoning": reasoning,
          },
          ensure_ascii=True,
        ),
      }
    )

  def _maybe_complete_finished_address_tib(self, state: SessionState, traces: list[dict[str, Any]]) -> bool:
    if state["current_stage"] != "address_tib":
      return False
    if self._current_stage_exchange_count(state, "address_tib") <= 0:
      return False

    self._finish_address_tib(
      state,
      traces,
      "single_turn_complete",
      "Addressed in the allotted single exchange turn.",
    )
    return True

  def _handle_address_tib(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    tib_description = ensure_str(state["address_tib_description"])
    if not tib_description:
      raise ConversationProcessingError("address_tib stage is active without address_tib_description.")

    turns = self._current_stage_exchange_count(state, "address_tib")
    if turns > 0:
      return self._route_after_address_tib(
        state,
        traces,
        "single_turn_complete",
        "Addressed in the allotted single exchange turn.",
      )

    if state["current_substage"] == "tib_plan" or not state["address_tib_plan"]:
      self._generate_address_tib_plan(state, traces)
      self._set_substage(state, "tib_deliver")
    else:
      self._set_substage(state, "tib_deliver")

    instruction = self._load_stage_prompt("address_tib", "deliver_utterance")
    return self._generate_therapist_utterance(state, instruction)
