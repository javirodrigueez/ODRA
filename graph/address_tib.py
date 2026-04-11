from __future__ import annotations

import json
from typing import Any

from agents.interfaces import ensure_str

from .state import DEFAULT_SUBSTAGE_BY_STAGE, SessionState, Stage, SubStage
from .support import ConversationProcessingError, GraphSupportMixin

ADDRESS_TIB_MAX_TURNS = 5


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

  def _route_after_address_tib(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    exit_reason: str,
    completion_reason: str,
  ) -> str:
    tib_name = ensure_str(state["address_tib_name"])
    tib_description = ensure_str(state["address_tib_description"])
    resume_stage = state["address_tib_resume_stage"] or "id_distortions"
    resume_substage = state["address_tib_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
    completion_note = completion_reason
    if exit_reason == "max_turn_cap":
      completion_note = "Maximum turns for addressing the TIB were used."
    subject = self._build_address_tib_resume_subject(tib_name, tib_description, completion_note)
    band = self._current_band(state["predicted_resistance"], state)

    self._set_resume_context(state, "address_tib", exit_reason, subject)
    self._clear_address_tib_state(state, clear_resume_cursor=True)

    routed_stage = resume_stage
    routed_substage = resume_substage
    if band == "contemplation":
      self._enter_mi(
        state,
        "mi_techniques",
        resume_stage,
        resume_substage,
        "Resuming through MI after the address_tib detour due to contemplation-level resistance.",
      )
      routed_stage = state["current_stage"]
      routed_substage = state["current_substage"]
    elif band == "precontemplation":
      self._enter_mi(
        state,
        "mi_topic_exploration",
        resume_stage,
        resume_substage,
        "Resuming through MI after the address_tib detour due to precontemplation-level resistance.",
      )
      routed_stage = state["current_stage"]
      routed_substage = state["current_substage"]
    else:
      self._set_stage(state, resume_stage, resume_substage)
      if resume_stage not in {"opening", "id_distortions", "cbt_technique"}:
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
            "resume_context_active": state["resume_context_active"],
            "resume_context_subject": state["resume_context_subject"],
          },
          ensure_ascii=True,
        ),
      }
    )
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

  def _update_address_tib_plan_if_needed(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    turns: int,
  ) -> None:
    instruction = self._load_stage_prompt("address_tib", "plan_update")
    parsed = self._run_therapist_structured(state, traces, "address_tib_plan_update", instruction)
    payload = self._require_dict(parsed, "address_tib_plan_update")
    updated = self._require_bool(payload, "updated", "address_tib_plan_update")
    reasoning = self._require_non_empty_string(payload, "reasoning", "address_tib_plan_update")
    trace_payload: dict[str, Any] = {
      "updated": updated,
      "tib_plan": "none",
      "reasoning": reasoning,
      "turn_count": turns,
      "turn_cap": ADDRESS_TIB_MAX_TURNS,
      "resume_stage": state["address_tib_resume_stage"],
      "resume_substage": state["address_tib_resume_substage"],
    }
    if updated:
      raw_plan = ensure_str(payload.get("tib_plan")).strip()
      if not raw_plan or raw_plan.lower() == "none":
        raise ConversationProcessingError(
          "address_tib_plan_update.tib_plan must be provided when updated=true."
        )
      plan = self._require_non_empty_string(payload, "tib_plan", "address_tib_plan_update")
      state["address_tib_plan"] = plan
      trace_payload["tib_plan"] = plan
    self._set_last_trace_output(traces, "address_tib_plan_update", trace_payload)

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

  def _handle_address_tib(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    tib_description = ensure_str(state["address_tib_description"])
    if not tib_description:
      raise ConversationProcessingError("address_tib stage is active without address_tib_description.")

    turns = self._current_stage_exchange_count(state, "address_tib")

    if state["current_substage"] == "tib_plan" or not state["address_tib_plan"]:
      self._generate_address_tib_plan(state, traces)
      self._set_substage(state, "tib_deliver")
    elif turns > 0:
      self._set_substage(state, "tib_check")
      instruction = self._load_stage_prompt("address_tib", "completion_check")
      parsed = self._run_therapist_structured(state, traces, "address_tib_completion_check", instruction)
      payload = self._require_dict(parsed, "address_tib_completion_check")
      covered = self._require_bool(payload, "covered", "address_tib_completion_check")
      covered_reasoning = self._require_non_empty_string(payload, "reasoning", "address_tib_completion_check")
      completion_payload = dict(payload)
      completion_payload.update(
        {
          "route_bucket": state["address_tib_route_bucket"],
          "tib_name": state["address_tib_name"],
          "turn_count": turns,
          "turn_cap": ADDRESS_TIB_MAX_TURNS,
          "resume_stage": state["address_tib_resume_stage"],
          "resume_substage": state["address_tib_resume_substage"],
        }
      )
      self._set_last_trace_output(traces, "address_tib_completion_check", completion_payload)
      if covered:
        self._set_substage(state, "tib_resume")
        return self._route_after_address_tib(state, traces, "covered", covered_reasoning)
      if turns >= ADDRESS_TIB_MAX_TURNS:
        traces.append(
          {
            "step": "address_tib_max_turn_cap",
            "input": tib_description,
            "output": json.dumps(
              {
                "route_bucket": state["address_tib_route_bucket"],
                "tib_name": state["address_tib_name"],
                "turn_count": turns,
                "turn_cap": ADDRESS_TIB_MAX_TURNS,
                "resume_stage": state["address_tib_resume_stage"],
                "resume_substage": state["address_tib_resume_substage"],
              },
              ensure_ascii=True,
            ),
          }
        )
        self._set_substage(state, "tib_resume")
        return self._route_after_address_tib(state, traces, "max_turn_cap", covered_reasoning)
      self._update_address_tib_plan_if_needed(state, traces, turns)
      self._set_substage(state, "tib_deliver")
    else:
      self._set_substage(state, "tib_deliver")

    instruction = self._load_stage_prompt("address_tib", "deliver_utterance")
    return self._generate_therapist_utterance(state, instruction)
