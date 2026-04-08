from __future__ import annotations

import json
from typing import Any

from agents.interfaces import ensure_str

from .state import DEFAULT_SUBSTAGE_BY_STAGE, SessionState, Stage, SubStage
from .support import ConversationProcessingError, GraphSupportMixin
from .closing import ClosingStageMixin

PATIENT_DEMAND_MAX_TURNS = 5


class PatientDemandStageMixin(GraphSupportMixin):

  def _generate_patient_demand_plan(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    demand: str,
  ) -> None:
    instruction = self._load_stage_prompt("patient_demand", "plan_generation")
    parsed = self._run_therapist_structured(state, traces, "patient_demand_plan_generation", instruction)
    payload = self._require_dict(parsed, "patient_demand_plan_generation")
    plan = self._require_non_empty_string(payload, "demand_plan", "patient_demand_plan_generation")
    state["patient_demand_plan"] = plan
    self._set_last_trace_output(traces, "patient_demand_plan_generation", {"demand_plan": plan})

  def _update_patient_demand_plan_if_needed(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    turns: int,
  ) -> None:
    instruction = self._load_stage_prompt("patient_demand", "plan_update")
    parsed = self._run_therapist_structured(state, traces, "patient_demand_plan_update", instruction)
    payload = self._require_dict(parsed, "patient_demand_plan_update")
    updated = self._require_bool(payload, "updated", "patient_demand_plan_update")
    reasoning = self._require_non_empty_string(payload, "reasoning", "patient_demand_plan_update")
    trace_payload: dict[str, Any] = {
      "updated": updated,
      "demand_plan": "none",
      "reasoning": reasoning,
      "turn_count": turns,
      "turn_cap": PATIENT_DEMAND_MAX_TURNS,
      "resume_stage": state["patient_demand_resume_stage"],
      "resume_substage": state["patient_demand_resume_substage"],
    }
    if updated:
      raw_plan = ensure_str(payload.get("demand_plan")).strip()
      if not raw_plan or raw_plan.lower() == "none":
        raise ConversationProcessingError(
          "patient_demand_plan_update.demand_plan must be provided when updated=true."
        )
      plan = self._require_non_empty_string(payload, "demand_plan", "patient_demand_plan_update")
      state["patient_demand_plan"] = plan
      trace_payload["demand_plan"] = plan
    self._set_last_trace_output(traces, "patient_demand_plan_update", trace_payload)

  def _enter_patient_demand(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    demand: str,
    reasoning: str,
    resume_stage: Stage,
    resume_substage: SubStage,
    *,
    end_session_attempt_active: bool = False,
  ) -> None:
    state["patient_demand_text"] = demand
    state["patient_demand_plan"] = ""
    state["patient_demand_reasoning"] = reasoning
    state["patient_demand_resume_stage"] = resume_stage
    state["patient_demand_resume_substage"] = resume_substage
    state["patient_demand_end_session_attempt_active"] = end_session_attempt_active
    if end_session_attempt_active:
      state["mi_end_session_demand_redirect_used"] = True
    self._set_stage(state, "patient_demand", "demand_plan")
    traces.append(
      {
        "step": "patient_demand_enter",
        "input": demand,
        "output": json.dumps(
          {
            "resume_stage": resume_stage,
            "resume_substage": resume_substage,
            "reasoning": reasoning,
            "end_session_attempt_active": end_session_attempt_active,
          },
          ensure_ascii=True,
        ),
      }
    )

  def _replace_active_patient_demand(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    demand: str,
    reasoning: str,
  ) -> None:
    state["patient_demand_text"] = demand
    state["patient_demand_plan"] = ""
    state["patient_demand_reasoning"] = reasoning
    state["patient_demand_end_session_attempt_active"] = False
    self._set_substage(state, "demand_plan")
    traces.append(
      {
        "step": "patient_demand_replaced",
        "input": demand,
        "output": json.dumps(
          {
            "resume_stage": state["patient_demand_resume_stage"],
            "resume_substage": state["patient_demand_resume_substage"],
            "reasoning": reasoning,
            "end_session_attempt_active": False,
          },
          ensure_ascii=True,
        ),
      }
    )

  def _resume_from_patient_demand(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    exit_reason: str,
  ) -> str:
    demand = state["patient_demand_text"]
    resume_stage = state["patient_demand_resume_stage"] or "id_distortions"
    resume_substage = state["patient_demand_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
    instruction = self._load_stage_prompt(
      "patient_demand",
      "resume_utterance",
      exit_reason=exit_reason,
      resume_stage=resume_stage,
      resume_substage=resume_substage,
    )
    utterance = self._generate_therapist_utterance(state, instruction)
    traces.append(
      {
        "step": "patient_demand_exit",
        "input": demand,
        "output": json.dumps(
          {
            "exit_reason": exit_reason,
            "resume_stage": resume_stage,
            "resume_substage": resume_substage,
          },
          ensure_ascii=True,
        ),
      }
    )
    self._set_stage(state, resume_stage, resume_substage)
    self._clear_patient_demand_state(state, clear_resume_cursor=True)
    return utterance

  def _exit_patient_demand_to_closing_fallback(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    exit_reason: str,
  ) -> str:
    demand = state["patient_demand_text"]
    resume_stage = state["patient_demand_resume_stage"] or "id_distortions"
    resume_substage = state["patient_demand_resume_substage"] or DEFAULT_SUBSTAGE_BY_STAGE[resume_stage]
    traces.append(
      {
        "step": "patient_demand_exit_to_closing_fallback",
        "input": demand,
        "output": json.dumps(
          {
            "exit_reason": exit_reason,
            "resume_stage": resume_stage,
            "resume_substage": resume_substage,
          },
          ensure_ascii=True,
        ),
      }
    )
    self._clear_patient_demand_state(state, clear_resume_cursor=True)
    state["mi_active"] = False
    self._set_stage(state, "closing", "closing_fallback")
    return self._handle_closing(state, traces)

  def _handle_patient_demand(self, state: SessionState, traces: list[dict[str, Any]]) -> str:
    demand = ensure_str(state["patient_demand_text"])
    if not demand:
      raise ConversationProcessingError("patient_demand stage is active without patient_demand_text.")

    turns = self._current_stage_exchange_count(state, "patient_demand")

    if state["current_substage"] == "demand_plan" or not state["patient_demand_plan"]:
      self._generate_patient_demand_plan(state, traces, demand)
      self._set_substage(state, "demand_deliver")
    elif turns > 0:
      self._set_substage(state, "demand_check")
      instruction = self._load_stage_prompt("patient_demand", "completion_check")
      parsed = self._run_therapist_structured(state, traces, "patient_demand_completion_check", instruction)
      payload = self._require_dict(parsed, "patient_demand_completion_check")
      covered = self._require_bool(payload, "covered", "patient_demand_completion_check")
      covered_reasoning = self._require_non_empty_string(payload, "reasoning", "patient_demand_completion_check")
      print(f"<DEBUG_PATIENT_DEMAND_COMPLETION_CHECK>covered={covered}, reasoning={covered_reasoning}</DEBUG_PATIENT_DEMAND_COMPLETION_CHECK>") #devlog
      completion_payload = dict(payload)
      completion_payload.update(
        {
          "turn_count": turns,
          "turn_cap": PATIENT_DEMAND_MAX_TURNS,
          "resume_stage": state["patient_demand_resume_stage"],
          "resume_substage": state["patient_demand_resume_substage"],
        }
      )
      self._set_last_trace_output(traces, "patient_demand_completion_check", completion_payload)
      if covered:
        self._set_substage(state, "demand_resume")
        return self._resume_from_patient_demand(state, traces, "covered")
      if turns >= PATIENT_DEMAND_MAX_TURNS:
        traces.append(
          {
            "step": "patient_demand_max_turn_cap",
            "input": demand,
            "output": json.dumps(
              {
                "turn_count": turns,
                "turn_cap": PATIENT_DEMAND_MAX_TURNS,
                "resume_stage": state["patient_demand_resume_stage"],
                "resume_substage": state["patient_demand_resume_substage"],
              },
              ensure_ascii=True,
            ),
          }
        )
        if state["patient_demand_end_session_attempt_active"]:
          return self._exit_patient_demand_to_closing_fallback(state, traces, "max_turn_cap")
        self._set_substage(state, "demand_resume")
        return self._resume_from_patient_demand(state, traces, "max_turn_cap")
      self._update_patient_demand_plan_if_needed(state, traces, turns)
      self._set_substage(state, "demand_deliver")
    else:
      self._set_substage(state, "demand_deliver")

    print(f"<DEBUG_PLAN>{state['patient_demand_plan']}</DEBUG_PLAN>") #devlog
    instruction = self._load_stage_prompt("patient_demand", "deliver_utterance")
    return self._generate_therapist_utterance(state, instruction)
