from __future__ import annotations

from typing import Literal, TypedDict

Stage = Literal[
  "opening",
  "mi_topic_exploration",
  "mi_techniques",
  "meta_conversation",
  "id_distortions",
  "cbt_technique",
  "homework",
  "closing",
  "anticipated_closing",
  "forced_closing",
]

SubStage = Literal[
  "open_initial",
  "open_mood",
  "open_problem",
  "mi_choose_topic",
  "mi_raise_topic",
  "mi_choose_technique",
  "mi_raise_technique",
  "mi_meta_open",
  "mi_meta_probe",
  "mi_meta_adjust",
  "mi_meta_close",
  "mi_end_session",
  "cognitive_bootstrap",
  "cognitive_review_active_question",
  "cognitive_assign_core_belief",
  "cognitive_assign_distortion",
  "cognitive_commit_set",
  "cognitive_check_completion",
  "cbt_select_techniques",
  "cbt_review_current",
  "cbt_decide_next",
  "cbt_validate_fit",
  "cbt_plan_current",
  "cbt_deliver_current",
  "homework_check_existing",
  "homework_review_response",
  "homework_apply_fallback",
  "homework_propose_tasks",
  "closing_share_summary",
  "closing_share_plan",
  "closing_review_plan",
  "closing_review_feedback",
  "closing_finalize_feedback",
  "closing_end_session",
  "forced_closing_end",
]

CBT_STAGES = {"id_distortions", "cbt_technique", "homework"}
MI_STAGES = {"mi_topic_exploration", "mi_techniques", "meta_conversation"}
CORE_BELIEF_KEYS = ["helplessness", "unlovability", "worthlessness"]
DISTORTION_KEYS = [
  "mental_filtering",
  "fortune_telling",
  "mind_reading",
  "personalization",
  "catastrophizing",
  "all_or_nothing_thinking",
  "should_statements",
  "overgeneralization",
  "labeling_and_mislabeling",
  "discounting_the_positive",
]
MI_TOPICS = ["Health", "Economy", "Interpersonal Relationships", "Law", "Education"]

DEFAULT_SUBSTAGE_BY_STAGE: dict[Stage, SubStage] = {
  "opening": "open_initial",
  "mi_topic_exploration": "mi_choose_topic",
  "mi_techniques": "mi_choose_technique",
  "meta_conversation": "mi_meta_open",
  "id_distortions": "cognitive_review_active_question",
  "cbt_technique": "cbt_select_techniques",
  "homework": "homework_check_existing",
  "closing": "closing_share_summary",
  "anticipated_closing": "mi_end_session",
  "forced_closing": "forced_closing_end",
}


class SessionState(TypedDict, total=False):
  current_stage: Stage
  current_substage: SubStage
  messages: list[dict]

  patient_profile: dict
  resistance: float
  predicted_resistance: float

  opening_patient_mood: str
  opening_patient_problem: str
  last_reasoning: str

  cognitive_model_sets: list[dict]
  working_cognitive_set: dict
  working_cognitive_summary: str
  cognitive_gap_plan: str
  active_cognitive_target: str
  active_cognitive_question: str
  active_cognitive_follow_up_attempts: int

  selected_techniques: list[dict]
  current_technique_index: int
  current_technique_plan: str
  current_technique_expected_outcome: str
  current_technique_current_progress: str
  technique_evaluations: list[dict]
  techniques_completed: int
  technique_turn_count: int
  technique_slot_replaced: dict

  mi_active: bool
  mi_context: list[dict]
  cbt_flow_saved_state: str
  cbt_resume_stage: str
  mi_topics_tried: list[str]
  mi_techniques_used: list[str]
  mi_techniques_paused_until_precontemplation: bool
  current_mi_topic: str
  last_mi_predicted_resistance: float

  proposed_tasks: list[dict]
  homework_proposal_attempts: int
  homework_waiting_for_patient: bool
  homework_fallback_done: bool
  agreed_tasks: list[dict]
  hw_reasoning: str
  homework_fallback_to_apply: bool

  session_summary: str
  next_session_plan: str
  patient_feedback: str
  closing_plan_refinements: int

  reasoning_traces: list[dict]
  utterance_index: int
  exchange_counts: dict
  active_exchange_stage: str

  final_patient_pending_end: bool
  session_ended: bool
  session_ended_reason: str
  conversation_error: str
  opening_fallback_triggered: bool

  max_turns: int
  min_turns: int
