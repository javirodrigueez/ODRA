from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

Stage = Literal[
  "opening",
  "mi_topic_exploration",
  "mi_techniques",
  "meta_conversation",
  "patient_demand",
  "id_distortions",
  "cbt_technique",
  "homework",
  "closing",
  "anticipated_closing",
  "forced_closing",
]

ConversationRole = Literal["system", "system_note", "user", "assistant", "therapist", "patient"]

SubStage = Literal[
  "open_initial",
  "open_mood",
  "open_problem",
  "mi_choose_topic",
  "mi_raise_topic",
  "mi_choose_technique",
  "mi_raise_technique",
  "mi_transition_to_cbt",
  "mi_meta_open",
  "mi_meta_probe",
  "mi_meta_adjust",
  "mi_meta_close",
  "mi_end_session",
  "demand_plan",
  "demand_deliver",
  "demand_check",
  "demand_resume",
  "cognitive_bootstrap",
  "cognitive_review_active_question",
  "cognitive_refinement",
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
  "homework_end_fallback",
  "homework_propose_tasks",
  "homework_transition_to_closing",
  "closing_share_summary",
  "closing_share_plan",
  "closing_review_plan",
  "closing_review_feedback",
  "closing_finalize_feedback",
  "closing_fallback",
  "closing_end_session",
  "forced_closing_end",
]

ALL_SUBSTAGES: tuple[SubStage, ...] = (
  "open_initial",
  "open_mood",
  "open_problem",
  "mi_choose_topic",
  "mi_raise_topic",
  "mi_choose_technique",
  "mi_raise_technique",
  "mi_transition_to_cbt",
  "mi_meta_open",
  "mi_meta_probe",
  "mi_meta_adjust",
  "mi_meta_close",
  "mi_end_session",
  "demand_plan",
  "demand_deliver",
  "demand_check",
  "demand_resume",
  "cognitive_bootstrap",
  "cognitive_review_active_question",
  "cognitive_refinement",
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
  "homework_end_fallback",
  "homework_propose_tasks",
  "homework_transition_to_closing",
  "closing_share_summary",
  "closing_share_plan",
  "closing_review_plan",
  "closing_review_feedback",
  "closing_finalize_feedback",
  "closing_fallback",
  "closing_end_session",
  "forced_closing_end",
)

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
MIN_COGNITIVE_SET_EXCHANGES = 3
MI_TOPICS = ["Health", "Economy", "Interpersonal Relationships", "Law", "Education"]

DEFAULT_SUBSTAGE_BY_STAGE: dict[Stage, SubStage] = {
  "opening": "open_initial",
  "mi_topic_exploration": "mi_choose_topic",
  "mi_techniques": "mi_choose_technique",
  "meta_conversation": "mi_meta_open",
  "patient_demand": "demand_plan",
  "id_distortions": "cognitive_review_active_question",
  "cbt_technique": "cbt_select_techniques",
  "homework": "homework_check_existing",
  "closing": "closing_share_summary",
  "anticipated_closing": "mi_end_session",
  "forced_closing": "forced_closing_end",
}


class ConversationMessage(TypedDict):
  utterance_index: int
  role: ConversationRole
  content: str
  stage: Stage
  substage: SubStage
  resistance: NotRequired[float | None]


class CognitiveModelSet(TypedDict):
  cognitive_model_set_id: int
  situation: str
  automatic_thoughts: list[str]
  emotions: list[str]
  behaviors: list[str]
  core_belief: str
  core_belief_probabilities: dict[str, float]
  distortion: str
  distortion_probabilities: dict[str, float]


class SelectedTechnique(TypedDict):
  name: str
  target_distortion: str
  rationale: str


class HomeworkTask(TypedDict):
  task_title: str
  task_description: str
  rationale: str
  expected_outcomes: str


class ReasoningTrace(TypedDict):
  utterance_index: int
  speaker: str
  traces: list[dict[str, Any]]


class PromptAuditRecord(TypedDict):
  utterance_index: int
  speaker: str
  request_kind: str
  prompt_tokens: int
  input_prompt: str


class SessionState(TypedDict):
  current_stage: Stage
  current_substage: SubStage
  messages: list[ConversationMessage]

  patient_profile: dict[str, Any]
  resistance_baseline: float
  resistance_recovery_rate: float
  resistance: float
  predicted_resistance: float | None

  opening_patient_mood: str
  opening_patient_problem: str
  last_reasoning: str
  conversation_rollup_summary: str
  conversation_rollup_until_utterance_index: int

  cognitive_model_sets: list[CognitiveModelSet]
  working_cognitive_set: CognitiveModelSet
  shelved_cognitive_sets: list[CognitiveModelSet]
  cognitive_gap_plan: str
  active_cognitive_target: str
  active_cognitive_question: str
  active_cognitive_follow_up_attempts: int
  working_cognitive_set_exchange_count: int

  selected_techniques: list[SelectedTechnique]
  current_technique_index: int
  current_technique_plan: str
  current_technique_expected_outcome: str
  current_technique_current_progress: str
  technique_evaluations: list[str | dict[str, Any]]
  techniques_completed: int
  technique_turn_count: int
  technique_slot_replaced: dict[str, bool]

  mi_active: bool
  mi_context: list[ConversationMessage]
  cbt_flow_saved_state: str
  cbt_resume_stage: Stage
  cbt_resume_substage: SubStage | Literal[""]
  mi_topics_tried: list[str]
  mi_techniques_used: list[str]
  mi_techniques_paused_until_precontemplation: bool
  current_mi_topic: str
  last_mi_predicted_resistance: float | None
  mi_end_session_demand_redirect_used: bool
  patient_demand_text: str
  patient_demand_plan: str
  patient_demand_reasoning: str
  patient_demand_resume_stage: Stage | Literal[""]
  patient_demand_resume_substage: SubStage | Literal[""]
  patient_demand_end_session_attempt_active: bool
  resume_context_active: bool
  resume_context_from_stage: Stage | Literal[""]
  resume_context_reason: str
  resume_context_subject: str

  proposed_tasks: list[HomeworkTask]
  homework_proposal_attempts: int
  homework_waiting_for_patient: bool
  homework_fallback_done: bool
  agreed_tasks: list[HomeworkTask]
  hw_reasoning: str
  homework_fallback_to_apply: bool

  session_summary: str
  next_session_plan: str
  patient_feedback: str
  closing_plan_refinements: int

  prompt_audit_records: list[PromptAuditRecord]
  reasoning_traces: list[ReasoningTrace]
  utterance_index: int
  exchange_counts: dict[str, list[int]]
  active_exchange_stage: Stage | Literal[""]
  active_exchange_substage: SubStage | Literal[""]
  high_resistance_redirect_triggered: bool
  high_resistance_level: str
  high_resistance_source_stage: Stage | Literal[""]
  high_resistance_source_substage: SubStage | Literal[""]
  high_resistance_reasoning: str

  final_patient_pending_end: bool
  session_ended: bool
  session_ended_reason: str
  conversation_error: str
  opening_fallback_triggered: bool

  max_turns: int
  min_turns: int
