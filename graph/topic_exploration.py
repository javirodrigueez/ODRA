from __future__ import annotations

from typing import Any, Literal

from agents.interfaces import ensure_str

from .state import SessionState
from .support import ConversationProcessingError, GraphSupportMixin

SUPERCLASS_TOPICS = ["Health", "Economy", "Interpersonal Relationships", "Law", "Education"]

_RAW_TOPIC_GRAPH: dict[str, list[str]] = {
  "Health": ["Mental Disorders", "Diseases", "Sexual Health", "Fitness", "Health Care", "Workplace Wellness"],
  "Interpersonal Relationships": ["Family", "Parenting", "Workplace Relationships"],
  "Law": ["Criminal Law", "Child Custody", "Traffic Ticket"],
  "Economy": ["Employment", "Personal Finance"],
  "Education": ["Student Affairs", "Exam", "Academic Achievement"],
  "Mental Disorders": ["Depression", "Chronodisruption", "Anxiety Disorders", "Cognitive Decline"],
  "Diseases": ["Infection", "Hypertension", "Flu", "Inflammation", "Liver Disease", "Lung Cancer", "COPD", "Asthma", "Stroke", "Diabetes", "Stroke", "Inflammation"],
  "Sexual Health": ["Birth Defects", "Maternal Health", "Preterm Birth", "Miscarriage", "Safe Sex"],
  "Fitness": ["Physical Activity", "Sport", "Physical Fitness", "Strength", "Flexibility", "Endurance"],
  "Health Care": ["Dentistry", "Caregiver Burden", "Independent Living", "Human Appearance", "Child Care", "Maternal Health"],
  "Parenting": ["Role Model", "Child Development", "Paternal Bond", "Child Care", "Habituation", "Child Custody"],
  "Family": ["Family Estrangement", "Family Disruption", "Divorce", "Child Custody"],
  "Criminal Law": ["Arrest", "Imprisonment", "Complaint"],
  "Personal Finance": ["Cost of Living", "Personal Budget", "Debt", "Income Deficit"],
  "Employment": ["Productivity", "Absenteeism", "Workplace Relationships", "Career Break", "Career Assessment", "Absence Rate", "Salary", "Workplace Wellness", "Workplace Incivility"],
  "Student Affairs": ["Attendance", "Suspension", "Exam", "Scholarship"],
}


def _dedupe(values: list[str]) -> list[str]:
  ordered: list[str] = []
  for value in values:
    if value not in ordered:
      ordered.append(value)
  return ordered


def _normalize_topic_graph(raw_graph: dict[str, list[str]]) -> dict[str, list[str]]:
  normalized = {topic: _dedupe(children) for topic, children in raw_graph.items()}
  for children in list(normalized.values()):
    for child in children:
      normalized.setdefault(child, [])
  return normalized


TOPIC_GRAPH = _normalize_topic_graph(_RAW_TOPIC_GRAPH)
COARSE_TOPICS = {topic for root in SUPERCLASS_TOPICS for topic in TOPIC_GRAPH.get(root, [])}

TOPIC_FOCUS_HINTS: dict[str, str] = {
  "Health": "Explore effects on wellbeing, symptoms, energy, and daily functioning.",
  "Economy": "Explore effects on money pressure, work stability, productivity, or future security.",
  "Interpersonal Relationships": "Explore effects on closeness, trust, family life, or social support.",
  "Law": "Explore worries about rules, consequences, safety, or losing control in public settings.",
  "Education": "Explore effects on focus, attendance, performance, or future opportunities.",
  "Mental Disorders": "Explore links with mood, anxiety, stress, sleep, or mental strain.",
  "Diseases": "Explore broader physical health risks and how the struggle may affect the body over time.",
  "Sexual Health": "Explore effects on intimacy, safety, pregnancy-related concerns, or reproductive health.",
  "Fitness": "Explore energy, movement, stamina, strength, and physical confidence.",
  "Health Care": "Explore self-care, appearance, independence, or day-to-day health habits.",
  "Family": "Explore tension, distance, support, and the impact at home.",
  "Parenting": "Explore the effect on parenting, reliability, modeling, and connection with children.",
  "Workplace Relationships": "Explore teamwork, friction, trust, and workplace connection.",
  "Criminal Law": "Explore fears about legal trouble, complaints, arrest, or other consequences.",
  "Child Custody": "Explore concerns about stability, parenting responsibility, and custody-related risk.",
  "Traffic Ticket": "Explore consequences of risky choices, attention lapses, or public safety concerns.",
  "Employment": "Explore effects on work performance, attendance, advancement, and job stability.",
  "Personal Finance": "Explore budgeting, debt, financial stress, and long-term security.",
  "Student Affairs": "Explore attendance, discipline, academic participation, and staying on track.",
  "Exam": "Explore preparation, focus, stress, and performance under pressure.",
  "Academic Achievement": "Explore grades, progress, confidence, and future academic opportunities.",
}


class TopicExplorationMixin(GraphSupportMixin):
  def _reset_mi_topic_exploration_state(self, state: SessionState) -> None:
    state["current_mi_topic"] = ""
    state["mi_topic_stack"] = []
    state["mi_topic_probabilities"] = {}
    state["mi_topics_tried"] = []
    state["last_mi_topic_action"] = ""
    state["current_mi_technique_selection"] = []
    state["mi_recovery_turns"] = 0
    state["cami_recovery_turns"] = 0

  def _normalize_mi_topic_stack(self, state: SessionState) -> list[str]:
    stack = [topic for topic in state.get("mi_topic_stack", []) if topic][:3]
    if not stack:
      current = ensure_str(state.get("current_mi_topic"))
      if current in SUPERCLASS_TOPICS:
        stack = [current]

    normalized: list[str] = []
    for topic in stack:
      if not normalized:
        if topic not in SUPERCLASS_TOPICS:
          break
        normalized.append(topic)
        continue
      parent = normalized[-1]
      if topic not in TOPIC_GRAPH.get(parent, []):
        break
      normalized.append(topic)

    state["mi_topic_stack"] = normalized
    state["current_mi_topic"] = normalized[-1] if normalized else ""
    return normalized

  def _topic_focus_hint(self, topic: str) -> str:
    hint = ensure_str(TOPIC_FOCUS_HINTS.get(topic))
    if hint:
      return hint
    return (
      f"Explore how the patient's current struggle connects to \"{topic}\", "
      "what burdens it creates, and what might improve with change."
    )

  def _format_topic_options(self, topics: list[str]) -> str:
    if not topics:
      return "- (none)"
    return "\n".join(f"- {topic}: {self._topic_focus_hint(topic)}" for topic in topics)

  def _recent_utterances_since_rollup_text(self, state: SessionState) -> str:
    recent_messages = self._recent_messages_since_rollup(state)
    if not recent_messages:
      return "(none beyond the rollup summary)"
    return self._format_conversation(recent_messages)

  def _normalize_probability_map(self, payload: dict[str, Any]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for topic in SUPERCLASS_TOPICS:
      raw_value: Any = payload.get(topic)
      try:
        extracted[topic] = max(float(raw_value), 0.0)
      except (TypeError, ValueError):
        extracted[topic] = 0.0

    max_value = max(extracted.values(), default=0.0)
    if max_value > 1.0:
      scale = 100.0 if max_value <= 100.0 else max_value
      extracted = {topic: value / scale for topic, value in extracted.items()}

    total = sum(extracted.values())
    if total <= 0.0:
      uniform = 1.0 / float(len(SUPERCLASS_TOPICS))
      return {topic: uniform for topic in SUPERCLASS_TOPICS}
    return {topic: value / total for topic, value in extracted.items()}

  def _root_topics_tried(self, state: SessionState) -> set[str]:
    return {topic for topic in state["mi_topics_tried"] if topic in SUPERCLASS_TOPICS}

  def _select_initial_superclass_topic(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    formatted_topics = self._format_topic_options(SUPERCLASS_TOPICS)
    instruction = self._load_stage_prompt(
      "mi_techniques_with_te",
      "superclass_probability",
      formatted_superclass_topics=formatted_topics,
      conversation_rollup_summary=ensure_str(state.get("conversation_rollup_summary"), "(none)"),
      recent_utterances_since_rollup=self._recent_utterances_since_rollup_text(state),
    )
    parsed = self._run_therapist_structured(state, traces, "mi_superclass_probability", instruction)
    payload = self._require_dict(parsed, "mi_superclass_probability")
    raw_probabilities = payload.get("probabilities")
    probability_payload = self._require_dict(raw_probabilities, "mi_superclass_probability.probabilities")
    normalized_probabilities = self._normalize_probability_map(probability_payload)
    ordered_topics = sorted(
      SUPERCLASS_TOPICS,
      key=lambda topic: (-normalized_probabilities[topic], SUPERCLASS_TOPICS.index(topic)),
    )
    chosen = ordered_topics[0]
    state["mi_topic_probabilities"] = normalized_probabilities
    self._set_last_trace_output(
      traces,
      "mi_superclass_probability",
      {
        **payload,
        "probabilities": normalized_probabilities,
        "chosen_topic": chosen,
      },
    )
    print(f"<DEBUG>Normalized superclass topic probabilities: {normalized_probabilities}</DEBUG>") #devlog
    self._apply_topic_transition(state, "Initialize", chosen)

  def _navigation_candidates(self, state: SessionState) -> dict[str, list[str]]:
    stack = self._normalize_mi_topic_stack(state)
    if not stack:
      return {}

    depth = len(stack)
    current = stack[-1]
    candidates: dict[str, list[str]] = {}
    if depth == 1:
      step_into = TOPIC_GRAPH.get(current, [])
      switch_topics = [topic for topic in SUPERCLASS_TOPICS if topic != current]
      tried_roots = self._root_topics_tried(state)
      switch_topics.sort(key=lambda topic: (topic in tried_roots, SUPERCLASS_TOPICS.index(topic)))
      if step_into:
        candidates["Step Into"] = step_into
      if switch_topics:
        candidates["Switch"] = switch_topics
      return candidates

    if depth == 2:
      parent = stack[0]
      step_into = TOPIC_GRAPH.get(current, [])
      switch_topics = [topic for topic in TOPIC_GRAPH.get(parent, []) if topic != current]
      step_out_topics = [topic for topic in SUPERCLASS_TOPICS if topic != parent]
      tried_roots = self._root_topics_tried(state)
      step_out_topics.sort(key=lambda topic: (topic in tried_roots, SUPERCLASS_TOPICS.index(topic)))
      if step_into:
        candidates["Step Into"] = step_into
      if switch_topics:
        candidates["Switch"] = switch_topics
      if step_out_topics:
        candidates["Step Out"] = step_out_topics
      return candidates

    parent = stack[1]
    switch_topics = [topic for topic in TOPIC_GRAPH.get(parent, []) if topic != current]
    step_out_topics = [topic for topic in TOPIC_GRAPH.get(stack[0], []) if topic != parent]
    if switch_topics:
      candidates["Switch"] = switch_topics
    if step_out_topics:
      candidates["Step Out"] = step_out_topics
    return candidates

  def _normalize_navigation_action(self, raw_action: str) -> Literal["Step Into", "Switch", "Step Out", "Stay"]:
    normalized = ensure_str(raw_action).strip().lower()
    if normalized in {"step into", "step_into", "stepin", "into"}:
      return "Step Into"
    if normalized in {"switch", "swap"}:
      return "Switch"
    if normalized in {"step out", "step_out", "stepout", "out"}:
      return "Step Out"
    return "Stay"

  def _sanitize_navigation_choice(
    self,
    current_topic: str,
    candidates: dict[str, list[str]],
    raw_action: str,
    raw_topic: str,
  ) -> tuple[Literal["Step Into", "Switch", "Step Out", "Stay"], str]:
    action = self._normalize_navigation_action(raw_action)
    if action == "Stay":
      return "Stay", current_topic
    available_topics = candidates.get(action, [])
    if not available_topics:
      return "Stay", current_topic
    if raw_topic in available_topics:
      return action, raw_topic
    return action, available_topics[0]

  def _apply_topic_transition(
    self,
    state: SessionState,
    action: Literal["Initialize", "Step Into", "Switch", "Step Out", "Stay"],
    topic: str,
  ) -> None:
    stack = self._normalize_mi_topic_stack(state)
    previous_topic = state["current_mi_topic"]
    if action == "Initialize":
      stack = [topic]
    elif action == "Step Into":
      stack = [*stack, topic][:3]
    elif action == "Switch":
      stack = [*stack[:-1], topic] if stack else [topic]
    elif action == "Step Out":
      if len(stack) == 2:
        stack = [topic]
      elif len(stack) >= 3:
        stack = [stack[0], topic]
      else:
        stack = [topic]

    state["mi_topic_stack"] = stack
    state["current_mi_topic"] = stack[-1] if stack else ""
    state["last_mi_topic_action"] = action
    if action in {"Step Into", "Switch", "Step Out"}:
      state["mi_topic_probabilities"] = {}
    if state["current_mi_topic"] and state["current_mi_topic"] != previous_topic and state["current_mi_topic"] not in state["mi_topics_tried"]:
      state["mi_topics_tried"].append(state["current_mi_topic"])
    elif action == "Initialize" and state["current_mi_topic"] and state["current_mi_topic"] not in state["mi_topics_tried"]:
      state["mi_topics_tried"].append(state["current_mi_topic"])

  def _navigate_existing_topic_tree(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    stack = self._normalize_mi_topic_stack(state)
    if not stack:
      self._select_initial_superclass_topic(state, traces)
      return

    current_topic = stack[-1]
    candidates = self._navigation_candidates(state)
    instruction = self._load_stage_prompt(
      "mi_techniques_with_te",
      "topic_navigation",
      explore_path=" -> ".join(stack),
      topic=current_topic,
      topic_focus=self._topic_focus_hint(current_topic),
      step_into_topics=self._format_topic_options(candidates.get("Step Into", [])),
      switch_topics=self._format_topic_options(candidates.get("Switch", [])),
      step_out_topics=self._format_topic_options(candidates.get("Step Out", [])),
    )
    parsed = self._run_therapist_structured(state, traces, "mi_topic_navigation", instruction)
    payload = self._require_dict(parsed, "mi_topic_navigation")
    action, topic = self._sanitize_navigation_choice(
      current_topic,
      candidates,
      ensure_str(payload.get("action")),
      ensure_str(payload.get("topic")),
    )
    self._set_last_trace_output(
      traces,
      "mi_topic_navigation",
      {
        **payload,
        "action": action,
        "topic": topic,
        "candidates": candidates,
      },
    )
    print(f"<DEBUG>MI topic navigation choice: action={action}, topic={topic}</DEBUG>") #devlog
    self._apply_topic_transition(state, action, topic)

  def _mi_topic_exploration_exhausted(self, state: SessionState) -> bool:
    return len(self._root_topics_tried(state)) >= len(SUPERCLASS_TOPICS)

  def _select_next_mi_topic(self, state: SessionState, traces: list[dict[str, Any]]) -> None:
    stack = self._normalize_mi_topic_stack(state)
    if not stack:
      self._select_initial_superclass_topic(state, traces)
    else:
      self._navigate_existing_topic_tree(state, traces)
    state["last_mi_predicted_resistance"] = state["predicted_resistance"]
