from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

from agents.interfaces import ensure_str

from .state import SessionState
from .support import ConversationProcessingError, GraphSupportMixin
from .topic_exploration import SUPERCLASS_TOPICS, TOPIC_FOCUS_HINTS, TOPIC_GRAPH

CAMI_WEIGHTED_TOPIC_GRAPH: dict[str, dict[str, int]] = {
  "Health": {
    "Mental Disorders": 2,
    "Diseases": 2,
    "Sexual Health": 2,
    "Fitness": 2,
    "Health Care": 2,
    "Workplace Wellness": 1,
    "Interpersonal Relationships": 3,
    "Law": 3,
    "Economy": 3,
    "Education": 3,
  },
  "Interpersonal Relationships": {
    "Parenting": 2,
    "Family": 2,
    "Health": 3,
    "Law": 3,
    "Economy": 3,
    "Education": 3,
    "Workplace Relationships": 1,
  },
  "Law": {
    "Child Custody": 1,
    "Traffic Ticket": 1,
    "Criminal Law": 2,
    "Health": 3,
    "Interpersonal Relationships": 3,
    "Economy": 3,
    "Education": 3,
  },
  "Economy": {
    "Personal Finance": 2,
    "Employment": 2,
    "Family": 3,
    "Law": 3,
    "Health": 3,
    "Interpersonal Relationships": 3,
    "Education": 3,
  },
  "Education": {
    "Academic Achievement": 1,
    "Student Affairs": 2,
    "Exam": 1,
    "Health": 3,
    "Interpersonal Relationships": 3,
    "Law": 3,
    "Economy": 3,
  },
  "Student Affairs": {
    "Attendance": 1,
    "Suspension": 1,
    "Scholarship": 1,
    "Education": 2,
  },
  "Exam": {
    "Education": 2,
  },
  "Academic Achievement": {
    "Education": 2,
  },
  "Scholarship": {
    "Student Affairs": 1,
  },
  "Suspension": {
    "Student Affairs": 1,
  },
  "Attendance": {
    "Student Affairs": 1,
  },
  "Employment": {
    "Economy": 2,
    "Productivity": 1,
    "Absenteeism": 1,
    "Career Assessment": 1,
    "Absence Rate": 1,
    "Career Break": 1,
    "Salary": 1,
    "Workplace Wellness": 1,
    "Workplace Relationships": 1,
    "Workplace Incivility": 1,
  },
  "Productivity": {
    "Employment": 1,
  },
  "Absenteeism": {
    "Employment": 1,
  },
  "Career Assessment": {
    "Employment": 1,
  },
  "Absence Rate": {
    "Employment": 1,
  },
  "Career Break": {
    "Employment": 1,
  },
  "Salary": {
    "Employment": 1,
  },
  "Workplace Wellness": {
    "Employment": 1,
    "Health": 1,
  },
  "Workplace Relationships": {
    "Employment": 1,
    "Interpersonal Relationships": 1,
  },
  "Workplace Incivility": {
    "Employment": 1,
  },
  "Personal Finance": {
    "Economy": 2,
    "Debt": 1,
    "Cost of Living": 1,
    "Income Deficit": 1,
    "Personal Budget": 1,
  },
  "Debt": {
    "Personal Finance": 1,
  },
  "Cost of Living": {
    "Personal Finance": 1,
  },
  "Income Deficit": {
    "Personal Finance": 1,
  },
  "Personal Budget": {
    "Personal Finance": 1,
  },
  "Criminal Law": {
    "Law": 2,
    "Arrest": 1,
    "Complaint": 1,
    "Imprisonment": 1,
  },
  "Arrest": {
    "Criminal Law": 1,
  },
  "Complaint": {
    "Criminal Law": 1,
  },
  "Imprisonment": {
    "Criminal Law": 1,
  },
  "Traffic Ticket": {
    "Law": 1,
  },
  "Child Custody": {
    "Law": 1,
    "Family": 1,
    "Parenting": 1,
  },
  "Family": {
    "Interpersonal Relationships": 2,
    "Family Estrangement": 1,
    "Family Disruption": 1,
    "Divorce": 1,
    "Child Custody": 1,
  },
  "Family Estrangement": {
    "Family": 1,
  },
  "Family Disruption": {
    "Family": 1,
  },
  "Divorce": {
    "Family": 1,
  },
  "Parenting": {
    "Interpersonal Relationships": 2,
    "Role Model": 1,
    "Child Development": 1,
    "Paternal Bond": 1,
    "Child Care": 1,
    "Habituation": 1,
    "Child Custody": 1,
  },
  "Role Model": {
    "Parenting": 1,
  },
  "Child Development": {
    "Parenting": 1,
  },
  "Paternal Bond": {
    "Parenting": 1,
  },
  "Child Care": {
    "Parenting": 1,
    "Health Care": 1,
  },
  "Habituation": {
    "Parenting": 1,
  },
  "Health Care": {
    "Health": 2,
    "Dentistry": 1,
    "Caregiver Burden": 1,
    "Independent Living": 1,
    "Human Appearance": 1,
    "Child Care": 1,
    "Maternal Health": 1,
  },
  "Dentistry": {
    "Health Care": 1,
  },
  "Caregiver Burden": {
    "Health Care": 1,
  },
  "Independent Living": {
    "Health Care": 1,
  },
  "Human Appearance": {
    "Health Care": 1,
  },
  "Fitness": {
    "Health": 2,
    "Strength": 1,
    "Flexibility": 1,
    "Endurance": 1,
    "Sport": 1,
    "Physical Activity": 1,
    "Physical Fitness": 1,
  },
  "Strength": {
    "Fitness": 1,
  },
  "Flexibility": {
    "Fitness": 1,
  },
  "Endurance": {
    "Fitness": 1,
  },
  "Sport": {
    "Fitness": 1,
  },
  "Physical Activity": {
    "Fitness": 1,
  },
  "Physical Fitness": {
    "Fitness": 1,
  },
  "Sexual Health": {
    "Health": 2,
    "Maternal Health": 1,
    "Miscarriage": 1,
    "Safe Sex": 1,
    "Birth Defects": 1,
    "Preterm Birth": 1,
  },
  "Maternal Health": {
    "Sexual Health": 1,
    "Health Care": 1,
  },
  "Miscarriage": {
    "Sexual Health": 1,
  },
  "Safe Sex": {
    "Sexual Health": 1,
  },
  "Birth Defects": {
    "Sexual Health": 1,
  },
  "Preterm Birth": {
    "Sexual Health": 1,
  },
  "Mental Disorders": {
    "Health": 2,
    "Depression": 1,
    "Chronodisruption": 1,
    "Anxiety Disorders": 1,
    "Cognitive Decline": 1,
  },
  "Depression": {
    "Mental Disorders": 1,
  },
  "Chronodisruption": {
    "Mental Disorders": 1,
  },
  "Anxiety Disorders": {
    "Mental Disorders": 1,
  },
  "Cognitive Decline": {
    "Mental Disorders": 1,
  },
  "Diseases": {
    "Health": 2,
    "Infection": 1,
    "Hypertension": 1,
    "Flu": 1,
    "Inflammation": 1,
    "Liver Disease": 1,
    "Lung Cancer": 1,
    "COPD": 1,
    "Asthma": 1,
    "Stroke": 1,
    "Diabetes": 1,
  },
  "Infection": {
    "Diseases": 1,
  },
  "Hypertension": {
    "Diseases": 1,
  },
  "Flu": {
    "Diseases": 1,
  },
  "Inflammation": {
    "Diseases": 1,
  },
  "Liver Disease": {
    "Diseases": 1,
  },
  "Lung Cancer": {
    "Diseases": 1,
  },
  "COPD": {
    "Diseases": 1,
  },
  "Asthma": {
    "Diseases": 1,
  },
  "Stroke": {
    "Diseases": 1,
  },
  "Diabetes": {
    "Diseases": 1,
  },
}

CANONICAL_COARSE_BY_LEAF: dict[str, str] = {
  "Child Care": "Parenting",
  "Child Custody": "Family",
  "Exam": "Student Affairs",
  "Maternal Health": "Sexual Health",
  "Workplace Relationships": "Employment",
  "Workplace Wellness": "Employment",
}


class TopicRanking(TypedDict):
  topic: str
  score: float


class PatientTopicEngagementResult(TypedDict):
  perceived_topic: str
  topic_distance: int
  engagement_level: int
  engagement_instruction: str
  engagement_analysis: str
  ranked_topics: list[TopicRanking]


class _PatientTopicExplorationHost(Protocol):
  cfg: Any
  _patient_topic_tokenizer: Any
  _patient_topic_model: Any
  _patient_topic_device: Any
  _patient_topic_leaf_topics: list[str] | None
  _patient_topic_passages: list[str] | None

  def _load_stage_prompt(self, stage: str, name: str, **values: Any) -> str:
    ...

  def _format_last_utterances(self, messages: list[Any], limit: int = 10) -> str:
    ...

  def _last_therapist_utterance(self, state: SessionState) -> str:
    ...

  def _run_patient_structured(
    self,
    state: SessionState,
    traces: list[dict[str, Any]],
    step_name: str,
    instruction: str,
    expected_type: type[Any] | tuple[type[Any], ...] | None = dict,
    engagement_block: str | None = None,
  ) -> Any:
    ...

  def _require_dict(self, payload: Any, step_name: str) -> dict[str, Any]:
    ...

  def _require_non_empty_string(self, payload: dict[str, Any], key: str, step_name: str) -> str:
    ...

  def _topic_exploration_runtime_paths(self) -> tuple[Path, Path]:
    ...

  def _ensure_patient_topic_classifier_assets(self) -> None:
    ...

  def _leaf_topic_description(self, leaf_topic: str) -> str:
    ...

  def _motivation_topic_contract(self, state: SessionState) -> tuple[str, str]:
    ...

  def _classify_last_therapist_utterance_topics(
    self,
    therapist_utterance: str,
    top_k: int = 5,
  ) -> list[TopicRanking]:
    ...

  def _build_patient_engagement_analysis_instruction(
    self,
    state: SessionState,
    perceived_topic: str,
    engagement_level: int,
    motivation_topic: str,
    motivation_text: str,
  ) -> str:
    ...


def _topic_children(topic: str) -> list[str]:
  return [child for child in TOPIC_GRAPH.get(topic, []) if child]


def _build_leaf_topics() -> list[str]:
  leaf_topics: list[str] = []
  for topic, children in TOPIC_GRAPH.items():
    if children:
      continue
    if topic in SUPERCLASS_TOPICS:
      continue
    leaf_topics.append(topic)
  return sorted(leaf_topics)


LEAF_TOPICS = _build_leaf_topics()


def _build_canonical_leaf_paths() -> dict[str, tuple[str, str, str]]:
  paths: dict[str, tuple[str, str, str]] = {}
  for superclass in SUPERCLASS_TOPICS:
    for child in _topic_children(superclass):
      grand_children = _topic_children(child)
      if grand_children:
        for leaf in grand_children:
          paths.setdefault(leaf, (superclass, child, leaf))
      else:
        coarse = CANONICAL_COARSE_BY_LEAF.get(child, child)
        if coarse == child:
          paths.setdefault(child, (superclass, child, child))
  for leaf, coarse in CANONICAL_COARSE_BY_LEAF.items():
    if coarse == leaf:
      continue
    for superclass in SUPERCLASS_TOPICS:
      if coarse in _topic_children(superclass):
        paths[leaf] = (superclass, coarse, leaf)
        break
  return paths


CANONICAL_LEAF_PATHS = _build_canonical_leaf_paths()


def dijkstra_topic_distance(graph: dict[str, dict[str, int]], start_node: str, target_node: str) -> int:
  if start_node not in graph or target_node not in graph:
    return 999
  if start_node == target_node:
    return 0

  distances = {node: 999 for node in graph}
  distances[start_node] = 0
  queue: list[tuple[int, str]] = [(0, start_node)]
  visited: set[str] = set()

  while queue:
    current_distance, current_node = heapq.heappop(queue)
    if current_node == target_node:
      return current_distance
    if current_node in visited:
      continue
    visited.add(current_node)
    for neighbor, weight in graph.get(current_node, {}).items():
      if neighbor in visited:
        continue
      distance = current_distance + int(weight)
      if distance < distances.get(neighbor, 999):
        distances[neighbor] = distance
        heapq.heappush(queue, (distance, neighbor))
  return 999


def engagement_level_from_distance(distance: int) -> int:
  if distance <= 0:
    return 4
  if distance <= 3:
    return 3
  if distance <= 5:
    return 2
  return 1


def canonical_path_for_leaf(leaf_topic: str) -> tuple[str, str, str]:
  path = CANONICAL_LEAF_PATHS.get(leaf_topic)
  if path is None:
    raise ConversationProcessingError(f"Unknown or non-leaf motivation_topic: {leaf_topic!r}")
  return path


def engagement_instruction_for_topic(
  engagement_level: int,
  motivation_topic: str,
  motivation_text: str,
  perceived_topic: str,
) -> str:
  superclass, coarse, leaf = canonical_path_for_leaf(motivation_topic)
  if engagement_level <= 1:
    return (
      f"Keep your response broad and somewhat vague. Avoid engaging deeply with {perceived_topic} "
      "and subtly steer the conversation toward other concerns that matter more to you."
    )
  if engagement_level == 2:
    return (
      f"Acknowledge the importance of {superclass}, but hint that your focus is really on a more "
      f"specific area within it: {coarse}."
    )
  if engagement_level == 3:
    return (
      f"Engage with {coarse}, while subtly signaling that there is an even more specific issue "
      f"within that area that matters to you: {leaf}."
    )
  return (
    f"Show that the therapist is on the right track about {leaf}. Let your tone and detail reflect "
    f"that this connects to what truly motivates you: {motivation_text}"
  )


class PatientTopicExplorationMixin(GraphSupportMixin):
  _patient_topic_tokenizer: Any = None
  _patient_topic_model: Any = None
  _patient_topic_device: Any = None
  _patient_topic_leaf_topics: list[str] | None = None
  _patient_topic_passages: list[str] | None = None

  def _reset_patient_topic_exploration_state(self, state: SessionState) -> None:
    state["last_patient_perceived_topic"] = ""
    state["last_patient_topic_distance"] = None
    state["last_patient_engagement_level"] = None
    state["last_patient_engagement_analysis"] = ""
    state["last_patient_engagement_instruction"] = ""

  def _topic_exploration_runtime_paths(self: _PatientTopicExplorationHost) -> tuple[Path, Path]:
    cfg_block = getattr(self.cfg, "topic_exploration", None)
    classifier_path = ensure_str(getattr(cfg_block, "patient_classifier_path", "") if cfg_block is not None else "")
    wikipedia_dir = ensure_str(getattr(cfg_block, "patient_wikipedia_dir", "") if cfg_block is not None else "")
    if not classifier_path:
      raise ConversationProcessingError("Missing topic_exploration.patient_classifier_path for mi_techniques_with_te.")
    if not wikipedia_dir:
      raise ConversationProcessingError("Missing topic_exploration.patient_wikipedia_dir for mi_techniques_with_te.")

    classifier = Path(classifier_path).expanduser()
    topic_dir = Path(wikipedia_dir).expanduser()
    if not classifier.exists():
      raise ConversationProcessingError(f"Patient topic classifier path does not exist: {classifier}")
    if not topic_dir.exists() or not topic_dir.is_dir():
      raise ConversationProcessingError(f"Patient topic corpus directory does not exist: {topic_dir}")
    return classifier, topic_dir

  def _ensure_patient_topic_classifier_assets(self: _PatientTopicExplorationHost) -> None:
    if self._patient_topic_model is not None and self._patient_topic_tokenizer is not None:
      return

    classifier_path, topic_dir = self._topic_exploration_runtime_paths()
    try:
      import torch  # pyright: ignore[reportMissingImports]
      from transformers import AutoModelForSequenceClassification, AutoTokenizer  # pyright: ignore[reportMissingImports]
    except Exception as exc:
      raise ConversationProcessingError(
        f"Patient topic exploration requires torch and transformers: {exc}"
      ) from exc

    passages: list[str] = []
    for leaf_topic in LEAF_TOPICS:
      topic_file = topic_dir / leaf_topic
      if not topic_file.exists():
        raise ConversationProcessingError(
          f"Missing patient topic file for {leaf_topic!r} at {topic_file}"
        )
      passages.append(
        f"{self._leaf_topic_description(leaf_topic)}\n{topic_file.read_text(encoding='utf-8').strip()}"
      )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
      tokenizer = AutoTokenizer.from_pretrained(str(classifier_path))
      model = AutoModelForSequenceClassification.from_pretrained(str(classifier_path)).to(device)
    except Exception as exc:
      raise ConversationProcessingError(
        f"Failed to load patient topic classifier from {classifier_path}: {exc}"
      ) from exc
    model.eval()

    self._patient_topic_tokenizer = tokenizer
    self._patient_topic_model = model
    self._patient_topic_device = device
    self._patient_topic_leaf_topics = list(LEAF_TOPICS)
    self._patient_topic_passages = passages

  def _leaf_topic_description(self, leaf_topic: str) -> str:
    superclass, coarse, leaf = canonical_path_for_leaf(leaf_topic)
    leaf_hint = ensure_str(TOPIC_FOCUS_HINTS.get(leaf))
    coarse_hint = ensure_str(TOPIC_FOCUS_HINTS.get(coarse))
    superclass_hint = ensure_str(TOPIC_FOCUS_HINTS.get(superclass))
    segments = [
      f"This topic is about {leaf} within {coarse} and the broader area of {superclass}.",
    ]
    if leaf_hint:
      segments.append(leaf_hint)
    if coarse_hint and coarse_hint not in segments:
      segments.append(coarse_hint)
    if superclass_hint and superclass_hint not in segments:
      segments.append(superclass_hint)
    if not leaf_hint:
      segments.append(
        f"Explore how the patient's current struggle affects {leaf}, what burden it creates, and what could improve with change."
      )
    return " ".join(segment.strip() for segment in segments if segment.strip())

  def _motivation_topic_contract(self, state: SessionState) -> tuple[str, str]:
    motivation_topic = ensure_str(state["patient_profile"].get("motivation_topic"))
    motivation_text = ensure_str(state["patient_profile"].get("motivation_text"))
    if not motivation_topic:
      raise ConversationProcessingError("Patient profile is missing required field motivation_topic.")
    if motivation_topic not in LEAF_TOPICS:
      raise ConversationProcessingError(
        f"Patient profile motivation_topic must be a canonical leaf topic, got {motivation_topic!r}."
      )
    if not motivation_text:
      raise ConversationProcessingError("Patient profile is missing required field motivation_text.")
    return motivation_topic, motivation_text

  def _classify_last_therapist_utterance_topics(
    self: _PatientTopicExplorationHost,
    therapist_utterance: str,
    top_k: int = 5,
  ) -> list[TopicRanking]:
    self._ensure_patient_topic_classifier_assets()
    tokenizer = self._patient_topic_tokenizer
    model = self._patient_topic_model
    device = self._patient_topic_device
    leaf_topics = self._patient_topic_leaf_topics or list(LEAF_TOPICS)
    passages = self._patient_topic_passages or []
    if not passages or not leaf_topics:
      raise ConversationProcessingError("Patient topic classifier assets were not initialized correctly.")

    try:
      import torch  # pyright: ignore[reportMissingImports]
    except Exception as exc:
      raise ConversationProcessingError(f"Patient topic exploration requires torch: {exc}") from exc

    query = ensure_str(therapist_utterance) or "(no therapist utterance yet)"
    paired_inputs = list(zip([query] * len(passages), passages))
    with torch.no_grad():
      encoded = tokenizer(
        paired_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
      )
      encoded = {key: value.to(device) for key, value in encoded.items()}
      logits = model(**encoded, return_dict=True).logits.view(-1).float()
      scores = cast(list[float], torch.sigmoid(logits).tolist())
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:max(1, top_k)]
    rankings: list[TopicRanking] = []
    for index in ranked_indices:
      rankings.append(
        {
          "topic": leaf_topics[index],
          "score": round(float(scores[index]), 6),
        }
      )
    return rankings

  def _build_patient_engagement_analysis_instruction(
    self: _PatientTopicExplorationHost,
    state: SessionState,
    perceived_topic: str,
    engagement_level: int,
    motivation_topic: str,
    motivation_text: str,
  ) -> str:
    return self._load_stage_prompt(
      "mi_techniques_with_te",
      "patient_engagement_analysis",
      recent_conversation=self._format_last_utterances(state["messages"]),
      last_utterance=self._last_therapist_utterance(state),
      perceived_topic=perceived_topic,
      engagement_level=str(engagement_level),
      motivation_topic=motivation_topic,
      motivation_text=motivation_text,
    )

  def _run_patient_topic_exploration_engagement(
    self: _PatientTopicExplorationHost,
    state: SessionState,
    traces: list[dict[str, Any]],
  ) -> PatientTopicEngagementResult:
    motivation_topic, motivation_text = self._motivation_topic_contract(state)
    ranked_topics = self._classify_last_therapist_utterance_topics(self._last_therapist_utterance(state))
    perceived_topic = ranked_topics[0]["topic"] if ranked_topics else motivation_topic
    topic_distance = dijkstra_topic_distance(CAMI_WEIGHTED_TOPIC_GRAPH, motivation_topic, perceived_topic)
    engagement_level = engagement_level_from_distance(topic_distance)
    engagement_instruction = engagement_instruction_for_topic(
      engagement_level,
      motivation_topic,
      motivation_text,
      perceived_topic,
    )

    analysis_instruction = self._build_patient_engagement_analysis_instruction(
      state,
      perceived_topic,
      engagement_level,
      motivation_topic,
      motivation_text,
    )
    parsed = self._run_patient_structured(
      state,
      traces,
      "patient_engagement_analysis",
      analysis_instruction,
    )
    payload = self._require_dict(parsed, "patient_engagement_analysis")
    engagement_analysis = self._require_non_empty_string(
      payload,
      "engagement_analysis",
      "patient_engagement_analysis",
    )

    state["last_patient_perceived_topic"] = perceived_topic
    state["last_patient_topic_distance"] = topic_distance
    state["last_patient_engagement_level"] = engagement_level
    state["last_patient_engagement_analysis"] = engagement_analysis
    state["last_patient_engagement_instruction"] = engagement_instruction

    trace_payload = {
      "perceived_topic": perceived_topic,
      "motivation_topic": motivation_topic,
      "topic_distance": topic_distance,
      "engagement_level": engagement_level,
      "ranked_topics": ranked_topics,
      "engagement_analysis": engagement_analysis,
    }
    traces.append(
      {
        "step": "patient_topic_engagement",
        "input": self._last_therapist_utterance(state),
        "output": json.dumps(trace_payload, ensure_ascii=True),
      }
    )
    return {
      "perceived_topic": perceived_topic,
      "topic_distance": topic_distance,
      "engagement_level": engagement_level,
      "engagement_instruction": engagement_instruction,
      "engagement_analysis": engagement_analysis,
      "ranked_topics": ranked_topics,
    }
