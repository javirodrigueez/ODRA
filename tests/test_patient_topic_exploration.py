import sys
import types
import unittest
from types import SimpleNamespace

if "dotenv" not in sys.modules:
  dotenv_module = types.ModuleType("dotenv")
  dotenv_module.load_dotenv = lambda *args, **kwargs: None
  sys.modules["dotenv"] = dotenv_module

if "langgraph.graph" not in sys.modules:
  langgraph_module = types.ModuleType("langgraph")
  langgraph_graph_module = types.ModuleType("langgraph.graph")
  langgraph_graph_module.END = "END"
  langgraph_graph_module.START = "START"
  langgraph_graph_module.StateGraph = object
  sys.modules["langgraph"] = langgraph_module
  sys.modules["langgraph.graph"] = langgraph_graph_module

if "langchain.messages" not in sys.modules:
  langchain_module = types.ModuleType("langchain")
  langchain_messages_module = types.ModuleType("langchain.messages")

  class _Message:
    def __init__(self, content: str):
      self.content = content

  langchain_messages_module.AIMessage = _Message
  langchain_messages_module.HumanMessage = _Message
  langchain_messages_module.SystemMessage = _Message
  sys.modules["langchain"] = langchain_module
  sys.modules["langchain.messages"] = langchain_messages_module

if "langchain_aws" not in sys.modules:
  langchain_aws_module = types.ModuleType("langchain_aws")
  langchain_aws_module.ChatBedrock = object
  sys.modules["langchain_aws"] = langchain_aws_module

if "langchain_core.prompts" not in sys.modules:
  langchain_core_module = types.ModuleType("langchain_core")
  langchain_core_prompts_module = types.ModuleType("langchain_core.prompts")

  class _PromptTemplate:
    def __init__(self, template: str):
      self.template = template

    @classmethod
    def from_template(cls, template: str) -> "_PromptTemplate":
      return cls(template)

    def format(self, **values: str) -> str:
      rendered = self.template
      for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
      return rendered

  langchain_core_prompts_module.PromptTemplate = _PromptTemplate
  sys.modules["langchain_core"] = langchain_core_module
  sys.modules["langchain_core.prompts"] = langchain_core_prompts_module

if "langchain_huggingface" not in sys.modules:
  langchain_huggingface_module = types.ModuleType("langchain_huggingface")
  langchain_huggingface_module.ChatHuggingFace = object
  langchain_huggingface_module.HuggingFacePipeline = object
  sys.modules["langchain_huggingface"] = langchain_huggingface_module

if "omegaconf" not in sys.modules:
  omegaconf_module = types.ModuleType("omegaconf")

  class _OmegaConf:
    @staticmethod
    def to_container(config, resolve: bool = True):
      return config

    @staticmethod
    def is_config(config) -> bool:
      return False

    @staticmethod
    def create(config):
      return config

  omegaconf_module.OmegaConf = _OmegaConf
  sys.modules["omegaconf"] = omegaconf_module

if "openai" not in sys.modules:
  openai_module = types.ModuleType("openai")
  openai_module.OpenAI = object
  sys.modules["openai"] = openai_module

from agents.patient import PatientAgent
from graph.graph import TherapyGraphBuilder
from graph.patient_topic_exploration import (
  CAMI_WEIGHTED_TOPIC_GRAPH,
  LEAF_TOPICS,
  canonical_path_for_leaf,
  dijkstra_topic_distance,
  engagement_instruction_for_topic,
  engagement_level_from_distance,
)
from graph.support import ConversationProcessingError


TEST_PROFILE = {
  "personal_info": {"name": "Alex", "age": 29},
  "more_information": {
    "past_history": "Often gets stuck in work-related stress loops.",
    "academic_occupational_functioning": "Currently employed full-time.",
    "social_support_system": "Has a partner and a sibling for support.",
  },
  "patterns": ["catastrophizing", "mind reading"],
  "presenting_problem": "I keep assuming every conflict at work means I'm failing.",
  "reason_seeking_counseling": "I want to stop spiraling every time something feels tense at work.",
  "attitude": "neutral",
  "emotional_recovery": "moderate",
  "motivation_topic": "Workplace Relationships",
  "motivation_text": "I care a lot about keeping trust with coworkers because conflict at work makes me shut down.",
}


class _FakeLLM:
  def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
    return "{}"


class PatientTopicExplorationTest(unittest.TestCase):
  def _make_builder(self) -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = dict(TEST_PROFILE)
    builder.cfg = SimpleNamespace(
      env=SimpleNamespace(max_turns=12, min_turns=0),
      topic_exploration=SimpleNamespace(patient_classifier_path="", patient_wikipedia_dir=""),
    )
    builder.patient = PatientAgent(_FakeLLM(), dict(TEST_PROFILE))
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}|{values}"
    return builder

  def _make_state(self, builder: TherapyGraphBuilder, *, stage: str = "mi_techniques_with_te") -> dict:
    state = builder.initial_state(max_turns=12)
    state["current_stage"] = stage
    state["current_substage"] = "mi_raise_cami_utterance" if stage == "mi_techniques_with_te" else "open_problem"
    state["messages"] = [
      {
        "utterance_index": 1,
        "role": "therapist",
        "content": "How is this affecting the way you relate to coworkers?",
        "stage": stage,
        "substage": state["current_substage"],
      },
    ]
    return builder._ensure_state_defaults(state)

  def test_leaf_topics_are_fine_grained(self) -> None:
    self.assertIn("Workplace Relationships", LEAF_TOPICS)
    self.assertIn("Academic Achievement", LEAF_TOPICS)
    self.assertNotIn("Employment", LEAF_TOPICS)
    self.assertNotIn("Health", LEAF_TOPICS)

  def test_dijkstra_topic_distance_matches_weighted_graph(self) -> None:
    self.assertEqual(
      dijkstra_topic_distance(CAMI_WEIGHTED_TOPIC_GRAPH, "Workplace Relationships", "Workplace Relationships"),
      0,
    )
    self.assertEqual(
      dijkstra_topic_distance(CAMI_WEIGHTED_TOPIC_GRAPH, "Family Estrangement", "Divorce"),
      2,
    )
    self.assertEqual(
      dijkstra_topic_distance(CAMI_WEIGHTED_TOPIC_GRAPH, "Role Model", "Workplace Relationships"),
      4,
    )

  def test_engagement_level_mapping_matches_cami_thresholds(self) -> None:
    self.assertEqual(engagement_level_from_distance(0), 4)
    self.assertEqual(engagement_level_from_distance(3), 3)
    self.assertEqual(engagement_level_from_distance(5), 2)
    self.assertEqual(engagement_level_from_distance(6), 1)

  def test_engagement_instruction_uses_canonical_path(self) -> None:
    self.assertEqual(
      canonical_path_for_leaf("Workplace Relationships"),
      ("Economy", "Employment", "Workplace Relationships"),
    )
    instruction = engagement_instruction_for_topic(
      2,
      "Workplace Relationships",
      TEST_PROFILE["motivation_text"],
      "Health",
    )
    self.assertIn("Economy", instruction)
    self.assertIn("Employment", instruction)

  def test_patient_agent_includes_engagement_block_only_when_provided(self) -> None:
    agent = PatientAgent(_FakeLLM(), dict(TEST_PROFILE))
    with_block = agent.build_utterance_messages(
      full_conversation="Therapist: Hello",
      last_utterances="Therapist: Hello",
      resistance_level=0.4,
      instruction="Respond naturally.",
      engagement_block="Stay broad and avoid the current topic.",
    )
    without_block = agent.build_utterance_messages(
      full_conversation="Therapist: Hello",
      last_utterances="Therapist: Hello",
      resistance_level=0.4,
      instruction="Respond naturally.",
    )
    self.assertIn("<engagement>", with_block[1]["content"])
    self.assertIn("Stay broad and avoid the current topic.", with_block[1]["content"])
    self.assertNotIn("<engagement>", without_block[1]["content"])

  def test_patient_turn_uses_engagement_blocks_during_cami_stage(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    captured: dict[str, object] = {}

    def fake_topic_engagement(current_state, traces):  # type: ignore[no-untyped-def]
      current_state["last_patient_perceived_topic"] = "Workplace Relationships"
      current_state["last_patient_topic_distance"] = 0
      current_state["last_patient_engagement_level"] = 4
      current_state["last_patient_engagement_analysis"] = "The therapist is finally focused on what matters at work."
      current_state["last_patient_engagement_instruction"] = "Show they are on the right track."
      return {
        "perceived_topic": "Workplace Relationships",
        "topic_distance": 0,
        "engagement_level": 4,
        "engagement_analysis": "The therapist is finally focused on what matters at work.",
        "engagement_instruction": "Show they are on the right track.",
        "ranked_topics": [{"topic": "Workplace Relationships", "score": 0.9}],
      }

    def fake_run_patient_structured(current_state, traces, step_name, instruction, expected_type=dict, engagement_block=None):  # type: ignore[no-untyped-def]
      captured["structured_step"] = step_name
      captured["structured_instruction"] = instruction
      captured["structured_engagement_block"] = engagement_block
      return {"r_change": 0.05, "reasoning": "The alignment lowered resistance."}

    def fake_generate_utterance(**kwargs):  # type: ignore[no-untyped-def]
      captured["utterance_engagement_block"] = kwargs.get("engagement_block")
      return "I guess that does hit closer to what bothers me."

    builder._run_patient_topic_exploration_engagement = fake_topic_engagement
    builder._run_patient_structured = fake_run_patient_structured
    builder._select_patient_action = lambda *args, **kwargs: "Acceptance"
    builder.patient.generate_utterance = fake_generate_utterance

    result = builder.patient_turn(state)

    self.assertEqual(captured["structured_step"], "resistance_update")
    self.assertEqual(
      captured["structured_engagement_block"],
      "The therapist is finally focused on what matters at work.",
    )
    self.assertEqual(captured["utterance_engagement_block"], "Show they are on the right track.")
    self.assertEqual(result["last_patient_engagement_level"], 4)
    self.assertEqual(result["messages"][-1]["role"], "patient")

  def test_non_topic_patient_turn_keeps_engagement_block_empty(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="opening")
    state["last_patient_perceived_topic"] = "Workplace Relationships"
    state["last_patient_topic_distance"] = 0
    state["last_patient_engagement_level"] = 4
    state["last_patient_engagement_analysis"] = "stale"
    state["last_patient_engagement_instruction"] = "stale"
    captured: dict[str, object] = {}

    def fake_run_patient_structured(current_state, traces, step_name, instruction, expected_type=dict, engagement_block=None):  # type: ignore[no-untyped-def]
      captured["structured_engagement_block"] = engagement_block
      return {"r_change": 0.0, "reasoning": "No major change."}

    def fake_generate_utterance(**kwargs):  # type: ignore[no-untyped-def]
      captured["utterance_engagement_block"] = kwargs.get("engagement_block")
      return "Okay."

    builder._run_patient_structured = fake_run_patient_structured
    builder._select_patient_action = lambda *args, **kwargs: "Acceptance"
    builder.patient.generate_utterance = fake_generate_utterance

    result = builder.patient_turn(state)

    self.assertIsNone(captured["structured_engagement_block"])
    self.assertIsNone(captured["utterance_engagement_block"])
    self.assertEqual(result["last_patient_perceived_topic"], "Workplace Relationships")
    self.assertEqual(result["last_patient_topic_distance"], 0)
    self.assertEqual(result["last_patient_engagement_level"], 4)

  def test_missing_motivation_text_raises_clear_error(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["patient_profile"]["motivation_text"] = ""
    with self.assertRaises(ConversationProcessingError):
      builder._run_patient_topic_exploration_engagement(state, [])

  def test_invalid_motivation_topic_raises_clear_error(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["patient_profile"]["motivation_topic"] = "Employment"
    with self.assertRaises(ConversationProcessingError):
      builder._run_patient_topic_exploration_engagement(state, [])

  def test_missing_classifier_config_raises_clear_error(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    with self.assertRaises(ConversationProcessingError):
      builder._run_patient_topic_exploration_engagement(state, [])


if __name__ == "__main__":
  unittest.main()
