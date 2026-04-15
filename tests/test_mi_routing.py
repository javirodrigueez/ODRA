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

from graph.graph import TherapyGraphBuilder


class MIRoutingTest(unittest.TestCase):
  def _make_builder(self, *, resistance_handling: str = "mi_techniques") -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = {}
    builder.cfg = SimpleNamespace(
      env=SimpleNamespace(max_turns=12, min_turns=0, resistance_handling=resistance_handling),
      topic_exploration=SimpleNamespace(patient_classifier_path="", patient_wikipedia_dir=""),
    )
    builder.mi_techniques = {"Open Question": "Ask an open question."}
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}"
    builder._append_mi_context_message = lambda *args, **kwargs: None
    builder._run_therapist_structured = lambda state, traces, step_name, instruction: {
      "reason_for_mi": "high resistance",
      "conclusions": "Patient softened.",
      "patient_state_change": "More willing to continue.",
    }
    return builder

  def _make_state(self, builder: TherapyGraphBuilder, *, stage: str = "id_distortions", predicted_resistance: float = 0.8) -> dict:
    state = builder.initial_state(max_turns=12)
    state["current_stage"] = stage
    state["current_substage"] = "cognitive_review_active_question" if stage == "id_distortions" else state["current_substage"]
    state["predicted_resistance"] = predicted_resistance
    return builder._ensure_state_defaults(state)

  def test_current_band_uses_global_thresholds(self) -> None:
    builder = self._make_builder()
    self.assertEqual(builder._current_band(0.66), "precontemplation")
    self.assertEqual(builder._current_band(0.65), "contemplation")
    self.assertEqual(builder._current_band(0.34), "contemplation")
    self.assertEqual(builder._current_band(0.33), "action")

  def test_precon_streak_enters_legacy_mi_mode(self) -> None:
    builder = self._make_builder(resistance_handling="mi_techniques")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["preconsecutive_turns"] = 4

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertEqual(state["current_substage"], "mi_choose_technique")

  def test_precon_streak_enters_cami_mode(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["preconsecutive_turns"] = 4

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques_with_te")
    self.assertEqual(state["current_substage"], "mi_choose_topic")

  def test_precon_streak_does_not_enter_mi_when_mode_is_none(self) -> None:
    builder = self._make_builder(resistance_handling="none")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["preconsecutive_turns"] = 4

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertEqual(state["preconsecutive_turns"], 5)

  def test_non_precon_band_resets_entry_streak(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, predicted_resistance=0.5)
    state["preconsecutive_turns"] = 4

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertEqual(state["preconsecutive_turns"], 0)

  def test_legacy_mi_mode_never_branches_to_topic_stage(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.9)
    state["current_substage"] = "mi_choose_technique"

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertEqual(state["current_substage"], "mi_choose_technique")

  def test_cami_exits_after_five_recovery_turns(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, stage="mi_techniques_with_te", predicted_resistance=0.5)
    state["current_substage"] = "mi_choose_topic"
    state["cami_recovery_turns"] = 4

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques_with_te")
    self.assertEqual(state["current_substage"], "mi_transition_to_cbt")
    self.assertEqual(state["cami_recovery_turns"], 0)

  def test_enter_mi_cami_preserves_topic_state_for_resume(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["current_mi_topic"] = "Employment"
    state["mi_topic_stack"] = ["Economy", "Employment"]
    state["mi_topics_tried"] = ["Economy", "Employment"]

    builder._enter_mi(
      state,
      "mi_techniques_with_te",
      "id_distortions",
      "cognitive_review_active_question",
      "resume CAMI",
    )

    self.assertEqual(state["current_mi_topic"], "Employment")
    self.assertEqual(state["mi_topic_stack"], ["Economy", "Employment"])
    self.assertEqual(state["mi_topics_tried"], ["Economy", "Employment"])


if __name__ == "__main__":
  unittest.main()
