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

  omegaconf_module.OmegaConf = _OmegaConf
  sys.modules["omegaconf"] = omegaconf_module

if "openai" not in sys.modules:
  openai_module = types.ModuleType("openai")
  openai_module.OpenAI = object
  sys.modules["openai"] = openai_module

from graph.graph import TherapyGraphBuilder
from graph.state import DEFAULT_SUBSTAGE_BY_STAGE


class OpeningStageTest(unittest.TestCase):
  def _make_builder(self, *, resistance_handling: str = "mi_techniques") -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = {}
    builder.cfg = SimpleNamespace(env=SimpleNamespace(max_turns=12, min_turns=0, resistance_handling=resistance_handling))
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}"
    builder._generate_therapist_utterance = lambda state, instruction: f"utterance:{instruction}"
    builder._handle_stage_dispatch = lambda state, traces: f"dispatch:{state['current_stage']}"
    builder._enter_mi = (
      lambda state, target_stage, resume_stage, resume_substage, reason:
      builder._set_stage(state, target_stage, DEFAULT_SUBSTAGE_BY_STAGE[target_stage])
    )
    return builder

  def _make_state(
    self,
    builder: TherapyGraphBuilder,
    opening_exchanges: int,
    predicted_resistance: float,
    *,
    preconsecutive_turns: int = 0,
  ) -> dict:
    state = builder.initial_state(max_turns=12)
    state["current_stage"] = "opening"
    state["current_substage"] = "open_problem"
    state["predicted_resistance"] = predicted_resistance
    state["preconsecutive_turns"] = preconsecutive_turns
    state["exchange_counts"]["opening"] = [opening_exchanges]
    return builder._ensure_state_defaults(state)

  def test_opening_does_not_fallback_before_five_exchanges(self) -> None:
    builder = self._make_builder()
    builder._check_readiness = lambda state, traces: False
    state = self._make_state(builder, opening_exchanges=4, predicted_resistance=0.95)

    result = builder._handle_opening(state, [])

    self.assertEqual(result, "utterance:opening/problem_question")
    self.assertEqual(state["current_stage"], "opening")
    self.assertFalse(state["opening_fallback_triggered"])

  def test_opening_cap_routes_to_id_distortions_without_resistance_gate(self) -> None:
    builder = self._make_builder()
    builder._check_readiness = lambda state, traces: False
    state = self._make_state(builder, opening_exchanges=5, predicted_resistance=0.95, preconsecutive_turns=4)

    result = builder._handle_opening(state, [])

    self.assertEqual(result, "dispatch:id_distortions")
    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertTrue(state["opening_fallback_triggered"])

  def test_opening_cap_routes_to_mi_techniques_after_precon_streak_in_legacy_mode(self) -> None:
    builder = self._make_builder(resistance_handling="mi_techniques")
    builder._check_readiness = lambda state, traces: False
    state = self._make_state(builder, opening_exchanges=5, predicted_resistance=0.95, preconsecutive_turns=5)

    result = builder._handle_opening(state, [])

    self.assertEqual(result, "dispatch:mi_techniques")
    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertTrue(state["opening_fallback_triggered"])

  def test_opening_cap_routes_to_cami_stage_after_precon_streak(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    builder._check_readiness = lambda state, traces: False
    state = self._make_state(builder, opening_exchanges=5, predicted_resistance=0.95, preconsecutive_turns=5)

    result = builder._handle_opening(state, [])

    self.assertEqual(result, "dispatch:mi_techniques_with_te")
    self.assertEqual(state["current_stage"], "mi_techniques_with_te")
    self.assertTrue(state["opening_fallback_triggered"])

  def test_opening_cap_ignores_precon_streak_when_resistance_handling_is_none(self) -> None:
    builder = self._make_builder(resistance_handling="none")
    builder._check_readiness = lambda state, traces: False
    state = self._make_state(builder, opening_exchanges=5, predicted_resistance=0.95, preconsecutive_turns=5)

    result = builder._handle_opening(state, [])

    self.assertEqual(result, "dispatch:id_distortions")
    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertTrue(state["opening_fallback_triggered"])

  def test_opening_cap_uses_total_session_opening_turns_not_only_current_stint(self) -> None:
    builder = self._make_builder()
    builder._check_readiness = lambda state, traces: False
    state = self._make_state(builder, opening_exchanges=1, predicted_resistance=0.60)
    state["exchange_counts"]["opening"] = [4, 1]

    result = builder._handle_opening(state, [])

    self.assertEqual(result, "dispatch:id_distortions")
    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertTrue(state["opening_fallback_triggered"])


if __name__ == "__main__":
  unittest.main()
