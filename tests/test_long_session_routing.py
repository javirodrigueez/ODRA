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


class LongSessionRoutingTest(unittest.TestCase):
  def _make_builder(self) -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = {}
    builder.cfg = SimpleNamespace(
      env=SimpleNamespace(max_turns=100, min_turns=0, long_session_turn_threshold=70, resistance_handling="mi_techniques"),
      topic_exploration=SimpleNamespace(patient_classifier_path="", patient_wikipedia_dir=""),
    )
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}"
    builder._handle_stage_dispatch = lambda state, traces: f"dispatch:{state['current_stage']}"
    builder._run_therapist_structured = lambda *args, **kwargs: {"apply_more": True}
    builder._generate_therapist_utterance = lambda state, instruction: instruction
    return builder

  def _make_state(self, builder: TherapyGraphBuilder) -> dict:
    state = builder.initial_state(max_turns=100)
    state["messages"] = [
      {
        "utterance_index": 1,
        "role": "therapist",
        "content": "Start",
        "stage": "opening",
        "substage": "open_initial",
      },
      {
        "utterance_index": 2,
        "role": "patient",
        "content": "Reply",
        "stage": "opening",
        "substage": "open_initial",
      },
    ]
    return builder._ensure_state_defaults(state)

  def test_cbt_threshold_routes_to_homework_after_current_technique_completes(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["current_stage"] = "cbt_technique"
    state["current_substage"] = "cbt_review_current"
    state["long_session_turn_threshold"] = 70
    state["utterance_index"] = 70
    state["selected_techniques"] = [{"name": "Technique A", "target_distortion": "", "rationale": ""}]
    state["current_technique_index"] = 0
    state["current_technique_plan"] = "Active plan"
    state["current_technique_current_progress"] = "Progress"
    state["current_technique_expected_outcome"] = "Outcome"
    state["technique_turn_count"] = 1
    builder._evaluate_technique_completion = lambda state, traces: True
    builder._evaluate_technique = lambda state, traces: None

    result = builder._handle_cbt_technique(state, [])

    self.assertEqual(result, "dispatch:homework")
    self.assertEqual(state["current_stage"], "homework")

  def test_cbt_threshold_does_not_start_new_technique(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["current_stage"] = "cbt_technique"
    state["current_substage"] = "cbt_decide_next"
    state["long_session_turn_threshold"] = 70
    state["utterance_index"] = 70
    state["selected_techniques"] = [{"name": "Technique A", "target_distortion": "", "rationale": ""}]
    state["current_technique_index"] = 1
    state["current_technique_plan"] = ""
    builder._select_additional_technique = lambda state, traces: (_ for _ in ()).throw(AssertionError("should not select"))

    result = builder._handle_cbt_technique(state, [])

    self.assertEqual(result, "dispatch:homework")
    self.assertEqual(state["current_stage"], "homework")

  def test_cbt_threshold_allows_current_technique_to_continue(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["current_stage"] = "cbt_technique"
    state["current_substage"] = "cbt_review_current"
    state["long_session_turn_threshold"] = 70
    state["utterance_index"] = 70
    state["selected_techniques"] = [{"name": "Technique A", "target_distortion": "", "rationale": ""}]
    state["current_technique_index"] = 0
    state["current_technique_plan"] = "Active plan"
    state["current_technique_current_progress"] = "Progress"
    state["current_technique_expected_outcome"] = "Outcome"
    state["technique_turn_count"] = 1
    builder._evaluate_technique_completion = lambda state, traces: False
    builder._run_therapist_structured = lambda *args, **kwargs: {
      "plan_description": "Active plan",
      "current_progress": "Progress",
    }

    result = builder._handle_cbt_technique(state, [])

    self.assertEqual(result, "cbt_technique/deliver_current")
    self.assertEqual(state["current_stage"], "cbt_technique")

  def test_hard_cap_routes_into_closing_fallback(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["current_stage"] = "mi_techniques"
    state["current_substage"] = "mi_choose_technique"
    state["max_turns"] = 100
    state["utterance_index"] = 98

    builder._maybe_force_closing(state)

    self.assertEqual(state["current_stage"], "closing")
    self.assertEqual(state["current_substage"], "closing_fallback")
    self.assertEqual(state["session_ended_reason"], "max_turns_forced_closing")
    self.assertTrue(state["final_patient_pending_end"])

  def test_long_session_threshold_blocks_mi_reentry_from_homework(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["current_stage"] = "homework"
    state["current_substage"] = "homework_review_response"
    state["predicted_resistance"] = 0.9
    state["preconsecutive_turns"] = 3
    state["utterance_index"] = 70
    state["long_session_turn_threshold"] = 70
    state["mi_active"] = True
    state["current_mi_technique_selection"] = ["Open Question"]
    state["mi_recovery_turns"] = 2
    state["cami_recovery_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "homework")
    self.assertEqual(state["current_substage"], "homework_review_response")
    self.assertFalse(state["mi_active"])
    self.assertEqual(state["current_mi_technique_selection"], [])
    self.assertEqual(state["mi_recovery_turns"], 0)
    self.assertEqual(state["cami_recovery_turns"], 0)

  def test_legacy_forced_closing_reroutes_to_closing_fallback(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["current_stage"] = "forced_closing"
    state["current_substage"] = "forced_closing_end"

    result = builder._handle_forced_closing(state, [])

    self.assertEqual(result, "closing/fallback_utterance")
    self.assertEqual(state["current_stage"], "closing")
    self.assertEqual(state["current_substage"], "closing_fallback")
    self.assertEqual(state["session_ended_reason"], "max_turns_forced_closing")


if __name__ == "__main__":
  unittest.main()
