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


class HomeworkStageTest(unittest.TestCase):
  def _make_builder(self) -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = {}
    builder.cfg = SimpleNamespace(
      env=SimpleNamespace(
        max_turns=12,
        min_turns=0,
        long_session_turn_threshold=70,
        resistance_handling="mi_techniques",
      ),
      topic_exploration=SimpleNamespace(patient_classifier_path="", patient_wikipedia_dir=""),
    )
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}"
    builder._generate_therapist_utterance = lambda state, instruction: instruction
    builder._handle_stage_dispatch = lambda state, traces: f"dispatch:{state['current_stage']}"
    return builder

  def _make_state(self, builder: TherapyGraphBuilder) -> dict:
    state = builder.initial_state(max_turns=12)
    state["current_stage"] = "homework"
    state["current_substage"] = "homework_review_response"
    state["messages"] = [
      {
        "utterance_index": 1,
        "role": "therapist",
        "content": "Try this exercise.",
        "stage": "homework",
        "substage": "homework_propose_tasks",
      },
      {
        "utterance_index": 2,
        "role": "patient",
        "content": "I don't think I can do that.",
        "stage": "homework",
        "substage": "homework_propose_tasks",
      },
    ]
    state["proposed_tasks"] = [
      {
        "task_title": "Thought Log",
        "task_description": "Note one difficult moment.",
        "rationale": "",
        "expected_outcomes": "",
      }
    ]
    state["homework_waiting_for_patient"] = True
    return builder._ensure_state_defaults(state)

  def test_rejected_homework_reproposes_tasks_without_error(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["homework_proposal_attempts"] = 1
    builder._check_homework_agreement = lambda state, traces: (False, [])
    builder._propose_homework_tasks = lambda state, traces: (
      state.__setitem__(
        "proposed_tasks",
        [
          {
            "task_title": "Smaller Task",
            "task_description": "Write one line.",
            "rationale": "",
            "expected_outcomes": "",
          }
        ],
      ),
      state.__setitem__("homework_proposal_attempts", state["homework_proposal_attempts"] + 1),
    )

    result = builder._handle_homework(state, [])

    self.assertEqual(result, "homework/propose_tasks_utterance")
    self.assertEqual(state["current_substage"], "homework_propose_tasks")
    self.assertTrue(state["homework_waiting_for_patient"])
    self.assertEqual(state["homework_proposal_attempts"], 2)

  def test_rejected_homework_at_cap_moves_into_fallback(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder)
    state["homework_proposal_attempts"] = 5
    builder._check_homework_agreement = lambda state, traces: (False, [])

    result = builder._handle_homework(state, [])

    self.assertEqual(result, "homework/fallback_task_proposal_utterance")
    self.assertEqual(state["current_substage"], "homework_apply_fallback")
    self.assertFalse(state["homework_fallback_to_apply"])
    self.assertTrue(state["homework_fallback_done"])
    self.assertEqual(state["agreed_tasks"], state["proposed_tasks"])


if __name__ == "__main__":
  unittest.main()
