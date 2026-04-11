import unittest
import sys
import types
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


class TIBRoutingTest(unittest.TestCase):
  def _make_builder(self, tib_payload: dict | None = None) -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = {}
    builder.cfg = SimpleNamespace(env=SimpleNamespace(max_turns=12, min_turns=0))
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}"
    if tib_payload is not None:
      builder._run_therapist_structured = lambda state, traces, step_name, instruction: tib_payload
    builder._enter_mi = (
      lambda state, target_stage, resume_stage, resume_substage, reason:
      builder._set_stage(state, target_stage, DEFAULT_SUBSTAGE_BY_STAGE[target_stage])
    )
    return builder

  def _make_state(
    self,
    builder: TherapyGraphBuilder,
    *,
    stage: str = "opening",
    substage: str = "open_initial",
  ) -> dict:
    state = builder.initial_state(max_turns=12)
    state["current_stage"] = stage
    state["current_substage"] = substage
    state["messages"] = [
      {
        "utterance_index": 1,
        "role": "therapist",
        "content": "Let's keep going.",
        "stage": stage,
        "substage": substage,
      },
      {
        "utterance_index": 2,
        "role": "patient",
        "content": "I don't know.",
        "stage": stage,
        "substage": substage,
      },
    ]
    return builder._ensure_state_defaults(state)

  def test_no_tib_keeps_current_flow(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": False,
        "tib_name": "none",
        "tib_description": None,
        "reasoning": "No interfering behavior is present.",
      }
    )
    state = self._make_state(builder)

    bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(bucket, "none")
    self.assertEqual(state["current_stage"], "opening")
    self.assertFalse(state["tib_redirect_triggered"])

  def test_end_session_routes_to_existing_meta_conversation_flow(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "end_session",
        "tib_description": "The patient says they want to stop for today.",
        "reasoning": "This is an explicit request to end the session.",
      }
    )
    state = self._make_state(builder, stage="opening", substage="open_problem")

    bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(bucket, "end_session")
    self.assertEqual(state["current_stage"], "meta_conversation")
    self.assertEqual(state["current_substage"], "mi_meta_open")
    self.assertTrue(state["mi_end_session_tib_redirect_used"])
    self.assertTrue(state["tib_redirect_triggered"])
    self.assertEqual(state["tib_redirect_level"], "yes")

  def test_exact_avoidance_routes_like_old_item_move(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "avoidance",
        "tib_description": "The patient asks to move away from the current topic.",
        "reasoning": "This is direct topic avoidance.",
      }
    )
    state = self._make_state(builder, stage="opening", substage="open_mood")

    bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(bucket, "avoidance")
    self.assertEqual(state["current_stage"], "opening")
    self.assertEqual(state["current_substage"], "open_problem")
    self.assertTrue(state["tib_redirect_triggered"])
    self.assertEqual(state["tib_redirect_level"], "partial")

  def test_avoidance_alias_routes_like_avoidance(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "social_avoidance_spike",
        "tib_description": "The patient backs away from the exercise and asks to skip it.",
        "reasoning": "The classifier named a more specific avoidance variant.",
      }
    )
    state = self._make_state(builder, stage="opening", substage="open_mood")

    bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(bucket, "avoidance")
    self.assertEqual(state["current_substage"], "open_problem")

  def test_other_tib_enters_address_tib_and_stores_payload(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "skepticism",
        "tib_description": "The patient says the exercise feels pointless.",
        "reasoning": "The patient doubts the usefulness of the current approach.",
      }
    )
    state = self._make_state(builder, stage="cbt_technique", substage="cbt_deliver_current")

    bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(bucket, "other")
    self.assertEqual(state["current_stage"], "address_tib")
    self.assertEqual(state["current_substage"], "tib_plan")
    self.assertEqual(state["address_tib_route_bucket"], "other")
    self.assertEqual(state["address_tib_name"], "skepticism")
    self.assertEqual(state["address_tib_description"], "The patient says the exercise feels pointless.")
    self.assertEqual(state["address_tib_reasoning"], "The patient doubts the usefulness of the current approach.")
    self.assertEqual(state["address_tib_resume_stage"], "cbt_technique")
    self.assertEqual(state["address_tib_resume_substage"], "cbt_deliver_current")

  def test_resume_state_normalizes_legacy_patient_demand_shape(self) -> None:
    builder = self._make_builder()
    saved_state = {
      "current_stage": "patient_demand",
      "current_substage": "demand_deliver",
      "messages": [
        {
          "utterance_index": 1,
          "role": "patient",
          "content": "I don't see how this helps.",
          "stage": "patient_demand",
          "substage": "demand_plan",
        }
      ],
      "patient_profile": {},
      "patient_demand_text": "The patient wants the therapist to explain why the exercise matters.",
      "patient_demand_plan": "Acknowledge the concern and reconnect to the current exercise.",
      "patient_demand_reasoning": "Legacy detour reasoning.",
      "patient_demand_resume_stage": "cbt_technique",
      "patient_demand_resume_substage": "cbt_deliver_current",
      "resume_context_from_stage": "patient_demand",
      "mi_end_session_demand_redirect_used": True,
      "high_resistance_redirect_triggered": True,
      "high_resistance_level": "partial",
      "high_resistance_source_stage": "patient_demand",
      "high_resistance_source_substage": "demand_check",
      "exchange_counts": {"patient_demand": [2]},
      "active_exchange_stage": "patient_demand",
      "active_exchange_substage": "demand_resume",
    }

    resumed = builder.resume_state(saved_state, max_turns=12)

    self.assertEqual(resumed["current_stage"], "address_tib")
    self.assertEqual(resumed["current_substage"], "tib_deliver")
    self.assertEqual(resumed["messages"][0]["stage"], "address_tib")
    self.assertEqual(resumed["messages"][0]["substage"], "tib_plan")
    self.assertEqual(resumed["address_tib_name"], "legacy_patient_demand")
    self.assertEqual(
      resumed["address_tib_description"],
      "The patient wants the therapist to explain why the exercise matters.",
    )
    self.assertEqual(
      resumed["address_tib_plan"],
      "Acknowledge the concern and reconnect to the current exercise.",
    )
    self.assertEqual(resumed["address_tib_resume_stage"], "cbt_technique")
    self.assertEqual(resumed["address_tib_resume_substage"], "cbt_deliver_current")
    self.assertEqual(resumed["address_tib_route_bucket"], "other")
    self.assertTrue(resumed["mi_end_session_tib_redirect_used"])
    self.assertTrue(resumed["tib_redirect_triggered"])
    self.assertEqual(resumed["tib_redirect_source_stage"], "address_tib")
    self.assertEqual(resumed["tib_redirect_source_substage"], "tib_check")
    self.assertEqual(resumed["resume_context_from_stage"], "address_tib")
    self.assertEqual(resumed["active_exchange_stage"], "address_tib")
    self.assertEqual(resumed["active_exchange_substage"], "tib_resume")
    self.assertIn("address_tib", resumed["exchange_counts"])
    self.assertNotIn("patient_demand", resumed["exchange_counts"])


if __name__ == "__main__":
  unittest.main()
