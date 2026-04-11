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

  def _arm_completed_address_tib_memory(
    self,
    state: dict,
    *,
    tib_name: str = "skepticism",
  ) -> None:
    state["last_completed_address_tib_name"] = tib_name
    state["last_completed_address_tib_route_bucket"] = "other"
    state["last_completed_address_tib_no_reroute_active"] = True

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

  def test_completed_address_tib_resumes_after_one_exchange_and_skips_tib_reroute(self) -> None:
    builder = self._make_builder()
    builder._maybe_handle_tib = (
      lambda state, traces:
      (_ for _ in ()).throw(AssertionError("_maybe_handle_tib should be skipped right after one-turn address_tib"))
    )
    builder._run_resistance_prediction = lambda state, traces: None
    builder._handle_pre_stage_routing = lambda state, traces: None
    builder._handle_stage_dispatch = lambda state, traces: "Resumed stage utterance"

    state = builder.initial_state(max_turns=12)
    state["current_stage"] = "address_tib"
    state["current_substage"] = "tib_deliver"
    state["predicted_resistance"] = 0.2
    state["address_tib_route_bucket"] = "other"
    state["address_tib_name"] = "skepticism"
    state["address_tib_description"] = "The patient says the exercise still feels pointless."
    state["address_tib_reasoning"] = "The patient continues doubting the usefulness of the current approach."
    state["address_tib_plan"] = "Acknowledge the skepticism and reconnect to the current exercise."
    state["address_tib_resume_stage"] = "cbt_technique"
    state["address_tib_resume_substage"] = "cbt_deliver_current"
    state["messages"] = [
      {
        "utterance_index": 1,
        "role": "therapist",
        "content": "I hear the skepticism, and I want to connect this back to what we're testing here.",
        "stage": "address_tib",
        "substage": "tib_deliver",
      },
      {
        "utterance_index": 2,
        "role": "patient",
        "content": "Okay, but I'm still not convinced.",
        "stage": "address_tib",
        "substage": "tib_deliver",
      },
    ]
    state["exchange_counts"] = {"address_tib": [1]}
    state = builder._ensure_state_defaults(state)

    updated = builder.therapist_turn(state)

    self.assertEqual(updated["current_stage"], "cbt_technique")
    self.assertEqual(updated["current_substage"], "cbt_deliver_current")
    self.assertEqual(updated["messages"][-1]["content"], "Resumed stage utterance")
    self.assertEqual(updated["messages"][-1]["stage"], "cbt_technique")
    self.assertEqual(updated["last_completed_address_tib_name"], "skepticism")
    self.assertTrue(updated["last_completed_address_tib_no_reroute_active"])
    self.assertEqual(updated["last_completed_address_tib_route_bucket"], "other")
    self.assertEqual(updated["address_tib_name"], "")
    self.assertEqual(updated["address_tib_description"], "")
    self.assertEqual(updated["address_tib_plan"], "")
    self.assertEqual(updated["reasoning_traces"][-1]["traces"][0]["step"], "address_tib_exit")

  def test_same_tib_is_handled_inline_without_reroute(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "skepticism",
        "tib_description": "The patient keeps saying the exercise feels pointless.",
        "reasoning": "The patient is repeating the same skepticism TIB.",
      }
    )
    state = self._make_state(builder, stage="cbt_technique", substage="cbt_deliver_current")
    self._arm_completed_address_tib_memory(state)
    traces: list[dict] = []

    bucket = builder._maybe_handle_tib(state, traces)

    self.assertEqual(bucket, "none")
    self.assertEqual(state["current_stage"], "cbt_technique")
    self.assertEqual(state["inline_tib_ack_name"], "skepticism")
    self.assertEqual(
      state["inline_tib_ack_description"],
      "The patient keeps saying the exercise feels pointless.",
    )
    self.assertTrue(state["last_completed_address_tib_no_reroute_active"])
    self.assertEqual(traces[-1]["step"], "address_tib_repeat_inline_suppressed")

  def test_same_tib_stays_inline_until_no_tib_then_can_route_again(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "skepticism",
        "tib_description": "The patient again doubts that the exercise will help.",
        "reasoning": "The same skepticism TIB is still present.",
      }
    )
    state = self._make_state(builder, stage="cbt_technique", substage="cbt_deliver_current")
    self._arm_completed_address_tib_memory(state)

    first_bucket = builder._maybe_handle_tib(state, [])
    builder._clear_inline_tib_ack(state)
    second_bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(first_bucket, "none")
    self.assertEqual(second_bucket, "none")
    self.assertTrue(state["last_completed_address_tib_no_reroute_active"])

    builder._run_therapist_structured = lambda state, traces, step_name, instruction: {
      "has_tib": False,
      "tib_name": "none",
      "tib_description": None,
      "reasoning": "The patient is no longer showing a TIB.",
    }
    cleared_bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(cleared_bucket, "none")
    self.assertFalse(state["last_completed_address_tib_no_reroute_active"])
    self.assertEqual(state["last_completed_address_tib_name"], "")

    builder._run_therapist_structured = lambda state, traces, step_name, instruction: {
      "has_tib": True,
      "tib_name": "skepticism",
      "tib_description": "The patient returns to saying the exercise feels pointless.",
      "reasoning": "The skepticism TIB has returned after a no-TIB turn.",
    }
    routed_bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(routed_bucket, "other")
    self.assertEqual(state["current_stage"], "address_tib")
    self.assertEqual(state["address_tib_name"], "skepticism")

  def test_different_other_tib_routes_normally_while_same_tib_memory_is_active(self) -> None:
    builder = self._make_builder(
      {
        "has_tib": True,
        "tib_name": "defensiveness",
        "tib_description": "The patient pushes back against the therapist's framing.",
        "reasoning": "This is a different TIB than the earlier skepticism.",
      }
    )
    state = self._make_state(builder, stage="cbt_technique", substage="cbt_deliver_current")
    self._arm_completed_address_tib_memory(state)

    bucket = builder._maybe_handle_tib(state, [])

    self.assertEqual(bucket, "other")
    self.assertEqual(state["current_stage"], "address_tib")
    self.assertEqual(state["address_tib_name"], "defensiveness")

  def test_inline_tib_ack_suffix_is_added_once_and_cleared(self) -> None:
    class _FakeTherapist:
      def __init__(self) -> None:
        self.last_instruction = ""

      def build_utterance_messages(self, **kwargs):
        return [{"role": "user", "content": kwargs["instruction"]}]

      def generate_utterance(self, **kwargs):
        self.last_instruction = kwargs["instruction"]
        return "Therapist reply"

    builder = self._make_builder()
    builder.therapist = _FakeTherapist()
    state = self._make_state(builder, stage="cbt_technique", substage="cbt_deliver_current")
    state["inline_tib_ack_name"] = "skepticism"
    state["inline_tib_ack_description"] = "The patient keeps doubting the value of the exercise."
    state["resume_context_active"] = True
    state["resume_context_from_stage"] = "address_tib"
    state["resume_context_reason"] = "single_turn_complete"
    state["resume_context_subject"] = "TIB name: skepticism"

    utterance = builder._generate_therapist_utterance(state, "Continue the CBT exercise.")

    self.assertEqual(utterance, "Therapist reply")
    self.assertIn("Do not reroute or start a separate TIB detour.", builder.therapist.last_instruction)
    self.assertIn("at most one sentence", builder.therapist.last_instruction)
    self.assertEqual(state["inline_tib_ack_name"], "")
    self.assertEqual(state["inline_tib_ack_description"], "")

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
