import sys
import types
import unittest
from types import SimpleNamespace


if "langgraph.graph" not in sys.modules:
  langgraph_module = types.ModuleType("langgraph")
  langgraph_graph_module = types.ModuleType("langgraph.graph")
  langgraph_graph_module.END = "END"
  langgraph_graph_module.START = "START"
  langgraph_graph_module.StateGraph = object
  sys.modules["langgraph"] = langgraph_module
  sys.modules["langgraph.graph"] = langgraph_graph_module

if "dotenv" not in sys.modules:
  dotenv_module = types.ModuleType("dotenv")
  dotenv_module.load_dotenv = lambda *args, **kwargs: None
  sys.modules["dotenv"] = dotenv_module

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
    @classmethod
    def from_template(cls, template):
      return cls(template)

    def __init__(self, template):
      self.template = template

    def format(self, **values):
      rendered = self.template
      for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
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
    def to_container(config, resolve=True):
      return config

    @staticmethod
    def is_config(config):
      return False

  omegaconf_module.OmegaConf = _OmegaConf
  sys.modules["omegaconf"] = omegaconf_module

if "openai" not in sys.modules:
  openai_module = types.ModuleType("openai")
  openai_module.OpenAI = object
  sys.modules["openai"] = openai_module


from graph.agent_graph_with_orch import AgentTherapyGraphWithOrchBuilder


class _FakeRepairLLM:
  def __init__(self, responses=None):
    self.responses = list(responses or [])
    self.calls = []

  def generate(self, messages, **kwargs):
    self.calls.append((messages, kwargs))
    if self.responses:
      return self.responses.pop(0)
    return '{"opening_info_update": {"mood": "sad", "problem": "work stress"}}'


class _FakeTherapist:
  def __init__(self, reasoning: str, utterance: str = "Visible response.", repair_responses=None):
    self.reasoning = reasoning
    self.utterance = utterance
    self.llm = _FakeRepairLLM(repair_responses)
    self.request_messages = []

  def build_agent_chatml_messages(self, history, use_ft_system_prompt=False):
    return [{"role": "system", "content": "SYSTEM"}, *history]

  def generate_agent_messages(self, messages, **kwargs):
    self.request_messages = messages
    raw = f"<think>\n{self.reasoning}\n</think>\n{self.utterance}"
    return self.utterance, self.reasoning, raw


def _builder(reasoning: str, repair_responses=None) -> AgentTherapyGraphWithOrchBuilder:
  builder = AgentTherapyGraphWithOrchBuilder.__new__(AgentTherapyGraphWithOrchBuilder)
  builder.cfg = SimpleNamespace(therapist={"type": "local"}, patient={"prompt_variant": "base"}, env={"resistance": False})
  builder.profile = {"idx": 1}
  builder.disable_tib = False
  builder.therapist = _FakeTherapist(reasoning, repair_responses=repair_responses)
  return builder


class AgentGraphWithOrchTest(unittest.TestCase):
  def test_parses_jsonl_reasoning_and_ignores_stage_for_structured_steps(self):
    builder = _builder(
      '{"stage": "Opening"}\n'
      '{"opening_info_update": {"mood": "sad", "problem": "work stress"}}\n'
      '{"technique_evaluation": {"evaluation": "The technique helped."}}'
    )
    state = {
      "messages": [{"utterance_index": 1, "role": "patient", "content": "I am stressed."}],
      "prompt_audit_records": [],
    }
    builder._begin_agent_orch_turn(state)
    traces = []

    parsed = builder._run_therapist_structured(
      state,
      traces,
      "opening_info_update",
      "instruction",
    )
    evaluation = builder._run_therapist_structured(
      state,
      traces,
      "technique_evaluation",
      "instruction",
      expected_type=str,
    )

    self.assertEqual(parsed, {"mood": "sad", "problem": "work stress"})
    self.assertEqual(evaluation, "The technique helped.")
    self.assertNotIn("stage", [trace["step"] for trace in traces])
    self.assertEqual(traces[0]["source"], "agent_reasoning")

  def test_missing_trace_is_repaired_and_metadata_is_recorded(self):
    builder = _builder(
      '{"stage": "Opening"}',
      repair_responses=['{"opening_info_update": {"mood": "sad", "problem": "work stress"}}'],
    )
    events = []
    builder.reasoning_retry_callback = events.append
    state = {"messages": [], "prompt_audit_records": []}
    builder._begin_agent_orch_turn(state)
    traces = []

    parsed = builder._run_therapist_structured(
      state,
      traces,
      "opening_info_update",
      "instruction",
    )

    self.assertEqual(parsed["problem"], "work stress")
    self.assertEqual(traces[0]["repair"]["attempts"], 1)
    self.assertEqual(traces[0]["repair"]["original"]["missing_step"], "opening_info_update")
    self.assertEqual(len(builder.therapist.llm.calls), 1)
    self.assertEqual(events[0]["event"], "reasoning_trace_repair_attempt")
    self.assertEqual(events[0]["step"], "opening_info_update")

  def test_text_reasoning_is_converted_once_before_structured_consumption(self):
    builder = _builder(
      "[stage] Opening\n"
      "[opening_info_update] Mood: sad. Problem: work stress\n"
      "[opening_readiness_check] Ready: yes. Issue: none. Reasoning: enough info.",
      repair_responses=[
        '{"traces": ['
        '{"name": "stage", "content": "Opening"},'
        '{"name": "opening_info_update", "content": {"mood": "sad", "problem": "work stress"}},'
        '{"name": "opening_readiness_check", "content": {"ready": true, "issue": "none", "reasoning": "enough info."}}'
        ']}'
      ],
    )
    builder.parse_text_reasoning = True
    state = {
      "messages": [{"utterance_index": 1, "role": "patient", "content": "I am stressed."}],
      "prompt_audit_records": [],
    }
    builder._begin_agent_orch_turn(state)
    traces = []

    update = builder._run_therapist_structured(state, traces, "opening_info_update", "instruction")
    readiness = builder._run_therapist_structured(state, traces, "opening_readiness_check", "instruction")

    self.assertEqual(update, {"mood": "sad", "problem": "work stress"})
    self.assertEqual(readiness["ready"], True)
    self.assertEqual(len(builder.therapist.llm.calls), 1)
    self.assertIn("text-to-JSON conversion", builder.therapist.llm.calls[0][0][0]["content"])

  def test_text_reasoning_missing_converted_trace_uses_single_trace_repair(self):
    builder = _builder(
      "[stage] Opening\n[opening_info_update] Mood: sad. Problem: work stress",
      repair_responses=[
        '{"traces": [{"name": "stage", "content": "Opening"}]}',
        '{"opening_info_update": {"mood": "sad", "problem": "work stress"}}',
      ],
    )
    builder.parse_text_reasoning = True
    state = {"messages": [], "prompt_audit_records": []}
    builder._begin_agent_orch_turn(state)
    traces = []

    update = builder._run_therapist_structured(state, traces, "opening_info_update", "instruction")

    self.assertEqual(update["problem"], "work stress")
    self.assertEqual(traces[0]["repair"]["attempts"], 1)
    self.assertEqual(len(builder.therapist.llm.calls), 2)


if __name__ == "__main__":
  unittest.main()
