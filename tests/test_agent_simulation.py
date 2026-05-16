import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock


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
    def __init__(self, template: str):
      self.template = template

    @classmethod
    def from_template(cls, template: str) -> "_PromptTemplate":
      return cls(template)

    def format(self, **values: str) -> str:
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
    def to_container(config, resolve: bool = True):
      return config

    @staticmethod
    def is_config(config) -> bool:
      return False

    @staticmethod
    def create(config):
      return SimpleNamespace(
        therapist=config.get("therapist", {}),
        patient=config.get("patient", {}),
        env=SimpleNamespace(**config.get("env", {})),
        topic_exploration=config.get("topic_exploration", {}),
      )

    @staticmethod
    def load(path):
      return {}

  omegaconf_module.OmegaConf = _OmegaConf
  sys.modules["omegaconf"] = omegaconf_module

if "openai" not in sys.modules:
  openai_module = types.ModuleType("openai")
  openai_module.OpenAI = object
  sys.modules["openai"] = openai_module


import agent_simulation
from agents.patient import PatientAgent
from agents.therapist import (
  LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
  TherapistAgent,
  parse_agent_therapist_response,
)
from agents.interfaces import LLMFactory, LocalLLM, OpenaiLLM
from graph.agent_graph import AgentTherapyGraphBuilder


def _profile() -> dict:
  return {
    "idx": 1,
    "attitude": "neutral",
    "emotional_recovery": "moderate",
    "personal_info": {"name": "Jason Crawford"},
    "presenting_problem": "Commute stress.",
    "reason_seeking_counseling": "Need help with stress.",
    "more_information": {"past_history": "None."},
    "patterns": ["catastrophizing"],
  }


class _FakeLLM:
  def __init__(self, response: str = "") -> None:
    self.response = response
    self.calls: list[tuple[list[dict], dict]] = []

  def generate(self, messages, **kwargs):
    self.calls.append((messages, kwargs))
    return self.response


class _RecordingCompletions:
  def __init__(self) -> None:
    self.kwargs = {}

  def create(self, **kwargs):
    self.kwargs = kwargs
    return SimpleNamespace(
      choices=[
        SimpleNamespace(
          message=SimpleNamespace(content="ok"),
        )
      ]
    )


class AgentTherapistPromptTest(unittest.TestCase):
  def test_builds_exact_agent_prompt_shape(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system.txt"
      prompt_path.write_text("AGENT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      with mock.patch("agents.therapist.AGENT_THERAPIST_SYSTEM_PROMPT", prompt_path):
        agent = TherapistAgent(_FakeLLM(), _profile())
        messages = agent.build_agent_utterance_messages("Patient: hello")

    self.assertIn("AGENT SYSTEM", messages[0]["content"])
    self.assertIn("Name: Jason Crawford", messages[0]["content"])
    self.assertEqual(
      messages[1]["content"],
      "<context>\nPatient: hello\n</context>\nContinue with the ongoing therapy session.",
    )

  def test_builds_agent_initial_prompt_from_opening_greeting(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system.txt"
      prompt_path.write_text("AGENT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      with mock.patch("agents.therapist.AGENT_THERAPIST_SYSTEM_PROMPT", prompt_path):
        agent = TherapistAgent(_FakeLLM(), _profile())
        messages = agent.build_agent_initial_utterance_messages()

    expected_prompt = (Path(__file__).resolve().parents[1] / "prompts" / "opening" / "initial_greeting.txt").read_text(
      encoding="utf-8"
    ).strip()
    self.assertIn("AGENT SYSTEM", messages[0]["content"])
    self.assertEqual(messages[1]["content"], expected_prompt)

  def test_uses_configured_agent_system_prompt_paths(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      regular_prompt_path = Path(tmp) / "custom_therapist_system.txt"
      ft_prompt_path = Path(tmp) / "custom_therapist_system_ft.txt"
      regular_prompt_path.write_text("CUSTOM REGULAR\n{patient_intake_form}", encoding="utf-8")
      ft_prompt_path.write_text("CUSTOM FT\n{patient_intake_form}", encoding="utf-8")
      agent = TherapistAgent(
        _FakeLLM(),
        _profile(),
        agent_system_prompt_path=regular_prompt_path,
        agent_ft_system_prompt_path=ft_prompt_path,
      )

      regular_messages = agent.build_agent_utterance_messages("Patient: hello")
      ft_messages = agent.build_agent_initial_utterance_messages(use_ft_system_prompt=True)

    self.assertIn("CUSTOM REGULAR", regular_messages[0]["content"])
    self.assertIn("Name: Jason Crawford", regular_messages[0]["content"])
    self.assertIn("CUSTOM FT", ft_messages[0]["content"])
    self.assertNotIn("CUSTOM REGULAR", ft_messages[0]["content"])

  def test_builds_local_agent_initial_prompt_with_ft_system_prompt(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      regular_prompt_path = Path(tmp) / "therapist_system.txt"
      ft_prompt_path = Path(tmp) / "therapist_system_ft.txt"
      regular_prompt_path.write_text("REGULAR SYSTEM\n{patient_intake_form}", encoding="utf-8")
      ft_prompt_path.write_text("FT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      with mock.patch("agents.therapist.AGENT_THERAPIST_SYSTEM_PROMPT", regular_prompt_path):
        with mock.patch("agents.therapist.AGENT_THERAPIST_FT_SYSTEM_PROMPT", ft_prompt_path):
          agent = TherapistAgent(_FakeLLM(), _profile())
          messages = agent.build_agent_initial_utterance_messages(use_ft_system_prompt=True)

    expected_prompt = (Path(__file__).resolve().parents[1] / "prompts" / "opening" / "initial_greeting.txt").read_text(
      encoding="utf-8"
    ).strip()
    self.assertIn("FT SYSTEM", messages[0]["content"])
    self.assertNotIn("REGULAR SYSTEM", messages[0]["content"])
    self.assertEqual(messages[1]["content"], expected_prompt)

  def test_builds_local_chatml_messages_with_ft_system_prompt(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      regular_prompt_path = Path(tmp) / "therapist_system.txt"
      ft_prompt_path = Path(tmp) / "therapist_system_ft.txt"
      regular_prompt_path.write_text("REGULAR SYSTEM\n{patient_intake_form}", encoding="utf-8")
      ft_prompt_path.write_text("FT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      with mock.patch("agents.therapist.AGENT_THERAPIST_SYSTEM_PROMPT", regular_prompt_path):
        with mock.patch("agents.therapist.AGENT_THERAPIST_FT_SYSTEM_PROMPT", ft_prompt_path):
          agent = TherapistAgent(_FakeLLM(), _profile())
          messages = agent.build_agent_chatml_messages(
            [{"role": "user", "content": "I am stuck."}],
            use_ft_system_prompt=True,
          )

    self.assertIn("FT SYSTEM", messages[0]["content"])
    self.assertNotIn("REGULAR SYSTEM", messages[0]["content"])
    self.assertEqual(messages[1], {"role": "user", "content": "I am stuck."})

  def test_parses_reasoning_block_without_leaking_into_utterance(self) -> None:
    reasoning, utterance = parse_agent_therapist_response("<think>plan\nline</think>Hi there.")

    self.assertEqual(reasoning, "plan\nline")
    self.assertEqual(utterance, "Hi there.")

  def test_parses_lone_closing_think_delimiter(self) -> None:
    reasoning, utterance = parse_agent_therapist_response("plan\nline</think>\nHi there.")

    self.assertEqual(reasoning, "plan\nline")
    self.assertEqual(utterance, "Hi there.")

  def test_missing_or_malformed_reasoning_block_is_treated_as_plain_utterance(self) -> None:
    self.assertEqual(parse_agent_therapist_response("Hi there."), ("", "Hi there."))
    self.assertEqual(
      parse_agent_therapist_response("<think>unfinished Hi there."),
      ("", "<think>unfinished Hi there."),
    )

  def test_agent_generation_preserves_local_thinking_output(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system.txt"
      prompt_path.write_text("AGENT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      llm = _FakeLLM("<think>reason</think>Hello")
      with mock.patch("agents.therapist.AGENT_THERAPIST_SYSTEM_PROMPT", prompt_path):
        agent = TherapistAgent(llm, _profile())
        utterance, reasoning, raw = agent.generate_agent_utterance("none")

    self.assertEqual(utterance, "Hello")
    self.assertEqual(reasoning, "reason")
    self.assertEqual(raw, "<think>reason</think>Hello")
    self.assertFalse(llm.calls[0][1]["strip_thinking"])

  def test_local_agent_default_max_tokens_can_be_overridden_by_config(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system_ft.txt"
      prompt_path.write_text("FT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      llm = _FakeLLM("<think>reason</think>Hello")
      with mock.patch("agents.therapist.AGENT_THERAPIST_FT_SYSTEM_PROMPT", prompt_path):
        agent = TherapistAgent(llm, _profile(), generation_defaults={"max_tokens": 123})
        agent.generate_agent_messages(
          [{"role": "user", "content": "hi"}],
          default_max_tokens=LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
        )

    self.assertEqual(llm.calls[0][1]["max_tokens"], 123)
    self.assertFalse(llm.calls[0][1]["strip_thinking"])

  def test_local_agent_default_max_tokens_is_large_without_config_override(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system_ft.txt"
      prompt_path.write_text("FT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      llm = _FakeLLM("<think>reason</think>Hello")
      with mock.patch("agents.therapist.AGENT_THERAPIST_FT_SYSTEM_PROMPT", prompt_path):
        agent = TherapistAgent(llm, _profile())
        agent.generate_agent_messages(
          [{"role": "user", "content": "hi"}],
          default_max_tokens=LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
        )

    self.assertEqual(llm.calls[0][1]["max_tokens"], LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS)
    self.assertFalse(llm.calls[0][1]["strip_thinking"])

  def test_local_llm_preserves_configured_max_model_len(self) -> None:
    with mock.patch("agents.interfaces.OpenAI", return_value=object()):
      llm = LLMFactory.create_from_config(
        {
          "type": "local",
          "base_url": "http://localhost:8000/v1",
          "model_id": "local-model",
          "max_model_len": "65536",
        }
      )

    self.assertIsInstance(llm, LocalLLM)
    self.assertEqual(llm.max_model_len, 65536)

  def test_openai_llm_uses_gpt_api_key_and_omits_extra_body(self) -> None:
    completions = _RecordingCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with mock.patch.dict(os.environ, {"GPT_API_KEY": "test-key"}):
      with mock.patch("agents.interfaces.OpenAI", return_value=fake_client) as openai_cls:
        llm = LLMFactory.create("openai", model_id="gpt-test")
        output = llm.generate(
          [{"role": "user", "content": "hello"}],
          max_tokens=12,
          temperature=0.2,
        )

    self.assertIsInstance(llm, OpenaiLLM)
    self.assertEqual(openai_cls.call_args.kwargs["api_key"], "test-key")
    self.assertEqual(openai_cls.call_args.kwargs["base_url"], "https://api.openai.com/v1")
    self.assertEqual(output, "ok")
    self.assertEqual(completions.kwargs["model"], "gpt-test")
    self.assertEqual(completions.kwargs["messages"], [{"role": "user", "content": "hello"}])
    self.assertEqual(completions.kwargs["max_tokens"], 12)
    self.assertEqual(completions.kwargs["temperature"], 0.2)
    self.assertNotIn("extra_body", completions.kwargs)


class AgentGraphTest(unittest.TestCase):
  def _builder(self, *, resistance: bool = False) -> AgentTherapyGraphBuilder:
    builder = AgentTherapyGraphBuilder.__new__(AgentTherapyGraphBuilder)
    builder.profile = _profile()
    builder.cfg = SimpleNamespace(
      therapist={"type": "deepseek"},
      patient={"prompt_variant": "base"},
      env={"resistance": resistance, "max_turns": 8, "min_turns": 0},
    )
    return builder

  def test_formats_last_10_context_with_therapist_reasoning(self) -> None:
    builder = self._builder()
    messages = [
      {"utterance_index": idx, "role": "patient" if idx % 2 == 0 else "therapist", "content": f"M{idx}"}
      for idx in range(1, 13)
    ]
    state = {
      "messages": messages,
      "reasoning_traces": [
        {
          "utterance_index": 3,
          "speaker": "therapist",
          "traces": [{"step": "agent_reasoning", "input": "old", "output": "Reason 3"}],
        },
        {
          "utterance_index": 11,
          "speaker": "therapist",
          "traces": [{"step": "agent_reasoning", "input": "new", "output": "Reason 11"}],
        },
      ],
    }

    context = builder._format_agent_prompt_context(state)
    context_lines = context.splitlines()

    self.assertFalse(any(line.endswith("M1") for line in context_lines))
    self.assertFalse(any(line.endswith("M2") for line in context_lines))
    self.assertIn("Therapist: <think>Reason 3</think> M3", context)
    self.assertIn("Therapist: <think>Reason 11</think> M11", context)
    self.assertEqual(len(context_lines), 10)

  def test_builds_recent_local_chatml_history_with_therapist_reasoning(self) -> None:
    builder = self._builder()
    messages = [
      {"utterance_index": idx, "role": "patient" if idx % 2 == 0 else "therapist", "content": f"M{idx}"}
      for idx in range(1, 25)
    ]
    state = {
      "messages": messages,
      "reasoning_traces": [
        {
          "utterance_index": 3,
          "speaker": "therapist",
          "traces": [{"step": "agent_reasoning", "input": "old", "output": "Reason 3"}],
        },
        {
          "utterance_index": 11,
          "speaker": "therapist",
          "traces": [{"step": "agent_reasoning", "input": "new", "output": "Reason 11"}],
        },
      ],
    }

    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system_ft.txt"
      prompt_path.write_text("FT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      with mock.patch("agents.therapist.AGENT_THERAPIST_FT_SYSTEM_PROMPT", prompt_path):
        builder.therapist = TherapistAgent(_FakeLLM(), _profile())
        request_messages = builder._build_local_therapist_request_messages(state)

    self.assertEqual(request_messages[0]["role"], "system")
    self.assertIn("FT SYSTEM", request_messages[0]["content"])
    self.assertEqual(len(request_messages), 21)
    self.assertEqual(request_messages[1], {"role": "assistant", "content": "M5"})
    self.assertEqual(request_messages[2], {"role": "user", "content": "M6"})
    self.assertFalse(any(message.get("content") == "M1" for message in request_messages))
    self.assertFalse(any(message.get("content") == "M4" for message in request_messages))
    self.assertEqual(
      request_messages[7],
      {"role": "assistant", "content": "<think>\nReason 11\n</think>\nM11"},
    )
    self.assertEqual(request_messages[-1], {"role": "user", "content": "M24"})

  def test_local_therapist_turn_uses_chatml_messages_without_continuation_instruction(self) -> None:
    builder = self._builder(resistance=False)
    builder.cfg.therapist = {"type": "local"}
    builder.therapist = _FakeLocalTherapist()
    builder.patient = _FakePatient()
    state = builder.initial_state(max_turns=6)
    state["messages"] = [{"utterance_index": 1, "role": "patient", "content": "I am stuck."}]
    state["utterance_index"] = 1

    result = builder.therapist_turn(state)

    self.assertEqual(result["messages"][-1]["content"], "Let's explore that.")
    self.assertEqual(builder.therapist.request_messages, [
      {"role": "system", "content": "SYSTEM"},
      {"role": "user", "content": "I am stuck."},
    ])
    self.assertTrue(builder.therapist.used_ft_system_prompt)
    self.assertEqual(builder.therapist.generation_kwargs["default_max_tokens"], LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS)
    self.assertIsNone(builder.therapist.legacy_context)

  def test_local_therapist_first_turn_uses_initial_greeting_prompt(self) -> None:
    builder = self._builder(resistance=False)
    builder.cfg.therapist = {"type": "local"}
    builder.therapist = _FakeLocalInitialTherapist()
    builder.patient = _FakePatient()

    result = builder.therapist_turn(builder.initial_state(max_turns=6))

    self.assertEqual(result["messages"][0]["content"], "Hello.")
    self.assertTrue(builder.therapist.initial_called)
    self.assertTrue(builder.therapist.used_ft_system_prompt)
    self.assertEqual(builder.therapist.generation_kwargs["default_max_tokens"], LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS)
    self.assertFalse(builder.therapist.chatml_called)

  def test_therapist_reasoning_is_not_saved_or_shown_to_patient(self) -> None:
    builder = self._builder(resistance=False)
    builder.therapist = TherapistAgent(
      _FakeLLM("Private therapist reasoning</think>\nVisible therapist utterance."),
      _profile(),
    )
    builder.patient = _RecordingPatient("Patient response.")
    state = builder.initial_state(max_turns=6)
    state["messages"] = [{"utterance_index": 1, "role": "patient", "content": "I am stuck."}]
    state["utterance_index"] = 1

    after_therapist = builder.therapist_turn(state)
    therapist_message = after_therapist["messages"][-1]
    therapist_trace = next(
      trace_group["traces"][0]
      for trace_group in after_therapist["reasoning_traces"]
      if trace_group["speaker"] == "therapist"
    )
    after_patient = builder.patient_turn(after_therapist)

    self.assertEqual(therapist_message["content"], "Visible therapist utterance.")
    self.assertNotIn("Private therapist reasoning", therapist_message["content"])
    self.assertEqual(therapist_trace["input"], "")
    self.assertEqual(therapist_trace["output"], "Private therapist reasoning")
    self.assertIsNone(builder.patient.last_kwargs["full_conversation"])
    self.assertNotIn("Private therapist reasoning", builder.patient.last_kwargs["last_utterances"])
    self.assertEqual(after_patient["messages"][-1]["content"], "Patient response.")

  def test_agent_patient_prompt_is_agent_only(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      agent_patient_prompt = Path(tmp) / "patient_system.txt"
      agent_patient_prompt.write_text(
        "AGENT PATIENT [/END]\n{persona}\n{more_info}\n{patterns}\n{presenting_problem}\n{reason_seeking_counseling}",
        encoding="utf-8",
      )
      builder = self._builder(resistance=False)
      builder.cfg = SimpleNamespace(
        therapist={"type": "deepseek"},
        patient={"type": "deepseek", "prompt_variant": "base"},
        env={"resistance": False, "max_turns": 8, "min_turns": 0},
      )
      with mock.patch("graph.agent_graph.AGENT_PATIENT_SYSTEM_PROMPT", agent_patient_prompt):
        with mock.patch("graph.agent_graph.LLMFactory.create_from_config", return_value=_FakeLLM()):
          builder._init_agents()

    standard_patient = PatientAgent(_FakeLLM(), _profile(), resistance_enabled=False)
    self.assertIn("AGENT PATIENT [/END]", builder.patient.system_prompt)
    self.assertNotIn("AGENT PATIENT [/END]", builder.patient.system_prompt_structured)
    self.assertNotIn("[/END]", standard_patient.system_prompt)

  def test_init_agents_uses_configured_system_prompt_paths(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      therapist_prompt = Path(tmp) / "therapist_system.txt"
      therapist_ft_prompt = Path(tmp) / "therapist_system_ft.txt"
      patient_prompt = Path(tmp) / "patient_system.txt"
      therapist_prompt.write_text("CFG THERAPIST\n{patient_intake_form}", encoding="utf-8")
      therapist_ft_prompt.write_text("CFG FT\n{patient_intake_form}", encoding="utf-8")
      patient_prompt.write_text(
        "CFG PATIENT\n{persona}\n{more_info}\n{patterns}\n{presenting_problem}\n{reason_seeking_counseling}",
        encoding="utf-8",
      )
      builder = self._builder(resistance=False)
      builder.cfg = SimpleNamespace(
        therapist={
          "type": "deepseek",
          "system_prompt_path": str(therapist_prompt),
          "ft_system_prompt_path": str(therapist_ft_prompt),
        },
        patient={
          "type": "deepseek",
          "prompt_variant": "base",
          "system_prompt_path": str(patient_prompt),
        },
        env={"resistance": False, "max_turns": 8, "min_turns": 0},
      )
      with mock.patch("graph.agent_graph.LLMFactory.create_from_config", return_value=_FakeLLM()):
        builder._init_agents()

      therapist_messages = builder.therapist.build_agent_utterance_messages("none")
      therapist_ft_messages = builder.therapist.build_agent_initial_utterance_messages(use_ft_system_prompt=True)
      self.assertIn("CFG THERAPIST", therapist_messages[0]["content"])
      self.assertIn("CFG FT", therapist_ft_messages[0]["content"])
      self.assertIn("CFG PATIENT", builder.patient.system_prompt)

  def test_cactus_patient_cli_mode_overrides_configured_patient_utterance_prompts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      therapist_prompt = Path(tmp) / "therapist_system.txt"
      patient_prompt = Path(tmp) / "patient_system.txt"
      cactus_system = Path(tmp) / "cactus_system_patient.txt"
      cactus_user = Path(tmp) / "cactus_utterance_patient.txt"
      therapist_prompt.write_text("CFG THERAPIST\n{patient_intake_form}", encoding="utf-8")
      patient_prompt.write_text(
        "CFG PATIENT\n{persona}\n{more_info}\n{patterns}\n{presenting_problem}\n{reason_seeking_counseling}",
        encoding="utf-8",
      )
      cactus_system.write_text("CACTUS SYSTEM", encoding="utf-8")
      cactus_user.write_text("CACTUS USER {attitude} {history}", encoding="utf-8")
      builder = self._builder(resistance=False)
      builder.cactus_patient = True
      builder.cfg = SimpleNamespace(
        therapist={
          "type": "deepseek",
          "system_prompt_path": str(therapist_prompt),
        },
        patient={
          "type": "deepseek",
          "prompt_variant": "base",
          "system_prompt_path": str(patient_prompt),
        },
        env={"resistance": False, "max_turns": 8, "min_turns": 0},
      )
      with mock.patch("graph.agent_graph.CACTUS_PATIENT_SYSTEM_PROMPT", cactus_system):
        with mock.patch("graph.agent_graph.CACTUS_PATIENT_USER_PROMPT", cactus_user):
          with mock.patch("graph.agent_graph.LLMFactory.create_from_config", return_value=_FakeLLM()):
            builder._init_agents()

      messages = builder.patient.build_utterance_messages(
        full_conversation=None,
        last_utterances="Therapist: Hello.",
        resistance_level=None,
        instruction="Respond naturally.",
      )

    self.assertEqual(builder._patient_prompt_ablation()["cactus_patient"], "true")
    self.assertEqual(messages[0]["content"], "CACTUS SYSTEM")
    self.assertIn("CACTUS USER Neutral Therapist: Hello.", messages[1]["content"])
    self.assertNotIn("CFG PATIENT", messages[0]["content"])

  def test_rejects_local_patient_config(self) -> None:
    builder = self._builder(resistance=False)
    builder.cfg.patient = {"type": "local", "prompt_variant": "base"}

    with self.assertRaisesRegex(ValueError, "patient.type: local"):
      builder._init_agents()

  def test_generic_patient_instruction_depends_only_on_resistance_mode(self) -> None:
    self.assertEqual(
      self._builder(resistance=True)._patient_utterance_instruction(),
      "Respond naturally and in character according to your current resistance level.",
    )
    self.assertEqual(
      self._builder(resistance=False)._patient_utterance_instruction(),
      "Respond naturally and in character without resistance framing.",
    )

  def test_therapist_end_tag_gets_one_final_patient_reply_and_no_stage_fields(self) -> None:
    builder = self._builder(resistance=False)
    builder.therapist = _FakeTherapist()
    builder.patient = _FakePatient()

    result = builder.create_chain().invoke(builder.initial_state(max_turns=6))

    self.assertEqual(result["session_ended_reason"], "therapist_requested_end")
    self.assertEqual([message["role"] for message in result["messages"]], ["therapist", "patient"])
    self.assertEqual(result["messages"][0]["content"], "Let's stop here for today. [/END]")
    self.assertEqual(result["messages"][1]["content"], "Okay, bye.")
    self.assertNotIn("stage", result["messages"][0])
    self.assertNotIn("substage", result["messages"][0])
    self.assertEqual(result["reasoning_traces"][0]["traces"][0]["output"], "Closing reason")

  def test_patient_end_tag_ends_session_and_is_preserved(self) -> None:
    builder = self._builder(resistance=False)
    builder.therapist = _FakeContinuingTherapist()
    builder.patient = _EndingPatient()

    result = builder.create_chain().invoke(builder.initial_state(max_turns=6))

    self.assertEqual(result["session_ended_reason"], "patient_requested_end")
    self.assertEqual([message["role"] for message in result["messages"]], ["therapist", "patient"])
    self.assertEqual(result["messages"][-1]["content"], "I think this is enough for today. [/END]")

  def test_patient_end_tag_takes_precedence_after_therapist_requested_end(self) -> None:
    builder = self._builder(resistance=False)
    builder.therapist = _FakeTherapist()
    builder.patient = _EndingPatient()

    result = builder.create_chain().invoke(builder.initial_state(max_turns=6))

    self.assertEqual(result["session_ended_reason"], "patient_requested_end")
    self.assertEqual(result["messages"][-1]["content"], "I think this is enough for today. [/END]")


class AgentTranscriptAndBatchTest(unittest.TestCase):
  def test_reasoning_transcript_flag_controls_txt_reasoning(self) -> None:
    result = {
      "messages": [
        {"utterance_index": 1, "role": "therapist", "content": "Hello"},
        {"utterance_index": 2, "role": "patient", "content": "Hi"},
      ],
      "reasoning_traces": [
        {
          "utterance_index": 1,
          "speaker": "therapist",
          "traces": [{"step": "agent_reasoning", "input": "none", "output": "Warm greeting"}],
        }
      ],
    }
    with tempfile.TemporaryDirectory() as tmp:
      plain = Path(tmp) / "plain.txt"
      with_reasoning = Path(tmp) / "reasoning.txt"
      agent_simulation._write_agent_transcript(plain, result, include_reasoning=False)
      agent_simulation._write_agent_transcript(with_reasoning, result, include_reasoning=True)

      plain_text = plain.read_text(encoding="utf-8")
      reasoning_text = with_reasoning.read_text(encoding="utf-8")

    self.assertNotIn("Warm greeting", plain_text)
    self.assertIn("<think>Warm greeting</think>", reasoning_text)
    self.assertIn("--> Therapist: Hello", reasoning_text)

  def test_multiprocessing_worker_receives_reasoning_flag(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      profiles = [{"idx": 1, "attitude": "neutral"}]
      manifest = agent_simulation._initialize_manifest(
        cfg_path="configs/test.yaml",
        profiles_path="profiles.json",
        output_dir=tmp,
        profiles=profiles,
        duplicate_idxs=[],
        resume_from_json=None,
      )
      _FakeTqdm.reset()
      _FakePool.include_reasoning_values = []
      _FakePool.cactus_patient_values = []

      with mock.patch("agent_simulation._load_tqdm", return_value=_FakeTqdm):
        with mock.patch("agent_simulation.multiprocessing.Manager", _FakeManager):
          with mock.patch("agent_simulation.multiprocessing.Pool", _FakePool):
            agent_simulation._run_batch_multiprocessing(
              cfg_path="configs/test.yaml",
              profiles_to_run=profiles,
              manifest=manifest,
              max_turns=10,
              min_turns=0,
              output_dir=tmp,
              n_processes=1,
              include_reasoning=True,
              cactus_patient=True,
            )

    self.assertEqual(_FakePool.include_reasoning_values, [True])
    self.assertEqual(_FakePool.cactus_patient_values, [True])
    self.assertTrue(manifest["profiles"]["1"]["executed"])


class _FakeTherapist:
  def generate_agent_initial_utterance(self):
    return "Let's stop here for today. [/END]", "Closing reason", "<think>Closing reason</think>Let's stop here for today. [/END]"

  def generate_agent_utterance(self, context: str):
    return "Let's stop here for today. [/END]", "Closing reason", "<think>Closing reason</think>Let's stop here for today. [/END]"


class _FakeContinuingTherapist:
  def generate_agent_initial_utterance(self):
    return "Hello.", "Greeting reason", "<think>Greeting reason</think>Hello."

  def generate_agent_utterance(self, context: str):
    return "Let's keep going.", "Continuing reason", "<think>Continuing reason</think>Let's keep going."


class _FakeLocalTherapist:
  def __init__(self) -> None:
    self.request_messages = []
    self.legacy_context = None
    self.used_ft_system_prompt = False
    self.generation_kwargs = {}

  def build_agent_chatml_messages(self, history, use_ft_system_prompt=False):
    self.used_ft_system_prompt = use_ft_system_prompt
    return [{"role": "system", "content": "SYSTEM"}, *history]

  def generate_agent_messages(self, messages, **kwargs):
    self.request_messages = messages
    self.generation_kwargs = kwargs
    return "Let's explore that.", "Local reason", "<think>Local reason</think>Let's explore that."

  def generate_agent_utterance(self, context: str):
    self.legacy_context = context
    raise AssertionError("local therapist turn should not use legacy context prompts")


class _FakeLocalInitialTherapist:
  def __init__(self) -> None:
    self.initial_called = False
    self.chatml_called = False
    self.used_ft_system_prompt = False
    self.generation_kwargs = {}

  def generate_agent_initial_utterance(self, **kwargs):
    self.initial_called = True
    self.used_ft_system_prompt = bool(kwargs.get("use_ft_system_prompt"))
    self.generation_kwargs = kwargs
    return "Hello.", "Opening reason", "<think>Opening reason</think>Hello."

  def generate_agent_messages(self, messages):
    self.chatml_called = True
    raise AssertionError("first local therapist turn should use the initial greeting prompt")


class _FakePatient:
  def generate_utterance(self, **kwargs):
    return "Okay, bye."


class _RecordingPatient:
  def __init__(self, response: str) -> None:
    self.response = response
    self.last_kwargs = {}

  def generate_utterance(self, **kwargs):
    self.last_kwargs = kwargs
    return self.response


class _EndingPatient:
  def generate_utterance(self, **kwargs):
    return "I think this is enough for today. [/END]"


class _FakeProgressQueue:
  def get_nowait(self):
    raise queue.Empty


class _FakeManager:
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, traceback) -> bool:
    return False

  def Queue(self) -> _FakeProgressQueue:
    return _FakeProgressQueue()


class _FakeTqdm:
  instances: list["_FakeTqdm"] = []

  @classmethod
  def reset(cls) -> None:
    cls.instances = []

  def __init__(self, *, total, desc, position, leave, unit) -> None:
    self.postfix = ""
    self.refreshed = False
    self.closed = False
    type(self).instances.append(self)

  def update(self, delta: int) -> None:
    return None

  def set_postfix_str(self, status: str) -> None:
    self.postfix = status

  def refresh(self) -> None:
    self.refreshed = True

  def close(self) -> None:
    self.closed = True

  @staticmethod
  def write(line: str) -> None:
    return None


class _FakePool:
  include_reasoning_values: list[bool] = []
  cactus_patient_values: list[bool] = []

  def __init__(self, processes: int) -> None:
    self.processes = processes

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, traceback) -> bool:
    return False

  def apply_async(self, func, args, callback, error_callback) -> None:
    worker_args = args[0]
    self.include_reasoning_values.append(worker_args[6])
    self.cactus_patient_values.append(worker_args[7])
    run_index = worker_args[0]
    profile = worker_args[2]
    callback(
      {
        "run_index": run_index,
        "profile": profile,
        "result": {
          "messages": [],
          "utterance_index": run_index,
          "session_ended_reason": "done",
          "conversation_error": "",
        },
      }
    )


if __name__ == "__main__":
  unittest.main()
