import json
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
from agents.interfaces import LocalLLM
from agents.therapist import TherapistAgent, parse_agent_therapist_response
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
    self.reasoning = ""
    self.calls: list[tuple[list[dict], dict]] = []

  def generate(self, messages, **kwargs):
    self.calls.append((messages, kwargs))
    return self.response

  def generate_with_reasoning(self, messages, **kwargs):
    self.calls.append((messages, kwargs))
    return {"content": self.response, "reasoning": self.reasoning}


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

  def test_parses_reasoning_block_without_leaking_into_utterance(self) -> None:
    reasoning, utterance = parse_agent_therapist_response("<think>plan\nline</think>Hi there.")

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

  def test_agent_generation_prefers_vllm_reasoning_field(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      prompt_path = Path(tmp) / "therapist_system.txt"
      prompt_path.write_text("AGENT SYSTEM\n{patient_intake_form}", encoding="utf-8")
      llm = _FakeLLM("Hello from content")
      llm.reasoning = "Parsed vLLM reasoning"
      with mock.patch("agents.therapist.AGENT_THERAPIST_SYSTEM_PROMPT", prompt_path):
        agent = TherapistAgent(llm, _profile())
        utterance, reasoning, raw = agent.generate_agent_utterance("none")

    self.assertEqual(utterance, "Hello from content")
    self.assertEqual(reasoning, "Parsed vLLM reasoning")
    self.assertEqual(raw, "Hello from content")

  def test_local_llm_extracts_vllm_reasoning_fields(self) -> None:
    llm = LocalLLM.__new__(LocalLLM)
    llm.model_id = "local-model"
    llm.chat_model = _FakeOpenAIChatModel(
      _FakeMessage(content="Final answer", reasoning="New reasoning", reasoning_content="Old reasoning")
    )

    response = llm.generate_with_reasoning([{"role": "user", "content": "hi"}])

    self.assertEqual(response, {"content": "Final answer", "reasoning": "New reasoning"})

  def test_local_llm_supports_legacy_reasoning_content_field(self) -> None:
    llm = LocalLLM.__new__(LocalLLM)
    llm.model_id = "local-model"
    llm.chat_model = _FakeOpenAIChatModel(
      _FakeMessage(content="Final answer", reasoning_content="Legacy reasoning")
    )

    response = llm.generate_with_reasoning([{"role": "user", "content": "hi"}])

    self.assertEqual(response, {"content": "Final answer", "reasoning": "Legacy reasoning"})


class AgentGraphTest(unittest.TestCase):
  def _builder(self, *, resistance: bool = False) -> AgentTherapyGraphBuilder:
    builder = AgentTherapyGraphBuilder.__new__(AgentTherapyGraphBuilder)
    builder.profile = _profile()
    builder.cfg = SimpleNamespace(
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
    self.assertEqual(result["messages"][0]["content"], "Let's stop here for today.")
    self.assertEqual(result["messages"][1]["content"], "Okay, bye.")
    self.assertNotIn("stage", result["messages"][0])
    self.assertNotIn("substage", result["messages"][0])
    self.assertEqual(result["reasoning_traces"][0]["traces"][0]["output"], "Closing reason")


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
            )

    self.assertEqual(_FakePool.include_reasoning_values, [True])
    self.assertTrue(manifest["profiles"]["1"]["executed"])


class _FakeTherapist:
  def generate_agent_initial_utterance(self):
    return "Let's stop here for today. [/END]", "Closing reason", "<think>Closing reason</think>Let's stop here for today. [/END]"

  def generate_agent_utterance(self, context: str):
    return "Let's stop here for today. [/END]", "Closing reason", "<think>Closing reason</think>Let's stop here for today. [/END]"


class _FakePatient:
  def generate_utterance(self, **kwargs):
    return "Okay, bye."


class _FakeMessage:
  def __init__(self, **fields) -> None:
    self.model_extra = {}
    for key, value in fields.items():
      setattr(self, key, value)


class _FakeChoice:
  def __init__(self, message) -> None:
    self.message = message


class _FakeResponse:
  def __init__(self, message) -> None:
    self.choices = [_FakeChoice(message)]


class _FakeChatCompletions:
  def __init__(self, message) -> None:
    self.message = message
    self.calls: list[dict] = []

  def create(self, **kwargs):
    self.calls.append(kwargs)
    return _FakeResponse(self.message)


class _FakeChat:
  def __init__(self, message) -> None:
    self.completions = _FakeChatCompletions(message)


class _FakeOpenAIChatModel:
  def __init__(self, message) -> None:
    self.chat = _FakeChat(message)


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

  def __init__(self, processes: int) -> None:
    self.processes = processes

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, traceback) -> bool:
    return False

  def apply_async(self, func, args, callback, error_callback) -> None:
    worker_args = args[0]
    self.include_reasoning_values.append(worker_args[6])
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
