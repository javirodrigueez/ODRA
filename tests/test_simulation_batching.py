import contextlib
import io
import json
import shutil
import sys
import types
import uuid
from pathlib import Path
import unittest
from unittest import mock

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

from simulation import (
  _advance_progress_bar,
  _deduplicate_profiles,
  _handle_completed_multiprocessing_result,
  _initialize_manifests,
  _manifest_path,
  _mark_profile_result,
  _matching_simulation_files,
  _resolve_profiles_path,
  _select_profiles_to_execute,
  _validate_multiprocessing_options,
  _write_manifest,
  run_simulation,
)


@contextlib.contextmanager
def _temporary_workspace_dir():
  path = Path.cwd() / f".tmp_test_{uuid.uuid4().hex}"
  path.mkdir()
  try:
    yield str(path)
  finally:
    shutil.rmtree(path, ignore_errors=True)


class SimulationBatchingTest(unittest.TestCase):
  def test_deduplicate_profiles_keeps_first_idx_and_assigns_missing_idx(self) -> None:
    profiles, duplicates = _deduplicate_profiles(
      [
        {"idx": 10, "attitude": "neutral"},
        {"idx": 10, "attitude": "negative"},
        {"attitude": "positive"},
      ]
    )

    self.assertEqual([profile["idx"] for profile in profiles], [10, 3])
    self.assertEqual(duplicates, ["10"])

  def test_manifest_path_uses_output_dir_prefix_and_attitude(self) -> None:
    self.assertEqual(
      _manifest_path("out/16_04-rh_none", "neutral"),
      Path("out/16_04-rh_none/out_16_04-rh_none_neutral.json"),
    )

  def test_initialize_manifests_creates_one_manifest_per_attitude(self) -> None:
    with _temporary_workspace_dir() as tmp:
      profiles = [
        {"idx": 1, "attitude": "neutral"},
        {"idx": 2, "attitude": "negative"},
      ]
      manifests = _initialize_manifests(
        cfg_path="configs/test.yaml",
        profiles_path="profiles.json",
        output_dir=tmp,
        profiles=profiles,
        duplicate_idxs=[],
        resume_from_json=None,
      )

      self.assertEqual(set(manifests), {"neutral", "negative"})
      self.assertEqual(set(manifests["neutral"]["profiles"]), {"1"})
      self.assertEqual(set(manifests["negative"]["profiles"]), {"2"})

  def test_select_profiles_to_execute_skips_executed_and_limits_new_profiles(self) -> None:
    with _temporary_workspace_dir() as tmp:
      profiles = [
        {"idx": 1, "attitude": "neutral"},
        {"idx": 2, "attitude": "negative"},
        {"idx": 3, "attitude": "neutral"},
      ]
      manifests = _initialize_manifests(
        cfg_path="configs/test.yaml",
        profiles_path="profiles.json",
        output_dir=tmp,
        profiles=profiles,
        duplicate_idxs=[],
        resume_from_json=None,
      )
      manifests["neutral"]["profiles"]["1"]["executed"] = True

      selected = _select_profiles_to_execute(profiles, manifests, n_profiles=2)

      self.assertEqual([profile["idx"] for profile in selected], [2, 3])

  def test_mark_profile_result_and_write_manifest_records_error_status(self) -> None:
    with _temporary_workspace_dir() as tmp:
      profile = {"idx": 1, "attitude": "neutral"}
      manifests = _initialize_manifests(
        cfg_path="configs/test.yaml",
        profiles_path="profiles.json",
        output_dir=tmp,
        profiles=[profile],
        duplicate_idxs=[],
        resume_from_json=None,
      )
      manifest = manifests["neutral"]

      _mark_profile_result(manifest, profile, tmp, {"conversation_error": "boom"})
      _write_manifest(_manifest_path(tmp, "neutral"), manifest)

      with _manifest_path(tmp, "neutral").open("r", encoding="utf-8") as f:
        saved = json.load(f)
      self.assertTrue(saved["profiles"]["1"]["executed"])
      self.assertFalse(saved["profiles"]["1"]["success"])
      self.assertEqual(saved["profiles"]["1"]["error"], "boom")
      self.assertEqual(saved["metadata"]["executed_count"], 1)
      self.assertEqual(saved["metadata"]["success_count"], 0)

  def test_matching_simulation_files_excludes_manifests_and_other_attitudes(self) -> None:
    with _temporary_workspace_dir() as tmp:
      base = Path(tmp)
      (base / "simulation_1_neutral.json").write_text("{}", encoding="utf-8")
      (base / "simulation_2_negative.json").write_text("{}", encoding="utf-8")
      (base / "out_tmp_neutral.json").write_text("{}", encoding="utf-8")

      matches = _matching_simulation_files(base, "neutral")

      self.assertEqual([path.name for path in matches], ["simulation_1_neutral.json"])

  def test_resolve_profiles_path_prefers_cli_value(self) -> None:
    self.assertEqual(
      _resolve_profiles_path("cli_profiles.json", "config_profiles.json"),
      "cli_profiles.json",
    )

  def test_resolve_profiles_path_requires_cli_or_config_value(self) -> None:
    with self.assertRaisesRegex(ValueError, "--profiles-path"):
      _resolve_profiles_path(None, "")

  def test_validate_multiprocessing_requires_positive_process_count(self) -> None:
    with self.assertRaisesRegex(ValueError, "--n-processes"):
      _validate_multiprocessing_options(True, None)
    with self.assertRaisesRegex(ValueError, "--n-processes"):
      _validate_multiprocessing_options(True, 0)

    _validate_multiprocessing_options(True, 1)
    _validate_multiprocessing_options(False, None)

  def test_run_simulation_verbose_uses_streaming_live_output(self) -> None:
    fake_chain = _FakeChain()
    fake_graph_module = types.ModuleType("graph.graph")
    fake_graph_module.TherapyGraphBuilder = _fake_builder_factory(fake_chain)

    stdout = io.StringIO()
    with mock.patch.dict(sys.modules, {"graph.graph": fake_graph_module}):
      with contextlib.redirect_stdout(stdout):
        result = run_simulation(
          "configs/test.yaml",
          {"idx": 1, "attitude": "neutral"},
          max_turns=10,
          min_turns=0,
          verbose=True,
        )

    self.assertTrue(fake_chain.stream_called)
    self.assertFalse(fake_chain.invoke_called)
    self.assertIn("Therapist", stdout.getvalue())
    self.assertIn("hello", stdout.getvalue())
    self.assertEqual(result["utterance_index"], 1)

  def test_run_simulation_non_verbose_uses_invoke(self) -> None:
    fake_chain = _FakeChain()
    fake_graph_module = types.ModuleType("graph.graph")
    fake_graph_module.TherapyGraphBuilder = _fake_builder_factory(fake_chain)

    with mock.patch.dict(sys.modules, {"graph.graph": fake_graph_module}):
      result = run_simulation(
        "configs/test.yaml",
        {"idx": 1, "attitude": "neutral"},
        max_turns=10,
        min_turns=0,
        verbose=False,
      )

    self.assertFalse(fake_chain.stream_called)
    self.assertTrue(fake_chain.invoke_called)
    self.assertEqual(result["utterance_index"], 2)

  def test_progress_bar_advances_by_utterance_delta_and_keeps_early_final_count(self) -> None:
    bar = _FakeProgressBar()
    progress_counts = {1: 0}

    _advance_progress_bar(progress_counts, {1: bar}, 1, 2, max_turns=10)
    _advance_progress_bar(progress_counts, {1: bar}, 1, 4, max_turns=10)
    _advance_progress_bar(progress_counts, {1: bar}, 1, 3, max_turns=10)

    self.assertEqual(bar.total_updated, 4)
    self.assertEqual(bar.updates, [2, 2])
    self.assertEqual(progress_counts[1], 4)

  def test_completed_multiprocessing_result_records_in_completion_order(self) -> None:
    with _temporary_workspace_dir() as tmp:
      first_profile = {"idx": 1, "attitude": "neutral"}
      second_profile = {"idx": 2, "attitude": "negative"}
      manifests = _initialize_manifests(
        cfg_path="configs/test.yaml",
        profiles_path="profiles.json",
        output_dir=tmp,
        profiles=[first_profile, second_profile],
        duplicate_idxs=[],
        resume_from_json=None,
      )
      written_lines: list[str] = []

      _handle_completed_multiprocessing_result(
        {
          "run_index": 2,
          "profile": second_profile,
          "result": {"utterance_index": 4, "exchange_counts": {}, "session_ended_reason": "done"},
        },
        total_to_run=2,
        output_dir=tmp,
        manifests=manifests,
        writer=written_lines.append,
      )
      _handle_completed_multiprocessing_result(
        {
          "run_index": 1,
          "profile": first_profile,
          "result": {"utterance_index": 3, "exchange_counts": {}, "session_ended_reason": "done"},
        },
        total_to_run=2,
        output_dir=tmp,
        manifests=manifests,
        writer=written_lines.append,
      )

      simulation_headers = [line for line in written_lines if line.startswith("Simulation ")]
      self.assertEqual(simulation_headers[0], "Simulation 2/2 \u2014 idx: 2 \u2014 attitude: negative")
      self.assertEqual(simulation_headers[1], "Simulation 1/2 \u2014 idx: 1 \u2014 attitude: neutral")
      self.assertTrue(manifests["negative"]["profiles"]["2"]["executed"])
      self.assertTrue(manifests["neutral"]["profiles"]["1"]["executed"])


class _FakeChain:
  def __init__(self) -> None:
    self.stream_called = False
    self.invoke_called = False

  def stream(self, initial_state, stream_mode):
    self.stream_called = True
    yield {
      "messages": [
        {
          "role": "therapist",
          "content": "hello",
          "stage": "opening",
        }
      ],
      "current_stage": "opening",
      "resistance": 0.2,
      "predicted_resistance": 0.1,
      "utterance_index": 1,
      "exchange_counts": {},
      "session_ended_reason": "done",
      "conversation_error": "",
    }

  def invoke(self, initial_state):
    self.invoke_called = True
    return {
      "messages": [],
      "utterance_index": 2,
      "exchange_counts": {},
      "session_ended_reason": "done",
      "conversation_error": "",
    }


class _FakeProgressBar:
  def __init__(self) -> None:
    self.updates: list[int] = []
    self.total_updated = 0

  def update(self, delta: int) -> None:
    self.updates.append(delta)
    self.total_updated += delta


def _fake_builder_factory(fake_chain: _FakeChain):
  class _FakeBuilder:
    def __init__(self, cfg_path, profile) -> None:
      self.cfg_path = cfg_path
      self.profile = profile

    def create_chain(self):
      return fake_chain

    def initial_state(self, max_turns, min_turns):
      return {
        "messages": [],
        "utterance_index": 0,
        "exchange_counts": {},
        "conversation_error": "",
      }

    def resume_state(self, saved_state, max_turns, min_turns):
      return dict(saved_state)

  return _FakeBuilder


if __name__ == "__main__":
  unittest.main()
