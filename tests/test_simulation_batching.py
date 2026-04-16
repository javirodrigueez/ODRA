import json
import sys
import tempfile
import types
from pathlib import Path
import unittest

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
  _deduplicate_profiles,
  _initialize_manifests,
  _manifest_path,
  _mark_profile_result,
  _matching_simulation_files,
  _select_profiles_to_execute,
  _write_manifest,
)


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
    with tempfile.TemporaryDirectory() as tmp:
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
    with tempfile.TemporaryDirectory() as tmp:
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
    with tempfile.TemporaryDirectory() as tmp:
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
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      (base / "simulation_1_neutral.json").write_text("{}", encoding="utf-8")
      (base / "simulation_2_negative.json").write_text("{}", encoding="utf-8")
      (base / "out_tmp_neutral.json").write_text("{}", encoding="utf-8")

      matches = _matching_simulation_files(base, "neutral")

      self.assertEqual([path.name for path in matches], ["simulation_1_neutral.json"])


if __name__ == "__main__":
  unittest.main()
