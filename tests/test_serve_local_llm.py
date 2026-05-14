import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


if "omegaconf" not in sys.modules:
  omegaconf_module = types.ModuleType("omegaconf")

  class _OmegaConf:
    @staticmethod
    def is_config(config) -> bool:
      return False

    @staticmethod
    def to_container(config, resolve: bool = True):
      return config

    @staticmethod
    def create(config):
      return SimpleNamespace(**config)

    @staticmethod
    def load(path):
      return {}

  omegaconf_module.OmegaConf = _OmegaConf
  sys.modules["omegaconf"] = omegaconf_module


from scripts.serve_local_llm import build_vllm_command


class ServeLocalLLMTest(unittest.TestCase):
  def test_build_vllm_command_uses_configured_max_model_len(self) -> None:
    cfg = SimpleNamespace(
      therapist={
        "type": "local",
        "base_url": "http://localhost:8000/v1",
        "model_id": "ft_models/example/merged_model/",
        "max_model_len": 65536,
      }
    )
    with mock.patch("scripts.serve_local_llm.load_runtime_config", return_value=cfg):
      command = build_vllm_command("config.yaml", "therapist", ["--gpu-memory-utilization", "0.9"])

    self.assertEqual(
      command,
      [
        "vllm",
        "serve",
        "ft_models/example/merged_model/",
        "--host",
        "localhost",
        "--port",
        "8000",
        "--served-model-name",
        "ft_models/example/merged_model/",
        "--max-model-len",
        "65536",
        "--gpu-memory-utilization",
        "0.9",
      ],
    )


if __name__ == "__main__":
  unittest.main()
