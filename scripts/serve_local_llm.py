from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime_config import load_runtime_config


def _config_value(block: Any, name: str, default: Any = None) -> Any:
  if isinstance(block, dict):
    return block.get(name, default)
  return getattr(block, name, default)


def _positive_int(value: Any, name: str) -> int | None:
  if value is None:
    return None
  if isinstance(value, str) and not value.strip():
    return None
  try:
    parsed = int(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be an integer.") from exc
  if parsed < 1:
    raise ValueError(f"{name} must be a positive integer.")
  return parsed


def _host_port_from_base_url(base_url: str) -> tuple[str, int]:
  parsed = urlparse(base_url)
  if not parsed.scheme or not parsed.netloc:
    raise ValueError(f"Invalid local LLM base_url: {base_url!r}")
  return parsed.hostname or "localhost", parsed.port or 8000


def build_vllm_command(cfg_path: str, role: str, extra_args: list[str]) -> list[str]:
  cfg = load_runtime_config(cfg_path)
  block = _config_value(cfg, role)
  if block is None:
    raise ValueError(f"Config has no {role!r} block.")

  llm_type = str(_config_value(block, "type", "") or "").strip().lower()
  if llm_type != "local":
    raise ValueError(f"{role}.type must be 'local' to launch a local vLLM server.")

  model_id = str(_config_value(block, "model_id", "") or "").strip()
  base_url = str(_config_value(block, "base_url", "") or "").strip()
  max_model_len = _positive_int(_config_value(block, "max_model_len"), f"{role}.max_model_len")
  if not model_id:
    raise ValueError(f"{role}.model_id is required.")
  if not base_url:
    raise ValueError(f"{role}.base_url is required.")

  host, port = _host_port_from_base_url(base_url)
  command = [
    "vllm",
    "serve",
    model_id,
    "--host",
    host,
    "--port",
    str(port),
    "--served-model-name",
    model_id,
  ]
  if max_model_len is not None:
    command.extend(["--max-model-len", str(max_model_len)])
  command.extend(extra_args)
  return command


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Launch a vLLM OpenAI-compatible server from a runtime config."
  )
  parser.add_argument("--cfg", default="configs/local_wo_resistance.yaml")
  parser.add_argument("--role", default="therapist", choices=("therapist", "therapist_struct", "patient", "patient_struct"))
  parser.add_argument("--dry-run", action="store_true", help="Print the vLLM command without executing it.")
  parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to vllm serve after '--'.")
  args = parser.parse_args()
  if args.extra_args and args.extra_args[0] == "--":
    args.extra_args = args.extra_args[1:]
  return args


def main() -> int:
  args = parse_args()
  command = build_vllm_command(args.cfg, args.role, args.extra_args)
  if args.dry_run:
    print(shlex.join(command))
    return 0
  os.execvp(command[0], command)
  return 1


if __name__ == "__main__":
  sys.exit(main())
