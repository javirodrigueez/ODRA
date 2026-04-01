from __future__ import annotations

from copy import deepcopy
from typing import Any

from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]


def _to_plain_container(value: Any) -> Any:
  if value is None:
    return None
  if OmegaConf.is_config(value):
    return OmegaConf.to_container(value, resolve=True)
  return deepcopy(value)


def _normalize_model_block(
  block_name: str,
  raw_block: Any,
  fallback_block: Any = None,
  *,
  include_profiles: bool = False,
) -> dict[str, Any]:
  raw_data = _to_plain_container(raw_block) or {}
  fallback_data = _to_plain_container(fallback_block) or {}

  if not isinstance(raw_data, dict):
    raise ValueError(f"Config block '{block_name}' must be a mapping.")
  if not isinstance(fallback_data, dict):
    raise ValueError(f"Fallback config for '{block_name}' must be a mapping.")

  normalized = dict(fallback_data)
  normalized.update({key: value for key, value in raw_data.items() if value is not None})
  normalized.setdefault("type", "")
  normalized.setdefault("model_id", "")
  normalized.setdefault("base_url", "")

  if include_profiles:
    normalized.setdefault("profiles", "")
  else:
    normalized.pop("profiles", None)

  return normalized


def normalize_runtime_config(cfg: Any) -> Any:
  raw_cfg = _to_plain_container(cfg) or {}
  if not isinstance(raw_cfg, dict):
    raise ValueError("Top-level config must be a mapping.")

  therapist = _normalize_model_block("therapist", raw_cfg.get("therapist"))
  therapist_struct = _normalize_model_block(
    "therapist_struct",
    raw_cfg.get("therapist_struct"),
    fallback_block=therapist,
  )

  patient = _normalize_model_block(
    "patient",
    raw_cfg.get("patient"),
    include_profiles=True,
  )
  patient_struct = _normalize_model_block(
    "patient_struct",
    raw_cfg.get("patient_struct"),
    fallback_block=patient,
  )

  env = _to_plain_container(raw_cfg.get("env")) or {}
  if not isinstance(env, dict):
    raise ValueError("Config block 'env' must be a mapping.")
  env.setdefault("max_turns", 80)
  env.setdefault("min_turns", 0)
  env.setdefault("output_dir", "")
  env.setdefault("verbose", False)
  env.setdefault("resume_file", "")

  return OmegaConf.create(
    {
      "therapist": therapist,
      "therapist_struct": therapist_struct,
      "patient": patient,
      "patient_struct": patient_struct,
      "env": env,
    }
  )


def load_runtime_config(config_path: str) -> Any:
  return normalize_runtime_config(OmegaConf.load(config_path))
