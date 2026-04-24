from __future__ import annotations

from copy import deepcopy
from typing import Any

from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]

from graph.resistance import (
  DEFAULT_PROMPT_INTERVALS,
  DEFAULT_PROMPT_VARIANT,
  PROMPT_INTERVALS,
  PROMPT_VARIANTS,
  normalize_prompt_intervals,
  normalize_prompt_variant,
)


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


def _coerce_bool(value: Any, default: bool) -> bool:
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
      return True
    if normalized in {"false", "0", "no", "n", "off"}:
      return False
  if isinstance(value, (int, float)):
    return bool(value)
  return default


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
  patient.setdefault("prompt_variant", DEFAULT_PROMPT_VARIANT)
  patient.setdefault("prompt_intervals", DEFAULT_PROMPT_INTERVALS)
  normalized_prompt_variant = normalize_prompt_variant(patient.get("prompt_variant"))
  if normalized_prompt_variant is None:
    allowed = ", ".join(PROMPT_VARIANTS)
    raise ValueError(f"Config block 'patient' has invalid prompt_variant {patient.get('prompt_variant')!r}. Allowed values: {allowed}")
  normalized_prompt_intervals = normalize_prompt_intervals(patient.get("prompt_intervals"))
  if normalized_prompt_intervals is None:
    allowed = ", ".join(PROMPT_INTERVALS)
    raise ValueError(
      f"Config block 'patient' has invalid prompt_intervals {patient.get('prompt_intervals')!r}. Allowed values: {allowed}"
    )
  patient["prompt_variant"] = normalized_prompt_variant
  patient["prompt_intervals"] = normalized_prompt_intervals
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
  env.setdefault("long_session_turn_threshold", 70)
  env.setdefault("output_dir", "")
  env.setdefault("verbose", False)
  env.setdefault("resume_file", "")
  env["resistance"] = _coerce_bool(env.get("resistance"), True)
  env.setdefault("resistance_prediction_clip_initial_turns", 5)
  env.setdefault("resistance_prediction_clip_initial_min_delta", -0.3)
  env.setdefault("resistance_prediction_clip_initial_max_delta", 0.3)
  env.setdefault("resistance_prediction_clip_later_min_delta", -0.15)
  env.setdefault("resistance_prediction_clip_later_max_delta", 0.15)
  resistance_handling = str(env.get("resistance_handling", "mi_techniques") or "mi_techniques").strip().lower()
  if resistance_handling not in {"none", "mi_techniques", "cami"}:
    resistance_handling = "mi_techniques"
  env["resistance_handling"] = resistance_handling

  topic_exploration = _to_plain_container(raw_cfg.get("topic_exploration")) or {}
  if not isinstance(topic_exploration, dict):
    raise ValueError("Config block 'topic_exploration' must be a mapping.")
  topic_exploration.setdefault("patient_classifier_path", "")
  topic_exploration.setdefault("patient_wikipedia_dir", "")

  return OmegaConf.create(
    {
      "therapist": therapist,
      "therapist_struct": therapist_struct,
      "patient": patient,
      "patient_struct": patient_struct,
      "topic_exploration": topic_exploration,
      "env": env,
    }
  )


def load_runtime_config(config_path: str) -> Any:
  return normalize_runtime_config(OmegaConf.load(config_path))
