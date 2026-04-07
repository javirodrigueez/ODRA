from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, NotRequired, TypedDict

EmotionalRecovery = Literal["high", "moderate", "low"]

RESISTANCE_BASELINE_BY_ATTITUDE: dict[str, float] = {
  "very_negative": 0.90,
  "negative": 0.70,
  "neutral": 0.50,
  "positive": 0.30,
  "very_positive": 0.10,
}

RESISTANCE_RECOVERY_RATE_BY_EMOTIONAL_RECOVERY: dict[EmotionalRecovery, float] = {
  "high": 0.30,
  "moderate": 0.15,
  "low": 0.05,
}


class HomeostaticResistanceUpdate(TypedDict):
  previous_r: float
  delta_llm: float
  baseline: float
  alpha: float
  decay_vector: float
  new_r: float


class TherapistResistanceUpdate(TypedDict):
  p_target: float
  signal_clarity: float
  new_p: float
  prediction_mode: NotRequired[str]
  previous_p: NotRequired[float]


class PatientResistanceState(TypedDict):
  patient_profile: dict[str, Any]
  resistance_baseline: float
  resistance_recovery_rate: float
  resistance: float


def _clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


def _coerce_bounded_float(value: Any, default: float, low: float, high: float) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    parsed = default
  return _clamp(parsed, low, high)


def normalize_emotional_recovery(value: Any) -> EmotionalRecovery:
  normalized = str(value or "").strip().lower()
  if normalized in RESISTANCE_RECOVERY_RATE_BY_EMOTIONAL_RECOVERY:
    return normalized  # type: ignore[return-value]
  return "moderate"


def normalize_patient_profile(profile: Any) -> dict[str, Any]:
  normalized = deepcopy(profile) if isinstance(profile, dict) else {}
  normalized["emotional_recovery"] = normalize_emotional_recovery(normalized.get("emotional_recovery"))
  return normalized


def resistance_baseline_for_attitude(value: Any) -> float:
  normalized = str(value or "").strip().lower()
  return RESISTANCE_BASELINE_BY_ATTITUDE.get(normalized, 0.55)


def resistance_recovery_rate_for_emotional_recovery(value: Any) -> float:
  normalized = normalize_emotional_recovery(value)
  return RESISTANCE_RECOVERY_RATE_BY_EMOTIONAL_RECOVERY[normalized]


def build_patient_resistance_state(profile: Any, resistance: Any) -> PatientResistanceState:
  normalized_profile = normalize_patient_profile(profile)
  baseline = resistance_baseline_for_attitude(normalized_profile.get("attitude"))
  recovery_rate = resistance_recovery_rate_for_emotional_recovery(normalized_profile.get("emotional_recovery"))
  current_resistance = _coerce_bounded_float(resistance, baseline, 0.0, 1.0)
  return {
    "patient_profile": normalized_profile,
    "resistance_baseline": baseline,
    "resistance_recovery_rate": recovery_rate,
    "resistance": current_resistance,
  }


def normalize_resistance_state_fields(state: dict[str, Any], profile: Any) -> dict[str, Any]:
  normalized_state = dict(state)
  raw_profile = normalized_state.get("patient_profile")
  if not isinstance(raw_profile, dict):
    raw_profile = profile
  resistance_state = build_patient_resistance_state(raw_profile, normalized_state.get("resistance"))
  normalized_state["patient_profile"] = resistance_state["patient_profile"]
  normalized_state["resistance_baseline"] = resistance_state["resistance_baseline"]
  normalized_state["resistance_recovery_rate"] = resistance_state["resistance_recovery_rate"]
  normalized_state["resistance"] = resistance_state["resistance"]
  normalized_state.pop("initial_resistance", None)
  return normalized_state


def normalize_predicted_resistance_state_fields(state: dict[str, Any]) -> dict[str, Any]:
  normalized_state = dict(state)
  raw_predicted = normalized_state.get("predicted_resistance")
  if raw_predicted is None:
    normalized_state["predicted_resistance"] = None
  else:
    normalized_state["predicted_resistance"] = _coerce_bounded_float(raw_predicted, 0.0, 0.0, 1.0)

  raw_last_mi = normalized_state.get("last_mi_predicted_resistance")
  if raw_last_mi is None:
    normalized_state["last_mi_predicted_resistance"] = normalized_state["predicted_resistance"]
  else:
    normalized_state["last_mi_predicted_resistance"] = _coerce_bounded_float(raw_last_mi, 0.0, 0.0, 1.0)

  normalized_state.pop("initial_predicted_resistance", None)
  return normalized_state


def apply_homeostatic_resistance_update(
  previous_r: float,
  delta_llm: float,
  baseline: float,
  alpha: float,
) -> HomeostaticResistanceUpdate:
  decay_vector = round(alpha * (previous_r - baseline), 4)
  raw_new_r = previous_r + delta_llm - decay_vector
  new_r = round(_clamp(raw_new_r, 0.0, 1.0), 4)
  return {
    "previous_r": round(previous_r, 4),
    "delta_llm": round(delta_llm, 4),
    "baseline": round(baseline, 4),
    "alpha": round(alpha, 4),
    "decay_vector": decay_vector,
    "new_r": new_r,
  }


def apply_bayesian_predicted_resistance_update(
  previous_p: float | None,
  p_target: float,
  signal_clarity: float,
) -> TherapistResistanceUpdate:
  bounded_target = round(_clamp(p_target, 0.0, 1.0), 4)
  bounded_clarity = round(_clamp(signal_clarity, 0.0, 1.0), 4)
  if previous_p is None:
    return {
      "prediction_mode": "initial_from_first_patient_utterance",
      "p_target": bounded_target,
      "signal_clarity": bounded_clarity,
      "new_p": bounded_target,
    }

  bounded_previous = round(_clamp(previous_p, 0.0, 1.0), 4)
  raw_new_p = bounded_previous + bounded_clarity * (bounded_target - bounded_previous)
  new_p = round(_clamp(raw_new_p, 0.0, 1.0), 4)
  return {
    "prediction_mode": "bayesian_update",
    "previous_p": bounded_previous,
    "p_target": bounded_target,
    "signal_clarity": bounded_clarity,
    "new_p": new_p,
  }
