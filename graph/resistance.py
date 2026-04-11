from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Literal, Mapping, NotRequired, TypeAlias, TypedDict, cast

EmotionalRecovery = Literal["high", "moderate", "low"]
ResistanceInterval = Literal["positive", "neutral", "negative"]
PatientAction = Literal[
  "Acceptance",
  "Acceptance with hesitation",
  "Hesitation",
  "Mild rejection",
  "Rejection",
  "Terminate",
]
PatientActionDistribution: TypeAlias = dict[PatientAction, float]

RESISTANCE_BASELINE_BY_ATTITUDE: dict[str, float] = {
  "very_negative": 1.0,
  "negative": 0.825,
  "neutral": 0.495,
  "positive": 0.165,
  "very_positive": 0.0,
}
RESISTANCE_PROMPT_NAME_BY_INTERVAL: dict[ResistanceInterval, str] = {
  "positive": "r_positive",
  "neutral": "r_neutral",
  "negative": "r_negative",
}
PATIENT_NON_TERMINATE_ACTIONS: tuple[PatientAction, ...] = (
  "Acceptance",
  "Acceptance with hesitation",
  "Hesitation",
  "Mild rejection",
  "Rejection",
)
PATIENT_ACTIONS: tuple[PatientAction, ...] = (
  "Acceptance",
  "Acceptance with hesitation",
  "Hesitation",
  "Mild rejection",
  "Rejection",
  "Terminate",
)
PATIENT_ACTION_DEFINITIONS: dict[PatientAction, str] = {
  "Acceptance": "Openly engage and collaborate with the therapist's proposed line.",
  "Acceptance with hesitation": "Engage with the therapist's proposed line, but let mild doubts or caution remain visible.",
  "Hesitation": "Stay in the exchange, but foreground meaningful doubts about the therapist's proposed line.",
  "Mild rejection": "Do not engage with the therapist's proposed line yet, but stay recoverable and open to further clarification.",
  "Rejection": "Refuse the therapist's proposed line clearly.",
  "Terminate": "Explicitly request ending the session now. If resistance is at the interval 0.88-1.0, the request should be more forceful, frustrated, or critical.",
}
PATIENT_ACTION_PRIORS_BY_INTERVAL: dict[ResistanceInterval, PatientActionDistribution] = {
  "positive": {
    "Acceptance": 45.0,
    "Acceptance with hesitation": 30.0,
    "Hesitation": 15.0,
    "Mild rejection": 7.0,
    "Rejection": 3.0,
    "Terminate": 0.0,
  },
  "neutral": {
    "Acceptance": 12.0,
    "Acceptance with hesitation": 23.0,
    "Hesitation": 30.0,
    "Mild rejection": 23.0,
    "Rejection": 12.0,
    "Terminate": 0.0,
  },
  "negative": {
    "Acceptance": 2.0,
    "Acceptance with hesitation": 6.0,
    "Hesitation": 12.0,
    "Mild rejection": 25.0,
    "Rejection": 40.0,
    "Terminate": 15.0,
  },
}
PATIENT_HIGH_NEGATIVE_ACTION_PRIOR: PatientActionDistribution = {
  "Acceptance": 0.0,
  "Acceptance with hesitation": 2.0,
  "Hesitation": 5.0,
  "Mild rejection": 10.0,
  "Rejection": 33.0,
  "Terminate": 50.0,
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
  normalized: dict[str, Any] = deepcopy(profile) if isinstance(profile, dict) else {}
  normalized["emotional_recovery"] = normalize_emotional_recovery(normalized.get("emotional_recovery"))
  return normalized


def resistance_baseline_for_attitude(value: Any) -> float:
  normalized = str(value or "").strip().lower()
  return RESISTANCE_BASELINE_BY_ATTITUDE.get(normalized, 0.55)


def resistance_recovery_rate_for_emotional_recovery(value: Any) -> float:
  normalized = normalize_emotional_recovery(value)
  return RESISTANCE_RECOVERY_RATE_BY_EMOTIONAL_RECOVERY[normalized]


def scaled_resistance_value(resistance_level: Any) -> float:
  bounded = _coerce_bounded_float(resistance_level, 0.0, 0.0, 1.0)
  return round(bounded * 10.0, 1)


def resistance_interval_for_level(resistance_level: float) -> ResistanceInterval:
  scaled = scaled_resistance_value(resistance_level)
  if scaled < 3.3:
    return "positive"
  if scaled < 6.6:
    return "neutral"
  return "negative"


def resistance_prompt_name_for_level(resistance_level: float) -> str:
  interval = resistance_interval_for_level(resistance_level)
  return RESISTANCE_PROMPT_NAME_BY_INTERVAL[interval]


def is_highest_negative_calibration_interval(resistance_level: float) -> bool:
  return scaled_resistance_value(resistance_level) >= 8.8


def action_prior_for_resistance(resistance_level: float) -> PatientActionDistribution:
  interval = resistance_interval_for_level(resistance_level)
  if interval == "negative" and is_highest_negative_calibration_interval(resistance_level):
    return dict(PATIENT_HIGH_NEGATIVE_ACTION_PRIOR)
  return dict(PATIENT_ACTION_PRIORS_BY_INTERVAL[interval])


def uniform_action_distribution(include_terminate: bool = False) -> PatientActionDistribution:
  if include_terminate:
    uniform_weight = round(100.0 / len(PATIENT_ACTIONS), 4)
    distribution: PatientActionDistribution = {
      action: uniform_weight for action in PATIENT_ACTIONS
    }
    distribution["Terminate"] = round(100.0 - uniform_weight * (len(PATIENT_ACTIONS) - 1), 4)
    return distribution

  distribution = {
    action: 20.0 for action in PATIENT_NON_TERMINATE_ACTIONS
  }
  distribution["Terminate"] = 0.0
  return distribution


def fallback_action_distribution_for_resistance(resistance_level: float) -> PatientActionDistribution:
  include_terminate = resistance_interval_for_level(resistance_level) == "negative"
  return uniform_action_distribution(include_terminate=include_terminate)


def should_select_patient_action(final_patient_pending_end: bool) -> bool:
  return not final_patient_pending_end


def normalize_patient_action(value: str | None) -> PatientAction | None:
  if value is None:
    return None
  normalized = str(value).strip()
  if normalized in PATIENT_ACTION_DEFINITIONS:
    return cast(PatientAction, normalized)
  return None


def action_definition_for_action(action: PatientAction | None) -> str:
  if action is None:
    return "No action override. Follow the stage instruction naturally."
  return PATIENT_ACTION_DEFINITIONS[action]


def validate_action_distribution(payload: Any) -> PatientActionDistribution | None:
  if not isinstance(payload, dict):
    return None

  distribution: PatientActionDistribution = {}
  for action in PATIENT_ACTIONS:
    if action not in payload:
      return None
    try:
      value = float(payload.get(action, 0.0))
    except (TypeError, ValueError):
      return None
    distribution[action] = max(0.0, value)

  if sum(distribution.values()) <= 0.0:
    return None
  return distribution


def merge_action_distributions(
  prior_distribution: Mapping[PatientAction, float],
  context_distribution: Mapping[PatientAction, float],
) -> PatientActionDistribution:
  merged_weights: PatientActionDistribution = {
    action: max(0.0, float(prior_distribution.get(action, 0.0)))
    + max(0.0, float(context_distribution.get(action, 0.0)))
    for action in PATIENT_ACTIONS
  }
  total = sum(merged_weights.values())
  if total <= 0.0:
    fallback = uniform_action_distribution()
    total = sum(fallback.values())
    return {
      action: round(fallback[action] / total, 6)
      for action in PATIENT_ACTIONS
    }
  return {
    action: round(merged_weights[action] / total, 6)
    for action in PATIENT_ACTIONS
  }


def sample_patient_action(
  action_distribution: Mapping[PatientAction, float],
  rng: random.Random | None = None,
) -> PatientAction:
  weights = [max(0.0, float(action_distribution.get(action, 0.0))) for action in PATIENT_ACTIONS]
  if sum(weights) <= 0.0:
    fallback = uniform_action_distribution()
    weights = [fallback[action] for action in PATIENT_ACTIONS]
  chooser = rng or random
  sampled = chooser.choices(PATIENT_ACTIONS, weights=weights, k=1)[0]
  return cast(PatientAction, sampled)


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
