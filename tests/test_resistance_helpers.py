import random
import unittest

from graph.resistance import (
  action_prior_for_resistance,
  fallback_action_distribution_for_resistance,
  is_highest_negative_calibration_interval,
  merge_action_distributions,
  resistance_interval_for_level,
  sample_patient_action,
  scaled_resistance_value,
  validate_action_distribution,
)


class ResistanceHelpersTest(unittest.TestCase):
  def test_scaled_resistance_value_boundaries(self) -> None:
    cases = {
      0.0: 0.0,
      0.11: 1.1,
      0.32: 3.2,
      0.33: 3.3,
      0.44: 4.4,
      0.55: 5.5,
      0.66: 6.6,
      0.88: 8.8,
      1.0: 10.0,
    }
    for raw_value, expected in cases.items():
      with self.subTest(raw_value=raw_value):
        self.assertEqual(scaled_resistance_value(raw_value), expected)

  def test_resistance_interval_boundaries(self) -> None:
    cases = {
      0.0: "positive",
      0.11: "positive",
      0.32: "positive",
      0.33: "neutral",
      0.44: "neutral",
      0.55: "neutral",
      0.66: "negative",
      0.88: "negative",
      1.0: "negative",
    }
    for raw_value, expected in cases.items():
      with self.subTest(raw_value=raw_value):
        self.assertEqual(resistance_interval_for_level(raw_value), expected)

  def test_action_prior_lookup_uses_interval_distribution(self) -> None:
    self.assertEqual(
      action_prior_for_resistance(0.10),
      {
        "Acceptance": 45.0,
        "Acceptance with hesitation": 30.0,
        "Hesitation": 15.0,
        "Mild rejection": 7.0,
        "Rejection": 3.0,
        "Terminate": 0.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.50),
      {
        "Acceptance": 12.0,
        "Acceptance with hesitation": 23.0,
        "Hesitation": 30.0,
        "Mild rejection": 23.0,
        "Rejection": 12.0,
        "Terminate": 0.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.70),
      {
        "Acceptance": 2.0,
        "Acceptance with hesitation": 6.0,
        "Hesitation": 12.0,
        "Mild rejection": 25.0,
        "Rejection": 40.0,
        "Terminate": 15.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.90),
      {
        "Acceptance": 0.0,
        "Acceptance with hesitation": 2.0,
        "Hesitation": 5.0,
        "Mild rejection": 10.0,
        "Rejection": 33.0,
        "Terminate": 50.0,
      },
    )

  def test_validate_action_distribution_requires_known_keys(self) -> None:
    valid = validate_action_distribution(
      {
        "Acceptance": 40,
        "Acceptance with hesitation": "30",
        "Hesitation": 20,
        "Mild rejection": 10,
        "Rejection": -5,
        "Terminate": 5,
      }
    )
    self.assertEqual(
      valid,
      {
        "Acceptance": 40.0,
        "Acceptance with hesitation": 30.0,
        "Hesitation": 20.0,
        "Mild rejection": 10.0,
        "Rejection": 0.0,
        "Terminate": 5.0,
      },
    )
    self.assertIsNone(
      validate_action_distribution(
        {
          "Acceptance": 50,
          "Acceptance with hesitation": 20,
          "Hesitation": 20,
          "Mild rejection": 10,
          "Rejection": 0,
        }
      )
    )

  def test_merge_action_distributions_normalizes(self) -> None:
    merged = merge_action_distributions(
      {
        "Acceptance": 45.0,
        "Acceptance with hesitation": 30.0,
        "Hesitation": 15.0,
        "Mild rejection": 7.0,
        "Rejection": 3.0,
        "Terminate": 0.0,
      },
      {
        "Acceptance": 20.0,
        "Acceptance with hesitation": 20.0,
        "Hesitation": 20.0,
        "Mild rejection": 20.0,
        "Rejection": 20.0,
        "Terminate": 0.0,
      },
    )
    self.assertAlmostEqual(sum(merged.values()), 1.0, places=6)
    self.assertAlmostEqual(merged["Acceptance"], 0.325, places=6)
    self.assertAlmostEqual(merged["Rejection"], 0.115, places=6)
    self.assertEqual(merged["Terminate"], 0.0)

  def test_sample_patient_action_respects_deterministic_weights(self) -> None:
    sampled = sample_patient_action(
      {
        "Acceptance": 0.0,
        "Acceptance with hesitation": 0.0,
        "Hesitation": 0.0,
        "Mild rejection": 0.0,
        "Rejection": 0.0,
        "Terminate": 1.0,
      },
      rng=random.Random(7),
    )
    self.assertEqual(sampled, "Terminate")

  def test_negative_high_interval_detection(self) -> None:
    self.assertFalse(is_highest_negative_calibration_interval(0.87))
    self.assertTrue(is_highest_negative_calibration_interval(0.88))
    self.assertTrue(is_highest_negative_calibration_interval(1.0))

  def test_fallback_action_distribution_respects_negative_interval(self) -> None:
    positive_fallback = fallback_action_distribution_for_resistance(0.20)
    negative_fallback = fallback_action_distribution_for_resistance(0.70)
    self.assertEqual(positive_fallback["Terminate"], 0.0)
    self.assertGreater(negative_fallback["Terminate"], 0.0)


if __name__ == "__main__":
  unittest.main()
