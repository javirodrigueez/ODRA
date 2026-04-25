import random
import unittest

from graph.resistance import (
  action_prior_for_resistance,
  apply_predicted_resistance_delta_clip,
  can_select_terminate_action,
  constrain_action_distribution_for_utterance,
  fallback_action_distribution_for_resistance,
  is_highest_negative_calibration_interval,
  merge_action_distributions,
  resistance_interval_for_level,
  resistance_prompt_relative_path_for_level,
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

  def test_resistance_prompt_relative_path_uses_root_files_for_default_baseline(self) -> None:
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.20),
      "patient_resistance/r_positive.txt",
    )
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.50),
      "patient_resistance/r_neutral.txt",
    )

  def test_resistance_prompt_relative_path_uses_variant_root_for_section_removals(self) -> None:
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.50, prompt_variant="no_fewshot"),
      "patient_resistance/no_fewshot/r_neutral.txt",
    )

  def test_resistance_prompt_relative_path_uses_seven_no_calib_bins(self) -> None:
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.10, prompt_variant="7_no_calibration"),
      "patient_resistance/7_no_calibration/r_bin_1.txt",
    )
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.11, prompt_variant="7_no_calibration"),
      "patient_resistance/7_no_calibration/r_bin_2.txt",
    )
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.55, prompt_variant="7_no_calibration"),
      "patient_resistance/7_no_calibration/r_bin_5.txt",
    )
    self.assertEqual(
      resistance_prompt_relative_path_for_level(0.88, prompt_variant="7_no_calibration"),
      "patient_resistance/7_no_calibration/r_bin_7.txt",
    )

  def test_predicted_resistance_delta_clip_exempts_initial_prediction(self) -> None:
    update = apply_predicted_resistance_delta_clip(
      {
        "prediction_mode": "initial_from_first_patient_utterance",
        "p_target": 0.8,
        "signal_clarity": 0.9,
        "new_p": 0.8,
      },
      previous_p=None,
      prediction_turn=1,
    )

    self.assertEqual(update["new_p"], 0.8)
    self.assertEqual(update["unclipped_new_p"], 0.8)
    self.assertIsNone(update["unclipped_delta"])
    self.assertIsNone(update["clipped_delta"])
    self.assertIsNone(update["delta_clip_min"])
    self.assertIsNone(update["delta_clip_max"])
    self.assertEqual(update["delta_clip_window"], "initial_prediction_exempt")
    self.assertFalse(update["delta_clip_applied"])

  def test_predicted_resistance_delta_clip_uses_initial_upper_bound(self) -> None:
    update = apply_predicted_resistance_delta_clip(
      {"p_target": 0.9, "signal_clarity": 1.0, "new_p": 0.8},
      previous_p=0.2,
      prediction_turn=2,
    )

    self.assertEqual(update["new_p"], 0.5)
    self.assertEqual(update["unclipped_new_p"], 0.8)
    self.assertEqual(update["unclipped_delta"], 0.6)
    self.assertEqual(update["clipped_delta"], 0.3)
    self.assertEqual(update["delta_clip_window"], "initial")
    self.assertTrue(update["delta_clip_applied"])

  def test_predicted_resistance_delta_clip_uses_initial_lower_bound(self) -> None:
    update = apply_predicted_resistance_delta_clip(
      {"p_target": 0.1, "signal_clarity": 1.0, "new_p": 0.1},
      previous_p=0.8,
      prediction_turn=2,
    )

    self.assertEqual(update["new_p"], 0.6)
    self.assertEqual(update["unclipped_delta"], -0.7)
    self.assertEqual(update["clipped_delta"], -0.2)
    self.assertEqual(update["delta_clip_window"], "initial")
    self.assertTrue(update["delta_clip_applied"])

  def test_predicted_resistance_delta_clip_uses_later_upper_bound(self) -> None:
    update = apply_predicted_resistance_delta_clip(
      {"p_target": 0.9, "signal_clarity": 1.0, "new_p": 0.8},
      previous_p=0.2,
      prediction_turn=6,
    )

    self.assertEqual(update["new_p"], 0.35)
    self.assertEqual(update["unclipped_delta"], 0.6)
    self.assertEqual(update["clipped_delta"], 0.15)
    self.assertEqual(update["delta_clip_window"], "later")
    self.assertTrue(update["delta_clip_applied"])

  def test_predicted_resistance_delta_clip_uses_later_lower_bound(self) -> None:
    update = apply_predicted_resistance_delta_clip(
      {"p_target": 0.1, "signal_clarity": 1.0, "new_p": 0.1},
      previous_p=0.8,
      prediction_turn=6,
    )

    self.assertEqual(update["new_p"], 0.7)
    self.assertEqual(update["unclipped_delta"], -0.7)
    self.assertEqual(update["clipped_delta"], -0.1)
    self.assertEqual(update["delta_clip_window"], "later")
    self.assertTrue(update["delta_clip_applied"])

  def test_predicted_resistance_delta_clip_leaves_in_range_delta_unchanged(self) -> None:
    update = apply_predicted_resistance_delta_clip(
      {"p_target": 0.5, "signal_clarity": 1.0, "new_p": 0.5},
      previous_p=0.4,
      prediction_turn=6,
    )

    self.assertEqual(update["new_p"], 0.5)
    self.assertEqual(update["unclipped_new_p"], 0.5)
    self.assertEqual(update["unclipped_delta"], 0.1)
    self.assertEqual(update["clipped_delta"], 0.1)
    self.assertEqual(update["delta_clip_min"], -0.1)
    self.assertEqual(update["delta_clip_max"], 0.15)
    self.assertEqual(update["delta_clip_window"], "later")
    self.assertFalse(update["delta_clip_applied"])

  def test_action_prior_lookup_uses_interval_distribution(self) -> None:
    self.assertEqual(
      action_prior_for_resistance(0.10),
      {
        "Acceptance": 60.0,
        "Hesitation": 30.0,
        "Mild rejection": 7.0,
        "Rejection": 3.0,
        "Terminate": 0.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.50),
      {
        "Acceptance": 30.0,
        "Hesitation": 40.0,
        "Mild rejection": 20.0,
        "Rejection": 10.0,
        "Terminate": 0.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.70),
      {
        "Acceptance": 10.0,
        "Hesitation": 30.0,
        "Mild rejection": 20.0,
        "Rejection": 40.0,
        "Terminate": 0.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.90),
      {
        "Acceptance": 5.0,
        "Hesitation": 15.0,
        "Mild rejection": 30.0,
        "Rejection": 50.0,
        "Terminate": 0.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.70, utterance_index=30),
      {
        "Acceptance": 10.0,
        "Hesitation": 30.0,
        "Mild rejection": 15.0,
        "Rejection": 35.0,
        "Terminate": 10.0,
      },
    )
    self.assertEqual(
      action_prior_for_resistance(0.90, utterance_index=30),
      {
        "Acceptance": 5.0,
        "Hesitation": 25.0,
        "Mild rejection": 15.0,
        "Rejection": 35.0,
        "Terminate": 20.0,
      },
    )

  def test_validate_action_distribution_requires_known_keys(self) -> None:
    valid = validate_action_distribution(
      {
        "Acceptance": 40,
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
          "Hesitation": 20,
          "Mild rejection": 10,
          "Rejection": 0,
        }
      )
    )

  def test_merge_action_distributions_normalizes(self) -> None:
    merged = merge_action_distributions(
      {
        "Acceptance": 60.0,
        "Hesitation": 30.0,
        "Mild rejection": 7.0,
        "Rejection": 3.0,
        "Terminate": 0.0,
      },
      {
        "Acceptance": 20.0,
        "Hesitation": 20.0,
        "Mild rejection": 20.0,
        "Rejection": 20.0,
        "Terminate": 0.0,
      },
    )
    self.assertAlmostEqual(sum(merged.values()), 1.0, places=6)
    self.assertAlmostEqual(merged["Acceptance"], 0.444444, places=6)
    self.assertAlmostEqual(merged["Rejection"], 0.127778, places=6)
    self.assertEqual(merged["Terminate"], 0.0)

  def test_sample_patient_action_respects_deterministic_weights(self) -> None:
    sampled = sample_patient_action(
      {
        "Acceptance": 0.0,
        "Hesitation": 0.0,
        "Mild rejection": 0.0,
        "Rejection": 0.0,
        "Terminate": 1.0,
      },
      rng=random.Random(7),
      utterance_index=30,
    )
    self.assertEqual(sampled, "Terminate")

  def test_sample_patient_action_hard_blocks_terminate_before_safe_threshold(self) -> None:
    sampled = sample_patient_action(
      {
        "Acceptance": 0.0,
        "Hesitation": 0.0,
        "Mild rejection": 0.0,
        "Rejection": 0.0,
        "Terminate": 1.0,
      },
      rng=random.Random(7),
      utterance_index=29,
    )
    self.assertNotEqual(sampled, "Terminate")

  def test_negative_high_interval_detection(self) -> None:
    self.assertFalse(is_highest_negative_calibration_interval(0.87))
    self.assertTrue(is_highest_negative_calibration_interval(0.88))
    self.assertTrue(is_highest_negative_calibration_interval(1.0))

  def test_fallback_action_distribution_respects_negative_interval(self) -> None:
    positive_fallback = fallback_action_distribution_for_resistance(0.20)
    negative_fallback = fallback_action_distribution_for_resistance(0.70)
    negative_post_threshold_fallback = fallback_action_distribution_for_resistance(0.70, utterance_index=30)
    high_negative_post_threshold_fallback = fallback_action_distribution_for_resistance(0.90, utterance_index=30)
    self.assertEqual(positive_fallback["Terminate"], 0.0)
    self.assertEqual(positive_fallback["Acceptance"], 25.0)
    self.assertEqual(positive_fallback["Hesitation"], 25.0)
    self.assertEqual(negative_fallback["Terminate"], 0.0)
    self.assertEqual(negative_fallback["Acceptance"], 10.0)
    self.assertEqual(negative_post_threshold_fallback["Terminate"], 10.0)
    self.assertEqual(high_negative_post_threshold_fallback["Terminate"], 20.0)

  def test_constrain_action_distribution_for_utterance_strips_terminate_when_blocked(self) -> None:
    constrained = constrain_action_distribution_for_utterance(
      {
        "Acceptance": 40.0,
        "Hesitation": 20.0,
        "Mild rejection": 10.0,
        "Rejection": 5.0,
        "Terminate": 25.0,
      },
      utterance_index=29,
    )
    self.assertEqual(constrained["Terminate"], 0.0)
    self.assertEqual(constrained["Acceptance"], 40.0)

  def test_can_select_terminate_action_uses_safe_threshold(self) -> None:
    self.assertFalse(can_select_terminate_action(29))
    self.assertTrue(can_select_terminate_action(30))


if __name__ == "__main__":
  unittest.main()
