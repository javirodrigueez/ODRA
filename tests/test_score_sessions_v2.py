import json
import tempfile
from pathlib import Path
import unittest

from evaluation.steering import score_sessions_v2
from evaluation.steering.steering_v2 import SteeringV2EvaluationError


def _write_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f)


def _fragment(expected_class: str = "neutral") -> dict:
  return {
    "fragment_index": 1,
    "fragment_key": "session.json#fragment-0001",
    "turn_count": 1,
    "expected_resistance_interval": expected_class,
    "resistance_concept_scores": {
      "positive": [{"score": 1.5}],
      "neutral": [{"score": 1.0}],
      "negative": [{"score": 0.5}],
    },
    "resistance_class_scores": {
      "positive": 1.5,
      "neutral": 1.0,
      "negative": 0.5,
    },
    "expected_resistance_concept_score": 1.0,
    "predicted_resistance_interval": "positive",
    "resistance_interval_match": False,
    "resistance_interval_tie": False,
    "context_relevance": 1.0,
    "patient_like_realism": 2.0,
    "harmonic_mean": 1.3333333333333333,
    "cost": 0.0,
    "evaluated_metrics": [
      "resistance_concept",
      "context_relevance",
      "patient_like_realism",
    ],
  }


class ScoreSessionsV2Test(unittest.TestCase):
  def test_score_sessions_folder_cross_validate_penalizes_existing_outputs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      sessions_dir = base / "sessions"
      output_dir = base / "rescored"
      _write_json(
        sessions_dir / "session.json",
        {
          "input_file": "session.json",
          "input_filename": "session.json",
          "matched_concepts": False,
          "evaluated_fragments": [_fragment()],
          "skipped_turns": [],
        },
      )

      summary = score_sessions_v2.score_sessions_folder(
        sessions_dir,
        output_dir,
        cross_validate=True,
      )
      rescored = score_sessions_v2.read_session_json(output_dir / "sessions" / "session.json")

    fragment = rescored["evaluated_fragments"][0]
    self.assertTrue(rescored["cross_validate"])
    self.assertTrue(fragment["cross_validation_penalized"])
    self.assertEqual(fragment["unpenalized_expected_resistance_concept_score"], 1.0)
    self.assertEqual(fragment["expected_resistance_concept_score"], 0.0)
    self.assertEqual(fragment["harmonic_mean"], 0.0)
    self.assertTrue(summary["cross_validate"])
    self.assertEqual(summary["mean_judge_scores"]["expected_resistance_concept_score"], 0.0)

  def test_score_sessions_folder_cross_validate_ignored_for_matched_concepts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      sessions_dir = base / "sessions"
      output_dir = base / "rescored"
      _write_json(
        sessions_dir / "session.json",
        {
          "input_file": "session.json",
          "input_filename": "session.json",
          "matched_concepts": True,
          "evaluated_fragments": [_fragment()],
          "skipped_turns": [],
        },
      )

      summary = score_sessions_v2.score_sessions_folder(
        sessions_dir,
        output_dir,
        cross_validate=True,
      )
      rescored = score_sessions_v2.read_session_json(output_dir / "sessions" / "session.json")

    self.assertFalse(rescored["cross_validate"])
    self.assertEqual(
      rescored["evaluated_fragments"][0]["expected_resistance_concept_score"],
      1.0,
    )
    self.assertFalse(summary["cross_validate"])

  def test_score_sessions_folder_cross_validate_fails_on_missing_class_scores(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      sessions_dir = base / "sessions"
      payload = _fragment()
      del payload["resistance_concept_scores"]["negative"]
      _write_json(
        sessions_dir / "session.json",
        {
          "input_file": "session.json",
          "input_filename": "session.json",
          "matched_concepts": False,
          "evaluated_fragments": [payload],
          "skipped_turns": [],
        },
      )

      with self.assertRaises(SteeringV2EvaluationError):
        score_sessions_v2.score_sessions_folder(sessions_dir, cross_validate=True)


if __name__ == "__main__":
  unittest.main()
