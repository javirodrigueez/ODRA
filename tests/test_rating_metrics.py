import json
import tempfile
from pathlib import Path
import unittest

from evaluation.rating_metrics import compute_scores


class RatingMetricsTest(unittest.TestCase):
  def _write_json(self, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
      json.dump(payload, f)

  def test_compute_scores_aggregates_ctrs_wai_panas_and_cost(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      self._write_json(
        base / "ctrs" / "simulation_1_neutral.json",
        {
          "general_1_understanding": 4,
          "general_2_interpersonal_effectiveness": 2,
          "general_3_collaboration": 3,
          "CBT_1_guided_discovery": 4,
          "CBT_2_focus": 5,
          "CBT_3_strategy": 2,
          "cost": 0.1,
        },
      )
      self._write_json(
        base / "ctrs" / "simulation_2_neutral.json",
        {
          "general_1_understanding": 2,
          "general_2_interpersonal_effectiveness": 4,
          "general_3_collaboration": 5,
          "CBT_1_guided_discovery": 2,
          "CBT_2_focus": 3,
          "CBT_3_strategy": 4,
          "cost": 0.2,
        },
      )
      self._write_json(
        base / "wai" / "simulation_1_neutral.json",
        {
          **{f"wai{i}": 4 for i in range(1, 13)},
          "wai4": 2,
          "wai10": 6,
          "cost": 0.3,
        },
      )

      before = {
        "attitude": "neutral",
        "cost": 0.4,
        "interested": [1, 3],
        "excited": [2],
        "strong": [2],
        "enthusiastic": [2],
        "proud": [2],
        "alert": [2],
        "inspired": [2],
        "determined": [2],
        "attentive": [2],
        "active": [2],
        "distressed": [5],
        "upset": [5],
        "guilty": [5],
        "scared": [5],
        "hostile": [5],
        "irritable": [5],
        "ashamed": [5],
        "nervous": [5],
        "jittery": [5],
        "afraid": [5],
      }
      after = dict(before)
      after["cost"] = 0.5
      for key in ("interested", "excited", "strong", "enthusiastic", "proud", "alert", "inspired", "determined", "attentive", "active"):
        after[key] = [4]
      for key in ("distressed", "upset", "guilty", "scared", "hostile", "irritable", "ashamed", "nervous", "jittery", "afraid"):
        after[key] = [3]
      self._write_json(base / "panas_before" / "simulation_1_neutral.json", before)
      self._write_json(base / "panas_after" / "simulation_1_neutral.json", after)

      scores = compute_scores(base)

      self.assertEqual(scores["ctrs"]["general_1_understanding"], 3)
      self.assertEqual(scores["ctrs"]["CBT_3_strategy"], 3)
      self.assertEqual(scores["wai"]["items"]["wai1"], 4)
      self.assertEqual(scores["wai"]["score_task"], 4)
      self.assertEqual(scores["wai"]["score_goal"], 4)
      self.assertEqual(scores["wai"]["score_bond"], 4)
      self.assertAlmostEqual(scores["panas"]["positive_change"]["neutral"], 2.0)
      self.assertEqual(scores["panas"]["negative_change"]["neutral"], -2.0)
      self.assertAlmostEqual(scores["cost"]["total"], 1.5)


if __name__ == "__main__":
  unittest.main()
