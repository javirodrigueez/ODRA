import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import unittest

from evaluation import run_batch_metrics
from scripts.data import create_cactus_set


def _intake_form(name: str) -> str:
  return f"""Name: {name}
Age: 31
Gender: female
Occupation: Analyst
Education: College
Marital Status: Single
Family Details: Lives with a roommate

2. Presenting Problem:
I feel stuck and worried about work.

3. Reason for Seeking Counseling:
I want help understanding why these thoughts keep coming up.

4. Past History:
No prior counseling.

5. Academic/occupational functioning level:
Work stress has increased.

6. Social Support System:
Has a close friend.
"""


def _row(case: int, attitude: str, technique: str = "Reality Testing") -> dict:
  return {
    "thought": f"case {case} thought",
    "patterns": ["catastrophizing", "mental filtering"],
    "intake_form": _intake_form(f"Patient {case}"),
    "cbt_technique": technique,
    "attitude": attitude,
    "dialogue": (
      f"Counselor: Hello Patient {case}.\n"
      f"Client: This is my {attitude} response.\n"
      "Counselor: What evidence supports that thought?"
    ),
  }


def _complete_rows(case_count: int) -> list[dict]:
  rows = []
  for case in range(case_count):
    for attitude in create_cactus_set.ATTITUDES:
      rows.append(_row(case, attitude))
  return rows


class CreateCactusSetTest(unittest.TestCase):
  def test_parse_dialogue_normalizes_roles(self) -> None:
    messages = create_cactus_set.parse_dialogue(
      "Counselor: Hi there.\n"
      "Client (sighs): I feel nervous.\n"
      "Client: I can keep talking."
    )

    self.assertEqual(
      messages,
      [
        {"role": "therapist", "content": "Hi there."},
        {"role": "patient", "content": "I feel nervous."},
        {"role": "patient", "content": "I can keep talking."},
      ],
    )

  def test_complete_case_groups_by_intake_thought_and_technique(self) -> None:
    rows = [
      _row(1, "positive", "Reality Testing"),
      _row(1, "negative", "Reality Testing"),
      _row(2, "positive", "Reality Testing"),
      _row(1, "neutral", "Reality Testing"),
      _row(1, "positive", "Alternative Perspective"),
      _row(1, "neutral", "Alternative Perspective"),
      _row(1, "negative", "Alternative Perspective"),
    ]

    groups = create_cactus_set.complete_case_groups(rows)

    self.assertEqual(len(groups), 2)
    first_group = groups[0]
    self.assertEqual(first_group.case_id, 0)
    self.assertEqual(
      {attitude: row.index for attitude, row in first_group.rows_by_attitude.items()},
      {"positive": 0, "neutral": 3, "negative": 1},
    )
    self.assertEqual(first_group.key[1], "case 1 thought")
    self.assertEqual(first_group.key[2], "Reality Testing")

  def test_write_cactus_set_creates_balanced_metric_compatible_files(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      output_dir = Path(tmp)
      stale_file = output_dir / "simulation_stale.json"
      keep_file = output_dir / "notes.txt"
      nested_dir = output_dir / "nested"
      nested_file = nested_dir / "simulation_nested.json"
      stale_file.write_text("{}", encoding="utf-8")
      keep_file.write_text("keep", encoding="utf-8")
      nested_dir.mkdir()
      nested_file.write_text("{}", encoding="utf-8")

      summary = create_cactus_set.write_cactus_set(_complete_rows(20), output_dir)

      self.assertEqual(summary["sampled_cases"], 20)
      self.assertEqual(summary["written_files"], 60)
      self.assertEqual(summary["removed_files"], 1)
      self.assertFalse(stale_file.exists())
      self.assertTrue(keep_file.exists())
      self.assertTrue(nested_file.exists())

      files = sorted(output_dir.glob("simulation_*.json"))
      self.assertEqual(len(files), 60)

      counts = Counter()
      case_ids_by_attitude = defaultdict(set)
      for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        attitude = payload["patient_profile"]["attitude"]
        case_id = path.stem.removeprefix("simulation_").rsplit("_", 1)[0]
        counts[attitude] += 1
        case_ids_by_attitude[attitude].add(case_id)

        self.assertEqual(set(payload["messages"][0]), {"role", "content"})
        self.assertEqual(payload["messages"][0]["role"], "therapist")
        self.assertIn("personal_info", payload["patient_profile"])
        self.assertIn("more_information", payload["patient_profile"])
        self.assertEqual(payload["patient_profile"]["emotional_recovery"], "moderate")

      self.assertEqual(counts, {"positive": 20, "neutral": 20, "negative": 20})
      self.assertEqual(
        case_ids_by_attitude["positive"],
        case_ids_by_attitude["neutral"],
      )
      self.assertEqual(
        case_ids_by_attitude["positive"],
        case_ids_by_attitude["negative"],
      )

      records = run_batch_metrics.discover_conversations(output_dir)
      self.assertEqual(len(records), 60)


if __name__ == "__main__":
  unittest.main()
