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


def _write_seed_file(path: Path, pairs: list[tuple[str, str]]) -> None:
  payload = [
    {
      "AI_client": {
        "intake_form": _intake_form(name),
        "attitude": attitude,
      }
    }
    for name, attitude in pairs
  ]
  path.write_text(json.dumps(payload), encoding="utf-8")


def _session_payload(name: str = "Rebecca Finch", attitude: str = "negative") -> dict:
  return {
    "example": {
      "AI_client": {
        "intake_form": _intake_form(name),
        "attitude": attitude,
      },
      "ground_truth_CBT": ["Alternative Perspective"],
    },
    "cbt_technique": "CBT technique:Decatastrophizing",
    "history": [
      {"role": "counselor", "message": "Hello, how are you feeling today?"},
      {"role": "client", "message": "I feel worried about work."},
      {"role": "counselor", "message": "Let's slow down and look at that thought."},
    ],
  }


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

  def test_seed_file_writes_referenced_name_attitude_pairs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      seed_file = base / "evaluation.json"
      output_dir = base / "out"
      _write_seed_file(
        seed_file,
        [
          ("Patient 1", "positive"),
          ("Patient 1", "neutral"),
          ("Patient 1", "negative"),
          ("Patient 1", "positive"),
        ],
      )
      rows = [
        _row(1, "positive", "Reality Testing"),
        _row(1, "neutral", "Reality Testing"),
        _row(1, "negative", "Reality Testing"),
        _row(1, "positive", "Alternative Perspective"),
        _row(2, "positive", "Reality Testing"),
      ]

      summary = create_cactus_set.write_cactus_set_from_seed_file(rows, output_dir, seed_file)

      self.assertEqual(summary["seed_references"], 4)
      self.assertEqual(summary["written_files"], 4)
      self.assertEqual(summary["ambiguous_matches"], 2)
      self.assertEqual(
        sorted(summary["files"]),
        [
          "simulation_0_positive.json",
          "simulation_0_positive_seed_3.json",
          "simulation_1_neutral.json",
          "simulation_2_negative.json",
        ],
      )

      first_positive = json.loads(
        (output_dir / "simulation_0_positive.json").read_text(encoding="utf-8")
      )
      duplicate_positive = json.loads(
        (output_dir / "simulation_0_positive_seed_3.json").read_text(encoding="utf-8")
      )
      self.assertEqual(first_positive["patient_profile"]["idx"], 0)
      self.assertEqual(duplicate_positive["patient_profile"]["idx"], 0)
      self.assertIn("Reality Testing", rows[first_positive["patient_profile"]["idx"]]["cbt_technique"])

      records = run_batch_metrics.discover_conversations(output_dir)
      self.assertEqual(len(records), 4)

  def test_seed_file_reports_missing_pairs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      seed_file = base / "evaluation.json"
      _write_seed_file(seed_file, [("Missing Patient", "positive")])

      with self.assertRaisesRegex(create_cactus_set.CactusSetError, "Missing Patient"):
        create_cactus_set.write_cactus_set_from_seed_file(
          [_row(1, "positive")],
          base / "out",
          seed_file,
        )

  def test_json_source_session_directory_writes_metric_compatible_files(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      source_dir = base / "sessions"
      output_dir = base / "out"
      source_dir.mkdir()
      output_dir.mkdir()
      (source_dir / "session_1_cactus.json").write_text(
        json.dumps(_session_payload()),
        encoding="utf-8",
      )
      (source_dir / "session 2.json").write_text(
        json.dumps(_session_payload(name="Alex Carter", attitude="positive")),
        encoding="utf-8",
      )
      stale_file = output_dir / "simulation_stale.json"
      keep_file = output_dir / "notes.txt"
      stale_file.write_text("{}", encoding="utf-8")
      keep_file.write_text("{}", encoding="utf-8")

      summary = create_cactus_set.write_cactus_set_from_json_source(source_dir, output_dir)

      self.assertEqual(summary["source_files"], 2)
      self.assertEqual(summary["written_files"], 2)
      self.assertEqual(summary["removed_files"], 1)
      self.assertFalse(stale_file.exists())
      self.assertTrue(keep_file.exists())
      self.assertEqual(
        sorted(summary["files"]),
        ["simulation_session_1_cactus.json", "simulation_session_2.json"],
      )

      payload = json.loads(
        (output_dir / "simulation_session_1_cactus.json").read_text(encoding="utf-8")
      )
      self.assertEqual(
        payload["messages"],
        [
          {"role": "therapist", "content": "Hello, how are you feeling today?"},
          {"role": "patient", "content": "I feel worried about work."},
          {
            "role": "therapist",
            "content": "Let's slow down and look at that thought.",
          },
        ],
      )
      self.assertEqual(payload["patient_profile"]["attitude"], "negative")
      self.assertEqual(payload["patient_profile"]["personal_info"]["name"], "Rebecca Finch")
      self.assertIn("work", payload["patient_profile"]["presenting_problem"])

      records = run_batch_metrics.discover_conversations(output_dir)
      self.assertEqual(len(records), 2)
      self.assertEqual(
        sorted(record.filename for record in records),
        ["simulation_session_1_cactus.json", "simulation_session_2.json"],
      )


if __name__ == "__main__":
  unittest.main()
