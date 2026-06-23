import json
import tempfile
from pathlib import Path
import unittest

from scripts.data import create_from_cactus


def _intake_form(name: str) -> str:
  return f"""Name:
{name}
Age:
44
Gender:
female
Occupation: Veterinarian
Education: Veterinary degree
Marital Status: Single
Family Details: 3 dogs and 2 cats

2. Presenting Problem
I called in sick at work and now I think I will be fired.

3. Reason for Seeking Counseling
The fear has become overwhelming.

4. Past History (including medical history)
I have not experienced similar problems before.

5. Academic/occupational functioning level: My work performance has been affected.
Interpersonal relationships: My relationships have been strained.
Daily life: My sleep quality has been affected.

6. Social Support System
I have supportive friends.
"""


class CreateFromCactusTest(unittest.TestCase):
  def test_json_source_ai_client_rows_preserve_source_attitudes(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      source = base / "eval_subset.json"
      output = base / "processed.json"
      source.write_text(
        json.dumps(
          [
            {
              "AI_client": {
                "intake_form": _intake_form("Rebecca Finch"),
                "attitude": "negative",
              },
              "ground_truth_CBT": ["Alternative Perspective"],
            },
            {
              "AI_client": {
                "intake_form": _intake_form("Rebecca Finch"),
                "attitude": "positive",
              },
              "ground_truth_CBT": ["Alternative Perspective"],
            },
          ]
        ),
        encoding="utf-8",
      )

      create_from_cactus.process_dataset(str(output), str(source))

      data = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(len(data), 2)
      self.assertEqual([entry["attitude"] for entry in data], ["negative", "positive"])
      self.assertEqual(data[0]["personal_info"]["name"], "Rebecca Finch")
      self.assertEqual(data[0]["personal_info"]["occupation"], "Veterinarian")
      self.assertEqual(data[0]["patterns"], [])
      self.assertEqual(data[0]["emotional_recovery"], "moderate")
      self.assertIn("fired", data[0]["presenting_problem"])
      self.assertIn("overwhelming", data[0]["reason_seeking_counseling"])

  def test_json_source_without_attitude_expands_all_attitudes(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      source = base / "raw_cactus_subset.json"
      output = base / "processed.json"
      source.write_text(
        json.dumps(
          [
            {
              "intake_form": _intake_form("Alex Carter"),
              "patterns": ["catastrophizing"],
            }
          ]
        ),
        encoding="utf-8",
      )

      create_from_cactus.process_dataset(str(output), str(source))

      data = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(len(data), 3)
      self.assertEqual([entry["attitude"] for entry in data], create_from_cactus.ATTITUDES)
      self.assertEqual(data[0]["patterns"], ["catastrophizing"])
      self.assertNotIn("cbt_plan", data[0])

  def test_include_cbt_plan_uses_top_level_field(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      source = base / "raw_cactus_subset.json"
      output = base / "processed.json"
      source.write_text(
        json.dumps(
          [
            {
              "intake_form": _intake_form("Alex Carter"),
              "patterns": ["catastrophizing"],
              "cbt_plan": "Use decatastrophizing to examine the feared outcome.",
            }
          ]
        ),
        encoding="utf-8",
      )

      create_from_cactus.process_dataset(str(output), str(source), include_cbt_plan=True)

      data = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(len(data), 3)
      self.assertEqual(data[0]["cbt_plan"], "Use decatastrophizing to examine the feared outcome.")

  def test_ground_truth_cbt_is_not_used_as_cbt_plan_fallback(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      source = base / "raw_cactus_subset.json"
      output = base / "processed.json"
      source.write_text(
        json.dumps(
          [
            {
              "AI_client": {
                "intake_form": _intake_form("Rebecca Finch"),
                "attitude": "negative",
              },
              "ground_truth_CBT": ["Alternative Perspective"],
            }
          ]
        ),
        encoding="utf-8",
      )

      create_from_cactus.process_dataset(str(output), str(source), include_cbt_plan=True)

      data = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(len(data), 1)
      self.assertEqual(data[0]["cbt_plan"], "")

  def test_json_source_directory_loads_cactus_session_records(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      source_dir = base / "cactus-chatgpt"
      source_dir.mkdir()
      output = base / "processed.json"
      (source_dir / "session_1.json").write_text(
        json.dumps(
          {
            "example": {
              "AI_client": {
                "intake_form": _intake_form("Rebecca Finch"),
                "attitude": "positive",
              },
              "ground_truth_CBT": ["Alternative Perspective"],
            },
            "cbt_plan": "Practice generating alternative perspectives.",
          }
        ),
        encoding="utf-8",
      )

      create_from_cactus.process_dataset(str(output), str(source_dir), include_cbt_plan=True)

      data = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(len(data), 1)
      self.assertEqual(data[0]["attitude"], "positive")
      self.assertEqual(data[0]["cbt_plan"], "Practice generating alternative perspectives.")

  def test_include_cbt_plan_full_source_preserves_rows_and_source_attitudes(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      output = base / "processed.json"
      rows = [
        {
          "intake_form": _intake_form("Shared Name"),
          "patterns": ["catastrophizing"],
          "attitude": "negative",
          "cbt_plan": "First plan.",
        },
        {
          "intake_form": _intake_form("Shared Name"),
          "patterns": ["catastrophizing"],
          "attitude": "positive",
          "cbt_plan": "Second plan.",
        },
      ]
      original_loader = create_from_cactus.load_huggingface_rows
      create_from_cactus.load_huggingface_rows = lambda: rows
      try:
        create_from_cactus.process_dataset(str(output), include_cbt_plan=True)
      finally:
        create_from_cactus.load_huggingface_rows = original_loader

      data = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(len(data), 2)
      self.assertEqual([entry["attitude"] for entry in data], ["negative", "positive"])
      self.assertEqual([entry["cbt_plan"] for entry in data], ["First plan.", "Second plan."])


if __name__ == "__main__":
  unittest.main()
