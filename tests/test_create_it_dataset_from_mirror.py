import json
from pathlib import Path
import tempfile
import unittest

from scripts.ft import create_it_dataset_from_mirror as builder


def _project_root(base: Path) -> Path:
  prompts = base / "ft_prompts"
  prompts.mkdir(parents=True)
  (prompts / "llava.txt").write_text(
    "LLAVA SYSTEM",
    encoding="utf-8",
  )
  return base


def _mirror_payload() -> dict:
  return {
    "patient_profile": {
      "personal_info": {
        "name": "Riley Park",
        "age": "33",
      },
      "reason_seeking_counseling": "Trouble with self-critical thoughts.",
    },
    "messages": [
      {"role": "therapist", "content": "T1."},
      {"role": "patient", "content": "P2."},
      {"role": "therapist", "content": "T3."},
    ],
  }


class CreateItDatasetFromMirrorTest(unittest.TestCase):
  def test_builds_turn_files_with_positional_indices(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      cactus_plan = base / "processed_cactus_plan.json"
      input_dir.mkdir()
      (input_dir / "simulation_001_positive.json").write_text(
        json.dumps(_mirror_payload()),
        encoding="utf-8",
      )
      cactus_plan.write_text(
        json.dumps(
          [
            {
              "personal_info": {
                "name": "Riley Park",
                "age": "33",
              },
              "cbt_plan": "Reality Testing\n\nCounseling plan:\n1. Test the evidence.",
            }
          ]
        ),
        encoding="utf-8",
      )

      count = builder.build_dataset(input_dir, output_dir, root, cactus_plan_path=cactus_plan)
      output_names = [path.name for path in sorted(output_dir.iterdir())]
      first = json.loads((output_dir / "simulation_001_positive.turn_0001.json").read_text(encoding="utf-8"))
      target = json.loads((output_dir / "simulation_001_positive.turn_0003.json").read_text(encoding="utf-8"))

    self.assertEqual(count, 2)
    self.assertEqual(output_names, [
      "simulation_001_positive.turn_0001.json",
      "simulation_001_positive.turn_0003.json",
    ])
    self.assertEqual([message["role"] for message in target["messages"]], ["system", "user", "assistant"])
    self.assertEqual(first["messages"][0]["content"], "LLAVA SYSTEM")
    self.assertIn("- Personal information: Name: Riley Park\nAge: 33", first["messages"][1]["content"])
    self.assertIn("- Reason for seeking counseling: Trouble with self-critical thoughts.", first["messages"][1]["content"])
    self.assertIn("- CBT Technique: Reality Testing", first["messages"][1]["content"])
    self.assertIn("Below is a conversation between the client and the psychotherapist.", first["messages"][1]["content"])
    self.assertIn(
      "Psychotherapist: T1.\nClient: P2.",
      target["messages"][1]["content"],
    )
    self.assertTrue(
      target["messages"][1]["content"].endswith(
        "Based on the conversation so far, respond as a psychotherapist conducting a CBT session."
      )
    )
    self.assertEqual(target["messages"][2]["content"], "T3.")


if __name__ == "__main__":
  unittest.main()
