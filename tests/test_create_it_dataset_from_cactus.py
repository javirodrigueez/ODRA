import json
from pathlib import Path
import tempfile
import unittest

from scripts.ft import create_it_dataset_from_cactus as builder


def _project_root(base: Path) -> Path:
  prompts = base / "ft_prompts"
  prompts.mkdir(parents=True)
  (prompts / "cactus.txt").write_text(
    "CACTUS SYSTEM\n\nClient:\n{patient_intake_form}",
    encoding="utf-8",
  )
  (prompts / "magnet.txt").write_text(
    "MAGNET SYSTEM\n\nClient:\n{patient_intake_form}",
    encoding="utf-8",
  )
  return base


def _cactus_payload() -> dict:
  return {
    "example": {
      "AI_client": {
        "intake_form": "Name:\nCasey\n\n2. Presenting Problem\nStress.",
      }
    },
    "messages": [
      {"utterance_index": 1, "role": "therapist", "content": "T1."},
      {"utterance_index": 2, "role": "patient", "content": "P2."},
      {"utterance_index": 3, "role": "therapist", "content": "T3."},
    ],
  }


def _magnet_payload() -> dict:
  return {
    "AI_client": {
      "intake_form": "Name:\nMorgan\n\n2. Presenting Problem\nWorry.",
    },
    "dialogue": [
      {"role": "counselor", "message": "M1."},
      {"role": "client", "message": "M2."},
      {"role": "counselor", "message": "M3."},
    ],
  }


class CreateItDatasetFromCactusTest(unittest.TestCase):
  def test_builds_chatml_session_from_cactus_messages(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "session_1.json").write_text(json.dumps(_cactus_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_dir, root)
      output = json.loads((output_dir / "session_1.json").read_text(encoding="utf-8"))

    self.assertEqual(count, 1)
    self.assertEqual([message["role"] for message in output["messages"]], ["system", "assistant", "user", "assistant"])
    self.assertIn("Name:\nCasey", output["messages"][0]["content"])
    self.assertEqual(output["messages"][-1]["content"], "T3.")

  def test_builds_turn_samples_from_magnet_dialogue(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "session_2.json").write_text(json.dumps(_magnet_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_dir, root, mode="turns")
      first = json.loads((output_dir / "session_2.turn_0001.json").read_text(encoding="utf-8"))
      target = json.loads((output_dir / "session_2.turn_0003.json").read_text(encoding="utf-8"))

    self.assertEqual(count, 2)
    self.assertEqual([message["role"] for message in target["messages"]], ["system", "user", "assistant"])
    self.assertEqual(first["messages"][0]["content"], "MAGNET SYSTEM")
    self.assertEqual(
      target["messages"][1]["content"],
      "Dialogue History: Therapist: M1.\nPatient: M2.\nClient Profile: Name:\nMorgan\n\n2. Presenting Problem\nWorry.",
    )
    self.assertEqual(target["messages"][2]["content"], "M3.")

  def test_sharegpt_template_is_supported_without_reasoning_flags(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "session_1.json").write_text(json.dumps(_cactus_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_dir, root, template="sharegpt")
      output = json.loads((output_dir / "session_1.json").read_text(encoding="utf-8"))

    self.assertEqual(count, 1)
    self.assertEqual([message["from"] for message in output["conversations"]], ["system", "gpt", "human", "gpt"])

  def test_turn_mode_accepts_plain_system_prompt_without_intake_placeholder(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      (root / "ft_prompts" / "cactus.txt").write_text("PLAIN CACTUS SYSTEM", encoding="utf-8")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "session_1.json").write_text(json.dumps(_cactus_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_dir, root, mode="turns")
      output = json.loads((output_dir / "session_1.turn_0003.json").read_text(encoding="utf-8"))

    self.assertEqual(count, 2)
    self.assertEqual(output["messages"][0]["content"], "PLAIN CACTUS SYSTEM")
    self.assertIn("Client Profile: Name:\nCasey", output["messages"][1]["content"])


if __name__ == "__main__":
  unittest.main()
