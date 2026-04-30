import json
from pathlib import Path
import tempfile
import unittest

from scripts.data import create_it_dataset_from_raw as builder


def _message(idx: int, role: str, content: str) -> dict:
  return {
    "utterance_index": idx,
    "role": role,
    "content": content,
    "stage": "ignored_stage",
    "substage": "ignored_substage",
  }


def _trace(step: str, output: object, input_text: str | None = None) -> dict:
  return {
    "step": step,
    "input": input_text or f"instruction for {step}",
    "output": output if isinstance(output, str) else json.dumps(output),
  }


def _trace_group(idx: int, speaker: str, traces: list[dict]) -> dict:
  return {
    "utterance_index": idx,
    "speaker": speaker,
    "traces": traces,
  }


def _project_root(base: Path) -> Path:
  agent_prompts = base / "agent_prompts"
  agent_prompts.mkdir(parents=True)
  (agent_prompts / "therapist_system.txt").write_text(
    "AGENT SYSTEM\n<patient_intake_form>\n{patient_intake_form}\n</patient_intake_form>",
    encoding="utf-8",
  )
  prompts = base / "prompts"
  prompts.mkdir(parents=True)
  (prompts / "therapist_system.txt").write_text(
    "SHOULD NOT BE USED\n{patient_intake_form}",
    encoding="utf-8",
  )
  return base


def _session_payload() -> dict:
  messages = [
    _message(1, "therapist", "T1 outside context window."),
    _message(2, "patient", "P2 outside context window."),
    _message(3, "therapist", "T3 historical response."),
    _message(4, "patient", "P4 with patient trace."),
    _message(5, "therapist", "T5 historical response without traces."),
    _message(6, "patient", "P6."),
    _message(7, "therapist", "T7."),
    _message(8, "patient", "P8."),
    _message(9, "therapist", "T9."),
    _message(10, "patient", "P10."),
    _message(11, "therapist", "T11."),
    _message(12, "patient", "P12."),
    _message(13, "therapist", "Target therapist response."),
  ]
  return {
    "patient_profile": {
      "personal_info": {
        "name": "Jason Crawford",
        "age": "44",
      },
      "presenting_problem": "This raw-only field should not appear in the instruction.",
      "reason_seeking_counseling": "Need help with commute stress.",
      "more_information": {
        "past_history": "No prior counseling.",
      },
    },
    "messages": messages,
    "reasoning_traces": [
      _trace_group(1, "therapist", []),
      _trace_group(2, "patient", [_trace("resistance_update", {"reasoning": "Patient-only reasoning."})]),
      _trace_group(3, "therapist", [_trace("resistance_prediction", {"reasoning": "Earlier therapist reasoning."})]),
      _trace_group(4, "patient", [_trace("resistance_update", {"reasoning": "Should stay out of context."})]),
      _trace_group(5, "therapist", []),
      _trace_group(
        13,
        "therapist",
        [
          _trace("opening_info_update", {"mood": "tense", "problem": "commute stress"}),
          _trace("unknown_string", "Done."),
        ],
      ),
    ],
  }


class CreateItDatasetFromRawTest(unittest.TestCase):
  def test_builds_jsonl_samples_from_raw_messages_and_traces(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      input_dir.mkdir()
      output_file = base / "out.jsonl"
      (input_dir / "simulation.json").write_text(json.dumps(_session_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_file, root)

      self.assertEqual(count, 2)
      rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    self.assertEqual(len(rows), 2)
    target_row = rows[1]
    self.assertEqual(set(target_row.keys()), {"instruction", "input", "output"})
    self.assertIn("AGENT SYSTEM", target_row["instruction"])
    self.assertNotIn("SHOULD NOT BE USED", target_row["instruction"])
    self.assertIn("Name: Jason Crawford", target_row["instruction"])
    self.assertIn("Reason Seeking Counseling:\nNeed help with commute stress.", target_row["instruction"])
    self.assertNotIn("This raw-only field should not appear", target_row["instruction"])

    context_text, continuation = target_row["input"].split("</context>\n", 1)
    self.assertEqual(continuation, "Continue with the ongoing therapy session")
    self.assertNotIn("T1 outside context window.", context_text)
    self.assertNotIn("P2 outside context window.", context_text)
    self.assertIn(
      "Therapist: <think>[resistance_prediction] Earlier therapist reasoning.</think> T3 historical response.",
      context_text,
    )
    self.assertIn("Patient: P4 with patient trace.", context_text)
    self.assertIn("Therapist: T5 historical response without traces.", context_text)
    self.assertNotIn("ignored_stage", context_text)
    self.assertNotIn("Should stay out of context.", context_text)

    self.assertIn(
      "<think>[opening_info_update] I updated the opening context from the patient response",
      target_row["output"],
    )
    self.assertIn("[unknown_string] I produced this result: Done.", target_row["output"])
    self.assertTrue(target_row["output"].endswith("</think>Target therapist response."))

  def test_missing_patient_profile_fails_fast(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      input_dir.mkdir()
      output_file = base / "out.jsonl"
      (input_dir / "simulation.json").write_text(
        json.dumps({"messages": [], "reasoning_traces": []}),
        encoding="utf-8",
      )

      with self.assertRaises(builder.DatasetBuildError):
        builder.build_dataset(input_dir, output_file, root)

      self.assertFalse(output_file.exists())


if __name__ == "__main__":
  unittest.main()
