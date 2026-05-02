import json
from pathlib import Path
import tempfile
import unittest

from scripts.ft import create_reasoning_instruction_dataset as builder


def _message(idx: int, role: str, content: str, stage: str = "opening", substage: str = "open_mood") -> dict:
  return {
    "utterance_index": idx,
    "role": role,
    "content": content,
    "stage": stage,
    "substage": substage,
  }


def _trace(step: str, output: object, input_text: str | None = None) -> dict:
  return {
    "step": step,
    "input": input_text or f"instruction for {step}",
    "output": output if isinstance(output, str) else json.dumps(output),
  }


def _project_root(base: Path) -> Path:
  prompts = base / "prompts"
  prompts.mkdir(parents=True)
  (prompts / "therapist_system.txt").write_text(
    "SYSTEM\n<patient_intake_form>\n{patient_intake_form}\n</patient_intake_form>",
    encoding="utf-8",
  )
  return base


def _context() -> dict:
  return {
    "current_stage": "id_distortions",
    "current_substage": "cognitive_review_active_question",
    "opening_patient_mood": "tense",
    "opening_patient_problem": "commute stress",
    "predicted_resistance": 0.2,
    "cognitive_model_sets": [
      {
        "cognitive_model_set_id": 1,
        "situation": "Driving to work",
        "automatic_thoughts": ["This ruins my day"],
        "emotions": ["tense"],
        "behaviors": ["snaps at coworkers"],
        "core_belief": "helplessness",
        "core_belief_probabilities": {
          "helplessness": 0.8,
          "unlovability": 0.1,
          "worthlessness": 0.1,
        },
        "distortion": "fortune_telling",
        "distortion_probabilities": {
          "fortune_telling": 0.7,
          "catastrophizing": 0.3,
        },
      }
    ],
    "working_cognitive_set": {
      "cognitive_model_set_id": 2,
      "situation": "",
      "automatic_thoughts": [],
      "emotions": [],
      "behaviors": [],
      "core_belief": "",
      "core_belief_probabilities": {},
      "distortion": "",
      "distortion_probabilities": {},
    },
    "shelved_cognitive_sets": [],
    "cognitive_gap_plan": "Ask about the thought.",
    "active_cognitive_target": "auto_thoughts",
    "active_cognitive_question": "What went through your mind?",
    "selected_techniques": [
      {
        "name": "Evidence-Based Questioning",
        "target_distortion": "fortune_telling",
        "rationale": "Tests the prediction.",
      }
    ],
    "current_technique": None,
    "current_technique_plan": "",
    "current_technique_expected_outcome": "",
    "current_technique_current_progress": "",
    "technique_evaluations": ["Good engagement."],
    "tasks": [
      {
        "task_title": "Mood note",
        "task_description": "Write one note.",
        "rationale": "Track the pattern.",
        "expected_outcomes": "More awareness.",
      }
    ],
    "address_tib_name": "",
  }


def _session_payload() -> dict:
  current_patient = _message(
    6,
    "patient",
    "Current patient text.",
    "id_distortions",
    "cognitive_review_active_question",
  )
  return {
    "patient_profile": {
      "personal_info": {
        "name": "Jason Crawford",
        "age": "44",
      },
      "reason_seeking_counseling": "Need help with commute stress.",
      "more_information": {
        "past_history": "No prior counseling.",
      },
    },
    "turns": [
      {
        "utterance_index": 1,
        "stage": "opening",
        "substage": "open_mood",
        "therapist_utterance": "Hi.",
        "last_utterances": [],
        "reasoning_traces": [],
        "context_before_therapist_turn": _context(),
      },
      {
        "utterance_index": 3,
        "stage": "opening",
        "substage": "open_mood",
        "therapist_utterance": "How are you?",
        "last_utterances": [_message(1, "therapist", "Hi."), _message(2, "patient", "Tense.")],
        "reasoning_traces": [],
        "context_before_therapist_turn": _context(),
      },
      {
        "utterance_index": 7,
        "stage": "id_distortions",
        "substage": "cognitive_review_active_question",
        "therapist_utterance": "Therapist response.",
        "last_utterances": [
          _message(1, "therapist", "Hi."),
          _message(2, "patient", "Tense."),
          _message(3, "therapist", "What happened?", "id_distortions", "cognitive_review_active_question"),
          _message(4, "patient", "Traffic.", "id_distortions", "cognitive_review_active_question"),
          _message(5, "therapist", "What went through your mind?", "id_distortions", "cognitive_review_active_question"),
          current_patient,
        ],
        "reasoning_traces": [
          _trace("resistance_prediction", {"reasoning": "Patient is open.", "new_p": 0.2}),
          _trace("opening_info_update", {"mood": "tense", "problem": "commute stress"}),
          _trace(
            "core_belief_probability",
            {"helplessness": 0.7, "unlovability": 0.2, "worthlessness": 0.1},
          ),
          _trace(
            "distortion_probability",
            {
              "mental_filtering": 0.0,
              "fortune_telling": 0.6,
              "mind_reading": 0.0,
              "personalization": 0.0,
              "catastrophizing": 0.3,
              "all_or_nothing_thinking": 0.1,
              "should_statements": 0.0,
              "overgeneralization": 0.0,
              "labeling_and_mislabeling": 0.0,
              "discounting_the_positive": 0.0,
            },
          ),
          _trace(
            "technique_selection",
            {
              "selected_techniques": [
                {
                  "name": "Evidence-Based Questioning",
                  "target_distortion": "fortune_telling",
                  "rationale": "It tests predictions.",
                }
              ]
            },
          ),
          _trace(
            "address_tib_exit",
            {
              "tib_name": "skepticism",
              "exit_reason": "single_turn_complete",
              "completion_note": "Addressed.",
              "predicted_resistance": 0.2,
              "band": "action",
              "resume_stage": "cbt_technique",
              "resume_substage": "cbt_review_current",
              "routed_stage": "cbt_technique",
              "routed_substage": "cbt_review_current",
              "resume_context_active": True,
              "resume_context_subject": "Skepticism was addressed.",
            },
          ),
          _trace(
            "technique_auto_complete",
            "Automatically marking technique as complete.",
            "Technique reached 12 exchange turns without completion.",
          ),
          _trace("unknown_json", {"foo": "bar"}),
          _trace("unknown_string", "Done."),
        ],
        "context_before_therapist_turn": _context(),
      },
    ],
  }


class CreateReasoningInstructionDatasetTest(unittest.TestCase):
  def test_builds_jsonl_samples_with_context_and_trace_fallbacks(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      input_dir.mkdir()
      output_file = base / "out.jsonl"
      (input_dir / "session.json").write_text(json.dumps(_session_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_file, root)

      self.assertEqual(count, 1)
      rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    self.assertEqual(len(rows), 1)
    row = rows[0]
    self.assertEqual(set(row.keys()), {"instruction", "input", "output"})
    self.assertIn("Name: Jason Crawford", row["instruction"])
    self.assertIn("Reason Seeking Counseling:\nNeed help with commute stress.", row["instruction"])
    self.assertIn("Past History:\nNo prior counseling.", row["instruction"])

    context_text, patient_text = row["input"].split("</context>\n", 1)
    self.assertIn("<context>\nThe current stage is: id_distortions.", row["input"])
    self.assertIn("Completed cognitive model sets: Cognitive model set 1.", row["input"])
    self.assertIn("Recent dialogue:", row["input"])
    self.assertIn("Therapist (utterance 5, id_distortions/cognitive_review_active_question): What went through your mind?", context_text)
    self.assertNotIn("Current patient text.", context_text)
    self.assertEqual(patient_text, "Current patient text.")

    self.assertIn("<think>[resistance_prediction] Patient is open.", row["output"])
    self.assertIn("[opening_info_update] I updated the opening context from the patient response", row["output"])
    self.assertIn("The highest-probability core belief is helplessness", row["output"])
    self.assertIn("The highest-probability distortion is fortune_telling", row["output"])
    self.assertIn("[technique_selection] I selected CBT techniques for the session", row["output"])
    self.assertIn("[address_tib_exit] I completed the TIB detour", row["output"])
    self.assertIn("[technique_auto_complete] I marked the current CBT technique complete automatically", row["output"])
    self.assertIn("[unknown_json] I produced this structured output", row["output"])
    self.assertIn('"foo": "bar"', row["output"])
    self.assertIn("[unknown_string] I produced this result: Done.", row["output"])
    self.assertTrue(row["output"].endswith("</think>Therapist response."))

  def test_missing_patient_profile_fails_fast(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      input_dir.mkdir()
      output_file = base / "out.jsonl"
      (input_dir / "session.json").write_text(json.dumps({"turns": []}), encoding="utf-8")

      with self.assertRaises(builder.DatasetBuildError):
        builder.build_dataset(input_dir, output_file, root)

      self.assertFalse(output_file.exists())


if __name__ == "__main__":
  unittest.main()
