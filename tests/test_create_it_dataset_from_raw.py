import json
from pathlib import Path
import tempfile
import unittest

from scripts.ft import create_it_dataset_from_raw as builder


def _message(idx: int, role: str, content: str, stage: str = "id_distortions") -> dict:
  return {
    "utterance_index": idx,
    "role": role,
    "content": content,
    "stage": stage,
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
  (agent_prompts / "therapist_system_ft.txt").write_text(
    "FT SYSTEM PROMPT\n{patient_intake_form}",
    encoding="utf-8",
  )
  (agent_prompts / "therapist_system.txt").write_text(
    "SHOULD NOT BE USED",
    encoding="utf-8",
  )

  knowledge = base / "data" / "knowledge"
  knowledge.mkdir(parents=True)
  (knowledge / "reasoning_trace_evaluation.json").write_text(
    json.dumps(
      {
        "stages": {
          "id_distortions": {
            "steps": {
              "resistance_prediction": {
                "fields": ["previous_resistance", "predicted_resistance", "reasoning"]
              },
              "tib_check": {
                "fields": ["has_tib", "tib_name", "tib_description", "reasoning"]
              },
              "cognitive_gap_response_review": {
                "fields": ["answered_satisfactorily", "identified_items", "reasoning"]
              },
            }
          }
        },
        "global_steps": {
          "address_tib_enter": {
            "fields": ["route_bucket", "tib_name", "resume_stage", "resume_substage", "reasoning"]
          },
          "address_tib_repeat_inline_suppressed": {
            "fields": [
              "tib_name",
              "current_stage",
              "current_substage",
              "remembered_tib_name",
              "remembered_tib_no_reroute_active",
            ]
          },
        },
        "excluded_for_experts": [
          "address_tib_enter",
          "address_tib_repeat_inline_suppressed",
        ],
      }
    ),
    encoding="utf-8",
  )
  return base


def _session_payload() -> dict:
  return {
    "patient_profile": {
      "personal_info": {
        "name": "Jason Crawford",
        "age": "44",
      },
      "reason_seeking_counseling": "Need help with commute stress.",
    },
    "messages": [
      _message(1, "therapist", "T1 plain opening.", "opening"),
      _message(2, "patient", "P2 with patient trace.", "opening"),
      _message(3, "therapist", "Target therapist response."),
      _message(4, "patient", "P4."),
      _message(5, "therapist", "T5 without traces.", "cbt_technique"),
    ],
    "reasoning_traces": [
      _trace_group(1, "therapist", [_trace("unknown_initial", "Initial therapist trace should be ignored.")]),
      _trace_group(2, "patient", [_trace("resistance_update", {"reasoning": "Patient-only reasoning."})]),
      _trace_group(
        3,
        "therapist",
        [
          _trace(
            "resistance_prediction",
            {
              "previous_p": 0.5,
              "new_p": 0.2,
              "p_target": 0.3,
              "prediction_mode": "debug_mode",
              "delta_clip_applied": False,
              "reasoning": "Patient is engaged.",
            },
          ),
          _trace(
            "tib_check",
            {
              "has_tib": True,
              "tib_name": "skepticism",
              "tib_description": "This should be excluded.",
              "route_bucket": "other",
              "reasoning": "The patient is skeptical.",
            },
          ),
          _trace(
            "cognitive_gap_response_review",
            {
              "answered_satisfactorily": True,
              "identified_items": {
                "situation": "Last week, a friend complimented me on a meal I made.",
                "auto_thoughts": ["She was just being polite because it was actually pretty bland."],
                "emotions": ["disappointment", "frustration"],
                "behaviors": ["The moment was ruined; likely withdrew."],
              },
              "reasoning": "The patient's response directly answers the therapist's question.",
              "extra_debug": "This should be excluded.",
            },
          ),
          _trace(
            "address_tib_enter",
            {
              "route_bucket": "other",
              "tib_name": "skepticism",
              "resume_stage": "id_distortions",
              "resume_substage": "cognitive_refinement",
              "reasoning": "This should become a message.",
            },
          ),
          _trace(
            "address_tib_repeat_inline_suppressed",
            {
              "tib_name": "skepticism",
              "current_stage": "id_distortions",
              "current_substage": "cognitive_refinement",
              "remembered_tib_name": "skepticism",
              "remembered_tib_no_reroute_active": True,
            },
          ),
          _trace("unknown_string", "Done."),
        ],
      ),
    ],
  }


class CreateItDatasetFromRawTest(unittest.TestCase):
  def test_builds_chatml_session_files_from_raw_messages_and_traces(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "simulation.json").write_text(json.dumps(_session_payload()), encoding="utf-8")

      count = builder.build_dataset(input_dir, output_dir, root, include_reasoning=True)
      output_payload = json.loads((output_dir / "simulation.json").read_text(encoding="utf-8"))

    self.assertEqual(count, 1)
    self.assertEqual(set(output_payload.keys()), {"messages"})
    messages = output_payload["messages"]
    self.assertEqual([message["role"] for message in messages], ["system", "assistant", "user", "assistant", "user", "assistant"])
    self.assertTrue(messages[0]["content"].startswith("FT SYSTEM PROMPT"))
    self.assertIn("Name: Jason Crawford", messages[0]["content"])
    self.assertEqual(messages[1]["content"], "T1 plain opening.")
    self.assertNotIn("<think>", messages[1]["content"])
    self.assertEqual(messages[2]["content"], "P2 with patient trace.")
    self.assertEqual(messages[5]["content"], "T5 without traces.")

    traced_content = messages[3]["content"]
    self.assertTrue(traced_content.startswith("<think>\n[stage] Distortions Identification\n"))
    self.assertTrue(traced_content.endswith("</think>\nTarget therapist response."))
    self.assertIn("[resistance_prediction] Previous Resistance: 0.5.", traced_content)
    self.assertIn("Predicted Resistance: 0.2.", traced_content)
    self.assertIn("Reasoning: Patient is engaged.", traced_content)
    self.assertIn("[cognitive_gap_response_review] Answered Satisfactorily: yes.", traced_content)
    self.assertIn("Identified Items Situation: Last week, a friend complimented me on a meal I made.", traced_content)
    self.assertIn(
      "[address_tib_enter] Message: As the therapy-interfering behavior skepticism was detected, I should address it",
      traced_content,
    )
    self.assertIn(
      "[address_tib_repeat_inline_suppressed] Message: I already tried to address the therapy-interfering behavior skepticism in my last utterance. Now, I will address it again to demonstrate validation, while continuing with the main line of the session.",
      traced_content,
    )
    self.assertIn("[unknown_string] Message: Done.", traced_content)

    self.assertNotIn("Patient-only reasoning", traced_content)
    self.assertNotIn("Initial therapist trace should be ignored.", traced_content)
    self.assertNotIn("instruction for", traced_content)
    self.assertNotIn("p_target", traced_content)
    self.assertNotIn("prediction_mode", traced_content)
    self.assertNotIn("delta_clip_applied", traced_content)
    self.assertNotIn("tib_description", traced_content)
    self.assertNotIn("route_bucket", traced_content)
    self.assertNotIn("extra_debug", traced_content)

  def test_reasoning_is_opt_in(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "simulation.json").write_text(json.dumps(_session_payload()), encoding="utf-8")

      builder.build_dataset(input_dir, output_dir, root)
      output_payload = json.loads((output_dir / "simulation.json").read_text(encoding="utf-8"))

    self.assertFalse(any("<think>" in message["content"] for message in output_payload["messages"]))

  def test_json_reasoning_uses_jsonl_trace_objects(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "simulation.json").write_text(json.dumps(_session_payload()), encoding="utf-8")

      builder.build_dataset(
        input_dir,
        output_dir,
        root,
        include_reasoning=True,
        reasoning_format="json",
      )
      output_payload = json.loads((output_dir / "simulation.json").read_text(encoding="utf-8"))

    traced_content = output_payload["messages"][3]["content"]
    think = traced_content.split("<think>\n", 1)[1].split("\n</think>", 1)[0]
    objects = [json.loads(line) for line in think.splitlines()]
    self.assertEqual(objects[0], {"stage": "Distortions Identification"})
    self.assertIn("resistance_prediction", objects[1])
    self.assertEqual(objects[1]["resistance_prediction"]["previous_resistance"], 0.5)
    self.assertIn("unknown_string", objects[-1])
    self.assertEqual(objects[-1]["unknown_string"], {"message": "Done."})

  def test_turn_mode_builds_one_file_per_therapist_target(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "simulation.json").write_text(json.dumps(_session_payload()), encoding="utf-8")

      count = builder.build_dataset(
        input_dir,
        output_dir,
        root,
        include_reasoning=True,
        reasoning_format="json",
        mode="turn",
      )
      first = json.loads((output_dir / "simulation.turn_0001.json").read_text(encoding="utf-8"))
      target = json.loads((output_dir / "simulation.turn_0003.json").read_text(encoding="utf-8"))
      output_names = [path.name for path in sorted(output_dir.iterdir())]

    self.assertEqual(count, 3)
    self.assertEqual(output_names, [
      "simulation.turn_0001.json",
      "simulation.turn_0003.json",
      "simulation.turn_0005.json",
    ])
    self.assertEqual([message["role"] for message in first["messages"]], ["system", "assistant"])
    self.assertEqual([message["role"] for message in target["messages"]], ["system", "user", "user", "assistant"])
    self.assertEqual(target["messages"][1]["content"], "Therapist: T1 plain opening.")
    self.assertEqual(target["messages"][2]["content"], "Patient: P2 with patient trace.")
    self.assertTrue(target["messages"][-1]["content"].startswith("<think>\n"))
    self.assertFalse(target["messages"][-1]["content"].split("</think>\n", 1)[1].startswith("Therapist:"))

  def test_missing_patient_profile_fails_fast(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      root = _project_root(base / "project")
      input_dir = base / "input"
      output_dir = base / "out"
      input_dir.mkdir()
      (input_dir / "simulation.json").write_text(
        json.dumps({"messages": [], "reasoning_traces": []}),
        encoding="utf-8",
      )

      with self.assertRaises(builder.DatasetBuildError):
        builder.build_dataset(input_dir, output_dir, root)

      self.assertFalse(output_dir.exists())


if __name__ == "__main__":
  unittest.main()
