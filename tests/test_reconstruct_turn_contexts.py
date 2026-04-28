import json
from pathlib import Path
import tempfile
import unittest

from scripts.data import reconstruct_turn_contexts


def _trace_step(step: str, output: object) -> dict:
  return {
    "step": step,
    "input": f"instruction for {step}",
    "output": output if isinstance(output, str) else json.dumps(output),
  }


def _therapist(idx: int, content: str, stage: str, substage: str) -> dict:
  return {
    "utterance_index": idx,
    "role": "therapist",
    "content": content,
    "stage": stage,
    "substage": substage,
  }


def _patient(idx: int, content: str, stage: str, substage: str) -> dict:
  return {
    "utterance_index": idx,
    "role": "patient",
    "content": content,
    "stage": stage,
    "substage": substage,
  }


def _trace_record(idx: int, traces: list[dict]) -> dict:
  return {
    "utterance_index": idx,
    "speaker": "therapist",
    "traces": traces,
  }


def _probabilities(keys: list[str], best: str) -> dict:
  return {key: 1.0 if key == best else 0.0 for key in keys}


def _sample_session() -> dict:
  return {
    "messages": [
      _therapist(1, "Hi.", "opening", "open_mood"),
      _patient(2, "I feel stuck.", "opening", "open_mood"),
      _therapist(3, "Tell me more about the feeling.", "id_distortions", "cognitive_review_active_question"),
      _patient(4, "It happens at dinner.", "id_distortions", "cognitive_review_active_question"),
      _therapist(5, "Let's look at that thought.", "cbt_technique", "cbt_review_current"),
      _patient(6, "The thought is strong.", "cbt_technique", "cbt_review_current"),
      _therapist(7, "Try this as homework.", "homework", "homework_propose_tasks"),
      _patient(8, "That sounds doable.", "homework", "homework_propose_tasks"),
      _therapist(9, "Good, let's close.", "closing", "closing_share_summary"),
      _patient(10, "Okay.", "closing", "closing_share_summary"),
      _therapist(11, "See you next time.", "closing", "closing_finalize_feedback"),
    ],
    "reasoning_traces": [
      _trace_record(1, []),
      _trace_record(
        3,
        [
          _trace_step("resistance_prediction", {"new_p": 0.4, "reasoning": "Patient is somewhat open."}),
          _trace_step("opening_info_update", {"mood": "stuck", "problem": "dinner routine"}),
          _trace_step("opening_readiness_check", {"ready": True, "reasoning": "Ready for CBT."}),
          _trace_step(
            "initial_cognitive_model_bootstrap",
            {
              "items": {
                "situation": "Dinner at home",
                "auto_thoughts": ["I am stuck"],
                "emotions": ["flat"],
                "behaviors": ["eat at the counter"],
              },
              "reasoning": "Opening material is enough.",
            },
          ),
          _trace_step(
            "cognitive_refinement_planning",
            {"target": "emotions", "plan": "Clarify the emotional intensity."},
          ),
        ],
      ),
      _trace_record(
        5,
        [
          _trace_step(
            "core_belief_probability",
            _probabilities(reconstruct_turn_contexts.CORE_BELIEF_KEYS, "helplessness"),
          ),
          _trace_step(
            "distortion_probability",
            _probabilities(reconstruct_turn_contexts.DISTORTION_KEYS, "all_or_nothing_thinking"),
          ),
          _trace_step(
            "technique_selection",
            {
              "selected_techniques": [
                {
                  "name": "Cognitive Restructuring",
                  "target_distortion": "all_or_nothing_thinking",
                  "rationale": "Targets rigid thinking.",
                }
              ]
            },
          ),
          _trace_step(
            "technique_plan",
            {
              "plan_description": "Examine the thought.",
              "expected_outcome": "A more flexible thought.",
              "current_progress": "Technique has not started.",
            },
          ),
        ],
      ),
      _trace_record(
        7,
        [
          _trace_step("technique_completion_check", {"completed": True, "reasoning": "Goal met."}),
          _trace_step("technique_evaluation", "The patient generated a flexible thought."),
          _trace_step(
            "homework_task_proposal",
            {
              "tasks": [
                {
                  "task_title": "Dinner Observation",
                  "task_description": "Observe dinner once.",
                  "rationale": "Practice noticing.",
                  "expected_outcomes": "More distance from the thought.",
                }
              ],
              "reasoning": "Task fits the work.",
            },
          ),
        ],
      ),
      _trace_record(
        9,
        [
          _trace_step(
            "homework_agreement_check",
            {
              "agreement_reached": True,
              "agreed_tasks": [
                {
                  "task_title": "Dinner Observation",
                  "task_description": "Observe dinner twice.",
                  "rationale": "Practice noticing.",
                  "expected_outcomes": "More distance from the thought.",
                },
                {
                  "task_title": "One note",
                  "task_description": "Write one sentence afterward.",
                  "rationale": "Capture data.",
                  "expected_outcomes": "A concrete record.",
                },
              ],
              "reasoning": "Patient agreed.",
            },
          )
        ],
      ),
      _trace_record(11, []),
    ],
  }


class ReconstructTurnContextsTest(unittest.TestCase):
  def test_reconstructs_context_before_each_therapist_turn(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "session.json"
      with path.open("w", encoding="utf-8") as f:
        json.dump(_sample_session(), f)

      reconstructed = reconstruct_turn_contexts.reconstruct_session(path)

    self.assertEqual(reconstructed["warnings"], [])
    self.assertEqual(len(reconstructed["turns"]), 6)

    first_context = reconstructed["turns"][0]["context_before_therapist_turn"]
    self.assertEqual(first_context["current_stage"], "opening")
    self.assertEqual(first_context["current_substage"], "open_initial")

    fifth_turn_context = reconstructed["turns"][2]["context_before_therapist_turn"]
    self.assertEqual(fifth_turn_context["opening_patient_mood"], "stuck")
    self.assertEqual(fifth_turn_context["opening_patient_problem"], "dinner routine")
    self.assertEqual(fifth_turn_context["predicted_resistance"], 0.4)
    self.assertEqual(fifth_turn_context["working_cognitive_set"]["situation"], "Dinner at home")
    self.assertEqual(fifth_turn_context["active_cognitive_question"], "Tell me more about the feeling.")

    homework_context = reconstructed["turns"][3]["context_before_therapist_turn"]
    self.assertEqual(homework_context["current_technique"]["name"], "Cognitive Restructuring")
    self.assertEqual(homework_context["current_technique_plan"], "Examine the thought.")
    self.assertEqual(len(homework_context["cognitive_model_sets"]), 1)

    final_context = reconstructed["turns"][-1]["context_before_therapist_turn"]
    self.assertEqual(
      [task["task_description"] for task in final_context["tasks"]],
      ["Observe dinner twice.", "Write one sentence afterward."],
    )
    self.assertNotIn("session_summary", final_context)
    self.assertNotIn("next_session_plan", final_context)
    self.assertEqual(len(reconstructed["turns"][-1]["last_utterances"]), 10)

  def test_malformed_trace_skips_turn_with_warning(self) -> None:
    payload = _sample_session()
    payload["reasoning_traces"][2]["traces"][-1] = _trace_step(
      "technique_plan",
      {"expected_outcome": "missing plan"},
    )

    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "session.json"
      with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

      reconstructed = reconstruct_turn_contexts.reconstruct_session(path)

    self.assertIn(5, reconstructed["skipped_turns"])
    self.assertTrue(any("technique_plan" in warning for warning in reconstructed["warnings"]))


if __name__ == "__main__":
  unittest.main()
