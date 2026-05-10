import json
import tempfile
from pathlib import Path
import unittest

from evaluation.rtraces import rtraces


def _write_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f)


def _trace_step(step: str, output: object) -> dict:
  raw_output = output if isinstance(output, str) else json.dumps(output)
  return {
    "step": step,
    "input": f"instruction for {step}",
    "output": raw_output,
  }


def _batch_output_line(custom_id: str, rating: int | float = 2) -> dict:
  return {
    "custom_id": custom_id,
    "response": {
      "body": {
        "choices": [
          {
            "message": {
              "role": "assistant",
              "content": f"Short explanation.\n\nRating: [[{rating}]]",
            }
          }
        ],
        "usage": {
          "prompt_tokens": 10,
          "completion_tokens": 5,
        },
      }
    },
    "error": None,
  }


def _sample_payload() -> dict:
  return {
    "messages": [
      {"utterance_index": 1, "role": "therapist", "content": "T1", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 2, "role": "patient", "content": "P1", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 3, "role": "therapist", "content": "T2", "stage": "id_distortions", "substage": "cognitive_review_active_question"},
      {"utterance_index": 4, "role": "patient", "content": "P2", "stage": "id_distortions", "substage": "cognitive_review_active_question"},
      {"utterance_index": 5, "role": "therapist", "content": "T3", "stage": "cbt_technique", "substage": "cbt_decide_next"},
      {"utterance_index": 6, "role": "patient", "content": "P3", "stage": "cbt_technique", "substage": "cbt_decide_next"},
      {"utterance_index": 7, "role": "therapist", "content": "Let us choose the next step.", "stage": "cbt_technique", "substage": "cbt_decide_next"},
    ],
    "reasoning_traces": [
      {
        "utterance_index": 7,
        "speaker": "therapist",
        "traces": [
          _trace_step(
            "technique_selection",
            {"selected_techniques": [{"name": "thought record", "target_distortion": "mind_reading"}]},
          ),
          _trace_step(
            "additional_technique_selection",
            {"name": "behavioral experiment", "target_distortion": "fortune_telling", "rationale": "Needs practice."},
          ),
          _trace_step(
            "apply_more_techniques_check",
            {"apply_more": True, "reasoning": "More work is useful."},
          ),
          _trace_step(
            "address_tib_exit",
            {
              "tib_name": "urgent_solutions",
              "exit_reason": "single_turn_complete",
              "completion_note": "Addressed briefly.",
              "predicted_resistance": 0.4,
              "resume_stage": "cbt_technique",
              "routed_stage": "cbt_technique",
            },
          ),
          _trace_step("unknown_step", {"message": "ignored"}),
        ],
      }
    ],
  }


class RTracesTest(unittest.TestCase):
  def test_load_trace_knowledge_contains_added_steps_and_expert_exclusions(self) -> None:
    knowledge = rtraces.load_trace_knowledge()

    self.assertIn("cognitive_gap_best_guess", knowledge.stage_steps["id_distortions"])
    self.assertIn("additional_technique_selection", knowledge.stage_steps["cbt_technique"])
    self.assertIn("apply_more_techniques_check", knowledge.stage_steps["cbt_technique"])
    self.assertIn("address_tib_exit", knowledge.global_steps)
    self.assertIn("anticipated_closing", knowledge.global_steps)
    self.assertIn("anticipated_closing", knowledge.excluded_for_experts)
    self.assertEqual(knowledge.min_traces_per_sample, 3)

  def test_extract_evaluable_turns_uses_stage_and_global_steps_and_context_turns(self) -> None:
    knowledge = rtraces.load_trace_knowledge()

    turns, skipped = rtraces.extract_evaluable_turns(_sample_payload(), "session.json", 2, knowledge)

    self.assertEqual(skipped, [])
    self.assertEqual(len(turns), 1)
    turn = turns[0]
    self.assertEqual(turn.context_utterance_count, 4)
    self.assertNotIn("Therapist #1", turn.context)
    self.assertIn("Therapist #3", turn.context)
    self.assertIn("Patient #6", turn.context)
    self.assertEqual(
      [trace["step"] for trace in turn.traces],
      [
        "technique_selection",
        "additional_technique_selection",
        "apply_more_techniques_check",
        "address_tib_exit",
      ],
    )
    self.assertIn("=== Trace 4 ===", turn.reasoning_traces_txt)
    self.assertIn("Exit Reason: single_turn_complete", turn.reasoning_traces_txt)

  def test_extract_evaluable_turns_skips_when_below_minimum_included_traces(self) -> None:
    knowledge = rtraces.load_trace_knowledge()
    payload = _sample_payload()
    payload["reasoning_traces"][0]["traces"] = [
      _trace_step("additional_technique_selection", {"name": "x", "target_distortion": "y", "rationale": "z"})
    ]

    turns, skipped = rtraces.extract_evaluable_turns(payload, "session.json", 5, knowledge)

    self.assertEqual(turns, [])
    self.assertEqual(skipped[0]["reason"], "insufficient included traces")
    self.assertEqual(skipped[0]["included_trace_count"], 1)

  def test_build_requests_creates_three_metric_requests_per_sampled_turn(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_file = base / "input" / "session.json"
      output_dir = base / "out"
      _write_json(input_file, _sample_payload())

      requests, sessions, skipped_files, sampling = rtraces.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        1,
        2,
      )

    self.assertEqual(skipped_files, [])
    self.assertEqual(len(requests), 3)
    self.assertEqual({request.metric for request in requests}, set(rtraces.METRIC_SPECS))
    turn = sessions[0]["evaluated_turns"][0]
    self.assertEqual(set(turn["batch_custom_ids"]), set(rtraces.METRIC_SPECS))
    self.assertEqual(sampling["selected_turns"], 1)
    self.assertIn("[Previous Context Start]", requests[0].body["messages"][0]["content"])

  def test_build_requests_from_batch_output_uses_observed_custom_ids_and_infers_missing_metrics(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_file = base / "input" / "session.json"
      output_dir = base / "out"
      output_jsonl = base / "batch_output.jsonl"
      _write_json(input_file, _sample_payload())
      with output_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_batch_output_line("rtraces:00000001:session:u0007:logic_consistency")) + "\n")
        f.write(json.dumps(_batch_output_line("rtraces:00000002:session:u0007:answer_explanation_alignment")) + "\n")

      requests, sessions, skipped_files, sampling, selection = rtraces.build_requests_from_batch_output(
        [input_file],
        output_dir,
        output_jsonl,
        "gpt-4o",
        1,
        2,
      )

    self.assertEqual(skipped_files, [])
    self.assertEqual(len(requests), 3)
    self.assertEqual(sampling["selection_source"], "batch_output_jsonl")
    self.assertEqual(sampling["selected_turns"], 1)
    self.assertEqual(selection["inferred_missing_custom_ids"], ["rtraces:00000000:session:u0007:faithfulness"])
    turn = sessions[0]["evaluated_turns"][0]
    self.assertEqual(
      turn["batch_custom_ids"],
      {
        "faithfulness": "rtraces:00000000:session:u0007:faithfulness",
        "logic_consistency": "rtraces:00000001:session:u0007:logic_consistency",
        "answer_explanation_alignment": "rtraces:00000002:session:u0007:answer_explanation_alignment",
      },
    )

  def test_score_turn_adds_metric_scores_and_harmonic_composite(self) -> None:
    turn = {
      "batch_custom_ids": {
        "faithfulness": "f",
        "logic_consistency": "l",
        "answer_explanation_alignment": "a",
      }
    }
    parsed = {
      "f": {"score": 2.0, "cost": 0.1, "choices": []},
      "l": {"score": 1.0, "cost": 0.2, "choices": []},
      "a": {"score": 2.0, "cost": 0.3, "choices": []},
    }

    scored = rtraces.score_turn(turn, parsed)

    self.assertEqual(scored["faithfulness"], 2.0)
    self.assertEqual(scored["logic_consistency"], 1.0)
    self.assertEqual(scored["answer_explanation_alignment"], 2.0)
    self.assertAlmostEqual(scored["harmonic_mean"], 1.5)
    self.assertAlmostEqual(scored["cost"], 0.6)

  def test_parse_batch_outputs_parses_rating_lines(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      output_jsonl = base / "output.jsonl"
      custom_id = "rtraces:00000001:session:u0007:faithfulness"
      with output_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_batch_output_line(custom_id, 1.5)) + "\n")

      parsed, report = rtraces.parse_batch_outputs(
        output_jsonl,
        {
          custom_id: {
            "custom_id": custom_id,
            "metric": "faithfulness",
            "min_score": 0.0,
            "max_score": 2.0,
          }
        },
        max_iter=1,
        input_cost_per_1m=1.0,
        output_cost_per_1m=1.0,
      )

    self.assertEqual(report["parse_errors"], [])
    self.assertEqual(parsed[custom_id]["score"], 1.5)
    self.assertGreater(parsed[custom_id]["cost"], 0.0)


if __name__ == "__main__":
  unittest.main()
