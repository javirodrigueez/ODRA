import json
import tempfile
from pathlib import Path
import unittest

from evaluation.steering import steering_v2


def _write_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f)


def _patient_trace(utterance_index: int, payload: dict, step_input: str = "") -> dict:
  return {
    "utterance_index": utterance_index,
    "speaker": "patient",
    "traces": [
      {
        "step": "resistance_update",
        "input": step_input,
        "output": json.dumps(payload),
      }
    ],
  }


def _conversation_with_resistances(values: list[float]) -> dict:
  messages = []
  traces = []
  for index, value in enumerate(values):
    therapist_index = index * 2 + 1
    patient_index = index * 2 + 2
    messages.append({"role": "therapist", "content": f"Question {index + 1}?"})
    messages.append({"role": "patient", "content": f"Answer {index + 1}."})
    traces.append(_patient_trace(patient_index, {"new_r": value, "reasoning": f"r={value}"}))
  return {
    "messages": messages,
    "patient_profile": {
      "personal_info": {"name": "Alex"},
      "presenting_problem": "Work stress.",
      "reason_seeking_counseling": "Needs help.",
      "patterns": [],
      "attitude": "neutral",
      "more_information": {},
    },
    "reasoning_traces": traces,
  }


def _conversation_without_traces(turn_count: int, attitude: str = "neutral") -> dict:
  messages = []
  for index in range(turn_count):
    messages.append({"role": "therapist", "content": f"Question {index + 1}?"})
    messages.append({"role": "patient", "content": f"Answer {index + 1}."})
  return {
    "messages": messages,
    "patient_profile": {
      "personal_info": {"name": "Alex"},
      "presenting_problem": "Work stress.",
      "reason_seeking_counseling": "Needs help.",
      "patterns": [],
      "attitude": attitude,
      "more_information": {},
    },
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


class SteeringV2Test(unittest.TestCase):
  def test_loads_concepts_from_knowledge_file(self) -> None:
    concepts = steering_v2.load_evaluation_concepts()

    self.assertEqual(set(concepts), {"positive", "neutral", "negative"})
    self.assertTrue(all(concepts[class_name] for class_name in concepts))

  def test_extract_turns_normalizes_message_indices_and_supports_r_change(self) -> None:
    payload = {
      "messages": [
        {"role": "therapist", "content": "How are you?"},
        {"role": "patient", "content": "I am not sure."},
      ],
      "reasoning_traces": [
        _patient_trace(
          2,
          {"r_change": 0.10, "reasoning": "more hesitant"},
          "Given the conversation and current resistance 0.55 to engage in treatment.",
        )
      ],
    }

    turns, skipped = steering_v2.extract_evaluable_turns(payload)

    self.assertEqual(skipped, [])
    self.assertEqual(len(turns), 1)
    self.assertEqual(turns[0].patient_utterance_index, 2)
    self.assertAlmostEqual(turns[0].expected_resistance_0_1, 0.65)
    self.assertEqual(turns[0].expected_resistance_interval, "neutral")
    self.assertEqual(turns[0].resistance_source, "resistance_update.r_change+input_current")

  def test_resistance_interval_boundaries(self) -> None:
    self.assertEqual(steering_v2.resistance_interval(0.33), "positive")
    self.assertEqual(steering_v2.resistance_interval(0.33001), "neutral")
    self.assertEqual(steering_v2.resistance_interval(0.66), "negative")

  def test_fragmenting_splits_on_interval_changes_and_chunks_runs(self) -> None:
    turns, skipped = steering_v2.extract_evaluable_turns(
      _conversation_with_resistances([0.10, 0.20, 0.50, 0.52, 0.80])
    )
    self.assertEqual(skipped, [])

    fragments = steering_v2.build_conversation_fragments("session.json", turns, n_per_batch=2)

    self.assertEqual([fragment.expected_resistance_interval for fragment in fragments], ["positive", "neutral", "negative"])
    self.assertEqual([len(fragment.turns) for fragment in fragments], [2, 2, 1])
    self.assertIn("Therapist: Question 1?", fragments[0].conversation_fragment)
    self.assertIn("Patient: Answer 2.", fragments[0].conversation_fragment)

  def test_extract_turns_no_intervals_ignores_missing_traces(self) -> None:
    turns, skipped = steering_v2.extract_evaluable_turns(
      _conversation_without_traces(2, attitude="positive"),
      no_intervals=True,
    )

    self.assertEqual(skipped, [])
    self.assertEqual(len(turns), 2)
    self.assertIsNone(turns[0].expected_resistance_0_1)
    self.assertIsNone(turns[0].expected_resistance_interval)
    self.assertIsNone(turns[0].resistance_source)
    self.assertEqual(turns[0].resistance_update_reasoning, "")

  def test_fragmenting_no_intervals_chunks_strictly_by_batch_size(self) -> None:
    payload = _conversation_without_traces(5, attitude="negative")
    turns, skipped = steering_v2.extract_evaluable_turns(payload, no_intervals=True)
    self.assertEqual(skipped, [])

    fragments = steering_v2.build_conversation_fragments(
      "session.json",
      turns,
      n_per_batch=2,
      no_intervals=True,
      expected_class=steering_v2.patient_profile_attitude(payload),
    )

    self.assertEqual([fragment.expected_resistance_interval for fragment in fragments], ["negative", "negative", "negative"])
    self.assertEqual([len(fragment.turns) for fragment in fragments], [2, 2, 1])

  def test_build_requests_counts_all_and_matched_concepts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_file = base / "input" / "session.json"
      output_dir = base / "out"
      _write_json(input_file, _conversation_with_resistances([0.20, 0.25]))

      concepts = {
        "positive": ["p1", "p2", "p3"],
        "neutral": ["n1", "n2", "n3"],
        "negative": ["x1", "x2", "x3"],
      }
      all_requests, all_sessions, _, _ = steering_v2.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        2,
        matched_concepts=False,
        concepts=concepts,
      )
      matched_requests, matched_sessions, _, _ = steering_v2.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        2,
        matched_concepts=True,
        concepts=concepts,
      )
      unified_requests, unified_sessions, _, unified_fragmenting = steering_v2.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        2,
        matched_concepts=False,
        unified_concepts=True,
        concepts=concepts,
      )
      matched_unified_requests, matched_unified_sessions, _, _ = steering_v2.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        2,
        matched_concepts=True,
        unified_concepts=True,
        concepts=concepts,
      )

    self.assertEqual(len(all_sessions[0]["evaluated_fragments"]), 1)
    self.assertEqual(len(matched_sessions[0]["evaluated_fragments"]), 1)
    self.assertEqual(len(unified_sessions[0]["evaluated_fragments"]), 1)
    self.assertEqual(len(matched_unified_sessions[0]["evaluated_fragments"]), 1)
    self.assertEqual(len(all_requests), 11)
    self.assertEqual(len(matched_requests), 5)
    self.assertEqual(len(unified_requests), 5)
    self.assertEqual(len(matched_unified_requests), 3)
    self.assertTrue(unified_fragmenting["unified_concepts"])
    positive_items = unified_sessions[0]["evaluated_fragments"][0]["batch_custom_ids"]["resistance_concepts"]["positive"]
    self.assertEqual(len(positive_items), 1)
    self.assertEqual(positive_items[0]["concept"], "p1. p2. p3")
    self.assertEqual(positive_items[0]["concept_count"], 3)
    self.assertTrue(positive_items[0]["unified_concepts"])

  def test_build_requests_no_intervals_uses_profile_attitude_and_ignores_matched(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_file = base / "input" / "session.json"
      output_dir = base / "out"
      _write_json(input_file, _conversation_without_traces(2, attitude="positive"))

      concepts = {
        "positive": ["p1", "p2", "p3"],
        "neutral": ["n1", "n2", "n3"],
        "negative": ["x1", "x2", "x3"],
      }
      requests, sessions, _, fragmenting = steering_v2.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        2,
        no_intervals=True,
        matched_concepts=True,
        concepts=concepts,
      )

    self.assertEqual(len(requests), 11)
    self.assertFalse(sessions[0]["matched_concepts"])
    self.assertTrue(sessions[0]["no_intervals"])
    self.assertTrue(fragmenting["no_intervals"])
    self.assertFalse(sessions[0]["fragmenting"]["split_on_resistance_interval_changes"])
    fragment = sessions[0]["evaluated_fragments"][0]
    self.assertEqual(fragment["expected_resistance_interval"], "positive")
    self.assertIsNone(fragment["mean_expected_resistance_0_1"])

  def test_build_requests_cross_validate_regular_interval_runs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_file = base / "input" / "session.json"
      output_dir = base / "out"
      _write_json(input_file, _conversation_with_resistances([0.20, 0.25]))

      requests, sessions, _, fragmenting = steering_v2.build_requests(
        [input_file],
        output_dir,
        "gpt-4o",
        1,
        2,
        cross_validate=True,
      )

    self.assertGreater(len(requests), 0)
    self.assertTrue(sessions[0]["cross_validate"])
    self.assertTrue(fragmenting["cross_validate"])
    self.assertEqual(sessions[0]["evaluated_fragments"][0]["expected_resistance_interval"], "positive")

  def test_split_request_shards_honors_request_limit(self) -> None:
    requests = [
      steering_v2.BatchRequest(
        custom_id=f"cid-{index}",
        metric="context_relevance",
        input_file="session.json",
        output_file="sessions/session.json",
        fragment_key=f"session.json#fragment-{index:04d}",
        fragment_index=index,
        min_score=0.0,
        max_score=2.0,
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}], "n": 1},
      )
      for index in range(5)
    ]

    shards = steering_v2.split_request_shards(requests, max_requests=2, max_bytes=10_000)

    self.assertEqual([len(shard) for shard in shards], [2, 2, 1])

  def test_split_request_shards_honors_byte_limit(self) -> None:
    requests = [
      steering_v2.BatchRequest(
        custom_id=f"cid-{index}",
        metric="context_relevance",
        input_file="session.json",
        output_file="sessions/session.json",
        fragment_key=f"session.json#fragment-{index:04d}",
        fragment_index=index,
        min_score=0.0,
        max_score=2.0,
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 100}], "n": 1},
      )
      for index in range(3)
    ]
    first_line_bytes = len(steering_v2.jsonl_line_text(requests[0]).encode("utf-8")) + 1

    shards = steering_v2.split_request_shards(
      requests,
      max_requests=50,
      max_bytes=first_line_bytes + 5,
    )

    self.assertEqual([len(shard) for shard in shards], [1, 1, 1])

  def test_score_fragment_aggregates_concepts_and_predicts_class(self) -> None:
    fragment = {
      "fragment_index": 1,
      "fragment_key": "session.json#fragment-0001",
      "turn_count": 1,
      "expected_resistance_interval": "neutral",
      "batch_custom_ids": {
        "resistance_concepts": {
          "positive": [{"concept_index": 0, "concept": "p", "custom_id": "p"}],
          "neutral": [{"concept_index": 0, "concept": "n", "custom_id": "n"}],
          "negative": [{"concept_index": 0, "concept": "x", "custom_id": "x"}],
        },
        "context_relevance": "context",
        "patient_like_realism": "realism",
      },
    }
    parsed = {
      "p": {"score": 0.5, "cost": 0.1, "choices": []},
      "n": {"score": 2.0, "cost": 0.1, "choices": []},
      "x": {"score": 1.0, "cost": 0.1, "choices": []},
      "context": {"score": 1.0, "cost": 0.1, "choices": []},
      "realism": {"score": 2.0, "cost": 0.1, "choices": []},
    }

    scored = steering_v2.score_fragment(fragment, parsed)

    self.assertEqual(scored["resistance_class_scores"]["neutral"], 2.0)
    self.assertEqual(scored["expected_resistance_concept_score"], 2.0)
    self.assertEqual(scored["predicted_resistance_interval"], "neutral")
    self.assertTrue(scored["resistance_interval_match"])
    self.assertFalse(scored["resistance_interval_tie"])
    self.assertAlmostEqual(scored["cost"], 0.5)

  def test_score_fragment_cross_validate_zeroes_for_stronger_competing_concept(self) -> None:
    fragment = {
      "fragment_index": 1,
      "fragment_key": "session.json#fragment-0001",
      "turn_count": 1,
      "expected_resistance_interval": "neutral",
      "batch_custom_ids": {
        "resistance_concepts": {
          "positive": [{"concept_index": 0, "concept": "p", "custom_id": "p"}],
          "neutral": [
            {"concept_index": 0, "concept": "n1", "custom_id": "n1"},
            {"concept_index": 1, "concept": "n2", "custom_id": "n2"},
          ],
          "negative": [{"concept_index": 0, "concept": "x", "custom_id": "x"}],
        },
        "context_relevance": "context",
        "patient_like_realism": "realism",
      },
    }
    parsed = {
      "p": {"score": 1.5, "cost": 0.1, "choices": []},
      "n1": {"score": 1.0, "cost": 0.1, "choices": []},
      "n2": {"score": 1.0, "cost": 0.1, "choices": []},
      "x": {"score": 0.5, "cost": 0.1, "choices": []},
      "context": {"score": 1.0, "cost": 0.1, "choices": []},
      "realism": {"score": 2.0, "cost": 0.1, "choices": []},
    }

    scored = steering_v2.score_fragment(fragment, parsed, cross_validate=True)

    self.assertEqual(scored["unpenalized_expected_resistance_concept_score"], 1.0)
    self.assertTrue(scored["cross_validation_penalized"])
    self.assertEqual(scored["expected_resistance_concept_score"], 0.0)
    self.assertEqual(scored["harmonic_mean"], 0.0)

  def test_score_fragment_cross_validate_ignored_for_matched_concepts(self) -> None:
    fragment = {
      "fragment_index": 1,
      "fragment_key": "session.json#fragment-0001",
      "turn_count": 1,
      "expected_resistance_interval": "neutral",
      "batch_custom_ids": {
        "resistance_concepts": {
          "positive": [],
          "neutral": [{"concept_index": 0, "concept": "n", "custom_id": "n"}],
          "negative": [],
        },
        "context_relevance": "context",
        "patient_like_realism": "realism",
      },
    }
    parsed = {
      "n": {"score": 1.25, "cost": 0.1, "choices": []},
      "context": {"score": 1.0, "cost": 0.1, "choices": []},
      "realism": {"score": 2.0, "cost": 0.1, "choices": []},
    }

    scored = steering_v2.score_fragment(fragment, parsed, matched_concepts=True, cross_validate=True)

    self.assertEqual(scored["expected_resistance_concept_score"], 1.25)
    self.assertNotIn("cross_validation_penalized", scored)

  def test_parse_all_shard_outputs_merges_results(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      manifest = {"shards": []}
      for shard_index, custom_id in enumerate(["cid-a", "cid-b"], start=1):
        shard_dir = base / "shards" / f"shard_{shard_index:05d}"
        shard_dir.mkdir(parents=True)
        request_map = {
          custom_id: {
            "custom_id": custom_id,
            "metric": "context_relevance",
            "min_score": 0.0,
            "max_score": 2.0,
          }
        }
        output_jsonl = shard_dir / "output.jsonl"
        _write_json(shard_dir / "request_map.json", request_map)
        with output_jsonl.open("w", encoding="utf-8") as f:
          f.write(json.dumps(_batch_output_line(custom_id)) + "\n")
        shard_manifest = {
          "version": 2,
          "shard_index": shard_index,
          "request_count": 1,
          "request_map": str(shard_dir / "request_map.json"),
          "output_jsonl": str(output_jsonl),
        }
        _write_json(shard_dir / "manifest.json", shard_manifest)
        manifest["shards"].append({"manifest": str(shard_dir / "manifest.json")})

      parsed, report = steering_v2.parse_all_shard_outputs(
        manifest,
        max_iter=1,
        input_cost_per_1m=1.0,
        output_cost_per_1m=1.0,
      )

    self.assertEqual(set(parsed), {"cid-a", "cid-b"})
    self.assertEqual(report["request_count"], 2)
    self.assertEqual(report["response_count"], 2)
    self.assertEqual(len(report["shards"]), 2)


if __name__ == "__main__":
  unittest.main()
