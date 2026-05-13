import json
import unittest

from scripts.ft import create_multiturn_replay_dataset as replay_dataset


class ReplayDatasetFormatTest(unittest.TestCase):
  def test_chatml_sample_appends_assistant_target(self) -> None:
    collector = replay_dataset.SampleCollector("chatml")
    collector.add(
      [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
      ],
      "assistant target",
      {"sample_kind": "utterance"},
    )

    self.assertEqual(len(collector.samples), 1)
    sample = collector.samples[0]
    self.assertEqual([message["role"] for message in sample["messages"]], ["system", "user", "assistant"])
    self.assertEqual(sample["messages"][-1]["content"], "assistant target")
    self.assertEqual(sample["metadata"], {"sample_kind": "utterance"})

  def test_sharegpt_sample_uses_from_value_schema(self) -> None:
    sample = replay_dataset.make_sample(
      [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
        {"role": "assistant", "content": "assistant target"},
      ],
      {"sample_kind": "reasoning_trace"},
      "sharegpt",
    )

    self.assertEqual(
      sample["conversations"],
      [
        {"from": "system", "value": "system prompt"},
        {"from": "human", "value": "user prompt"},
        {"from": "gpt", "value": "assistant target"},
      ],
    )
    self.assertEqual(sample["metadata"], {"sample_kind": "reasoning_trace"})

  def test_request_trace_requires_parseable_json_output(self) -> None:
    self.assertTrue(
      replay_dataset.is_request_trace(
        {"input": "instruction", "output": json.dumps({"reasoning": "ok"})}
      )
    )
    self.assertFalse(
      replay_dataset.is_request_trace(
        {"input": "synthetic event", "output": "Shelved the current topic."}
      )
    )
    self.assertFalse(
      replay_dataset.is_request_trace(
        {"input": "", "output": json.dumps({"reasoning": "ok"})}
      )
    )

  def test_prompts_text_structured_suffix_requests_plain_text(self) -> None:
    suffix = replay_dataset.load_prompt_from_root(
      replay_dataset.PROMPTS_TEXT_DIR,
      "shared/structured_suffix.txt",
    )

    self.assertIn("Respond in plain text only", suffix)
    self.assertNotIn("Respond with valid JSON only", suffix)


if __name__ == "__main__":
  unittest.main()
