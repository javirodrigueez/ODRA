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

  def test_compact_prompt_omits_last_utterances_block(self) -> None:
    prompt = replay_dataset.compact_user_prompt(
      full_conversation="Patient: hello",
      last_reasoning="previous reasoning",
      session_context='{"current_stage": "opening"}',
      guidelines="be concise",
      instruction="Do the step.",
      suffix=replay_dataset.dataset_suffix(
        "Your output should use the most recent exchanges <last_utterances>, and the instruction."
      ),
    )

    self.assertIn("<full_conversation>", prompt)
    self.assertIn("<last_reasoning>", prompt)
    self.assertIn("<session_context>", prompt)
    self.assertNotIn("<last_utterances>", prompt)

  def test_recent_conversation_uses_configured_window(self) -> None:
    messages = [
      {"role": "therapist" if index % 2 else "patient", "content": f"message {index}"}
      for index in range(1, 13)
    ]

    rendered = replay_dataset.format_recent_conversation(messages, 10)

    lines = rendered.splitlines()
    self.assertFalse(any(line.endswith("message 1") for line in lines))
    self.assertFalse(any(line.endswith("message 2") for line in lines))
    self.assertTrue(any(line.endswith("message 3") for line in lines))
    self.assertTrue(any(line.endswith("message 12") for line in lines))
    self.assertEqual(len(lines), 10)

  def test_compact_system_prompt_excludes_long_knowledge_blocks(self) -> None:
    system_prompt = replay_dataset.render_compact_system_prompt(
      replay_dataset.COMPACT_THERAPIST_SYSTEM_TEMPLATE,
      {
        "personal_info": {"name": "Laura"},
        "reason_seeking_counseling": "Running stress.",
        "more_information": {},
      },
    )

    self.assertIn("Name: Laura", system_prompt)
    self.assertNotIn("<cbt_knowledge>", system_prompt)
    self.assertLess(len(system_prompt), 1000)

  def test_fallback_policy_excludes_fallback_rows_by_default(self) -> None:
    collector = replay_dataset.SampleCollector("chatml")
    messages = [{"role": "user", "content": "prompt"}]

    collector.add(messages, "target", {"sample_kind": "utterance", "replay_status": "fallback_unemitted_utterance"})
    collector.add(messages, "target", {"sample_kind": "utterance"})

    self.assertEqual(len(collector.samples), 1)
    self.assertNotIn("replay_status", collector.samples[0]["metadata"])

  def test_fallback_policy_include_keeps_fallback_rows(self) -> None:
    collector = replay_dataset.SampleCollector("chatml", fallback_policy="include")

    collector.add(
      [{"role": "user", "content": "prompt"}],
      "target",
      {"sample_kind": "utterance", "replay_status": "fallback_unemitted_utterance"},
    )

    self.assertEqual(len(collector.samples), 1)
    self.assertEqual(collector.samples[0]["metadata"]["replay_status"], "fallback_unemitted_utterance")


if __name__ == "__main__":
  unittest.main()
