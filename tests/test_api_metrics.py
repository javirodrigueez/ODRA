import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation import run_api_metrics as api_metrics
from evaluation import run_batch_metrics as batch_metrics


def _write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f)


def _sample_conversation(name: str = "Alex", attitude: str = "neutral") -> dict:
  return {
    "messages": [
      {"role": "therapist", "content": "Hi, how are you feeling today?"},
      {"role": "patient", "content": "I feel stuck and worried about work."},
      {"role": "therapist", "content": "What evidence supports and challenges that thought?"},
    ],
    "patient_profile": {
      "idx": 1,
      "attitude": attitude,
      "personal_info": {"name": name, "age": 31},
      "more_information": {
        "past_history": "No prior counseling.",
        "academic_occupational_functioning": "Work stress has increased.",
        "social_support_system": "Has a close friend.",
      },
    },
  }


class _FakeChatCompletions:
  def __init__(self):
    self.calls = []

  def create(self, **body):
    self.calls.append(body)
    n = int(body.get("n", 1))
    return {
      "id": f"chatcmpl-{len(self.calls)}",
      "choices": [
        {"message": {"role": "assistant", "content": "4, strict but adequate"}}
        for _ in range(n)
      ],
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
      },
    }


class _FakeClient:
  def __init__(self):
    completions = _FakeChatCompletions()
    self.chat = type("Chat", (), {"completions": completions})()
    self.completions = completions


class ApiMetricsTest(unittest.TestCase):
  def test_chunk_filenames_balances_requested_processes(self) -> None:
    chunks = api_metrics.chunk_filenames([f"c{i}.json" for i in range(10)], 4)
    self.assertEqual([len(chunk) for chunk in chunks], [3, 3, 2, 2])

  def test_dry_run_reports_process_chunks(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      for index in range(5):
        _write_json(input_dir / f"conversation_{index}.json", _sample_conversation(name=f"Alex {index}"))

      plan = api_metrics.run_api_metrics(
        input_dir,
        output_dir,
        1,
        n_processes=4,
        exc_wai=True,
        exc_panas=True,
        dry_run=True,
      )

    self.assertTrue(plan["dry_run"])
    self.assertEqual(plan["n_processes"], 4)
    self.assertEqual([chunk["conversation_count"] for chunk in plan["chunks"]], [2, 1, 1, 1])
    self.assertEqual(plan["phases"]["main"]["request_count"], len(batch_metrics.CTRS_KEYS) * 5)

  def test_run_api_metrics_writes_outputs_from_chat_completions(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      fake_client = _FakeClient()

      with mock.patch.dict(os.environ, {"REASONCBT_DISABLE_TQDM": "1"}):
        with mock.patch.object(api_metrics, "make_openai_client", return_value=fake_client):
          manifest = api_metrics.run_api_metrics(
            input_dir,
            output_dir,
            1,
            n_processes=1,
            exc_wai=True,
            exc_panas=True,
          )

      ctrs = json.loads((output_dir / "ctrs" / "conversation.json").read_text(encoding="utf-8"))

    self.assertEqual(manifest["status"], "completed")
    self.assertEqual(set(batch_metrics.CTRS_KEYS), set(key for key in ctrs if key != "cost"))
    self.assertGreater(ctrs["cost"], 0)
    self.assertEqual(len(fake_client.completions.calls), len(batch_metrics.CTRS_KEYS))


if __name__ == "__main__":
  unittest.main()
