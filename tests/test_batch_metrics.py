import json
import tempfile
from pathlib import Path
import unittest

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


def _choice(content: str) -> dict:
  return {"message": {"role": "assistant", "content": content}}


def _panas_content(score: int = 3) -> str:
  return "\n".join(
    f"{feeling}, short explanation for {feeling}, {score}"
    for feeling in batch_metrics.FEELINGS
  )


class _Obj:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def model_dump(self):
    return dict(self.__dict__)


class _Content:
  def __init__(self, text: str):
    self._text = text

  def read(self) -> bytes:
    return self._text.encode("utf-8")


class FakeOpenAIClient:
  def __init__(self, *, fail_custom_id_contains: str | None = None):
    self.files = self.Files(self)
    self.batches = self.Batches(self)
    self.fail_custom_id_contains = fail_custom_id_contains
    self.uploaded_files: dict[str, str] = {}
    self.download_files: dict[str, str] = {}
    self.batch_to_output: dict[str, str] = {}
    self.batch_to_error: dict[str, str | None] = {}
    self.batch_create_count = 0
    self.file_create_count = 0

  class Files:
    def __init__(self, client: "FakeOpenAIClient"):
      self.client = client

    def create(self, *, file, purpose):
      assert purpose == "batch"
      self.client.file_create_count += 1
      file_id = f"file-{self.client.file_create_count}"
      self.client.uploaded_files[file_id] = file.read().decode("utf-8")
      return _Obj(id=file_id, purpose=purpose)

    def content(self, file_id: str):
      return _Content(self.client.download_files[file_id])

  class Batches:
    def __init__(self, client: "FakeOpenAIClient"):
      self.client = client

    def create(self, *, input_file_id, endpoint, completion_window, metadata):
      assert endpoint == batch_metrics.BATCH_ENDPOINT
      assert completion_window == batch_metrics.COMPLETION_WINDOW
      self.client.batch_create_count += 1
      batch_id = f"batch-{self.client.batch_create_count}"
      output_file_id = f"out-{self.client.batch_create_count}"
      error_file_id = None
      output_lines = []
      error_lines = []
      for line in self.client.uploaded_files[input_file_id].splitlines():
        request = json.loads(line)
        custom_id = request["custom_id"]
        if self.client.fail_custom_id_contains and self.client.fail_custom_id_contains in custom_id:
          error_lines.append(
            json.dumps(
              {
                "id": f"req-{custom_id}",
                "custom_id": custom_id,
                "response": None,
                "error": {"code": "test_failure", "message": "forced failure"},
              }
            )
          )
          continue
        output_lines.append(json.dumps(self._response_line(custom_id, request)))
      if error_lines:
        error_file_id = f"err-{self.client.batch_create_count}"
        self.client.download_files[error_file_id] = "\n".join(error_lines) + "\n"
      self.client.download_files[output_file_id] = "\n".join(reversed(output_lines)) + "\n"
      self.client.batch_to_output[batch_id] = output_file_id
      self.client.batch_to_error[batch_id] = error_file_id
      return _Obj(
        id=batch_id,
        status="completed",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        request_counts={
          "total": len(output_lines) + len(error_lines),
          "completed": len(output_lines),
          "failed": len(error_lines),
        },
      )

    def retrieve(self, batch_id: str):
      output_file_id = self.client.batch_to_output[batch_id]
      error_file_id = self.client.batch_to_error[batch_id]
      failed = 1 if error_file_id else 0
      completed = len(self.client.download_files[output_file_id].splitlines())
      return _Obj(
        id=batch_id,
        status="completed",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        request_counts={
          "total": completed + failed,
          "completed": completed,
          "failed": failed,
        },
      )

    def _response_line(self, custom_id: str, request: dict) -> dict:
      body = request["body"]
      max_iter = body["n"]
      metric_group = custom_id.split(":")[1]
      if metric_group in {"ctrs", "wai"}:
        content = "4, strict but adequate"
      elif metric_group in {"panas_before", "panas_after"}:
        content = _panas_content(3 if metric_group == "panas_before" else 4)
      else:
        content = "1, unknown"
      return {
        "id": f"batch-req-{custom_id}",
        "custom_id": custom_id,
        "response": {
          "status_code": 200,
          "request_id": f"req-{custom_id}",
          "body": {
            "choices": [_choice(content) for _ in range(max_iter)],
            "usage": {
              "prompt_tokens": 100,
              "completion_tokens": 20,
              "total_tokens": 120,
            },
          },
        },
        "error": None,
      }


class BatchMetricsTest(unittest.TestCase):
  def test_build_requests_counts_and_top_level_discovery(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      nested = input_dir / "nested"
      _write_json(input_dir / "conversation_a.json", _sample_conversation())
      _write_json(input_dir / "conversation_b.json", _sample_conversation(name="Blair"))
      _write_json(nested / "ignored.json", _sample_conversation(name="Nested"))
      output_dir = base / "out"
      batch_metrics.ensure_output_dirs(output_dir)

      records = batch_metrics.discover_conversations(input_dir)
      self.assertEqual([record.filename for record in records], ["conversation_a.json", "conversation_b.json"])

      phase1 = batch_metrics.build_panas_before_requests(records, output_dir, "gpt-4o", 2)
      before_payloads = {
        record.filename: {
          **{feeling: [3] for feeling in batch_metrics.FEELINGS},
          "explanations": [],
          "attitude": record.attitude,
          "cost": 0.1,
        }
        for record in records
      }
      phase2 = batch_metrics.build_main_requests(records, output_dir, before_payloads, "gpt-4o", 2)

      self.assertEqual(len(phase1), 2)
      self.assertEqual(len(phase2), 38)
      self.assertEqual(len({request.custom_id for request in phase1 + phase2}), 40)
      self.assertTrue(all(request.body["model"] == "gpt-4o" for request in phase1 + phase2))
      self.assertTrue(all(request.body["n"] == 2 for request in phase1 + phase2))

  def test_default_panas_after_prompt_uses_legacy_dialogue_input(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      batch_metrics.ensure_output_dirs(output_dir)
      records = batch_metrics.discover_conversations(input_dir)

      phase2 = batch_metrics.build_main_requests(
        records,
        output_dir,
        {},
        "gpt-4o",
        1,
        {"ctrs", "wai"},
      )

      self.assertEqual(len(phase2), 1)
      prompt = phase2[0].body["messages"][0]["content"]
      self.assertIn("Here is the counseling session:", prompt)
      self.assertIn("Therapist: Hi, how are you feeling today?", prompt)
      self.assertNotIn("{dialogue}", prompt)
      self.assertNotIn("<presession_questionnaire>", prompt)

  def test_panas_v2_prompt_uses_presession_questionnaire_input(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      batch_metrics.ensure_output_dirs(output_dir)
      records = batch_metrics.discover_conversations(input_dir)
      before_payloads = {
        "conversation.json": {
          **{feeling: [2] for feeling in batch_metrics.FEELINGS},
          "explanations": [],
          "attitude": "neutral",
          "cost": 0.1,
        }
      }

      phase2 = batch_metrics.build_main_requests(
        records,
        output_dir,
        before_payloads,
        "gpt-4o",
        1,
        {"ctrs", "wai"},
        panas_v2=True,
      )

      self.assertEqual(len(phase2), 1)
      prompt = phase2[0].body["messages"][0]["content"]
      self.assertIn("<presession_questionnaire>", prompt)
      self.assertIn("interested: 2", prompt)
      self.assertIn("<conversation>", prompt)
      self.assertIn("Therapist: Hi, how are you feeling today?", prompt)
      self.assertNotIn("{conversation}", prompt)
      self.assertNotIn("{presession_questionnaire}", prompt)

  def test_panas_v2_flag_is_parsed_and_part_of_signature(self) -> None:
    args = batch_metrics.parse_args(["input", "--panas-v2"])
    self.assertTrue(args.panas_v2)

    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      records = batch_metrics.discover_conversations(input_dir)

      default_signature = batch_metrics.compute_signature(
        input_dir,
        output_dir,
        records,
        "gpt-4o",
        1,
      )
      v2_signature = batch_metrics.compute_signature(
        input_dir,
        output_dir,
        records,
        "gpt-4o",
        1,
        panas_v2=True,
      )

      self.assertEqual(default_signature["panas_after_prompt"], "panas_after.txt")
      self.assertEqual(v2_signature["panas_after_prompt"], "panas_after_v2.txt")

  def test_skip_existing_complete_outputs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      batch_metrics.ensure_output_dirs(output_dir)
      _write_json(
        output_dir / "ctrs" / "conversation.json",
        {**{key: 4 for key in batch_metrics.CTRS_KEYS}, "cost": 1.0},
      )
      records = batch_metrics.discover_conversations(input_dir)
      before_payloads = {
        "conversation.json": {
          **{feeling: [3] for feeling in batch_metrics.FEELINGS},
          "explanations": [],
          "attitude": "neutral",
          "cost": 0.1,
        }
      }

      phase2 = batch_metrics.build_main_requests(records, output_dir, before_payloads, "gpt-4o", 1)

      self.assertEqual(len([request for request in phase2 if request.metric_group == "ctrs"]), 0)
      self.assertEqual(len([request for request in phase2 if request.metric_group == "wai"]), 12)
      self.assertEqual(len([request for request in phase2 if request.metric_group == "panas_after"]), 1)

  def test_parse_phase_outputs_with_shuffled_lines_and_costs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      records = batch_metrics.discover_conversations(input_dir)
      record_map = {record.filename: record for record in records}
      spec_panas = batch_metrics.BatchRequest(
        custom_id="main:panas_after:00000:conversation:panas_after",
        phase="main",
        metric_group="panas_after",
        input_file="conversation.json",
        output_file="panas_after/conversation.json",
        item_key="panas_after",
        body={},
      )
      ctrs_specs = [
        batch_metrics.BatchRequest(
          custom_id=f"main:ctrs:00000:conversation:{key}",
          phase="main",
          metric_group="ctrs",
          input_file="conversation.json",
          output_file="ctrs/conversation.json",
          item_key=key,
          body={},
        )
        for key in batch_metrics.CTRS_KEYS
      ]
      wai_specs = [
        batch_metrics.BatchRequest(
          custom_id=f"main:wai:00000:conversation:{key}",
          phase="main",
          metric_group="wai",
          input_file="conversation.json",
          output_file="wai/conversation.json",
          item_key=key,
          body={},
        )
        for key in batch_metrics.WAI_KEYS
      ]
      request_map = {
        spec.custom_id: spec.map_entry()
        for spec in [*ctrs_specs, *wai_specs, spec_panas]
      }
      output_jsonl = base / "output.jsonl"
      lines = [
        {
          "custom_id": spec_panas.custom_id,
          "response": {
            "status_code": 200,
            "body": {
              "choices": [_choice(_panas_content(5)), _choice(_panas_content(4))],
              "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
          },
          "error": None,
        },
      ]
      for spec in wai_specs:
        lines.append(
          {
            "custom_id": spec.custom_id,
            "response": {
              "status_code": 200,
              "body": {
                "choices": [_choice("3, explanation"), _choice("5, explanation")],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
              },
            },
            "error": None,
          }
        )
      for spec in ctrs_specs:
        lines.append(
          {
            "custom_id": spec.custom_id,
            "response": {
              "status_code": 200,
              "body": {
                "choices": [_choice("2, explanation"), _choice("6, explanation")],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
              },
            },
            "error": None,
          }
        )
      output_jsonl.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

      outputs, report = batch_metrics.parse_phase_outputs(
        output_jsonl,
        request_map,
        record_map,
        2,
        input_cost_per_1m=1.0,
        output_cost_per_1m=10.0,
      )

      self.assertEqual(outputs["ctrs/conversation.json"]["general_1_understanding"], 4)
      self.assertEqual(outputs["wai/conversation.json"]["wai1"], 4)
      self.assertEqual(outputs["panas_after/conversation.json"]["interested"], [5, 4])
      self.assertEqual(len(outputs["panas_after/conversation.json"]["explanations"]), 40)
      self.assertAlmostEqual(outputs["ctrs/conversation.json"]["cost"], 0.00036)
      self.assertEqual(report["parsed_requests"], 19)

  def test_run_batch_metrics_writes_current_output_contract(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      client = FakeOpenAIClient()

      manifest = batch_metrics.run_batch_metrics(
        input_dir,
        output_dir,
        2,
        client=client,
        poll_interval=0,
      )

      self.assertEqual(manifest["status"], "completed")
      self.assertEqual(client.batch_create_count, 2)
      self.assertTrue((output_dir / "ctrs" / "conversation.json").is_file())
      self.assertTrue((output_dir / "wai" / "conversation.json").is_file())
      self.assertTrue((output_dir / "panas_before" / "conversation.json").is_file())
      self.assertTrue((output_dir / "panas_after" / "conversation.json").is_file())
      ctrs = json.loads((output_dir / "ctrs" / "conversation.json").read_text(encoding="utf-8"))
      wai = json.loads((output_dir / "wai" / "conversation.json").read_text(encoding="utf-8"))
      before = json.loads((output_dir / "panas_before" / "conversation.json").read_text(encoding="utf-8"))
      after = json.loads((output_dir / "panas_after" / "conversation.json").read_text(encoding="utf-8"))
      self.assertEqual(set(batch_metrics.CTRS_KEYS), set(key for key in ctrs if key != "cost"))
      self.assertEqual(set(batch_metrics.WAI_KEYS), set(key for key in wai if key != "cost"))
      self.assertEqual(before["interested"], [3, 3])
      self.assertEqual(after["interested"], [4, 4])
      self.assertIn("explanations", after)
      self.assertGreater(ctrs["cost"], 0)

  def test_batch_failure_keeps_final_outputs_unwritten(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      client = FakeOpenAIClient(fail_custom_id_contains="panas_before")

      with self.assertRaises(batch_metrics.BatchMetricsError):
        batch_metrics.run_batch_metrics(input_dir, output_dir, 1, client=client, poll_interval=0)

      self.assertFalse((output_dir / "panas_before" / "conversation.json").exists())
      artifacts = list((output_dir / "_batch_artifacts").glob("*/failure_report.json"))
      self.assertEqual(len(artifacts), 1)

  def test_invalid_profile_fails_before_submission(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      payload = _sample_conversation()
      del payload["patient_profile"]["attitude"]
      _write_json(input_dir / "conversation.json", payload)
      client = FakeOpenAIClient()

      with self.assertRaises(batch_metrics.BatchMetricsError):
        batch_metrics.run_batch_metrics(input_dir, output_dir, 1, client=client, poll_interval=0)

      self.assertEqual(client.batch_create_count, 0)

  def test_malformed_metric_score_fails(self) -> None:
    with self.assertRaises(batch_metrics.BatchMetricsError):
      batch_metrics.parse_score_prefix("no numeric score here", "ctrs", "custom-id")

  def test_resume_from_manifest_does_not_submit_duplicate_phase(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      input_dir = base / "input"
      output_dir = base / "out"
      _write_json(input_dir / "conversation.json", _sample_conversation())
      client1 = FakeOpenAIClient()
      records = batch_metrics.discover_conversations(input_dir)
      batch_metrics.ensure_output_dirs(output_dir)
      signature = batch_metrics.compute_signature(input_dir.resolve(), output_dir.resolve(), records, "gpt-4o", 1)
      artifact_dir, manifest, _ = batch_metrics.load_or_create_manifest(output_dir.resolve(), signature)
      staged = {}
      phase1_specs = batch_metrics.build_panas_before_requests(records, output_dir.resolve(), "gpt-4o", 1)
      batch_metrics.run_phase(
        client1,
        "panas_before",
        phase1_specs,
        {record.filename: record for record in records},
        1,
        1.25,
        5.0,
        0,
        26,
        artifact_dir,
        manifest,
        staged,
      )
      self.assertEqual(client1.batch_create_count, 1)

      client2 = FakeOpenAIClient()
      final_manifest = batch_metrics.run_batch_metrics(
        input_dir,
        output_dir,
        1,
        client=client2,
        poll_interval=0,
      )

      self.assertEqual(final_manifest["status"], "completed")
      self.assertEqual(client2.batch_create_count, 1)
      self.assertTrue((output_dir / "panas_before" / "conversation.json").is_file())
      self.assertTrue((output_dir / "panas_after" / "conversation.json").is_file())


if __name__ == "__main__":
  unittest.main()
