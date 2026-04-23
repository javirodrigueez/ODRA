from pathlib import Path
import shutil
import unittest
import uuid

from scripts.helpers.apply_idx_mapping import copy_simulation_idx_files


class ApplyIdxMappingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = Path.cwd() / f".tmp_test_{uuid.uuid4().hex}"
    self.input_dir = self.tmp / "input"
    self.output_dir = self.tmp / "output"
    (self.input_dir / "json").mkdir(parents=True)
    (self.input_dir / "transcripts").mkdir()

  def tearDown(self) -> None:
    shutil.rmtree(self.tmp, ignore_errors=True)

  def _write(self, relative_path: str, content: str) -> Path:
    path = self.input_dir / relative_path
    path.write_text(content, encoding="utf-8")
    return path

  def test_copies_matching_json_and_transcript_files_to_output_dir(self) -> None:
    self._write("json/simulation_0_negative.json", "json-zero")
    self._write("transcripts/simulation_0_negative.txt", "txt-zero")
    self._write("json/simulation_1_positive.json", "json-one")

    result = copy_simulation_idx_files(self.input_dir, self.output_dir, {"0": "42"})

    self.assertEqual(len(result.plans), 2)
    self.assertEqual((self.input_dir / "json/simulation_0_negative.json").read_text(), "json-zero")
    self.assertEqual((self.input_dir / "transcripts/simulation_0_negative.txt").read_text(), "txt-zero")
    self.assertEqual((self.output_dir / "json/simulation_42_negative.json").read_text(), "json-zero")
    self.assertEqual((self.output_dir / "transcripts/simulation_42_negative.txt").read_text(), "txt-zero")
    self.assertFalse((self.output_dir / "json/simulation_1_positive.json").exists())

  def test_handles_idx_swaps_in_output_dir(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")
    self._write("json/simulation_1_negative.json", "one")

    copy_simulation_idx_files(self.input_dir, self.output_dir, {"0": "1", "1": "0"})

    self.assertEqual((self.input_dir / "json/simulation_0_negative.json").read_text(), "zero")
    self.assertEqual((self.input_dir / "json/simulation_1_negative.json").read_text(), "one")
    self.assertEqual((self.output_dir / "json/simulation_1_negative.json").read_text(), "zero")
    self.assertEqual((self.output_dir / "json/simulation_0_negative.json").read_text(), "one")

  def test_copies_unchanged_idx_to_output_dir(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")

    result = copy_simulation_idx_files(self.input_dir, self.output_dir, {"0": "0"})

    self.assertEqual(len(result.plans), 1)
    self.assertEqual((self.input_dir / "json/simulation_0_negative.json").read_text(), "zero")
    self.assertEqual((self.output_dir / "json/simulation_0_negative.json").read_text(), "zero")

  def test_dry_run_does_not_copy_files(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")

    result = copy_simulation_idx_files(self.input_dir, self.output_dir, {"0": "42"}, dry_run=True)

    self.assertEqual(len(result.plans), 1)
    self.assertTrue((self.input_dir / "json/simulation_0_negative.json").exists())
    self.assertFalse((self.output_dir / "json/simulation_42_negative.json").exists())

  def test_refuses_to_overwrite_existing_output_file(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")
    (self.output_dir / "json").mkdir(parents=True)
    (self.output_dir / "json/simulation_1_negative.json").write_text("existing", encoding="utf-8")

    with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
      copy_simulation_idx_files(self.input_dir, self.output_dir, {"0": "1"})


if __name__ == "__main__":
  unittest.main()
