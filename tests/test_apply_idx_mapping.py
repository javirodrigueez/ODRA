import shutil
from pathlib import Path
import unittest
import uuid

from scripts.helpers.apply_idx_mapping import rename_simulation_idx_files


class ApplyIdxMappingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = Path.cwd() / f".tmp_test_{uuid.uuid4().hex}"
    (self.tmp / "json").mkdir(parents=True)
    (self.tmp / "transcripts").mkdir()

  def tearDown(self) -> None:
    shutil.rmtree(self.tmp, ignore_errors=True)

  def _write(self, relative_path: str, content: str) -> Path:
    path = self.tmp / relative_path
    path.write_text(content, encoding="utf-8")
    return path

  def test_renames_matching_json_and_transcript_files(self) -> None:
    self._write("json/simulation_0_negative.json", "json-zero")
    self._write("transcripts/simulation_0_negative.txt", "txt-zero")
    self._write("json/simulation_1_positive.json", "json-one")

    result = rename_simulation_idx_files(self.tmp, {"0": "42"})

    self.assertEqual(len(result.plans), 2)
    self.assertFalse((self.tmp / "json/simulation_0_negative.json").exists())
    self.assertFalse((self.tmp / "transcripts/simulation_0_negative.txt").exists())
    self.assertEqual((self.tmp / "json/simulation_42_negative.json").read_text(), "json-zero")
    self.assertEqual((self.tmp / "transcripts/simulation_42_negative.txt").read_text(), "txt-zero")
    self.assertEqual((self.tmp / "json/simulation_1_positive.json").read_text(), "json-one")

  def test_handles_idx_swaps_without_overwriting(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")
    self._write("json/simulation_1_negative.json", "one")

    rename_simulation_idx_files(self.tmp, {"0": "1", "1": "0"})

    self.assertEqual((self.tmp / "json/simulation_1_negative.json").read_text(), "zero")
    self.assertEqual((self.tmp / "json/simulation_0_negative.json").read_text(), "one")

  def test_dry_run_does_not_rename_files(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")

    result = rename_simulation_idx_files(self.tmp, {"0": "42"}, dry_run=True)

    self.assertEqual(len(result.plans), 1)
    self.assertTrue((self.tmp / "json/simulation_0_negative.json").exists())
    self.assertFalse((self.tmp / "json/simulation_42_negative.json").exists())

  def test_refuses_to_overwrite_unmapped_target_file(self) -> None:
    self._write("json/simulation_0_negative.json", "zero")
    self._write("json/simulation_1_negative.json", "one")

    with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
      rename_simulation_idx_files(self.tmp, {"0": "1"})


if __name__ == "__main__":
  unittest.main()
