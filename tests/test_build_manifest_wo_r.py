import json
import shutil
from pathlib import Path
import unittest
import uuid

from scripts.helpers.build_manifest import build_manifest


class BuildManifestWithoutResistanceTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = Path.cwd() / f".tmp_test_{uuid.uuid4().hex}"
    self.tmp.mkdir()

  def tearDown(self) -> None:
    shutil.rmtree(self.tmp, ignore_errors=True)

  def _write_json(self, name: str, payload: object) -> str:
    path = self.tmp / name
    with path.open("w", encoding="utf-8") as f:
      json.dump(payload, f)
    return str(path)

  def test_profile_matching_marks_target_idx_when_source_idx_differs(self) -> None:
    target_profiles = [
      {"idx": 10, "attitude": "positive", "personal_info": {"name": "Ada"}, "presenting_problem": "a"},
      {"idx": 20, "attitude": "negative", "personal_info": {"name": "Ben"}, "presenting_problem": "b"},
    ]
    already_exec_profiles = [
      {"idx": 999, "attitude": "negative", "personal_info": {"name": "Ben"}, "presenting_problem": "b"},
    ]

    manifest, unmatched, idx_mapping = build_manifest(
      profiles_to_exec_path=self._write_json("target.json", target_profiles),
      already_exec_paths=[self._write_json("already.json", already_exec_profiles)],
      output_dir=str(self.tmp / "out"),
      cfg_path="configs/test.yaml",
    )

    self.assertEqual(unmatched, [])
    self.assertFalse(manifest["profiles"]["10"]["executed"])
    self.assertTrue(manifest["profiles"]["20"]["executed"])
    self.assertEqual(manifest["metadata"]["executed_count"], 1)
    self.assertEqual(manifest["metadata"]["total_profiles"], 2)
    self.assertEqual(idx_mapping, {"999": "20"})
    self.assertEqual(manifest["metadata"]["idx_mapping_count"], 1)

  def test_idx_matching_marks_matching_target_idx(self) -> None:
    target_profiles = [
      {"idx": 1, "attitude": "positive", "personal_info": {"name": "Ada"}},
      {"idx": 2, "attitude": "neutral", "personal_info": {"name": "Ben"}},
    ]

    manifest, unmatched, idx_mapping = build_manifest(
      profiles_to_exec_path=self._write_json("target.json", target_profiles),
      already_exec_paths=[self._write_json("already.json", [{"idx": 2}])],
      output_dir=str(self.tmp / "out"),
      match_key="idx",
    )

    self.assertEqual(unmatched, [])
    self.assertFalse(manifest["profiles"]["1"]["executed"])
    self.assertTrue(manifest["profiles"]["2"]["executed"])
    self.assertEqual(idx_mapping, {"2": "2"})


if __name__ == "__main__":
  unittest.main()
