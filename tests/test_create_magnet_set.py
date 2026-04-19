import json
import tempfile
from pathlib import Path
import unittest

from evaluation import run_batch_metrics
from scripts.data import create_magnet_set


def _intake_form(name: str = "Laura Saunders") -> str:
    return f"""Name:
{name}
Age:
45
Gender:
female
Occupation: Office Job
Education: College Graduate
Marital Status: Single
Family Details: Lives alone

2. Presenting Problem
I am struggling with the thought that I can't run far.

3. Reason for Seeking Counseling
This negative belief is affecting my overall mood.

4. Past History (including medical history)
I have not experienced similar problems before.

5. Academic/occupational functioning level: My job performance has not been impacted.
Interpersonal relationships: I have not experienced any strain.
Daily life: My self-esteem and motivation levels have been affected.

6. Social Support System
I have a close friend who is supportive.
"""


def _magnet_session(name: str = "Laura Saunders", attitude: str = "positive") -> dict:
    return {
        "AI_client": {
            "intake_form": _intake_form(name),
            "attitude": attitude,
            "dialogue_history_init": (
                "Counselor: Hi Laura, it's nice to meet you.\n"
                "Client: I keep feeling like I can't run very far."
            ),
            "dialogue_history": None,
        },
        "dialogue": [
            {"role": "counselor", "message": "Hi Laura, it's nice to meet you."},
            {"role": "client", "message": "I keep feeling like I can't run very far."},
            {"role": "counselor", "message": "What evidence supports that thought?"},
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CreateMagnetSetTest(unittest.TestCase):
    def test_filter_writes_evaluation_compatible_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "input"
            output_dir = base / "output"
            seed_file = base / "seed.json"

            _write_json(input_dir / "session_1_magnet.json", _magnet_session())
            _write_json(input_dir / "session_2_magnet.json", _magnet_session("Other Patient"))
            _write_json(seed_file, [_magnet_session()])

            summary = create_magnet_set.filter_magnet_sessions(
                input_dir,
                output_dir,
                seed_file,
            )

            self.assertEqual(summary["reference_seeds"], 1)
            self.assertEqual(summary["scanned_files"], 2)
            self.assertEqual(summary["written_files"], 1)
            self.assertEqual(summary["skipped_files"], 1)

            output_file = output_dir / "session_1_magnet.json"
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["messages"],
                [
                    {"role": "therapist", "content": "Hi Laura, it's nice to meet you."},
                    {"role": "patient", "content": "I keep feeling like I can't run very far."},
                    {"role": "therapist", "content": "What evidence supports that thought?"},
                ],
            )
            self.assertEqual(payload["patient_profile"]["attitude"], "positive")
            self.assertEqual(payload["patient_profile"]["personal_info"]["name"], "Laura Saunders")
            self.assertIn("run far", payload["patient_profile"]["presenting_problem"])
            self.assertIn("overall mood", payload["patient_profile"]["reason_seeking_counseling"])
            self.assertIn("past_history", payload["patient_profile"]["more_information"])

            records = run_batch_metrics.discover_conversations(output_dir)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].filename, "session_1_magnet.json")
            self.assertEqual(records[0].attitude, "positive")

    def test_falls_back_to_initial_dialogue_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "input"
            output_dir = base / "output"
            seed_file = base / "seed.json"
            session = _magnet_session()
            del session["dialogue"]

            _write_json(input_dir / "session_1_magnet.json", session)
            _write_json(seed_file, [session])

            summary = create_magnet_set.filter_magnet_sessions(
                input_dir,
                output_dir,
                seed_file,
            )

            self.assertEqual(summary["written_files"], 1)
            payload = json.loads((output_dir / "session_1_magnet.json").read_text(encoding="utf-8"))
            self.assertEqual(
                payload["messages"],
                [
                    {"role": "therapist", "content": "Hi Laura, it's nice to meet you."},
                    {"role": "patient", "content": "I keep feeling like I can't run very far."},
                ],
            )
            self.assertEqual(len(run_batch_metrics.discover_conversations(output_dir)), 1)


if __name__ == "__main__":
    unittest.main()
