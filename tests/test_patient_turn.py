import unittest

from graph.resistance import should_select_patient_action


class PatientTurnTest(unittest.TestCase):
  def test_action_selection_is_skipped_for_final_closing(self) -> None:
    self.assertFalse(should_select_patient_action(True))
    self.assertTrue(should_select_patient_action(False))


if __name__ == "__main__":
  unittest.main()
