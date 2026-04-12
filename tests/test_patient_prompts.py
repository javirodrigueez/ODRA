import unittest
from pathlib import Path

from graph.resistance import resistance_prompt_name_for_level


TEST_PROFILE = {
  "presenting_problem": "I keep assuming that small mistakes mean something is deeply wrong with me.",
  "reason_seeking_counseling": "I want to stop spiraling every time something goes wrong.",
}
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def render_prompt_text(relative_path: str, **values: str) -> str:
  template = (PROMPTS_DIR / relative_path).read_text(encoding="utf-8")
  rendered = template
  for key, value in values.items():
    rendered = rendered.replace(f"{{{key}}}", value)
  return rendered.replace("{{", "{").replace("}}", "}")


class PatientPromptTest(unittest.TestCase):
  def test_tib_check_prompt_uses_new_tib_contract(self) -> None:
    rendered = render_prompt_text("shared/tib_check.txt")
    self.assertIn('"has_tib": false', rendered)
    self.assertIn('"tib_name": "none"', rendered)
    self.assertIn('"tib_description": null', rendered)

  def test_address_tib_plan_generation_prompt_receives_new_variables(self) -> None:
    rendered = render_prompt_text(
      "address_tib/plan_generation.txt",
      address_tib_route_bucket="other",
      address_tib_name="skepticism",
      address_tib_description="The patient says the exercise feels pointless.",
    )
    self.assertIn("therapy-interfering behavior", rendered)
    self.assertIn("address_tib_route_bucket", rendered)
    self.assertIn("address_tib_name", rendered)
    self.assertIn("address_tib_description", rendered)

  def test_patient_resistance_update_prompt_receives_new_variables(self) -> None:
    rendered = render_prompt_text(
      "shared/patient_resistance_update.txt",
      resistance="0.42",
      presenting_problem=TEST_PROFILE["presenting_problem"],
      reason_for_seeking_counseling=TEST_PROFILE["reason_seeking_counseling"],
      last_utterance="Let's look at what went through your mind right then.",
    )
    self.assertIn(TEST_PROFILE["presenting_problem"], rendered)
    self.assertIn(TEST_PROFILE["reason_seeking_counseling"], rendered)
    self.assertIn("Let's look at what went through your mind right then.", rendered)

  def test_patient_utterance_core_prompt_receives_instruction_reasoning_and_action(self) -> None:
    rendered = render_prompt_text(
      "shared/patient_utterance_core.txt",
      patient_utterance_instruction="Respond to the CBT exercise with realistic engagement.",
      resistance_update_reasoning="The therapist felt aligned with what I came here for, so I softened slightly.",
      selected_action="Hesitation",
      selected_action_definition="Stay in the exchange, but foreground meaningful doubts about the therapist's proposed line.",
    )
    self.assertIn("Respond to the CBT exercise with realistic engagement.", rendered)
    self.assertIn("The therapist felt aligned with what I came here for", rendered)
    self.assertIn("Hesitation", rendered)
    self.assertIn("foreground meaningful doubts", rendered)

  def test_patient_action_distribution_prompt_lists_all_actions(self) -> None:
    rendered = render_prompt_text(
      "shared/patient_action_distribution.txt",
      patient_utterance_instruction="Respond naturally.",
      resistance_update_reasoning="The therapist's line did not help much.",
    )
    self.assertIn('"Acceptance": 0', rendered)
    self.assertIn('"Hesitation": 0', rendered)
    self.assertIn('"Mild rejection": 0', rendered)
    self.assertIn('"Rejection": 0', rendered)
    self.assertIn('"Terminate": 0', rendered)

  def test_patient_utterance_core_prompt_mentions_terminate_action(self) -> None:
    rendered = render_prompt_text(
      "shared/patient_utterance_core.txt",
      patient_utterance_instruction="Respond naturally.",
      resistance_update_reasoning="The therapist is making me want to stop.",
      selected_action="Terminate",
      selected_action_definition="Explicitly request ending the session now.",
    )
    self.assertIn("Action: Terminate", rendered)
    self.assertIn("Definition: Explicitly request ending the session now.", rendered)

  def test_topic_exploration_patient_resistance_update_prompt_mentions_engagement_alignment(self) -> None:
    rendered = render_prompt_text(
      "mi_topic_exploration/patient_resistance_update.txt",
      resistance="0.42",
      presenting_problem=TEST_PROFILE["presenting_problem"],
      reason_for_seeking_counseling=TEST_PROFILE["reason_seeking_counseling"],
      last_utterance="How is this affecting the way you relate to coworkers?",
      perceived_topic="Workplace Relationships",
      motivation_topic="Workplace Relationships",
      engagement_level="4",
    )
    self.assertIn("Workplace Relationships", rendered)
    self.assertIn("Use the <engagement> block", rendered)
    self.assertIn("engagement level", rendered)

  def test_topic_exploration_patient_utterance_core_prompt_mentions_engagement_block(self) -> None:
    rendered = render_prompt_text(
      "mi_topic_exploration/patient_utterance_core.txt",
      patient_utterance_instruction="Respond to the motivational discussion in line with your current resistance level.",
      resistance_update_reasoning="The therapist is finally speaking to what matters at work.",
      selected_action="Acceptance",
      selected_action_definition="Openly engage and collaborate with the therapist's proposed line.",
    )
    self.assertIn("Use the <engagement> block", rendered)
    self.assertIn("Acceptance", rendered)


if __name__ == "__main__":
  unittest.main()
