import sys
import types
import unittest
from pathlib import Path

if "dotenv" not in sys.modules:
  dotenv_module = types.ModuleType("dotenv")
  dotenv_module.load_dotenv = lambda *args, **kwargs: None
  sys.modules["dotenv"] = dotenv_module

if "langchain.messages" not in sys.modules:
  langchain_module = types.ModuleType("langchain")
  langchain_messages_module = types.ModuleType("langchain.messages")

  class _Message:
    def __init__(self, content: str):
      self.content = content

  langchain_messages_module.AIMessage = _Message
  langchain_messages_module.HumanMessage = _Message
  langchain_messages_module.SystemMessage = _Message
  sys.modules["langchain"] = langchain_module
  sys.modules["langchain.messages"] = langchain_messages_module

if "langchain_aws" not in sys.modules:
  langchain_aws_module = types.ModuleType("langchain_aws")
  langchain_aws_module.ChatBedrock = object
  sys.modules["langchain_aws"] = langchain_aws_module

if "langchain_core.prompts" not in sys.modules:
  langchain_core_module = types.ModuleType("langchain_core")
  langchain_core_prompts_module = types.ModuleType("langchain_core.prompts")

  class _PromptTemplate:
    def __init__(self, template: str):
      self.template = template

    @classmethod
    def from_template(cls, template: str) -> "_PromptTemplate":
      return cls(template)

    def format(self, **values: str) -> str:
      rendered = self.template
      for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
      return rendered

  langchain_core_prompts_module.PromptTemplate = _PromptTemplate
  sys.modules["langchain_core"] = langchain_core_module
  sys.modules["langchain_core.prompts"] = langchain_core_prompts_module

if "langchain_huggingface" not in sys.modules:
  langchain_huggingface_module = types.ModuleType("langchain_huggingface")
  langchain_huggingface_module.ChatHuggingFace = object
  langchain_huggingface_module.HuggingFacePipeline = object
  sys.modules["langchain_huggingface"] = langchain_huggingface_module

if "omegaconf" not in sys.modules:
  omegaconf_module = types.ModuleType("omegaconf")

  class _OmegaConf:
    @staticmethod
    def to_container(config, resolve: bool = True):
      return config

    @staticmethod
    def is_config(config) -> bool:
      return False

    @staticmethod
    def create(config):
      return config

  omegaconf_module.OmegaConf = _OmegaConf
  sys.modules["omegaconf"] = omegaconf_module

if "openai" not in sys.modules:
  openai_module = types.ModuleType("openai")
  openai_module.OpenAI = object
  sys.modules["openai"] = openai_module

from agents.patient import PatientAgent


TEST_PROFILE = {
  "personal_info": {
    "name": "Alex",
    "age": "32",
  },
  "presenting_problem": "I keep assuming that small mistakes mean something is deeply wrong with me.",
  "reason_seeking_counseling": "I want to stop spiraling every time something goes wrong.",
  "patterns": ["catastrophizing"],
  "more_information": {
    "past_history": "Longstanding self-doubt.",
    "academic_occupational_functioning": "Works full time.",
    "social_support_system": "Has a close friend.",
  },
  "attitude": "neutral",
}
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class FakeLLM:
  def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
    return "{}"


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
      "mi_techniques_with_te/patient_resistance_update.txt",
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
      "mi_techniques_with_te/patient_utterance_core.txt",
      patient_utterance_instruction="Respond to the motivational discussion in line with your current resistance level.",
      resistance_update_reasoning="The therapist is finally speaking to what matters at work.",
      selected_action="Acceptance",
      selected_action_definition="Openly engage and collaborate with the therapist's proposed line.",
    )
    self.assertIn("Use the <engagement> block", rendered)
    self.assertIn("Acceptance", rendered)

  def test_no_fewshot_variant_prompt_omits_examples_only(self) -> None:
    rendered = render_prompt_text(
      "patient_resistance/no_fewshot/r_neutral.txt",
      scaled_resistance="5.0",
    )

    self.assertIn("Calibration rules", rendered)
    self.assertNotIn("In-context examples", rendered)
    self.assertIn("The patient should not", rendered)

  def test_no_calibration_variant_prompt_omits_calibration_only(self) -> None:
    rendered = render_prompt_text(
      "patient_resistance/no_calibration/r_neutral.txt",
      scaled_resistance="5.0",
    )

    self.assertNotIn("Calibration rules", rendered)
    self.assertIn("In-context examples", rendered)
    self.assertIn("The patient should not", rendered)

  def test_no_should_not_variant_prompt_omits_should_not_only(self) -> None:
    rendered = render_prompt_text(
      "patient_resistance/no_should_not/r_neutral.txt",
      scaled_resistance="5.0",
    )

    self.assertIn("Calibration rules", rendered)
    self.assertIn("In-context examples", rendered)
    self.assertNotIn("The patient should not", rendered)

  def test_no_fewshot_no_calibration_variant_prompt_omits_both_sections(self) -> None:
    rendered = render_prompt_text(
      "patient_resistance/no_fewshot_no_calibration/r_neutral.txt",
      scaled_resistance="5.0",
    )

    self.assertNotIn("Calibration rules", rendered)
    self.assertNotIn("In-context examples", rendered)
    self.assertIn("The patient should not", rendered)

  def test_weak_patient_prompt_uses_attitude_block_without_resistance_block(self) -> None:
    profile = dict(TEST_PROFILE)
    profile["attitude"] = "very_positive"
    agent = PatientAgent(FakeLLM(), profile, resistance_enabled=False)

    messages = agent.build_utterance_messages(
      full_conversation="Therapist: Hello.",
      last_utterances="Therapist: Hello.",
      resistance_level=0.95,
      instruction="Respond naturally.",
    )

    user_prompt = messages[1]["content"]
    self.assertIn("<attitude_towards_therapy>\nPositive\n</attitude_towards_therapy>", user_prompt)
    self.assertNotIn("<resistance_level>", user_prompt)
    self.assertNotIn("Patient attitude towards therapy:", user_prompt)

  def test_weak_structured_patient_prompt_uses_same_attitude_block(self) -> None:
    profile = dict(TEST_PROFILE)
    profile["attitude"] = "very_negative"
    agent = PatientAgent(FakeLLM(), profile, resistance_enabled=False)

    messages = agent.build_structured_messages(
      full_conversation="Therapist: Hello.",
      last_utterances="Therapist: Hello.",
      resistance_level=0.05,
      instruction="Return JSON.",
    )

    user_prompt = messages[1]["content"]
    self.assertIn("<attitude_towards_therapy>\nNegative\n</attitude_towards_therapy>", user_prompt)
    self.assertNotIn("<resistance_level>", user_prompt)

  def test_weak_patient_utterance_core_omits_action_and_resistance_reasoning(self) -> None:
    rendered = render_prompt_text(
      "shared/patient_utterance_core_weak.txt",
      patient_utterance_instruction="Respond to the CBT exercise with realistic engagement.",
    )

    self.assertIn("Respond to the CBT exercise with realistic engagement.", rendered)
    self.assertIn("<attitude_towards_therapy>", rendered)
    self.assertNotIn("Selected utterance action", rendered)
    self.assertNotIn("resistance update", rendered.lower())

  def test_unimplemented_three_no_calib_prompt_raises_clear_missing_file_error(self) -> None:
    agent = PatientAgent(
      FakeLLM(),
      dict(TEST_PROFILE),
      prompt_variant="no_fewshot",
      prompt_intervals="3_no_calib",
    )

    with self.assertRaisesRegex(
      FileNotFoundError,
      "patient_resistance/no_fewshot_3_no_calib/r_neutral.txt",
    ):
      agent.build_utterance_messages(
        full_conversation="Therapist: Hello.",
        last_utterances="Therapist: Hello.",
        resistance_level=0.50,
        instruction="Respond naturally.",
      )


if __name__ == "__main__":
  unittest.main()
