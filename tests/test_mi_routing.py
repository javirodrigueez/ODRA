import sys
import types
import json
import unittest
from types import SimpleNamespace

if "dotenv" not in sys.modules:
  dotenv_module = types.ModuleType("dotenv")
  dotenv_module.load_dotenv = lambda *args, **kwargs: None
  sys.modules["dotenv"] = dotenv_module

if "langgraph.graph" not in sys.modules:
  langgraph_module = types.ModuleType("langgraph")
  langgraph_graph_module = types.ModuleType("langgraph.graph")
  langgraph_graph_module.END = "END"
  langgraph_graph_module.START = "START"
  langgraph_graph_module.StateGraph = object
  sys.modules["langgraph"] = langgraph_module
  sys.modules["langgraph.graph"] = langgraph_graph_module

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

from graph.graph import TherapyGraphBuilder
from runtime_config import normalize_runtime_config


class MIRoutingTest(unittest.TestCase):
  def _make_builder(self, *, resistance_handling: str = "mi_techniques", resistance: bool = True) -> TherapyGraphBuilder:
    builder = TherapyGraphBuilder.__new__(TherapyGraphBuilder)
    builder.profile = {}
    builder.cfg = SimpleNamespace(
      env=SimpleNamespace(
        max_turns=12,
        min_turns=0,
        long_session_turn_threshold=70,
        resistance_handling=resistance_handling,
        resistance=resistance,
      ),
      topic_exploration=SimpleNamespace(patient_classifier_path="", patient_wikipedia_dir=""),
    )
    builder.mi_techniques = {"Open Question": "Ask an open question."}
    builder._load_stage_prompt = lambda stage, name, **values: f"{stage}/{name}"
    builder._append_mi_context_message = lambda *args, **kwargs: None
    builder._run_therapist_structured = lambda state, traces, step_name, instruction: {
      "reason_for_mi": "high resistance",
      "conclusions": "Patient softened.",
      "patient_state_change": "More willing to continue.",
    }
    return builder

  def _make_state(self, builder: TherapyGraphBuilder, *, stage: str = "id_distortions", predicted_resistance: float = 0.8) -> dict:
    state = builder.initial_state(max_turns=12)
    state["current_stage"] = stage
    state["current_substage"] = "cognitive_review_active_question" if stage == "id_distortions" else state["current_substage"]
    state["predicted_resistance"] = predicted_resistance
    return builder._ensure_state_defaults(state)

  def test_current_band_uses_global_thresholds(self) -> None:
    builder = self._make_builder()
    self.assertEqual(builder._current_band(0.66), "precontemplation")
    self.assertEqual(builder._current_band(0.65), "contemplation")
    self.assertEqual(builder._current_band(0.34), "contemplation")
    self.assertEqual(builder._current_band(0.33), "action")

  def test_precon_streak_enters_legacy_mi_mode(self) -> None:
    builder = self._make_builder(resistance_handling="mi_techniques")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["preconsecutive_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertEqual(state["current_substage"], "mi_choose_technique")

  def test_precon_streak_enters_cami_mode(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["preconsecutive_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques_with_te")
    self.assertEqual(state["current_substage"], "mi_choose_topic")

  def test_precon_streak_does_not_enter_mi_when_mode_is_none(self) -> None:
    builder = self._make_builder(resistance_handling="none")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["preconsecutive_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertEqual(state["preconsecutive_turns"], 3)

  def test_non_precon_band_resets_entry_streak(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, predicted_resistance=0.5)
    state["preconsecutive_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "id_distortions")
    self.assertEqual(state["preconsecutive_turns"], 0)

  def test_legacy_mi_mode_never_branches_to_topic_stage(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.9)
    state["current_substage"] = "mi_choose_technique"

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertEqual(state["current_substage"], "mi_choose_technique")

  def test_legacy_mi_exits_after_three_below_precontemplation_turns(self) -> None:
    builder = self._make_builder(resistance_handling="mi_techniques")
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.5)
    state["current_substage"] = "mi_choose_technique"
    state["mi_recovery_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertEqual(state["current_substage"], "mi_transition_to_cbt")
    self.assertEqual(state["mi_recovery_turns"], 0)

  def test_action_band_no_longer_exits_legacy_mi_immediately(self) -> None:
    builder = self._make_builder(resistance_handling="mi_techniques")
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.2)
    state["current_substage"] = "mi_choose_technique"

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques")
    self.assertEqual(state["current_substage"], "mi_choose_technique")
    self.assertEqual(state["mi_recovery_turns"], 1)

  def test_cami_exits_after_three_recovery_turns(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, stage="mi_techniques_with_te", predicted_resistance=0.5)
    state["current_substage"] = "mi_choose_topic"
    state["cami_recovery_turns"] = 2

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "mi_techniques_with_te")
    self.assertEqual(state["current_substage"], "mi_transition_to_cbt")
    self.assertEqual(state["cami_recovery_turns"], 0)

  def test_enter_mi_cami_preserves_topic_state_for_resume(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, predicted_resistance=0.8)
    state["current_mi_topic"] = "Employment"
    state["mi_topic_stack"] = ["Economy", "Employment"]
    state["mi_topics_tried"] = ["Economy", "Employment"]

    builder._enter_mi(
      state,
      "mi_techniques_with_te",
      "id_distortions",
      "cognitive_review_active_question",
      "resume CAMI",
    )

    self.assertEqual(state["current_mi_topic"], "Employment")
    self.assertEqual(state["mi_topic_stack"], ["Economy", "Employment"])
    self.assertEqual(state["mi_topics_tried"], ["Economy", "Employment"])

  def test_runtime_config_sets_default_long_session_threshold(self) -> None:
    cfg = normalize_runtime_config({"env": {}})
    self.assertEqual(cfg["env"]["long_session_turn_threshold"], 70)

  def test_runtime_config_sets_default_resistance_enabled(self) -> None:
    cfg = normalize_runtime_config({"env": {}})
    self.assertTrue(cfg["env"]["resistance"])

  def test_runtime_config_sets_default_patient_prompt_ablation(self) -> None:
    cfg = normalize_runtime_config({"patient": {}})
    self.assertEqual(cfg["patient"]["prompt_variant"], "base")

  def test_runtime_config_preserves_disabled_resistance(self) -> None:
    cfg = normalize_runtime_config({"env": {"resistance": False}})
    self.assertFalse(cfg["env"]["resistance"])

  def test_runtime_config_rejects_unknown_patient_prompt_variant(self) -> None:
    with self.assertRaisesRegex(ValueError, "prompt_variant"):
      normalize_runtime_config({"patient": {"prompt_variant": "unknown"}})

  def test_runtime_config_rejects_patient_prompt_intervals_field(self) -> None:
    with self.assertRaisesRegex(ValueError, "prompt_intervals"):
      normalize_runtime_config({"patient": {"prompt_intervals": "3_with_calib"}})

  def test_disabled_resistance_overrides_resistance_handling_mode(self) -> None:
    builder = self._make_builder(resistance_handling="cami", resistance=False)

    self.assertEqual(builder._resistance_handling_mode(), "none")

  def test_runtime_config_sets_default_resistance_prediction_clip_values(self) -> None:
    cfg = normalize_runtime_config({"env": {}})

    self.assertEqual(cfg["env"]["resistance_prediction_clip_initial_turns"], 5)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_initial_min_delta"], -0.3)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_initial_max_delta"], 0.3)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_later_min_delta"], -0.15)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_later_max_delta"], 0.15)

  def test_runtime_config_preserves_custom_long_session_threshold(self) -> None:
    cfg = normalize_runtime_config({"env": {"long_session_turn_threshold": 55}})
    self.assertEqual(cfg["env"]["long_session_turn_threshold"], 55)

  def test_runtime_config_preserves_custom_resistance_prediction_clip_values(self) -> None:
    cfg = normalize_runtime_config(
      {
        "env": {
          "resistance_prediction_clip_initial_turns": 4,
          "resistance_prediction_clip_initial_min_delta": -0.25,
          "resistance_prediction_clip_initial_max_delta": 0.35,
          "resistance_prediction_clip_later_min_delta": -0.05,
          "resistance_prediction_clip_later_max_delta": 0.12,
        }
      }
    )

    self.assertEqual(cfg["env"]["resistance_prediction_clip_initial_turns"], 4)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_initial_min_delta"], -0.25)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_initial_max_delta"], 0.35)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_later_min_delta"], -0.05)
    self.assertEqual(cfg["env"]["resistance_prediction_clip_later_max_delta"], 0.12)

  def test_initial_state_uses_configured_long_session_threshold(self) -> None:
    builder = self._make_builder()
    builder.cfg.env.long_session_turn_threshold = 33

    state = builder.initial_state(max_turns=12)

    self.assertEqual(state["long_session_turn_threshold"], 33)

  def test_resistance_prediction_turn_uses_patient_message_count(self) -> None:
    builder = self._make_builder()
    state = builder.initial_state(max_turns=12)
    state["messages"] = [
      {"utterance_index": 1, "role": "therapist", "content": "T1", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 2, "role": "patient", "content": "P1", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 3, "role": "therapist", "content": "T2", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 4, "role": "patient", "content": "P2", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 5, "role": "therapist", "content": "T3", "stage": "opening", "substage": "open_mood"},
      {"utterance_index": 6, "role": "patient", "content": "P3", "stage": "opening", "substage": "open_mood"},
    ]
    state["reasoning_traces"] = []
    state = builder._ensure_state_defaults(state)

    self.assertEqual(builder._current_resistance_prediction_turn(state), 3)

  def test_resistance_prediction_turn_uses_existing_prediction_traces_as_fallback(self) -> None:
    builder = self._make_builder()
    state = builder.initial_state(max_turns=12)
    state["messages"] = [
      {"utterance_index": 1, "role": "patient", "content": "P1", "stage": "opening", "substage": "open_mood"},
    ]
    state["reasoning_traces"] = [
      {
        "utterance_index": index,
        "speaker": "therapist",
        "traces": [{"step": "resistance_prediction", "input": "x", "output": "{}"}],
      }
      for index in (2, 4, 6, 8, 10)
    ]
    state = builder._ensure_state_defaults(state)

    self.assertEqual(builder._current_resistance_prediction_turn(state), 6)

  def test_resume_state_preserves_saved_long_session_threshold(self) -> None:
    builder = self._make_builder()
    saved_state = builder.initial_state(max_turns=12)
    saved_state["long_session_turn_threshold"] = 44

    resumed = builder.resume_state(saved_state, max_turns=12)

    self.assertEqual(resumed["long_session_turn_threshold"], 44)

  def test_resume_state_restores_legacy_mi_recovery_streak(self) -> None:
    builder = self._make_builder()
    saved_state = builder.initial_state(max_turns=12)
    saved_state["current_stage"] = "mi_techniques"
    saved_state["current_substage"] = "mi_choose_technique"
    saved_state["messages"] = [
      {
        "utterance_index": 1,
        "role": "therapist",
        "content": "Let's step back.",
        "stage": "mi_techniques",
        "substage": "mi_raise_technique",
      },
      {
        "utterance_index": 2,
        "role": "patient",
        "content": "Maybe.",
        "stage": "mi_techniques",
        "substage": "mi_raise_technique",
      },
      {
        "utterance_index": 3,
        "role": "therapist",
        "content": "What feels a bit more open here?",
        "stage": "mi_techniques",
        "substage": "mi_raise_technique",
      },
      {
        "utterance_index": 4,
        "role": "patient",
        "content": "I guess there might be something to try.",
        "stage": "mi_techniques",
        "substage": "mi_raise_technique",
      },
    ]
    saved_state["reasoning_traces"] = [
      {
        "utterance_index": 3,
        "speaker": "therapist",
        "traces": [
          {
            "step": "resistance_prediction",
            "input": "prediction",
            "output": json.dumps({"new_p": 0.5}),
          }
        ],
      }
    ]

    resumed = builder.resume_state(saved_state, max_turns=12)

    self.assertEqual(resumed["mi_recovery_turns"], 1)
    self.assertEqual(resumed["cami_recovery_turns"], 0)

  def test_mi_timeout_instruction_applies_only_when_pending(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.8)
    state["current_substage"] = "mi_raise_technique"

    unchanged, used = builder._instruction_with_long_session_timeout(state, "base")
    self.assertEqual(unchanged, "base")
    self.assertFalse(used)

    state["long_session_timeout_instruction_pending"] = True
    enriched, used = builder._instruction_with_long_session_timeout(state, "base")

    self.assertIn("time is running short for this session", enriched)
    self.assertTrue(used)

  def test_hard_cap_closing_fallback_uses_timeout_instruction(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="closing", predicted_resistance=0.2)
    state["current_stage"] = "closing"
    state["current_substage"] = "closing_fallback"
    state["session_ended_reason"] = "max_turns_forced_closing"

    enriched, used = builder._instruction_with_long_session_timeout(state, "closing/fallback_utterance")

    self.assertTrue(used)
    self.assertIn("Close for today", enriched)

  def test_pre_stage_routing_moves_pending_mi_timeout_to_homework(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.8)
    state["long_session_mi_route_to_homework_pending"] = True

    builder._handle_pre_stage_routing(state, [])

    self.assertEqual(state["current_stage"], "homework")
    self.assertFalse(state["long_session_mi_route_to_homework_pending"])

  def test_mi_threshold_marks_timeout_exit_and_defers_homework_handoff(self) -> None:
    builder = self._make_builder()
    state = self._make_state(builder, stage="mi_techniques", predicted_resistance=0.8)
    state["long_session_turn_threshold"] = 5
    state["utterance_index"] = 5
    state["current_substage"] = "mi_choose_technique"
    captured: list[str] = []
    def _select_techniques(state: dict, traces: list[dict], **kwargs: str) -> list[str]:
      state["current_mi_technique_selection"] = ["Open Question"]
      state["mi_techniques_used"].append("Open Question")
      return ["Open Question"]
    builder._select_mi_techniques = _select_techniques
    def _generate(state: dict, instruction: str) -> str:
      captured.append(instruction)
      state["long_session_timeout_instruction_pending"] = False
      return instruction
    builder._generate_therapist_utterance = _generate

    result = builder._handle_mi_techniques(state, [])

    self.assertEqual(result, "mi_techniques/raise_technique_utterance")
    self.assertEqual(captured, ["mi_techniques/raise_technique_utterance"])
    self.assertTrue(state["long_session_mi_route_to_homework_pending"])
    self.assertFalse(state["long_session_timeout_instruction_pending"])

  def test_cami_threshold_marks_timeout_exit_and_defers_homework_handoff(self) -> None:
    builder = self._make_builder(resistance_handling="cami")
    state = self._make_state(builder, stage="mi_techniques_with_te", predicted_resistance=0.8)
    state["long_session_turn_threshold"] = 5
    state["utterance_index"] = 5
    state["mi_topic_stack"] = ["Work"]
    state["current_mi_topic"] = "Work"
    builder._select_next_mi_topic = lambda *args, **kwargs: None
    builder._mi_topic_exploration_exhausted = lambda *args, **kwargs: False
    def _select_techniques(state: dict, traces: list[dict], **kwargs: str) -> list[str]:
      state["current_mi_technique_selection"] = ["Open Question"]
      state["mi_techniques_used"].append("Open Question")
      return ["Open Question"]
    builder._select_mi_techniques = _select_techniques
    def _generate(state: dict, instruction: str) -> str:
      state["long_session_timeout_instruction_pending"] = False
      return instruction
    builder._generate_therapist_utterance = _generate

    result = builder._handle_mi_techniques_with_te(state, [])

    self.assertEqual(result, "mi_techniques_with_te/raise_utterance")
    self.assertTrue(state["long_session_mi_route_to_homework_pending"])
    self.assertFalse(state["long_session_timeout_instruction_pending"])


if __name__ == "__main__":
  unittest.main()
