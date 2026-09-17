import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same optional-dependency stubs as the other repo tests (langchain stack is not needed here).
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
        rendered = rendered.replace(f"{{{key}}}", str(value))
      return rendered

  langchain_core_prompts_module.PromptTemplate = _PromptTemplate
  sys.modules["langchain_core"] = langchain_core_module
  sys.modules["langchain_core.prompts"] = langchain_core_prompts_module

if "langchain_huggingface" not in sys.modules:
  langchain_huggingface_module = types.ModuleType("langchain_huggingface")
  langchain_huggingface_module.ChatHuggingFace = object
  langchain_huggingface_module.HuggingFacePipeline = object
  sys.modules["langchain_huggingface"] = langchain_huggingface_module

from agents.translator import MANUAL_PROMPT_TEMPLATE, Translator, chunk_text  # noqa: E402
from graph.zh_therapist import PSYCHE_R1_THINK_PREFIX, ZhTherapistGraphBuilder  # noqa: E402


PROFILE = {"idx": "1", "attitude": "neutral", "personal_info": {"name": "Alex"}}
SYSTEM_PROMPT = "系统提示"
GREETING_PROMPT = "请生成开场问候。"


class FakeTranslator:
  def __init__(self) -> None:
    self.calls: list[tuple[str, str, str]] = []

  def translate(self, text: str, source: str, target: str) -> str:
    self.calls.append((text, source, target))
    return f"[{source}->{target}] {text}"

  def en_to_zh(self, text: str) -> str:
    return self.translate(text, "en", "zh")

  def zh_to_en(self, text: str) -> str:
    return self.translate(text, "zh", "en")


class FakeZhModel:
  def __init__(self, replies: list[str]) -> None:
    self.replies = list(replies)
    self.requests: list[list[dict]] = []

  def __call__(self, messages: list[dict]) -> str:
    self.requests.append(messages)
    return self.replies.pop(0)


def make_builder(variant: str, replies: list[str], resistance: bool = False, history_limit: int = 0) -> ZhTherapistGraphBuilder:
  builder = ZhTherapistGraphBuilder.__new__(ZhTherapistGraphBuilder)
  builder.cfg = SimpleNamespace(env={"resistance": resistance}, patient={})
  builder.profile = PROFILE
  builder.cactus_patient = False
  builder.zh_variant = variant
  builder.zh_history_limit = history_limit
  builder.zh_system_prompt = SYSTEM_PROMPT
  builder.zh_greeting_prompt = GREETING_PROMPT
  builder.translator = FakeTranslator()
  builder.zh_model = FakeZhModel(replies)
  builder._generate_zh = builder.zh_model
  builder.moderator_inputs = []
  builder._moderator_detects_closing = lambda utterance: builder.moderator_inputs.append(utterance) or False
  return builder


def state_with(messages: list[dict], max_turns: int = 50) -> dict:
  return {
    "messages": messages,
    "reasoning_traces": [],
    "patient_profile": PROFILE,
    "utterance_index": len(messages),
    "max_turns": max_turns,
    "min_turns": 0,
  }


class TranslatorTest(unittest.TestCase):
  def test_chunk_text_keeps_short_paragraphs_and_splits_long_ones_by_sentence(self) -> None:
    self.assertEqual(chunk_text("one\n\ntwo", 100), ["one", "two"])
    chunks = chunk_text("Aaa aaa. Bbb bbb! Ccc ccc?", 12)
    self.assertEqual(chunks, ["Aaa aaa.", "Bbb bbb!", "Ccc ccc?"])
    self.assertEqual(chunk_text("第一句。第二句！", 4), ["第一句。", "第二句！"])

  def test_build_messages_chat_template_and_manual_modes(self) -> None:
    translator = Translator.__new__(Translator)
    translator.prompt_mode = "chat_template"
    messages = translator.build_messages("hello", "en", "zh")
    self.assertEqual(
      messages,
      [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "zh", "text": "hello"}]}],
    )
    translator.prompt_mode = "manual"
    manual = translator.build_messages("hello", "zh", "en")
    self.assertEqual(manual[0]["content"], MANUAL_PROMPT_TEMPLATE.format(source="Chinese", target="English", text="hello"))

  def test_translate_joins_chunk_translations(self) -> None:
    translator = Translator.__new__(Translator)
    translator.chunk_chars = 5
    translator._translate_chunk = lambda text, source, target: text.upper()
    self.assertEqual(translator.translate("abc\ndef", "en", "zh"), "ABC\nDEF")
    self.assertEqual(translator.translate("   ", "en", "zh"), "")


class ZhRequestBuildingTest(unittest.TestCase):
  def test_soulchat2_greeting_request_has_system_and_plain_greeting_cue(self) -> None:
    builder = make_builder("soulchat2", [])
    messages = builder._build_zh_request_messages(builder._ensure_state_defaults(state_with([])))
    self.assertEqual(messages, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": GREETING_PROMPT}])

  def test_psyche_r1_wraps_every_user_turn_and_keeps_answers_only(self) -> None:
    builder = make_builder("psyche_r1", [])
    messages = builder._build_zh_request_messages(builder._ensure_state_defaults(state_with([])))
    self.assertEqual(messages[1]["content"], f"{PSYCHE_R1_THINK_PREFIX}{GREETING_PROMPT}")

    history = [
      {"utterance_index": 1, "role": "therapist", "content": "Hello.", "content_zh": "你好。"},
      {"utterance_index": 2, "role": "patient", "content": "I feel down.", "content_zh": "我很低落。"},
      {"utterance_index": 3, "role": "therapist", "content": "Tell me more.", "content_zh": "多说一些。"},
      {"utterance_index": 4, "role": "patient", "content": "Work is hard.", "content_zh": "工作很难。"},
    ]
    messages = builder._build_zh_request_messages(builder._ensure_state_defaults(state_with(history)))
    self.assertEqual([m["role"] for m in messages], ["system", "assistant", "user", "assistant", "user"])
    self.assertEqual(messages[1]["content"], "你好。")
    self.assertEqual(messages[2]["content"], f"{PSYCHE_R1_THINK_PREFIX}我很低落。")
    self.assertEqual(messages[4]["content"], f"{PSYCHE_R1_THINK_PREFIX}工作很难。")

  def test_history_limit_keeps_last_messages_only(self) -> None:
    builder = make_builder("soulchat2", [], history_limit=2)
    history = [
      {"utterance_index": 1, "role": "therapist", "content": "Hello.", "content_zh": "你好。"},
      {"utterance_index": 2, "role": "patient", "content": "I feel down.", "content_zh": "我很低落。"},
      {"utterance_index": 3, "role": "therapist", "content": "Tell me more.", "content_zh": "多说一些。"},
      {"utterance_index": 4, "role": "patient", "content": "Work is hard.", "content_zh": "工作很难。"},
    ]
    messages = builder._build_zh_request_messages(builder._ensure_state_defaults(state_with(history)))
    self.assertEqual([m["content"] for m in messages[1:]], ["多说一些。", "工作很难。"])

  def test_normalize_messages_preserves_content_zh(self) -> None:
    builder = make_builder("soulchat2", [])
    state = builder._ensure_state_defaults(
      state_with(
        [
          {"utterance_index": 1, "role": "therapist", "content": "Hello.", "content_zh": "你好。"},
          {"utterance_index": 2, "role": "patient", "content": "Hi."},
          {"role": "narrator", "content": "ignored"},
        ]
      )
    )
    self.assertEqual(state["messages"][0]["content_zh"], "你好。")
    self.assertNotIn("content_zh", state["messages"][1])
    self.assertEqual(len(state["messages"]), 2)


class ZhTherapistTurnTest(unittest.TestCase):
  def test_greeting_turn_translates_and_stores_original(self) -> None:
    builder = make_builder("soulchat2", ["你好，很高兴见到你。"])
    state = builder.therapist_turn(state_with([]))
    message = state["messages"][-1]
    self.assertEqual(message["content"], "[zh->en] 你好，很高兴见到你。")
    self.assertEqual(message["content_zh"], "你好，很高兴见到你。")
    self.assertEqual(builder.moderator_inputs, ["[zh->en] 你好，很高兴见到你。"])
    self.assertEqual(state["reasoning_traces"][0]["traces"][0]["output"], "")
    self.assertFalse(state["session_ended"])

  def test_patient_utterances_are_translated_once_and_reused(self) -> None:
    builder = make_builder("soulchat2", ["回复一", "回复二"])
    state = builder.therapist_turn(
      state_with(
        [
          {"utterance_index": 1, "role": "therapist", "content": "Hello.", "content_zh": "你好。"},
          {"utterance_index": 2, "role": "patient", "content": "I feel down."},
        ]
      )
    )
    self.assertEqual(state["messages"][1]["content_zh"], "[en->zh] I feel down.")
    self.assertEqual(builder.zh_model.requests[0][2]["content"], "[en->zh] I feel down.")

    state["messages"].append({"utterance_index": 4, "role": "patient", "content": "Work is hard."})
    state["utterance_index"] = 4
    state = builder.therapist_turn(state)
    en_to_zh_calls = [call for call in builder.translator.calls if call[1] == "en"]
    self.assertEqual([call[0] for call in en_to_zh_calls], ["I feel down.", "Work is hard."])
    self.assertEqual(state["messages"][1]["content_zh"], "[en->zh] I feel down.")
    self.assertEqual(state["messages"][-1]["content_zh"], "回复二")

  def test_psyche_r1_reasoning_is_kept_in_chinese_and_stripped_from_utterance(self) -> None:
    builder = make_builder("psyche_r1", ["<think>来访者情绪低落。</think>\n我听到你最近很辛苦。"])
    state = builder.therapist_turn(state_with([]))
    self.assertEqual(state["messages"][-1]["content_zh"], "我听到你最近很辛苦。")
    self.assertEqual(state["messages"][-1]["content"], "[zh->en] 我听到你最近很辛苦。")
    self.assertEqual(state["reasoning_traces"][0]["traces"][0]["output"], "来访者情绪低落。")
    zh_to_en_calls = [call for call in builder.translator.calls if call[1] == "zh"]
    self.assertEqual(len(zh_to_en_calls), 1)

  def test_empty_utterance_retries_once_then_errors(self) -> None:
    builder = make_builder("psyche_r1", ["<think>只有推理</think>", "<think>还是只有推理</think>"])
    state = builder.therapist_turn(state_with([]))
    self.assertEqual(len(builder.zh_model.requests), 2)
    self.assertTrue(state["session_ended"])
    self.assertEqual(state["session_ended_reason"], "conversation_error")
    self.assertIn("empty utterance", state["conversation_error"])

  def test_moderator_closing_sets_pending_end(self) -> None:
    builder = make_builder("soulchat2", ["今天就到这里，再见。"])
    builder._moderator_detects_closing = lambda utterance: True
    state = builder.therapist_turn(state_with([{"utterance_index": 1, "role": "patient", "content": "Thanks."}]))
    self.assertTrue(state["final_patient_pending_end"])
    self.assertEqual(state["session_ended_reason"], "moderator_detected_closing")
    self.assertFalse(state["session_ended"])


if __name__ == "__main__":
  unittest.main()
