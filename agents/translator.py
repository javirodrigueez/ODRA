"""EN<->ZH utterance translator backed by a vLLM-served TranslateGemma (google/translategemma-*-it).

TranslateGemma carries the language pair in `source_lang_code` / `target_lang_code` keys *inside* the
user content part; its bundled `chat_template.jinja` builds the translation instruction from them.
That does not survive vLLM's OpenAI chat endpoint: `_parse_chat_message_content_part`
(vllm/entrypoints/chat_utils.py) rebuilds every text part as `{"type": "text", "text": ...}` and drops
all other keys before the template runs, so the template resolves an empty language code and the
request fails with HTTP 400 `'dict object' has no attribute ''`. The stripping is unconditional for
text parts -- `--chat-template-content-format openai|string|auto` does not change it -- and a plain
string message fails too, because the template also requires a one-item iterable content. So neither
chat-endpoint shape can work with this model.

Fix: render the model's own chat template client-side (PROMPT_TEMPLATE below reproduces
`chat_template.jinja` byte for byte for a single text turn) and send it to `/v1/completions`, which
tokenizes the literal string and does no content-part rewriting. `add_special_tokens: False` is
required because the rendered text already starts with `<bos>`; leaving it at vLLM's default `True`
would prepend a second one. Verified token-identical to `apply_chat_template(..., tokenize=True)`.

Input context is only 4K tokens, so long texts are translated paragraph by paragraph.
"""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI  # pyright: ignore[reportMissingImports]

from .interfaces import ensure_str

# Language names exactly as google/translategemma-12b-it's chat_template.jinja `languages` map has them.
LANGUAGE_NAMES = {"en": "English", "zh": "Chinese"}
# Byte-for-byte reproduction of chat_template.jinja rendered with add_generation_prompt=True for a
# single {"type": "text", ...} user turn. Do not reword: this is the model's own instruction text.
PROMPT_TEMPLATE = (
  "<bos><start_of_turn>user\n"
  "You are a professional {source_lang} ({source}) to {target_lang} ({target}) translator. "
  "Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
  "while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
  "Produce only the {target_lang} translation, without any additional explanations or commentary. "
  "Please translate the following {source_lang} text into {target_lang}:\n\n\n"
  "{text}<end_of_turn>\n"
  "<start_of_turn>model\n"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")


def chunk_text(text: str, max_chars: int) -> list[str]:
  """Split into paragraph chunks of at most max_chars (falling back to sentence splits)."""
  chunks: list[str] = []
  for paragraph in text.split("\n"):
    paragraph = paragraph.strip()
    if not paragraph:
      continue
    if len(paragraph) <= max_chars:
      chunks.append(paragraph)
      continue
    current = ""
    for sentence in SENTENCE_SPLIT_RE.split(paragraph):
      if not sentence:
        continue
      if current and len(current) + len(sentence) > max_chars:
        chunks.append(current)
        current = sentence
      else:
        current = f"{current}{sentence}" if current else sentence
    if current:
      chunks.append(current)
  return chunks


class Translator:
  def __init__(
    self,
    base_url: str,
    model_id: str,
    max_tokens: int = 1024,
    chunk_chars: int = 1200,
    timeout: float = 300,
  ):
    self.model_id = model_id
    self.max_tokens = int(max_tokens)
    self.chunk_chars = int(chunk_chars)
    self.client = OpenAI(api_key="test", base_url=base_url, timeout=timeout, max_retries=3)

  @classmethod
  def from_config(cls, config: dict[str, Any] | None) -> "Translator":
    config = dict(config or {})
    if not config.get("base_url") or not config.get("model_id"):
      raise ValueError("Config block 'translator' requires base_url and model_id.")
    return cls(
      base_url=str(config["base_url"]),
      model_id=str(config["model_id"]),
      max_tokens=int(config.get("max_tokens") or 1024),
      chunk_chars=int(config.get("chunk_chars") or 1200),
      timeout=float(config.get("timeout") or 300),
    )

  def build_prompt(self, text: str, source: str, target: str) -> str:
    return PROMPT_TEMPLATE.format(
      source=source,
      target=target,
      source_lang=LANGUAGE_NAMES.get(source, source),
      target_lang=LANGUAGE_NAMES.get(target, target),
      text=text.strip(),  # the template applies `| trim` to the text
    )

  def _translate_chunk(self, text: str, source: str, target: str) -> str:
    response = self.client.completions.create(
      model=self.model_id,
      prompt=self.build_prompt(text, source, target),
      temperature=0.0,
      max_tokens=self.max_tokens,
      extra_body={"add_special_tokens": False},  # the rendered prompt already carries <bos>
    )
    return ensure_str(response.choices[0].text).strip()

  def translate(self, text: str, source: str, target: str) -> str:
    chunks = chunk_text(ensure_str(text), self.chunk_chars)
    return "\n".join(self._translate_chunk(chunk, source, target) for chunk in chunks).strip()

  def en_to_zh(self, text: str) -> str:
    return self.translate(text, "en", "zh")

  def zh_to_en(self, text: str) -> str:
    return self.translate(text, "zh", "en")
