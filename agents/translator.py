"""EN<->ZH utterance translator backed by a vLLM-served TranslateGemma (google/translategemma-*-it).

TranslateGemma expects a single user message whose content is a one-entry list carrying
`source_lang_code` / `target_lang_code` (its chat template renders the translation instruction from
those fields). If the deployed vLLM version strips those extra keys from content parts, switch the
config to `prompt_mode: manual`, which renders the same instruction as a plain string message.
Input context is only 2K tokens, so long texts are translated paragraph by paragraph.
"""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI  # pyright: ignore[reportMissingImports]

from .interfaces import ensure_str

LANGUAGE_NAMES = {"en": "English", "zh": "Chinese"}
MANUAL_PROMPT_TEMPLATE = (
  "You are a professional {source} to {target} translator. Your goal is to accurately convey the meaning "
  "and nuances of the original {source} text while adhering to {target} grammar, vocabulary, and cultural "
  "sensitivities. Produce only the {target} translation, without any additional explanations or commentary.\n\n"
  "Please translate the following {source} text into {target}:\n\n{text}"
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
    prompt_mode: str = "chat_template",
    chunk_chars: int = 1200,
    timeout: float = 300,
  ):
    self.model_id = model_id
    self.max_tokens = int(max_tokens)
    self.prompt_mode = prompt_mode
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
      prompt_mode=str(config.get("prompt_mode") or "chat_template"),
      chunk_chars=int(config.get("chunk_chars") or 1200),
      timeout=float(config.get("timeout") or 300),
    )

  def build_messages(self, text: str, source: str, target: str) -> list[dict[str, Any]]:
    if self.prompt_mode == "manual":
      prompt = MANUAL_PROMPT_TEMPLATE.format(
        source=LANGUAGE_NAMES.get(source, source),
        target=LANGUAGE_NAMES.get(target, target),
        text=text,
      )
      return [{"role": "user", "content": prompt}]
    return [
      {
        "role": "user",
        "content": [
          {"type": "text", "source_lang_code": source, "target_lang_code": target, "text": text},
        ],
      }
    ]

  def _translate_chunk(self, text: str, source: str, target: str) -> str:
    response = self.client.chat.completions.create(
      model=self.model_id,
      messages=self.build_messages(text, source, target),
      temperature=0.0,
      max_tokens=self.max_tokens,
    )
    return ensure_str(response.choices[0].message.content)

  def translate(self, text: str, source: str, target: str) -> str:
    chunks = chunk_text(ensure_str(text), self.chunk_chars)
    return "\n".join(self._translate_chunk(chunk, source, target) for chunk in chunks).strip()

  def en_to_zh(self, text: str) -> str:
    return self.translate(text, "en", "zh")

  def zh_to_en(self, text: str) -> str:
    return self.translate(text, "zh", "en")
