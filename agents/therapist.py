from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_core.prompts import PromptTemplate

from .interfaces import Agent, LLM, StructuredOutputError, load_prompt, parse_json_response


class TherapistAgent(Agent):
  def __init__(self, llm: LLM, profile: dict):
    super().__init__(llm)
    self.profile = profile
    prompt_path = Path(__file__).parent.parent / "prompts" / "therapist_system.txt"
    self.system_prompt = self.load_prompt(prompt_path)

  def _build_user_prompt(
    self,
    instruction: str,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    suffix: str,
  ) -> str:
    return (
      f"<full_conversation>\n{full_conversation}\n</full_conversation>\n\n"
      f"<last_utterances>\n{last_utterances}\n</last_utterances>\n\n"
      f"<last_reasoning>\n{last_reasoning}\n</last_reasoning>\n\n"
      f"<session_context>\n{session_context or 'N/A'}\n</session_context>\n\n"
      f"Instruction: {instruction}\n"
      f"{suffix}"
    )

  def generate_utterance(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
    max_tokens: int = 5000,
    temperature: float = 0.7,
  ) -> str:
    suffix_path = Path(__file__).parent.parent / "prompts" / "shared" / "utterance_suffix.txt"
    messages = [
      {"role": "system", "content": self.system_prompt},
      {
        "role": "user",
        "content": self._build_user_prompt(
          instruction=instruction,
          full_conversation=full_conversation,
          last_utterances=last_utterances,
          last_reasoning=last_reasoning,
          session_context=session_context,
          suffix=self.load_prompt(suffix_path),
        ),
      },
    ]
    return self.llm.generate(messages, max_tokens=max_tokens, temperature=temperature).strip()

  def generate_structured(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
    max_retries: int = 2,
    expected_type: Optional[type | tuple[type, ...]] = dict,
  ) -> tuple[Any, dict]:
    suffix_path = Path(__file__).parent.parent / "prompts" / "shared" / "structured_suffix.txt"
    messages = [
      {"role": "system", "content": self.system_prompt},
      {
        "role": "user",
        "content": self._build_user_prompt(
          instruction=instruction,
          full_conversation=full_conversation,
          last_utterances=last_utterances,
          last_reasoning=last_reasoning,
          session_context=session_context,
          suffix=self.load_prompt(suffix_path),
        ),
      },
    ]

    attempts = 0
    last_raw = ""
    last_error = "invalid response"
    for _ in range(max_retries + 1):
      attempts += 1
      raw = self.llm.generate(messages, temperature=0.2, max_tokens=5000)
      last_raw = raw or ""
      try:
        parsed, parse_meta = parse_json_response(last_raw)
      except StructuredOutputError as exc:
        last_error = str(exc)
      else:
        if expected_type is None or isinstance(parsed, expected_type):
          return parsed, {
            "raw": last_raw,
            "attempts": attempts,
            "parse_mode": parse_meta.get("mode", "direct"),
          }
        expected_name = getattr(expected_type, "__name__", str(expected_type))
        last_error = f"expected {expected_name}, got {type(parsed).__name__}"
      messages.append(
        {
          "role": "user",
          "content": (
            "Your previous response was not valid JSON with the requested schema. "
            "Reply again with JSON only."
          ),
        }
      )

    raise StructuredOutputError(
      f"Therapist structured response failed after {attempts} attempts: {last_error}. Last raw response: {last_raw!r}"
    )

  # def generate(self, context: str, wrap_up: bool = False) -> str:
  #   instruction = "Continue the therapy session naturally."
  #   if wrap_up:
  #     instruction += " Begin wrapping up the session naturally."
  #   return self.generate_utterance(
  #     conversation=context,
  #     current_stage="closing" if wrap_up else "opening",
  #     session_context="legacy_mode",
  #     instruction=instruction,
  #   )

  # def generateOpening(self, context: str = "") -> str:
  #   return self.generate_utterance(
  #     conversation=context,
  #     current_stage="opening",
  #     session_context="legacy_mode",
  #     instruction="Produce a brief welcome greeting to start the therapy session.",
  #   )
