from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_core.prompts import PromptTemplate

from .interfaces import Agent, LLM, StructuredOutputError, load_prompt, parse_json_response

ATTITUDE_DESCRIPTIONS = {
  "negative": "Express reluctance, skepticism, or emotional resistance toward the therapeutic process",
  "positive": "Be open, cooperative, and willing to engage with the therapeutic process",
  "neutral": "Be balanced and measured in your emotional responses",
}

class PatientAgent(Agent):
  def __init__(self, llm: LLM, profile: dict):
    super().__init__(llm)
    self.profile = profile
    prompt_filename = "patient_system.txt"
    if not (Path(__file__).parent.parent / "prompts" / prompt_filename).exists():
      prompt_filename = "patient_system.txt"
    self._system_template = self._build_template(prompt_filename)

  def _build_template(self, filename: str) -> PromptTemplate:
    template = load_prompt(filename)
    prompt_template = PromptTemplate.from_template(template)
    attitude = self.profile.get("attitude", "neutral")
    return prompt_template.partial(
      persona=self._format_persona(),
      more_info=self._format_more_info(),
      patterns=self._format_patterns(),
      presenting_problem=self.profile["presenting_problem"],
      reason_seeking_counseling=self.profile["reason_seeking_counseling"],
      attitude=attitude,
      attitude_description=ATTITUDE_DESCRIPTIONS.get(attitude, "Respond naturally"),
    )

  def _format_persona(self) -> str:
    info = self.profile["personal_info"]
    lines = []
    for key, val in info.items():
      label = key.replace("_", " ").title()
      lines.append(f"{label}: {val}")
    return "\n".join(lines)

  def _format_more_info(self) -> str:
    info = self.profile["more_information"]
    sections = [
      ("Past History", info.get("past_history", "")),
      ("Academic/Occupational Functioning", info.get("academic_occupational_functioning", "")),
      ("Social Support System", info.get("social_support_system", "")),
    ]
    return "\n\n".join(f"{title}:\n{content}" for title, content in sections if content)

  def _format_patterns(self) -> str:
    return "\n".join(f"- {p}" for p in self.profile["patterns"])

  def _get_system_prompt(self, conversation: str, resistance_level: float = 0.5, current_stage: str = "opening") -> str:
    return self._system_template.format(
      conversation=conversation,
      resistance_level=f"{resistance_level:.2f}",
      current_stage=current_stage,
    )

  def generate_utterance(
    self,
    conversation: str,
    resistance_level: float,
    instruction: str,
    current_stage: str = "opening",
    max_tokens: int = 5000,
    temperature: float = 0.8,
  ) -> str:
    system_prompt = self._get_system_prompt(conversation, resistance_level=resistance_level, current_stage=current_stage)
    messages = [
      {"role": "system", "content": system_prompt},
      {
        "role": "user",
        "content": (
          f"{instruction}\n\n"
          "Produce only the patient's next utterance. Do not output JSON, analysis, or speaker labels."
        ),
      },
    ]
    return self.llm.generate(messages, temperature=temperature, max_tokens=max_tokens).strip()

  def generate_structured(
    self,
    conversation: str,
    resistance_level: float,
    instruction: str,
    current_stage: str = "opening",
    max_retries: int = 2,
    expected_type: Optional[type | tuple[type, ...]] = dict,
  ) -> tuple[Any, dict]:
    system_prompt = self._get_system_prompt(conversation, resistance_level=resistance_level, current_stage=current_stage)
    messages = [
      {"role": "system", "content": system_prompt},
      {
        "role": "user",
        "content": (
          f"{instruction}\n\n"
          "Respond with valid JSON only. No markdown fences, no prose outside JSON."
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
      f"Patient structured response failed after {attempts} attempts: {last_error}. Last raw response: {last_raw!r}"
    )

  # def generate(self, context: str) -> str:
  #   return self.generate_utterance(
  #     conversation=context,
  #     resistance_level=0.5,
  #     current_stage="opening",
  #     instruction="Continue the conversation naturally.",
  #   )

  # def generateOpening(self, context: str) -> str:
  #   return self.generate_utterance(
  #     conversation=context,
  #     resistance_level=0.5,
  #     current_stage="opening",
  #     instruction="Your therapist just greeted you. Respond briefly (1-2 sentences).",
  #   )

  # def generateClosing(self, context: str) -> str:
  #   return self.generate_utterance(
  #     conversation=context,
  #     resistance_level=0.5,
  #     current_stage="closing",
  #     instruction="The therapist is ending the session. Produce a brief closing response.",
  #   )
