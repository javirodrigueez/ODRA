from __future__ import annotations

from pathlib import Path

from typing import Any

from langchain_core.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]

from .interfaces import Agent, ChatMessage, LLM, StructuredOutputError, load_prompt, parse_json_response

class PatientAgent(Agent):
  def __init__(
    self,
    llm: LLM,
    profile: dict[str, Any],
    generation_defaults: dict[str, Any] | None = None,
  ):
    super().__init__(
      llm,
      generation_defaults=generation_defaults,
    )
    self.profile = profile
    system_path = Path(__file__).parent.parent / "prompts" / "patient_system.txt"
    self.system_prompt = self._build_system_prompt(system_path)

  def _build_system_prompt(self, filename: str | Path) -> str:
    template = load_prompt(filename)
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(
        persona=self._format_persona(),
        more_info=self._format_more_info(),
        patterns=self._format_patterns(),
        presenting_problem=self.profile["presenting_problem"],
        reason_seeking_counseling=self.profile["reason_seeking_counseling"],
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

  def _resistance_prompt_name(self, resistance_level: float) -> str:
    if resistance_level < 0.20:
      return "r_very_low"
    if resistance_level < 0.40:
      return "r_low"
    if resistance_level < 0.60:
      return "r_medium"
    if resistance_level < 0.80:
      return "r_high"
    return "r_very_high"

  def _prompt_resistance_level(self, resistance_level: float) -> float:
    return max(0, min(10, float(resistance_level) * 10))

  def _render_resistance_prompt(self, resistance_level: float) -> str:
    prompt_name = self._resistance_prompt_name(float(resistance_level))
    template = load_prompt(f"patient_resistance/{prompt_name}.txt")
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(
      resistance_level=self._prompt_resistance_level(round(resistance_level)),
    )

  def _build_user_prompt(
    self,
    instruction: str,
    full_conversation: str,
    last_utterances: str,
    resistance_level: float,
    suffix: str,
    structured: bool = False,
  ) -> str:
    resistance_prompt = self._render_resistance_prompt(resistance_level) if structured == False else f"Your resistance level towards therapy (bounded in the interval 0.0,1.0) is: {resistance_level:.2f}"
    return (
      f"<full_conversation>\n{full_conversation}\n</full_conversation>\n\n"
      f"<last_utterances>\n{last_utterances}\n</last_utterances>\n\n"
      f"<resistance_level>\n{resistance_prompt}\n</resistance_level>\n\n"
      f"Instruction: {instruction}\n"
      f"{suffix}"
    )

  def build_utterance_messages(
    self,
    full_conversation: str,
    last_utterances: str,
    resistance_level: float,
    instruction: str,
  ) -> list[ChatMessage]:
    suffix = load_prompt("shared/patient_utterance_suffix.txt")
    user_prompt = self._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      resistance_level=resistance_level,
      suffix=suffix,
    )
    return [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": user_prompt},
    ]

  def build_structured_messages(
    self,
    full_conversation: str,
    last_utterances: str,
    resistance_level: float,
    instruction: str,
  ) -> list[ChatMessage]:
    suffix = load_prompt("shared/patient_structured_suffix.txt")
    user_prompt = self._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      resistance_level=resistance_level,
      suffix=suffix,
      structured=True,
    )
    return [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": user_prompt},
    ]

  def generate_utterance(
    self,
    full_conversation: str,
    last_utterances: str,
    resistance_level: float,
    instruction: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> str:
    messages = self.build_utterance_messages(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      resistance_level=resistance_level,
      instruction=instruction,
    )
    generation_kwargs = self._build_generation_kwargs(
      default_values={"max_tokens": 5000, "temperature": 0.8},
      override_values={
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
      },
    )
    return self.llm.generate(messages, **generation_kwargs).strip()

  def generate_structured(
    self,
    full_conversation: str,
    last_utterances: str,
    resistance_level: float,
    instruction: str,
    max_retries: int = 2,
    expected_type: type[Any] | tuple[type[Any], ...] | None = dict,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> tuple[Any, dict[str, Any]]:
    retry_suffix = load_prompt("shared/patient_structured_suffix.txt")
    messages = self.build_structured_messages(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      resistance_level=resistance_level,
      instruction=instruction,
    )

    attempts = 0
    last_raw = ""
    last_error = "invalid response"
    for _ in range(max_retries + 1):
      attempts += 1
      generation_kwargs = self._build_generation_kwargs(
        default_values={"max_tokens": 8192, "temperature": 0.2},
        override_values={
          "max_tokens": max_tokens,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "min_p": min_p,
          "presence_penalty": presence_penalty,
          "frequency_penalty": frequency_penalty,
          "repetition_penalty": repetition_penalty,
        },
        include_generation_defaults=False,
      )
      raw = self.llm.generate(messages, **generation_kwargs)
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
          "content": self._build_user_prompt(
            instruction=(
              f"{instruction}\n\n"
              "Your previous response was not valid JSON with the requested schema. Reply again."
            ),
            full_conversation=full_conversation,
            last_utterances=last_utterances,
            resistance_level=resistance_level,
            suffix=retry_suffix,
            structured=True,
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
