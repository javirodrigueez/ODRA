from __future__ import annotations

from typing import Any

from .interfaces import Agent, ChatMessage, LLM, StructuredOutputError, load_prompt, parse_json_response, render_prompt


class TherapistAgent(Agent):
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
    self.system_prompt = render_prompt(
      "therapist_system.txt",
      patient_intake_form=self._format_patient_intake_form(),
    )

  def _format_patient_intake_form(self) -> str:
    personal_info = self.profile.get("personal_info", {})
    more_info = self.profile.get("more_information", {})

    sections: list[str] = []
    if isinstance(personal_info, dict) and personal_info:
      personal_lines = [
        f"{str(key).replace('_', ' ').title()}: {value}"
        for key, value in personal_info.items()
      ]
      sections.append("Personal Info:\n" + "\n".join(personal_lines))

    reason = self.profile.get("reason_seeking_counseling", "")
    if reason:
      sections.append(f"Reason Seeking Counseling:\n{reason}")

    if isinstance(more_info, dict) and more_info:
      more_info_lines = [
        f"{str(key).replace('_', ' ').title()}:\n{value}"
        for key, value in more_info.items()
        if value
      ]
      if more_info_lines:
        sections.append("More Information:\n" + "\n\n".join(more_info_lines))

    return "\n\n".join(sections)

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
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> str:
    suffix = load_prompt("shared/utterance_suffix.txt")
    messages: list[ChatMessage] = [
      {"role": "system", "content": self.system_prompt},
      {
        "role": "user",
        "content": self._build_user_prompt(
          instruction=instruction,
          full_conversation=full_conversation,
          last_utterances=last_utterances,
          last_reasoning=last_reasoning,
          session_context=session_context,
          suffix=suffix,
        ),
      },
    ]
    generation_kwargs = self._build_generation_kwargs(
      default_values={"max_tokens": 8192, "temperature": 0.7},
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
    last_reasoning: str,
    session_context: str,
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
    suffix = load_prompt("shared/structured_suffix.txt")
    retry_suffix = load_prompt("shared/structured_suffix.txt")
    messages: list[ChatMessage] = [
      {"role": "system", "content": self.system_prompt},
      {
        "role": "user",
        "content": self._build_user_prompt(
          instruction=instruction,
          full_conversation=full_conversation,
          last_utterances=last_utterances,
          last_reasoning=last_reasoning,
          session_context=session_context,
          suffix=suffix,
        ),
      },
    ]

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
            last_reasoning=last_reasoning,
            session_context=session_context,
            suffix=retry_suffix,
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
