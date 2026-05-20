from __future__ import annotations

from pathlib import Path

from typing import Any

from langchain_core.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]

from graph.resistance import (
  DEFAULT_PROMPT_VARIANT,
  normalize_prompt_variant,
  resistance_prompt_name_for_variant,
  resistance_prompt_relative_path_for_level,
  scaled_resistance_value,
)

from .interfaces import Agent, ChatMessage, LLM, StructuredOutputError, load_prompt, parse_json_response

class PatientAgent(Agent):
  def __init__(
    self,
    llm: LLM,
    profile: dict[str, Any],
    generation_defaults: dict[str, Any] | None = None,
    resistance_enabled: bool = True,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    utterance_system_prompt_path: str | Path | None = None,
    utterance_user_prompt_path: str | Path | None = None,
  ):
    super().__init__(
      llm,
      generation_defaults=generation_defaults,
    )
    self.profile = profile
    self.resistance_enabled = bool(resistance_enabled)
    normalized_prompt_variant = normalize_prompt_variant(prompt_variant)
    if normalized_prompt_variant is None:
      raise ValueError(f"Unknown patient prompt_variant: {prompt_variant!r}")
    self.prompt_variant = normalized_prompt_variant
    system_filename = "patient_system.txt" if self.resistance_enabled else "patient_system_weak.txt"
    system_structured_filename = (
      "patient_system_structured.txt" if self.resistance_enabled else "patient_system_structured_weak.txt"
    )
    system_path = (
      Path(utterance_system_prompt_path)
      if utterance_system_prompt_path is not None
      else Path(__file__).parent.parent / "prompts" / system_filename
    )
    system_structured_path = Path(__file__).parent.parent / "prompts" / system_structured_filename
    self.system_prompt = self._build_system_prompt(system_path)
    self.system_prompt_structured = self._build_system_prompt(system_structured_path)
    self.utterance_user_prompt_path = Path(utterance_user_prompt_path) if utterance_user_prompt_path is not None else None

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
    return resistance_prompt_name_for_variant(resistance_level, self.prompt_variant)

  def _prompt_resistance_level(self, resistance_level: float) -> float:
    return scaled_resistance_value(resistance_level)

  def prompt_ablation_settings(self) -> dict[str, str]:
    return {
      "prompt_variant": self.prompt_variant,
    }

  def _resistance_prompt_relative_path(self, resistance_level: float) -> str:
    return resistance_prompt_relative_path_for_level(
      resistance_level,
      prompt_variant=self.prompt_variant,
    )

  def _render_resistance_prompt(self, resistance_level: float) -> str:
    relative_path = self._resistance_prompt_relative_path(float(resistance_level))
    try:
      template = load_prompt(relative_path)
    except FileNotFoundError as exc:
      raise FileNotFoundError(
        f"Missing patient resistance prompt for prompt_variant='{self.prompt_variant}': {relative_path}"
      ) from exc
    if not template:
      raise ValueError(
        f"Selected patient resistance prompt is empty for prompt_variant='{self.prompt_variant}': {relative_path}"
      )
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template.format(
      scaled_resistance=f"{self._prompt_resistance_level(resistance_level):.1f}",
    )

  def _attitude_towards_therapy(self) -> str:
    attitude = str(self.profile.get("attitude") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if attitude in {"positive", "very_positive"}:
      return "Positive"
    if attitude in {"negative", "very_negative"}:
      return "Negative"
    return "Neutral"

  def _render_patient_state_block(self, resistance_level: float | None, structured: bool) -> str:
    if not self.resistance_enabled:
      return (
        "<attitude_towards_therapy>\n"
        f"{self._attitude_towards_therapy()}\n"
        "</attitude_towards_therapy>\n\n"
      )

    resolved_resistance = 0.0 if resistance_level is None else float(resistance_level)
    resistance_prompt = (
      self._render_resistance_prompt(resolved_resistance)
      if structured == False
      else f"Your resistance level towards therapy (bounded in the interval 0.0,1.0) is: {resolved_resistance:.2f}"
    )
    return f"<resistance_level>\n{resistance_prompt}\n</resistance_level>\n\n"

  def _build_user_prompt(
    self,
    instruction: str,
    full_conversation: str | None,
    last_utterances: str,
    resistance_level: float | None,
    suffix: str,
    engagement_block: str | None = None,
    structured: bool = False,
  ) -> str:
    conversation_section = ""
    if full_conversation:
      conversation_section = f"<full_conversation>\n{full_conversation}\n</full_conversation>\n\n"
    engagement_section = ""
    if engagement_block:
      engagement_section = f"<engagement>\n{engagement_block}\n</engagement>\n\n"
    patient_state_block = self._render_patient_state_block(resistance_level, structured)
    return (
      f"{conversation_section}"
      f"<last_utterances>\n{last_utterances}\n</last_utterances>\n\n"
      f"{patient_state_block}"
      f"{engagement_section}"
      f"Instruction: {instruction}\n"
      f"{suffix}"
    )

  def _build_utterance_user_prompt(
    self,
    last_utterances: str,
    resistance_level: float | None,
    instruction: str,
  ) -> str:
    if self.utterance_user_prompt_path is None:
      raise ValueError("utterance_user_prompt_path is required for custom utterance user prompts.")
    template = load_prompt(self.utterance_user_prompt_path)
    prompt_template = PromptTemplate.from_template(template)
    resolved_resistance = 0.0 if resistance_level is None else float(resistance_level)
    resistance = self._render_resistance_prompt(resolved_resistance) if self.resistance_enabled else ""
    custom_instruction = self._custom_utterance_instruction(instruction)
    return prompt_template.format(
      persona=self._format_persona(),
      more_info=self._format_more_info(),
      patterns=self._format_patterns(),
      presenting_problem=self.profile["presenting_problem"],
      reason_seeking_counseling=self.profile["reason_seeking_counseling"],
      attitude=self._attitude_towards_therapy(),
      resistance=resistance,
      history=last_utterances,
      instruction=custom_instruction,
    )

  def _custom_utterance_instruction(self, instruction: str) -> str:
    if not self.resistance_enabled:
      return instruction
    marker = "Last resistance update reasoning"
    compressed = instruction
    if marker in compressed:
      compressed = marker + compressed.split(marker, 1)[1]
    return compressed.replace("your <resistance_level>", "your resistance").strip()

  def build_utterance_messages(
    self,
    full_conversation: str | None,
    last_utterances: str,
    resistance_level: float | None,
    instruction: str,
    engagement_block: str | None = None,
  ) -> list[ChatMessage]:
    if self.utterance_user_prompt_path is not None:
      user_prompt = self._build_utterance_user_prompt(
        last_utterances=last_utterances,
        resistance_level=resistance_level,
        instruction=instruction,
      )
    else:
      suffix = load_prompt("shared/patient_utterance_suffix.txt")
      user_prompt = self._build_user_prompt(
        instruction=instruction,
        full_conversation=full_conversation,
        last_utterances=last_utterances,
        resistance_level=resistance_level,
        suffix=suffix,
        engagement_block=engagement_block,
      )
    #breakpoint() #devlog
    return [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": user_prompt},
    ]

  def build_structured_messages(
    self,
    full_conversation: str | None,
    last_utterances: str,
    resistance_level: float | None,
    instruction: str,
    engagement_block: str | None = None,
  ) -> list[ChatMessage]:
    suffix = load_prompt("shared/patient_structured_suffix.txt")
    user_prompt = self._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      resistance_level=resistance_level,
      suffix=suffix,
      engagement_block=engagement_block,
      structured=True,
    )
    return [
      {"role": "system", "content": self.system_prompt_structured},
      {"role": "user", "content": user_prompt},
    ]

  def generate_utterance(
    self,
    full_conversation: str | None,
    last_utterances: str,
    resistance_level: float | None,
    instruction: str,
    engagement_block: str | None = None,
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
      engagement_block=engagement_block,
    )
    generation_kwargs = self._build_generation_kwargs(
      default_values={"max_tokens": 4096, "temperature": 0.8},
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
    full_conversation: str | None,
    last_utterances: str,
    resistance_level: float | None,
    instruction: str,
    engagement_block: str | None = None,
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
      engagement_block=engagement_block,
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
            engagement_block=engagement_block,
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
