from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .interfaces import Agent, ChatMessage, LLM, StructuredOutputError, coerce_bool, load_prompt, parse_json_response, render_prompt


AGENT_THERAPIST_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "therapist_system.txt"
AGENT_THERAPIST_FT_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "ft_prompts" / "standard.txt"
AGENT_THERAPIST_CONTINUATION = "Continue with the ongoing therapy session."
AGENT_INITIAL_GREETING_PROMPT = "opening/initial_greeting.txt"
AGENT_THERAPIST_DEFAULT_MAX_TOKENS = 8192
LOCAL_AGENT_THERAPIST_DEFAULT_MAX_TOKENS = 65536
AGENT_THERAPIST_MAX_FORMAT_RETRIES = 5
AGENT_THERAPIST_STRICT_USER_INSTRUCTION = "Generate next therapist utterance STRICTLY following format <think>...</think> for reasoning, and then the therapist utterance without any prefix like 'Therapist:'."
AGENT_THERAPIST_LENGTH_RESTRICTION = "Maintain the utterance under 70 words."
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
THINK_DASH_RE = re.compile(r"<think>(.*?)---(.*)", re.DOTALL | re.IGNORECASE)


def parse_agent_therapist_response(raw_text: str) -> tuple[str, str]:
  text = (raw_text or "").strip()
  match = THINK_BLOCK_RE.search(text)
  if match is not None:
    reasoning = match.group(1).strip()
    utterance = f"{text[:match.start()]}{text[match.end():]}".strip()
    return reasoning, utterance

  dash_match = THINK_DASH_RE.search(text)
  if dash_match is not None:
    return dash_match.group(1).strip(), dash_match.group(2).strip()

  close_match = THINK_CLOSE_RE.search(text)
  if close_match is None:
    return "", text

  reasoning = text[:close_match.start()].strip()
  utterance = text[close_match.end():].strip()
  return reasoning, utterance


def _has_reasoning_and_utterance(reasoning: str, utterance: str) -> bool:
  return bool(reasoning.strip() and utterance.strip())


class TherapistAgent(Agent):
  def __init__(
    self,
    llm: LLM,
    profile: dict[str, Any],
    generation_defaults: dict[str, Any] | None = None,
    agent_system_prompt_path: str | Path | None = None,
    agent_ft_system_prompt_path: str | Path | None = None,
    user_instruction: bool = False,
    length_restriction: bool = False,
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
    self.system_prompt_structured = render_prompt(
      "therapist_system_structured.txt",
      patient_intake_form=self._format_patient_intake_form(),
    )
    self.agent_system_prompt_path = (
      Path(agent_system_prompt_path) if agent_system_prompt_path else AGENT_THERAPIST_SYSTEM_PROMPT
    )
    self.agent_ft_system_prompt_path = (
      Path(agent_ft_system_prompt_path) if agent_ft_system_prompt_path else AGENT_THERAPIST_FT_SYSTEM_PROMPT
    )
    self.agent_system_prompt: str | None = None
    self.agent_ft_system_prompt: str | None = None
    self.user_instruction = coerce_bool(user_instruction, False)
    self.length_restriction = coerce_bool(length_restriction, False)

  def _render_agent_system_prompt(self, prompt_path: Path) -> str:
    template = load_prompt(prompt_path)
    return template.replace(
        "{patient_intake_form}",
        self._format_patient_intake_form(),
      )

  def _build_agent_system_prompt(self) -> str:
    if self.agent_system_prompt is None:
      self.agent_system_prompt = self._render_agent_system_prompt(self.agent_system_prompt_path)
    return self.agent_system_prompt

  def _build_agent_ft_system_prompt(self) -> str:
    if self.agent_ft_system_prompt is None:
      self.agent_ft_system_prompt = self._render_agent_system_prompt(self.agent_ft_system_prompt_path)
    return self.agent_ft_system_prompt

  def _build_agent_system_prompt_for_mode(self, use_ft_system_prompt: bool) -> str:
    if use_ft_system_prompt:
      return self._build_agent_ft_system_prompt()
    return self._build_agent_system_prompt()

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
    utterance: bool = False,
  ) -> str:
    if utterance:
      guidelines = (
        "Remember to strictly follow these rules:\n"
        "* Style. Use a standard conversational register. Avoid abusing from stylistic elements such as metaphors.\n"
        "* Dialogue flow. Follow the natural flow of the therapy session. **The counselor must not end every utterance with a question. The dialogue should not feel like an interview.**"
        "* Repetition. Use different phrasing and vocabulary to avoid sounding repetitive in subsequent turns. However, maintain the same core content of the message.",
        "* Output. Respond only with the next utterance in plain text."
        # "* Natural signals (e.g., 'mm-hm', 'um', 'yeah','right','...'). Avoid using them with same phrasing repeatedly."
      )
    else:
      guidelines = (
        "Remember to strictly follow these rules:\n"
        "* Output. Respond only with the requested structured data in valid JSON format (without markdown fences). Do not include any text that is not part of the JSON structure."
      )
    return (
      f"<full_conversation>\n{full_conversation}\n</full_conversation>\n\n"
      f"<last_utterances>\n{last_utterances}\n</last_utterances>\n\n"
      f"<last_reasoning>\n{last_reasoning}\n</last_reasoning>\n\n"
      f"<session_context>\n{session_context or 'N/A'}\n</session_context>\n\n"
      f"<guidelines>\n{guidelines}\n</guidelines>\n\n"
      f"Instruction: {instruction}\n"
      f"{suffix}"
    )

  def build_utterance_messages(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
  ) -> list[ChatMessage]:
    suffix = load_prompt("shared/utterance_suffix.txt")
    user_prompt = self._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      suffix=suffix,
      utterance=True,
    )
    return [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": user_prompt},
    ]

  def build_agent_utterance_messages(self, context: str) -> list[ChatMessage]:
    context_text = context.strip() if context.strip() else "none"
    user_prompt = f"<context>\n{context_text}\n</context>\n{AGENT_THERAPIST_CONTINUATION}"
    return [
      {"role": "system", "content": self._build_agent_system_prompt()},
      {"role": "user", "content": user_prompt},
    ]

  def build_agent_initial_utterance_messages(self, use_ft_system_prompt: bool = False) -> list[ChatMessage]:
    return [
      {"role": "system", "content": self._build_agent_system_prompt_for_mode(use_ft_system_prompt)},
      {"role": "user", "content": load_prompt(AGENT_INITIAL_GREETING_PROMPT)},
    ]

  def build_agent_chatml_messages(
    self,
    history: list[ChatMessage],
    use_ft_system_prompt: bool = False,
  ) -> list[ChatMessage]:
    return [
      {"role": "system", "content": self._build_agent_system_prompt_for_mode(use_ft_system_prompt)},
      *history,
    ]

  def _strict_user_instruction_prompt(self, patient: str) -> str:
    instruction = AGENT_THERAPIST_STRICT_USER_INSTRUCTION
    if self.length_restriction:
      instruction = f"{instruction} {AGENT_THERAPIST_LENGTH_RESTRICTION}"
    return f"Patient:{patient}\n{instruction}"

  def _with_strict_user_instruction(self, messages: list[ChatMessage]) -> list[ChatMessage]:
    instructed_messages = [dict(message) for message in messages]
    for index in range(len(instructed_messages) - 1, -1, -1):
      if instructed_messages[index]["role"] == "user":
        patient = instructed_messages[index]["content"].strip()
        if patient.lower().startswith("patient:"):
          patient = patient.split(":", 1)[1].lstrip()
        instructed_messages[index] = {
          "role": "user",
          "content": self._strict_user_instruction_prompt(patient),
        }
        break
    else:
      instructed_messages.append(
        {
          "role": "user",
          "content": self._strict_user_instruction_prompt(""),
        }
      )
    return instructed_messages

  def _generate_agent_response(
    self,
    messages: list[ChatMessage],
    default_max_tokens: int = AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> tuple[str, str, str]:
    generation_kwargs = self._build_generation_kwargs(
      default_values={"max_tokens": default_max_tokens, "temperature": 0.7},
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
    generation_kwargs["strip_thinking"] = False
    request_messages = self._with_strict_user_instruction(messages) if self.user_instruction else messages
    raw = ""
    reasoning = ""
    utterance = ""
    for attempt in range(AGENT_THERAPIST_MAX_FORMAT_RETRIES + 1):
      raw = self.llm.generate(request_messages, **generation_kwargs).strip()
      reasoning, utterance = parse_agent_therapist_response(raw)
      if _has_reasoning_and_utterance(reasoning, utterance):
        break
      if attempt < AGENT_THERAPIST_MAX_FORMAT_RETRIES:
        request_messages = self._with_strict_user_instruction(messages)
    return utterance, reasoning, raw

  def generate_agent_messages(
    self,
    messages: list[ChatMessage],
    default_max_tokens: int = AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> tuple[str, str, str]:
    return self._generate_agent_response(
      messages,
      default_max_tokens=default_max_tokens,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      min_p=min_p,
      presence_penalty=presence_penalty,
      frequency_penalty=frequency_penalty,
      repetition_penalty=repetition_penalty,
    )

  def generate_agent_initial_utterance(
    self,
    use_ft_system_prompt: bool = False,
    default_max_tokens: int = AGENT_THERAPIST_DEFAULT_MAX_TOKENS,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> tuple[str, str, str]:
    return self._generate_agent_response(
      self.build_agent_initial_utterance_messages(use_ft_system_prompt=use_ft_system_prompt),
      default_max_tokens=default_max_tokens,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      min_p=min_p,
      presence_penalty=presence_penalty,
      frequency_penalty=frequency_penalty,
      repetition_penalty=repetition_penalty,
    )

  def generate_agent_utterance(
    self,
    context: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    repetition_penalty: float | None = None,
  ) -> tuple[str, str, str]:
    return self._generate_agent_response(
      self.build_agent_utterance_messages(context),
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      min_p=min_p,
      presence_penalty=presence_penalty,
      frequency_penalty=frequency_penalty,
      repetition_penalty=repetition_penalty,
    )

  def build_structured_messages(
    self,
    full_conversation: str,
    last_utterances: str,
    last_reasoning: str,
    session_context: str,
    instruction: str,
  ) -> list[ChatMessage]:
    suffix = load_prompt("shared/structured_suffix.txt")
    user_prompt = self._build_user_prompt(
      instruction=instruction,
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      suffix=suffix,
    )
    return [
      {"role": "system", "content": self.system_prompt_structured},
      {"role": "user", "content": user_prompt},
    ]

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
    messages = self.build_utterance_messages(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
      instruction=instruction,
    )
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
    retry_suffix = load_prompt("shared/structured_suffix.txt")
    messages = self.build_structured_messages(
      full_conversation=full_conversation,
      last_utterances=last_utterances,
      last_reasoning=last_reasoning,
      session_context=session_context,
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
