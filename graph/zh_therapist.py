"""Agent-simulation builder that drives a Chinese therapist model (SoulChat2.0 / Psyche-R1) with translation.

Only the therapist side changes with respect to graph/agent_graph.py: the patient, resistance, closing
moderator, manifests and resume machinery are reused unchanged. Every therapist turn:

  1. translates any patient utterance that has no `content_zh` yet (EN->ZH, cached on the message),
  2. builds the Chinese ChatML request following each model's repo conventions,
  3. generates with the vLLM server, parses `<think>` reasoning (Psyche-R1 only),
  4. translates the utterance ZH->EN into `content` (seen by patient, moderator and CTRS) and keeps the
     original as `content_zh`. Reasoning traces stay in Chinese.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]
from openai import OpenAI  # pyright: ignore[reportMissingImports]

from agents.interfaces import ChatMessage, ensure_str, extract_generation_config, load_prompt
from agents.therapist import parse_agent_therapist_response
from agents.translator import Translator

from .agent_graph import (
  AGENT_END_TAG,
  CONVERSATION_ROLES,
  PATIENT_ROLE,
  THERAPIST_ROLE,
  AgentConversationProcessingError,
  AgentTherapyGraphBuilder,
)

ROOT = Path(__file__).resolve().parent.parent
ZH_VARIANTS = ("soulchat2", "psyche_r1")
DEFAULT_SYSTEM_PROMPTS = {
  "soulchat2": ROOT / "agent_prompts" / "soulchat2_system_zh.txt",
  "psyche_r1": ROOT / "ft_prompts" / "standard_zh.txt",
}
# Repo conventions: SoulChat2.0 demo uses max_tokens=4096; Psyche-R1 run.py uses max_new_tokens=2048.
DEFAULT_MAX_TOKENS = {"soulchat2": 4096, "psyche_r1": 2048}
GREETING_PROMPT_ZH = ROOT / "prompts" / "opening" / "initial_greeting_zh.txt"
# Verbatim from Psyche-R1 run.py (recommended for multi-turn dialogue).
PSYCHE_R1_THINK_PREFIX = (
  "你必须在<think>和</think>标签内给出你的推理过程，然后，在</think>标签后给出最终的答案。"
  "\n\n以下是我的问题：\n"
)
OPENAI_SAMPLING_KEYS = {"seed", "temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"}
ZH_MAX_EMPTY_RETRIES = 1


class ZhTherapistGraphBuilder(AgentTherapyGraphBuilder):
  def __init__(self, config_path: str, profile: dict[str, Any], *, cactus_patient: bool | None = None):
    raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    therapist_cfg = raw_cfg.get("therapist") or {}
    self.zh_variant = ensure_str(therapist_cfg.get("zh_variant")).lower()
    if self.zh_variant not in ZH_VARIANTS:
      raise ValueError(f"therapist.zh_variant must be one of {ZH_VARIANTS}, got {self.zh_variant!r}.")
    self.zh_history_limit = int(therapist_cfg.get("history_limit") or 0)
    self.translator_cfg = raw_cfg.get("translator") or {}
    super().__init__(config_path, profile, cactus_patient=cactus_patient)

  def _init_agents(self) -> None:
    super()._init_agents()
    therapist_cfg = self.cfg.therapist
    self.zh_model_id = ensure_str(therapist_cfg.model_id)
    self.zh_client = OpenAI(api_key="test", base_url=ensure_str(therapist_cfg.base_url), timeout=600, max_retries=3)
    self.zh_generation = extract_generation_config(therapist_cfg)
    self.zh_generation.setdefault("max_tokens", DEFAULT_MAX_TOKENS[self.zh_variant])
    system_prompt_path = ensure_str(therapist_cfg.get("ft_system_prompt_path")) or DEFAULT_SYSTEM_PROMPTS[self.zh_variant]
    self.zh_system_prompt = load_prompt(Path(system_prompt_path))
    self.zh_greeting_prompt = load_prompt(GREETING_PROMPT_ZH)
    self.translator = Translator.from_config(self.translator_cfg)

  # Keep the Chinese originals alive across _ensure_state_defaults and resumes.
  def _normalize_messages(self, raw_messages: Any) -> list[dict[str, Any]]:
    normalized = super()._normalize_messages(raw_messages)
    kept = [
      item
      for item in (raw_messages if isinstance(raw_messages, list) else [])
      if isinstance(item, dict)
      and ensure_str(item.get("role")).strip().lower() in CONVERSATION_ROLES
      and ensure_str(item.get("content"))
    ]
    for message, raw in zip(normalized, kept):
      content_zh = ensure_str(raw.get("content_zh"))
      if content_zh:
        message["content_zh"] = content_zh
    return normalized

  def _translate_pending_patient_messages(self, state: dict[str, Any]) -> None:
    for message in state["messages"]:
      if message.get("role") == PATIENT_ROLE and not ensure_str(message.get("content_zh")):
        message["content_zh"] = self.translator.en_to_zh(ensure_str(message.get("content")))

  def _wrap_user_content(self, text: str) -> str:
    if self.zh_variant == "psyche_r1":
      return f"{PSYCHE_R1_THINK_PREFIX}{text}"
    return text

  def _build_zh_request_messages(self, state: dict[str, Any]) -> list[ChatMessage]:
    messages: list[ChatMessage] = [{"role": "system", "content": self.zh_system_prompt}]
    history = state["messages"]
    if not history:
      messages.append({"role": "user", "content": self._wrap_user_content(self.zh_greeting_prompt)})
      return messages
    if self.zh_history_limit > 0:
      history = history[-self.zh_history_limit:]
    for message in history:
      content_zh = ensure_str(message.get("content_zh")) or ensure_str(message.get("content"))
      if message.get("role") == PATIENT_ROLE:
        messages.append({"role": "user", "content": self._wrap_user_content(content_zh)})
      elif message.get("role") == THERAPIST_ROLE:
        # Psyche-R1 assistant turns keep only the answer; the <think> block is not replayed.
        messages.append({"role": "assistant", "content": content_zh})
    return messages

  def _generate_zh(self, messages: list[ChatMessage]) -> str:
    create_kwargs: dict[str, Any] = {"model": self.zh_model_id, "messages": messages}
    extra_body = {key: value for key, value in self.zh_generation.items() if key not in OPENAI_SAMPLING_KEYS}
    create_kwargs.update({key: value for key, value in self.zh_generation.items() if key in OPENAI_SAMPLING_KEYS})
    if extra_body:
      create_kwargs["extra_body"] = extra_body
    response = self.zh_client.chat.completions.create(**create_kwargs)
    return ensure_str(response.choices[0].message.content)

  def _parse_zh_response(self, raw: str) -> tuple[str, str]:
    if self.zh_variant == "psyche_r1":
      return parse_agent_therapist_response(raw)
    return "", raw.strip()

  def _generate_zh_utterance(self, messages: list[ChatMessage]) -> tuple[str, str, str]:
    raw = reasoning = utterance = ""
    for _attempt in range(ZH_MAX_EMPTY_RETRIES + 1):
      raw = self._generate_zh(messages)
      reasoning, utterance = self._parse_zh_response(raw)
      if utterance:
        break
    return utterance, reasoning, raw

  def therapist_turn(self, state: dict[str, Any]) -> dict[str, Any]:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s
    if self._end_for_max_turns_if_needed(s):
      return s

    traces: list[dict[str, Any]] = []
    try:
      self._translate_pending_patient_messages(s)
      request_messages = self._build_zh_request_messages(s)
      utterance_zh, reasoning_zh, _raw = self._generate_zh_utterance(request_messages)
      if not utterance_zh:
        raise AgentConversationProcessingError("Therapist generated an empty utterance.")
      utterance = self.translator.zh_to_en(utterance_zh)
      if not utterance:
        raise AgentConversationProcessingError("Translator returned an empty therapist utterance.")
      requested_end = AGENT_END_TAG in utterance
      traces.append(
        {
          "step": "agent_reasoning",
          "input": ensure_str(request_messages[-1]["content"]),
          "output": reasoning_zh,
        }
      )
      idx = self._append_message(s, THERAPIST_ROLE, utterance)
      s["messages"][-1]["content_zh"] = utterance_zh
      self._append_trace(s, idx, THERAPIST_ROLE, traces)
      moderator_detected_end = self._moderator_detects_closing(utterance)
      if requested_end:
        if self._max_turns_reached(s):
          s["session_ended"] = True
          s["session_ended_reason"] = "therapist_requested_end"
        else:
          s["final_patient_pending_end"] = True
        return s
      if moderator_detected_end:
        if self._max_turns_reached(s):
          s["session_ended"] = True
        else:
          s["final_patient_pending_end"] = True
        s["session_ended_reason"] = "moderator_detected_closing"
        return s
      self._end_for_max_turns_if_needed(s)
    except AgentConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s
