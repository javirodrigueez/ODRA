"""Patient (client) modeling that replicates the MIRROR paper, for use with our therapist side.

See mirror/llm_therapist/src/agents/client.py (GPTClient) and mirror/llm_therapist/src/run.py
(TherapySession._exchange_statements) for the reference implementation this module mirrors.
Only the patient/client side is replaced; the therapist turn is untouched.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from agents.interfaces import Agent, ChatMessage, LLMFactory, ensure_str, extract_generation_config, load_prompt
from agents.therapist import TherapistAgent

from .agent_graph import (
  AgentConversationProcessingError,
  AgentTherapyGraphBuilder,
  MODERATOR_MAX_ATTEMPTS,
  MODERATOR_MODEL_ID,
  MODERATOR_SYSTEM_PROMPT,
  PATIENT_ROLE,
  THERAPIST_ROLE,
)

MIRROR_CLIENT_SYSTEM_PROMPT = Path(__file__).resolve().parent.parent / "agent_prompts" / "mirror_client_system.txt"

# Patient-side counterpart of the therapist closing moderator in agent_graph.py. The MIRROR client
# (gpt-3.5) appends [/END] to ~1/4 of its utterances regardless of content, so the tag alone is not a
# reliable end signal; it is only honored when this judge confirms the client is really closing.
PATIENT_MODERATOR_USER_PROMPT_TEMPLATE = (
  "Does this client utterance clearly indicate that the client considers the therapy session finished? "
  "Answer Yes only if the client explicitly says goodbye, says they have nothing more to discuss, "
  "or states they feel resolved and want to stop the session now. "
  "Answer No if the client is still describing problems, feelings, or thoughts, answering the therapist's "
  "question, or agreeing to continue working.\n\n"
  "Therapist utterance:\n{therapist_utterance}\n\n"
  "Client utterance:\n{utterance}"
)

# Copied verbatim from mirror/llm_therapist/src/agents/client.py
QUERY = """### Personal Information ###: 
{client_information}

### Personality Traits ###: {personality_trait}
### Distorted Thoughts ###: {distorted_thoughts}
### Reason for Seeking Counseling ###: {reason_counseling}

### Counseling Dialogue History ###:
{history}"""


def get_resistance_desc(resistance_type: str) -> str | None:
  # Emotional/Cognitive/Behavioral text copied verbatim from mirror/llm_therapist/src/run.py.
  # "No Resistance" is a deliberate deviation: MIRROR's run.py returns None there (its dataset
  # never sets that label), while we use the description from mirror/mirror/agent/personality.py.
  if resistance_type == "Emotional Resistance":
    return "This client tends to avoid expressing their emotions, finding it particularly challenging to display vulnerability or anxiety. They struggle to discuss past emotional wounds and often present an indifferent or detached attitude toward their own emotional experiences."
  if resistance_type == "Cognitive Resistance":
    return "The client demonstrates a reluctance to acknowledge their own problems, often maintaining a strong negative self-image. They are quick to reject the therapist’s advice and persist in holding onto distorted or unhelpful thought patterns throughout the therapeutic process."
  if resistance_type == "Behavioral Resistance":
    return "The client frequently skips therapy sessions and does not complete assigned tasks. They approach therapy with skepticism, expressing doubt about its effectiveness. A deep-seated fear of change leads them to avoid new experiences and resist stepping out of their comfort zone."
  if resistance_type == "No Resistance":
    return (
      "The client exhibits the personality trait of 'No Resistance'. "
      "Open to discussing feelings. Eager to explore new ideas. Willing to accept feedback. "
      "Shows commitment to the therapy process."
    )
  return None


def remove_text_in_parentheses(text: str) -> str:
  # Regex copied verbatim from mirror/llm_therapist/src/run.py.
  text = re.sub(r"\[[^)]*\]", "", text)
  return re.sub(r"[\s]+", " ", text).strip()


class MirrorPatientAgent(Agent):
  def __init__(self, llm: Any, profile: dict[str, Any], generation_defaults: dict[str, Any] | None = None):
    super().__init__(llm, generation_defaults=generation_defaults)
    self.profile = profile

  def _client_information(self) -> str:
    personal_info = self.profile.get("personal_info") or {}
    normalized = {str(key).lower().replace("_", " "): value for key, value in personal_info.items()}
    return str(normalized)

  def _personality_trait(self) -> str | None:
    resistance_type = ensure_str(self.profile.get("mirror_resistance_type"))
    if not resistance_type:
      attitude = str(self.profile.get("attitude") or "").strip().lower()
      resistance_type = "No Resistance" if attitude == "positive" else "Cognitive Resistance"
    return get_resistance_desc(resistance_type)

  def _history_text(self, messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
      role = ensure_str(message.get("role")).strip().lower()
      if role == THERAPIST_ROLE:
        lines.append(f"Counselor: {ensure_str(message.get('content'))}")
      elif role == PATIENT_ROLE:
        content = ensure_str(message.get("mirror_raw")) or ensure_str(message.get("content"))
        lines.append(f"Client: {content}")
    return "\n".join(lines)

  def build_messages(self, messages: list[dict[str, Any]]) -> list[ChatMessage]:
    user_prompt = QUERY.format(
      client_information=self._client_information(),
      personality_trait=self._personality_trait(),
      distorted_thoughts=ensure_str(self.profile.get("presenting_problem")),
      reason_counseling=ensure_str(self.profile.get("reason_seeking_counseling")),
      history=self._history_text(messages),
    )
    return [
      {"role": "system", "content": load_prompt(MIRROR_CLIENT_SYSTEM_PROMPT)},
      {"role": "user", "content": user_prompt},
    ]

  def generate(self, messages: list[ChatMessage]) -> str:
    generation_kwargs = self._build_generation_kwargs(
      default_values={
        "temperature": 0.7,
        "max_tokens": 128,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
      },
    )
    return self.llm.generate(messages, **generation_kwargs)


class MirrorAgentTherapyGraphBuilder(AgentTherapyGraphBuilder):
  def _resistance_enabled(self) -> bool:
    return False

  def _init_agents(self) -> None:
    therapist_llm = LLMFactory.create_from_config(self.cfg.therapist)
    patient_llm = LLMFactory.create_from_config(self.cfg.patient)
    self.therapist = TherapistAgent(
      therapist_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.therapist),
      agent_system_prompt_path=self._config_value(self.cfg.therapist, "system_prompt_path") or None,
      agent_ft_system_prompt_path=self._config_value(self.cfg.therapist, "ft_system_prompt_path") or None,
      user_instruction=self._config_bool(self.cfg.therapist, "user_instruction", False),
      length_restriction=self._config_bool(self.cfg.therapist, "length_restriction", False),
    )
    self.patient = MirrorPatientAgent(
      patient_llm,
      self.profile,
      generation_defaults=extract_generation_config(self.cfg.patient),
    )

  def _patient_prompt_ablation(self) -> dict[str, str]:
    settings = dict(super()._patient_prompt_ablation())
    settings["mirror_patient"] = "true"
    return settings

  def _normalize_messages(self, raw_messages: Any) -> list[dict[str, Any]]:
    normalized = super()._normalize_messages(raw_messages)
    if not isinstance(raw_messages, list):
      return normalized

    raw_by_index: dict[int, dict[str, Any]] = {}
    for item in raw_messages:
      if not isinstance(item, dict):
        continue
      try:
        utterance_index = int(item.get("utterance_index"))
      except (TypeError, ValueError):
        continue
      raw_by_index[utterance_index] = item

    for message in normalized:
      raw_item = raw_by_index.get(int(message.get("utterance_index", 0)))
      if raw_item is None:
        continue
      if "mirror_raw" in raw_item:
        message["mirror_raw"] = ensure_str(raw_item.get("mirror_raw"))
      if "stage_direction" in raw_item:
        message["stage_direction"] = ensure_str(raw_item.get("stage_direction"))
    return normalized

  def _moderator_confirms_patient_end(self, state: dict[str, Any], patient_utterance: str) -> bool:
    messages: list[ChatMessage] = [
      {"role": "system", "content": MODERATOR_SYSTEM_PROMPT},
      {
        "role": "user",
        "content": PATIENT_MODERATOR_USER_PROMPT_TEMPLATE.format(
          therapist_utterance=self._last_therapist_utterance(state),
          utterance=patient_utterance,
        ),
      },
    ]
    for _attempt in range(MODERATOR_MAX_ATTEMPTS):
      try:
        response = self._closing_moderator_client().chat.completions.create(
          model=MODERATOR_MODEL_ID,
          messages=messages,
          temperature=0,
          max_tokens=3,
          extra_body={
            "provider": {
              "order": ["deepseek", "atlas-cloud/fp8", "novita/fp8"],
              "allow_fallbacks": False,
            }
          },
        )
        decision = self._parse_closing_moderator_response(response.choices[0].message.content)
      except Exception:
        decision = None
      if decision is not None:
        return decision
    return False

  def patient_turn(self, state: dict[str, Any]) -> dict[str, Any]:
    s = self._ensure_state_defaults(state)
    if s["session_ended"]:
      return s
    if self._end_for_max_turns_if_needed(s):
      return s

    traces: list[dict[str, Any]] = []
    try:
      messages = self.patient.build_messages(s["messages"])
      raw = ensure_str(self.patient.generate(messages))

      try:
        statement = raw.split(":", 1)[1].strip()
      except Exception:
        statement = raw.strip()

      traces.append(
        {
          "step": "mirror_patient_utterance",
          "input": messages[-1]["content"],
          "output": raw,
        }
      )

      requested_end = "[/END]" in statement or "END" in statement
      statement = statement.replace("[/END]", "")
      verbal = remove_text_in_parentheses(statement)
      stage_direction = " ".join(re.findall(r"\[[^\]]*\]", statement))
      if not verbal:
        # Purely non-verbal reply (e.g. "[Nods silently]"): MIRROR would hand the counselor an empty
        # string; we pass the stage direction itself so the text-only therapist sees the cue.
        verbal = stage_direction.strip()
      if not verbal:
        raise AgentConversationProcessingError(f"Patient generated an empty utterance. Raw output: {raw!r}")

      idx = self._append_message(s, PATIENT_ROLE, verbal)
      s["messages"][-1]["mirror_raw"] = statement
      s["messages"][-1]["stage_direction"] = stage_direction
      self._append_trace(s, idx, PATIENT_ROLE, traces)

      patient_count = sum(1 for message in s["messages"] if message.get("role") == PATIENT_ROLE)
      if requested_end and patient_count >= 2:
        moderator_confirmed = self._moderator_confirms_patient_end(s, verbal)
        traces.append(
          {
            "step": "patient_end_moderator",
            "input": verbal,
            "output": "Yes" if moderator_confirmed else "No",
          }
        )
        if moderator_confirmed:
          s["session_ended"] = True
          s["session_ended_reason"] = "patient_requested_end"
          s["final_patient_pending_end"] = False
          return s
      if s["final_patient_pending_end"]:
        s["session_ended"] = True
        if not s["session_ended_reason"]:
          s["session_ended_reason"] = "therapist_requested_end"
        s["final_patient_pending_end"] = False
        return s
      self._end_for_max_turns_if_needed(s)
    except AgentConversationProcessingError as exc:
      self._end_session_with_error(s, traces, exc)
    return s
