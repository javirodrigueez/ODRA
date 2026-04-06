from __future__ import annotations

import re
from typing import Any

from agents.interfaces import ChatMessage, ensure_str

from .state import SessionState

PROMPT_AUDIT_TOKEN_THRESHOLD = 5000


def count_text_tokens(text: str) -> int:
  normalized = ensure_str(text)
  if not normalized:
    return 0
  try:
    import tiktoken  # pyright: ignore[reportMissingImports]

    try:
      encoding = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(normalized))
  except Exception:
    return max(1, (len(normalized) + 3) // 4)


def render_raw_llm_input_prompt(messages: list[ChatMessage]) -> str:
  rendered: list[str] = []
  for message in messages:
    role = ensure_str(message.get("role"), "user")
    content = ensure_str(message.get("content"))
    rendered.append(f"<{role}>\n{content}\n</{role}>")
  return "\n\n".join(rendered)


def summarize_prompt_blocks(content: str) -> list[str]:
  normalized = ensure_str(content)
  if not normalized:
    return ["content_tokens=0"]

  block_pattern = re.compile(r"<([a-zA-Z0-9_]+)>\s*(.*?)\s*</\1>", re.DOTALL)
  summary_lines: list[str] = []
  cursor = 0
  free_text_index = 1

  for match in block_pattern.finditer(normalized):
    prefix = normalized[cursor:match.start()].strip()
    if prefix:
      summary_lines.append(
        f"free_text_{free_text_index}_tokens={count_text_tokens(prefix)}"
      )
      free_text_index += 1

    block_name = match.group(1)
    block_content = match.group(2).strip()
    summary_lines.append(
      f"{block_name}_tokens={count_text_tokens(block_content)}"
    )
    cursor = match.end()

  suffix = normalized[cursor:].strip()
  if suffix:
    summary_lines.append(
      f"free_text_{free_text_index}_tokens={count_text_tokens(suffix)}"
    )

  if not summary_lines:
    return [f"content_tokens={count_text_tokens(normalized)}"]
  return summary_lines


def render_llm_input_prompt(messages: list[ChatMessage]) -> str:
  rendered: list[str] = []
  for message in messages:
    role = ensure_str(message.get("role"), "user")
    content = ensure_str(message.get("content"))
    block_summary = "\n".join(summarize_prompt_blocks(content))
    rendered.append(
      f"<{role}>\n"
      f"total_tokens={count_text_tokens(content)}\n"
      f"{block_summary}\n"
      f"</{role}>"
    )
  return "\n\n".join(rendered)


def record_large_prompt_request(
  state: SessionState,
  messages: list[ChatMessage],
  speaker: str,
  request_kind: str,
) -> None:
  raw_input_prompt = render_raw_llm_input_prompt(messages)
  prompt_tokens = count_text_tokens(raw_input_prompt)
  if prompt_tokens <= PROMPT_AUDIT_TOKEN_THRESHOLD:
    return
  state["prompt_audit_records"].append(
    {
      "utterance_index": int(state.get("utterance_index", 0)),
      "speaker": speaker,
      "request_kind": request_kind,
      "prompt_tokens": prompt_tokens,
      "input_prompt": render_llm_input_prompt(messages),
    }
  )
