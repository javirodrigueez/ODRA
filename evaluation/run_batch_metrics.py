#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
  from dotenv import load_dotenv
except ImportError:  # pragma: no cover - kept for clearer CLI failure modes.
  load_dotenv = None  # type: ignore[assignment]


MODEL_NAME = "gpt-4o"
BATCH_ENDPOINT = "/v1/chat/completions"
AZURE_BATCH_ENDPOINT = "/chat/completions"
COMPLETION_WINDOW = "24h"
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-21"
DEFAULT_ENQUEUED_TOKEN_LIMIT = 100_000_000
DEFAULT_MAX_ITER = 3
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_LOCAL_TIMEOUT_HOURS = 26.0
DEFAULT_INPUT_COST_PER_1M = 1.25
DEFAULT_OUTPUT_COST_PER_1M = 5.0
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024

# Anthropic judge (--judge sonnet): Message Batches API, one request per sample (no `n`).
JUDGE_MODELS = {"gpt": MODEL_NAME, "sonnet": "claude-sonnet-5-5"}
# Batch prices (50% of standard): gpt-4o $2.50/$10 -> $1.25/$5; Sonnet 5.5 $2/$10 -> $1/$5.
JUDGE_COSTS_PER_1M = {"gpt": (DEFAULT_INPUT_COST_PER_1M, DEFAULT_OUTPUT_COST_PER_1M), "sonnet": (1.0, 5.0)}
CACHE_WRITE_COST_MULTIPLIER = 1.25  # Anthropic 5-minute cache writes, relative to the input price.
CACHE_READ_COST_MULTIPLIER = 0.1  # Anthropic cache reads, relative to the input price.
ANTHROPIC_BATCH_ENDPOINT = "/v1/messages/batches"
ANTHROPIC_MAX_TOKENS = 4096  # Judge outputs are ~100 tokens (CTRS/WAI) and ~1k tokens (PANAS).
# Sonnet 5.5 rejects temperature and `thinking: disabled`; between_tools is its no-thinking mode.
ANTHROPIC_THINKING = {"type": "between_tools"}
ANTHROPIC_EFFORT = "high"
# Top-level automatic prompt caching: the max_iter identical samples of an item can reuse the cached prompt.
ANTHROPIC_CACHE_CONTROL = {"type": "ephemeral"}
ANTHROPIC_MAX_BATCH_REQUESTS = 100_000
ANTHROPIC_MAX_BATCH_BYTES = 256 * 1024 * 1024
# ~5 bytes/token in the JSONL (~6 for escaped non-ASCII), so this keeps one batch below 256 MB.
ANTHROPIC_MAX_BATCH_TOKENS = 20_000_000

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_ROOT = Path(__file__).resolve().parent
CTRS_PROMPTS_ROOT = EVALUATION_ROOT / "prompts" / "ctrs"

CTRS_KEYS = [
  "general_1_understanding",
  "general_2_interpersonal_effectiveness",
  "general_3_collaboration",
  "CBT_1_guided_discovery",
  "CBT_2_focus",
  "CBT_3_strategy",
]
WAI_KEYS = [f"wai{i}" for i in range(1, 13)]
FEELINGS = [
  "interested",
  "excited",
  "strong",
  "enthusiastic",
  "proud",
  "alert",
  "inspired",
  "determined",
  "attentive",
  "active",
  "distressed",
  "upset",
  "guilty",
  "scared",
  "hostile",
  "irritable",
  "ashamed",
  "nervous",
  "jittery",
  "afraid",
]
PANAS_ENTRY_START_RE = re.compile(
  r"(?i)(?:^|[\r\n]+|(?<=[.!?])\s+)("
  + "|".join(re.escape(feeling) for feeling in FEELINGS)
  + r")\s*,"
)

FINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
FAILED_STATUSES = {"failed", "expired", "cancelled"}
DONE_MANIFEST_STATUSES = {"completed", "failed"}


class BatchMetricsError(RuntimeError):
  pass


@dataclass(frozen=True)
class ConversationRecord:
  path: Path
  filename: str
  data: dict[str, Any]
  history: str
  intake_form: str
  attitude: str


@dataclass(frozen=True)
class BatchRequest:
  custom_id: str
  phase: str
  metric_group: str
  input_file: str
  output_file: str
  item_key: str
  body: dict[str, Any]

  def jsonl_line(self) -> dict[str, Any]:
    return {
      "custom_id": self.custom_id,
      "method": "POST",
      "url": BATCH_ENDPOINT,
      "body": self.body,
    }

  def map_entry(self) -> dict[str, Any]:
    return {
      "custom_id": self.custom_id,
      "phase": self.phase,
      "metric_group": self.metric_group,
      "input_file": self.input_file,
      "output_file": self.output_file,
      "item_key": self.item_key,
    }


@dataclass(frozen=True)
class BatchShard:
  name: str
  specs: list[BatchRequest]
  enqueued_tokens: int


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def atomic_write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, ensure_ascii=True)
      f.write("\n")
    os.replace(tmp_name, path)
  except Exception:
    try:
      os.unlink(tmp_name)
    except FileNotFoundError:
      pass
    raise


def atomic_write_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(text)
    os.replace(tmp_name, path)
  except Exception:
    try:
      os.unlink(tmp_name)
    except FileNotFoundError:
      pass
    raise


def to_plain(value: Any) -> Any:
  if hasattr(value, "model_dump"):
    return value.model_dump()
  if isinstance(value, dict):
    return {key: to_plain(item) for key, item in value.items()}
  if isinstance(value, list):
    return [to_plain(item) for item in value]
  return value


def get_field(value: Any, key: str, default: Any = None) -> Any:
  if isinstance(value, dict):
    return value.get(key, default)
  return getattr(value, key, default)


def render_template(template: str, values: dict[str, str]) -> str:
  rendered = template
  for key, value in values.items():
    rendered = rendered.replace("{" + key + "}", value)
  return rendered


def load_prompt(path: Path) -> str:
  with path.open("r", encoding="utf-8") as f:
    return f.read()


def generate_history(history: list[dict[str, Any]]) -> str:
  return "\n".join(
    f"{str(message.get('role', '')).capitalize()}: {message.get('content', '')}"
    for message in history
  )


def format_persona(info: dict[str, Any]) -> str:
  lines = []
  for key, val in info.items():
    label = key.replace("_", " ").title()
    lines.append(f"{label}: {val}")
  return "\n".join(lines)


def format_more_info(info: dict[str, Any]) -> str:
  sections = [
    ("Past History", info.get("past_history", "")),
    ("Academic/Occupational Functioning", info.get("academic_occupational_functioning", "")),
    ("Social Support System", info.get("social_support_system", "")),
  ]
  return "\n".join(f"{title}:\n{content}" for title, content in sections if content)


def format_presession_questionnaire(data: dict[str, Any]) -> str:
  output = []
  for feeling in FEELINGS:
    value = data.get(feeling)
    if isinstance(value, list) and value and isinstance(value[0], int):
      output.append(f"{feeling}: {value[0]}")
  return "\n".join(output)


def _number(value: Any) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def is_ctrs_complete(path: Path) -> bool:
  if not path.is_file():
    return False
  try:
    data = read_json(path)
  except (OSError, json.JSONDecodeError):
    return False
  if not isinstance(data, dict) or _number(data.get("cost")) is None:
    return False
  return all(_number(data.get(key)) is not None for key in CTRS_KEYS)


def is_wai_complete(path: Path) -> bool:
  if not path.is_file():
    return False
  try:
    data = read_json(path)
  except (OSError, json.JSONDecodeError):
    return False
  if not isinstance(data, dict) or _number(data.get("cost")) is None:
    return False
  return all(_number(data.get(key)) is not None for key in WAI_KEYS)


def _is_score_list(value: Any) -> bool:
  return isinstance(value, list) and any(isinstance(item, int) for item in value)


def is_panas_complete(path: Path) -> bool:
  if not path.is_file():
    return False
  try:
    data = read_json(path)
  except (OSError, json.JSONDecodeError):
    return False
  if not isinstance(data, dict):
    return False
  if _number(data.get("cost")) is None or not str(data.get("attitude") or "").strip():
    return False
  if not isinstance(data.get("explanations"), list):
    return False
  return all(_is_score_list(data.get(feeling)) for feeling in FEELINGS)


def output_is_complete(metric_group: str, path: Path) -> bool:
  if metric_group == "ctrs":
    return is_ctrs_complete(path)
  if metric_group == "wai":
    return is_wai_complete(path)
  if metric_group in {"panas_before", "panas_after"}:
    return is_panas_complete(path)
  raise ValueError(f"Unknown metric group: {metric_group}")


def validate_conversation_file(path: Path) -> ConversationRecord:
  try:
    payload = read_json(path)
  except json.JSONDecodeError as exc:
    raise BatchMetricsError(f"Invalid JSON file: {path}: {exc}") from exc

  if not isinstance(payload, dict):
    raise BatchMetricsError(f"Conversation file must contain a JSON object: {path}")

  messages = payload.get("messages")
  if not isinstance(messages, list):
    raise BatchMetricsError(f"Conversation file is missing messages list: {path}")
  for index, message in enumerate(messages):
    if not isinstance(message, dict) or "role" not in message or "content" not in message:
      raise BatchMetricsError(f"Invalid message at index {index} in: {path}")

  patient_profile = payload.get("patient_profile")
  if not isinstance(patient_profile, dict):
    raise BatchMetricsError(f"Conversation file is missing patient_profile: {path}")

  required_keys = ("personal_info", "more_information", "attitude")
  missing = [key for key in required_keys if key not in patient_profile]
  if missing:
    raise BatchMetricsError(
      f"Conversation file has incomplete patient_profile ({', '.join(missing)}): {path}"
    )
  if not isinstance(patient_profile.get("personal_info"), dict):
    raise BatchMetricsError(f"patient_profile.personal_info must be an object: {path}")
  if not isinstance(patient_profile.get("more_information"), dict):
    raise BatchMetricsError(f"patient_profile.more_information must be an object: {path}")

  intake_form = "\n".join(
    [
      format_persona(patient_profile["personal_info"]),
      format_more_info(patient_profile["more_information"]),
    ]
  )
  attitude = str(patient_profile.get("attitude") or "").strip()
  if not attitude:
    raise BatchMetricsError(f"patient_profile.attitude must not be empty: {path}")

  return ConversationRecord(
    path=path,
    filename=path.name,
    data=payload,
    history=generate_history(messages),
    intake_form=intake_form,
    attitude=attitude,
  )


def discover_conversations(input_dir: Path) -> list[ConversationRecord]:
  if not input_dir.is_dir():
    raise BatchMetricsError(f"Input directory does not exist: {input_dir}")

  files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
  if not files:
    raise BatchMetricsError(f"No JSON conversation files found in: {input_dir}")
  return [validate_conversation_file(path) for path in files]


def _is_relative_to(path: Path, parent: Path) -> bool:
  try:
    path.relative_to(parent)
  except ValueError:
    return False
  return True


def load_run_from_json_paths(path: Path, input_dir: Path) -> list[Path]:
  try:
    payload = read_json(path)
  except json.JSONDecodeError as exc:
    raise BatchMetricsError(f"Invalid --run-from-json file: {path}: {exc}") from exc

  if not isinstance(payload, dict):
    raise BatchMetricsError(f"--run-from-json must contain a JSON object: {path}")
  failed_simulations = payload.get("failed_simulations")
  if not isinstance(failed_simulations, list):
    raise BatchMetricsError(f"--run-from-json is missing failed_simulations list: {path}")

  selected: list[Path] = []
  seen: set[Path] = set()
  for index, item in enumerate(failed_simulations):
    if not isinstance(item, dict):
      raise BatchMetricsError(f"Invalid failed_simulations item at index {index}: {path}")
    status = item.get("status")
    if not isinstance(status, dict):
      raise BatchMetricsError(f"Missing status object at failed_simulations index {index}: {path}")
    output_json = status.get("output_json")
    if not isinstance(output_json, str) or not output_json.strip():
      raise BatchMetricsError(f"Missing status.output_json at failed_simulations index {index}: {path}")

    conversation_path = Path(output_json).expanduser().resolve()
    if not _is_relative_to(conversation_path, input_dir):
      raise BatchMetricsError(
        f"Listed simulation is outside input_dir: {conversation_path} not under {input_dir}"
      )
    if not conversation_path.is_file():
      raise BatchMetricsError(f"Listed simulation JSON does not exist: {conversation_path}")
    if conversation_path in seen:
      continue
    selected.append(conversation_path)
    seen.add(conversation_path)

  if not selected:
    raise BatchMetricsError(f"--run-from-json did not list any simulation JSON paths: {path}")
  return selected


def discover_conversations_from_run_json(input_dir: Path, run_from_json: Path) -> list[ConversationRecord]:
  if not input_dir.is_dir():
    raise BatchMetricsError(f"Input directory does not exist: {input_dir}")
  paths = load_run_from_json_paths(run_from_json, input_dir)
  return [validate_conversation_file(path) for path in paths]


def remove_metric_outputs_for_records(
  output_dir: Path,
  records: list[ConversationRecord],
  excluded_metrics: set[str] | None = None,
) -> list[str]:
  excluded = excluded_metrics or set()
  groups = []
  for group in ("ctrs", "wai", "panas_before", "panas_after"):
    if group in excluded or (group in {"panas_before", "panas_after"} and "panas" in excluded):
      continue
    groups.append(group)

  removed: list[str] = []
  for record in records:
    for group in groups:
      path = output_dir / group / record.filename
      if path.is_file():
        path.unlink()
        removed.append((Path(group) / record.filename).as_posix())
  return removed


def ensure_output_dirs(output_dir: Path) -> None:
  for group in ("ctrs", "wai", "panas_before", "panas_after"):
    (output_dir / group).mkdir(parents=True, exist_ok=True)


def make_chat_body(prompt: str, model: str, max_iter: int) -> dict[str, Any]:
  return {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "n": max_iter,
  }


def safe_custom_id_part(value: str) -> str:
  safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
  return safe[:80] or "item"


def make_custom_id(phase: str, metric_group: str, index: int, filename: str, item_key: str) -> str:
  name = safe_custom_id_part(Path(filename).stem)
  item = safe_custom_id_part(item_key)
  return f"{phase}:{metric_group}:{index:05d}:{name}:{item}"


def build_panas_before_requests(
  records: list[ConversationRecord],
  output_dir: Path,
  model: str,
  max_iter: int,
  force_recompute: bool = False,
) -> list[BatchRequest]:
  prompt_template = load_prompt(EVALUATION_ROOT / "PANAS" / "prompts" / "panas_before.txt")
  requests = []
  for index, record in enumerate(records):
    output_file = Path("panas_before") / record.filename
    if not force_recompute and output_is_complete("panas_before", output_dir / output_file):
      continue
    prompt = render_template(prompt_template, {"intake_form": record.intake_form})
    requests.append(
      BatchRequest(
        custom_id=make_custom_id("panas_before", "panas_before", index, record.filename, "panas_before"),
        phase="panas_before",
        metric_group="panas_before",
        input_file=record.filename,
        output_file=output_file.as_posix(),
        item_key="panas_before",
        body=make_chat_body(prompt, model, max_iter),
      )
    )
  return requests


def panas_after_prompt_filename(panas_v2: bool) -> str:
  return "panas_after_v2.txt" if panas_v2 else "panas_after.txt"


def render_panas_after_prompt(
  prompt_template: str,
  record: ConversationRecord,
  before_payload: dict[str, Any] | None,
  *,
  panas_v2: bool,
) -> str:
  if panas_v2:
    if before_payload is None:
      raise BatchMetricsError(
        f"PANAS-after v2 requires a PANAS-before result for {record.filename}, but none is available."
      )
    return render_template(
      prompt_template,
      {
        "intake_form": record.intake_form,
        "conversation": record.history,
        "presession_questionnaire": format_presession_questionnaire(before_payload),
      },
    )

  return render_template(
    prompt_template,
    {
      "intake_form": record.intake_form,
      "dialogue": record.history,
    },
  )


def build_main_requests(
  records: list[ConversationRecord],
  output_dir: Path,
  before_payloads: dict[str, dict[str, Any]],
  model: str,
  max_iter: int,
  excluded_metrics: set[str] | None = None,
  panas_v2: bool = False,
  force_recompute: bool = False,
) -> list[BatchRequest]:
  requests: list[BatchRequest] = []
  excluded = excluded_metrics or set()
  ctrs_templates = {
    key: load_prompt(CTRS_PROMPTS_ROOT / f"{key}.txt")
    for key in CTRS_KEYS
  } if "ctrs" not in excluded else {}
  wai_templates = {
    key: load_prompt(EVALUATION_ROOT / "WAI" / "prompts" / f"{key}.txt")
    for key in WAI_KEYS
  } if "wai" not in excluded else {}
  panas_after_template = (
    load_prompt(EVALUATION_ROOT / "PANAS" / "prompts" / panas_after_prompt_filename(panas_v2))
    if "panas" not in excluded
    else ""
  )

  for index, record in enumerate(records):
    ctrs_output_file = Path("ctrs") / record.filename
    if "ctrs" not in excluded and (force_recompute or not output_is_complete("ctrs", output_dir / ctrs_output_file)):
      for key in CTRS_KEYS:
        prompt = render_template(ctrs_templates[key], {"conversation": record.history})
        requests.append(
          BatchRequest(
            custom_id=make_custom_id("main", "ctrs", index, record.filename, key),
            phase="main",
            metric_group="ctrs",
            input_file=record.filename,
            output_file=ctrs_output_file.as_posix(),
            item_key=key,
            body=make_chat_body(prompt, model, max_iter),
          )
        )

    wai_output_file = Path("wai") / record.filename
    if "wai" not in excluded and (force_recompute or not output_is_complete("wai", output_dir / wai_output_file)):
      for key in WAI_KEYS:
        prompt = render_template(wai_templates[key], {"conversation": record.history})
        requests.append(
          BatchRequest(
            custom_id=make_custom_id("main", "wai", index, record.filename, key),
            phase="main",
            metric_group="wai",
            input_file=record.filename,
            output_file=wai_output_file.as_posix(),
            item_key=key,
            body=make_chat_body(prompt, model, max_iter),
          )
        )

    panas_after_output_file = Path("panas_after") / record.filename
    if "panas" not in excluded and (force_recompute or not output_is_complete("panas_after", output_dir / panas_after_output_file)):
      before_payload = before_payloads.get(record.filename)
      prompt = render_panas_after_prompt(
        panas_after_template,
        record,
        before_payload,
        panas_v2=panas_v2,
      )
      requests.append(
        BatchRequest(
          custom_id=make_custom_id("main", "panas_after", index, record.filename, "panas_after"),
          phase="main",
          metric_group="panas_after",
          input_file=record.filename,
          output_file=panas_after_output_file.as_posix(),
          item_key="panas_after",
          body=make_chat_body(prompt, model, max_iter),
        )
      )

  return requests


def request_map_from_specs(specs: list[BatchRequest]) -> dict[str, dict[str, Any]]:
  return {spec.custom_id: spec.map_entry() for spec in specs}


def write_jsonl(path: Path, specs: list[BatchRequest]) -> None:
  lines = [json.dumps(spec.jsonl_line(), ensure_ascii=True) for spec in specs]
  atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def anthropic_custom_ids(index: int, max_iter: int) -> list[str]:
  # Anthropic custom ids must match ^[a-zA-Z0-9_-]{1,64}$; request_map keeps the link to the original id.
  return [f"r{index:06d}_s{sample}" for sample in range(max_iter)]


def anthropic_params(body: dict[str, Any]) -> dict[str, Any]:
  messages = body["messages"]
  system = "\n\n".join(str(message["content"]) for message in messages if message["role"] == "system")
  params = {
    "model": body["model"],
    "max_tokens": ANTHROPIC_MAX_TOKENS,
    "messages": [
      {"role": message["role"], "content": message["content"]}
      for message in messages
      if message["role"] != "system"
    ],
    "thinking": ANTHROPIC_THINKING,
    "output_config": {"effort": ANTHROPIC_EFFORT},
    "cache_control": ANTHROPIC_CACHE_CONTROL,
  }
  if system:
    params["system"] = system
  return params


def write_anthropic_jsonl(path: Path, specs: list[BatchRequest], request_map: dict[str, dict[str, Any]]) -> None:
  lines = [
    json.dumps({"custom_id": anthropic_id, "params": anthropic_params(spec.body)}, ensure_ascii=True)
    for spec in specs
    for anthropic_id in request_map[spec.custom_id]["anthropic_custom_ids"]
  ]
  atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def judge_shard_options(judge: str, max_iter: int, token_limit: int) -> tuple[int, int, int | None]:
  # Returns (token limit, requests per spec, max requests per batch).
  if judge == "sonnet":
    return min(token_limit, ANTHROPIC_MAX_BATCH_TOKENS), max_iter, ANTHROPIC_MAX_BATCH_REQUESTS
  return token_limit, 1, None


def _get_token_encoding(model: str) -> Any | None:
  try:
    import tiktoken  # pyright: ignore[reportMissingImports]
  except ImportError:
    return None
  try:
    return tiktoken.encoding_for_model(model)
  except KeyError:
    return tiktoken.get_encoding("cl100k_base")


def _estimate_text_tokens(text: str, encoding: Any | None) -> int:
  if encoding is not None:
    return len(encoding.encode(text))
  return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list[Any], model: str) -> int:
  encoding = _get_token_encoding(model)
  tokens_per_message = 3
  tokens_per_name = 1
  total = 3
  for message in messages:
    if not isinstance(message, dict):
      total += _estimate_text_tokens(str(message), encoding)
      continue
    total += tokens_per_message
    for key, value in message.items():
      if value is None:
        continue
      total += _estimate_text_tokens(str(value), encoding)
      if key == "name":
        total += tokens_per_name
  return total


def estimate_request_enqueued_tokens(spec: BatchRequest) -> int:
  model = str(spec.body.get("model") or MODEL_NAME)
  messages = spec.body.get("messages")
  if isinstance(messages, list):
    return estimate_messages_tokens(messages, model)
  return _estimate_text_tokens(json.dumps(spec.body, sort_keys=True), _get_token_encoding(model))


def shard_specs_by_enqueued_tokens(
  phase_name: str,
  specs: list[BatchRequest],
  token_limit: int,
  samples: int = 1,
  max_requests: int | None = None,
) -> list[BatchShard]:
  if token_limit < 1:
    raise BatchMetricsError(f"--limit must be a positive integer, got: {token_limit}")
  shards: list[BatchShard] = []
  current_specs: list[BatchRequest] = []
  current_tokens = 0

  for spec in specs:
    spec_tokens = estimate_request_enqueued_tokens(spec) * samples
    if spec_tokens > token_limit:
      raise BatchMetricsError(
        f"Request {spec.custom_id} has {spec_tokens} estimated enqueued tokens, "
        f"which exceeds the per-batch limit of {token_limit}."
      )
    too_many_requests = max_requests is not None and (len(current_specs) + 1) * samples > max_requests
    if current_specs and (current_tokens + spec_tokens > token_limit or too_many_requests):
      shards.append(
        BatchShard(
          name=f"{phase_name}/shard_{len(shards) + 1:04d}",
          specs=current_specs,
          enqueued_tokens=current_tokens,
        )
      )
      current_specs = []
      current_tokens = 0
    current_specs.append(spec)
    current_tokens += spec_tokens

  if current_specs:
    shards.append(
      BatchShard(
        name=f"{phase_name}/shard_{len(shards) + 1:04d}",
        specs=current_specs,
        enqueued_tokens=current_tokens,
      )
    )
  return shards


def summarize_batch_plan(
  phase_name: str,
  specs: list[BatchRequest],
  token_limit: int,
  *,
  status: str = "planned",
  samples: int = 1,
  max_requests: int | None = None,
) -> dict[str, Any]:
  shards = shard_specs_by_enqueued_tokens(phase_name, specs, token_limit, samples, max_requests) if specs else []
  return {
    "phase": phase_name,
    "status": status if specs or status != "planned" else "skipped",
    "request_count": len(specs) * samples,
    "batch_count": len(shards),
    "enqueued_tokens": sum(shard.enqueued_tokens for shard in shards),
    "batches": [
      {
        "name": shard.name,
        "request_count": len(shard.specs) * samples,
        "enqueued_tokens": shard.enqueued_tokens,
      }
      for shard in shards
    ],
  }


def shard_progress(items: list[Any], description: str) -> Iterable[Any]:
  try:
    from tqdm.auto import tqdm  # pyright: ignore[reportMissingImports]
  except ImportError:
    return items
  return tqdm(
    items,
    total=len(items),
    desc=description,
    unit="shard",
    disable=not os.sys.stderr.isatty(),
  )


def validate_batch_input(
  path: Path,
  request_count: int,
  max_requests: int = MAX_BATCH_REQUESTS,
  max_bytes: int = MAX_BATCH_FILE_BYTES,
) -> None:
  if request_count > max_requests:
    raise BatchMetricsError(
      f"Batch has {request_count} requests; maximum supported per batch is {max_requests}."
    )
  size = path.stat().st_size
  if size > max_bytes:
    raise BatchMetricsError(
      f"Batch input file is {size} bytes; maximum supported size is {max_bytes} bytes."
    )


def parse_score_prefix(text: str, metric_group: str, custom_id: str) -> int:
  prefix = str(text).split(",", 1)[0].strip()
  match = re.search(r"-?\d+", prefix)
  if match is None:
    raise BatchMetricsError(f"Could not parse {metric_group} score for {custom_id}: {text!r}")
  return int(match.group(0))


def _parse_panas_score_line(line: str) -> tuple[str, str, int] | None:
  stripped = line.strip()
  if not stripped:
    return None

  parts = [part.strip() for part in stripped.split(",")]
  if len(parts) >= 3:
    emotion = re.sub(r"[^a-zA-Z0-9]", "", parts[0]).lower()
    explanation = ",".join(parts[1:-1]).strip()
    score_str = re.sub(r"[^a-zA-Z0-9]", "", parts[-1])
  else:
    words = stripped.split()
    if len(words) < 2:
      return None
    emotion = re.sub(r"[^a-zA-Z0-9]", "", words[0]).lower()
    explanation = ""
    score_str = re.sub(r"[^a-zA-Z0-9]", "", words[-1])

  if emotion not in FEELINGS or not score_str.isdigit():
    return None
  return emotion, explanation, int(score_str)


def iter_panas_score_lines(content: Any) -> Iterable[str]:
  text = str(content or "")
  matches = list(PANAS_ENTRY_START_RE.finditer(text))
  if len(matches) <= 1:
    yield from text.splitlines()
    return

  for index, match in enumerate(matches):
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    segment = text[match.start() : end].strip()
    if segment:
      yield segment


def parse_panas_choices(choices: list[dict[str, Any]], max_iter: int, custom_id: str) -> dict[str, Any]:
  score_dict: dict[str, Any] = {feeling: [] for feeling in FEELINGS}
  explanations: list[dict[str, Any]] = []
  if len(choices) < max_iter:
    raise BatchMetricsError(
      f"PANAS response for {custom_id} returned {len(choices)} choices; expected {max_iter}."
    )

  for choice_index in range(max_iter):
    choice = choices[choice_index]
    message = choice.get("message") if isinstance(choice, dict) else get_field(choice, "message", {})
    content = get_field(message, "content", "") or ""
    for line in iter_panas_score_lines(content):
      parsed = _parse_panas_score_line(line)
      if parsed is None:
        continue
      emotion, explanation, value = parsed
      score_dict[emotion].append(value)
      explanations.append(
        {
          "emotion": emotion,
          "explanation": explanation,
          "score": value,
        }
      )

  missing = [feeling for feeling in FEELINGS if len(score_dict[feeling]) < max_iter]
  if missing:
    raise BatchMetricsError(
      f"PANAS response for {custom_id} is missing scores for: {', '.join(missing)}"
    )
  score_dict["explanations"] = explanations
  return score_dict


def response_usage_cost(
  body: dict[str, Any],
  input_cost_per_1m: float,
  output_cost_per_1m: float,
) -> float:
  usage = body.get("usage") or {}
  prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
  completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
  # Anthropic-only usage fields (absent for OpenAI, so these terms are 0 there).
  cache_write_tokens = usage.get("cache_creation_input_tokens", 0) or 0
  cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
  return (
    (float(prompt_tokens) * input_cost_per_1m)
    + (float(cache_write_tokens) * input_cost_per_1m * CACHE_WRITE_COST_MULTIPLIER)
    + (float(cache_read_tokens) * input_cost_per_1m * CACHE_READ_COST_MULTIPLIER)
    + (float(completion_tokens) * output_cost_per_1m)
  ) / 1_000_000


def parse_batch_output_lines(path: Path) -> dict[str, dict[str, Any]]:
  results: dict[str, dict[str, Any]] = {}
  with path.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
      if not line.strip():
        continue
      try:
        payload = json.loads(line)
      except json.JSONDecodeError as exc:
        raise BatchMetricsError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
      custom_id = payload.get("custom_id")
      if not isinstance(custom_id, str) or not custom_id:
        raise BatchMetricsError(f"Batch output line is missing custom_id at {path}:{line_number}")
      if custom_id in results:
        raise BatchMetricsError(f"Duplicate custom_id in batch output: {custom_id}")
      results[custom_id] = payload
  return results


def collect_batch_output_custom_ids(paths: list[Path]) -> set[str]:
  custom_ids: set[str] = set()
  for path in paths:
    with path.open("r", encoding="utf-8") as f:
      for line_number, line in enumerate(f, start=1):
        if not line.strip():
          continue
        try:
          payload = json.loads(line)
        except json.JSONDecodeError as exc:
          raise BatchMetricsError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        custom_id = payload.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
          raise BatchMetricsError(f"Batch output line is missing custom_id at {path}:{line_number}")
        if custom_id in custom_ids:
          raise BatchMetricsError(f"Duplicate custom_id in resume output: {custom_id}")
        custom_ids.add(custom_id)
  return custom_ids


def filter_specs_missing_from_resume(
  specs: list[BatchRequest],
  resume_custom_ids: set[str],
) -> tuple[list[BatchRequest], list[BatchRequest]]:
  resumed_specs = [spec for spec in specs if spec.custom_id in resume_custom_ids]
  pending_specs = [spec for spec in specs if spec.custom_id not in resume_custom_ids]
  return resumed_specs, pending_specs


def write_matching_batch_output_lines(
  output_jsonls: list[Path],
  request_map: dict[str, dict[str, Any]],
  destination: Path,
) -> None:
  wanted_ids = set(request_map)
  seen_ids: set[str] = set()
  lines: list[str] = []
  for output_jsonl in output_jsonls:
    with output_jsonl.open("r", encoding="utf-8") as f:
      for line_number, line in enumerate(f, start=1):
        if not line.strip():
          continue
        try:
          payload = json.loads(line)
        except json.JSONDecodeError as exc:
          raise BatchMetricsError(f"Invalid JSONL at {output_jsonl}:{line_number}: {exc}") from exc
        custom_id = payload.get("custom_id")
        if custom_id not in wanted_ids:
          continue
        if custom_id in seen_ids:
          raise BatchMetricsError(f"Duplicate custom_id in resume output: {custom_id}")
        seen_ids.add(str(custom_id))
        lines.append(json.dumps(payload, ensure_ascii=True))
  atomic_write_text(destination, "\n".join(lines) + ("\n" if lines else ""))


def parse_phase_outputs(
  output_jsonl: Path,
  request_map: dict[str, dict[str, Any]],
  records_by_filename: dict[str, ConversationRecord],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  lines_by_id = parse_batch_output_lines(output_jsonl)
  unknown = sorted(set(lines_by_id) - set(request_map))
  missing = sorted(set(request_map) - set(lines_by_id))
  if unknown:
    raise BatchMetricsError(f"Batch output contains unknown custom_id values: {', '.join(unknown[:5])}")
  if missing:
    raise BatchMetricsError(f"Batch output is missing responses for: {', '.join(missing[:5])}")

  outputs: dict[str, dict[str, Any]] = {}
  parse_report: dict[str, Any] = {
    "parsed_requests": 0,
    "output_files": {},
    "errors": [],
  }

  for custom_id, spec in request_map.items():
    line = lines_by_id[custom_id]
    error = line.get("error")
    if error:
      raise BatchMetricsError(f"Batch request failed for {custom_id}: {error}")

    response = line.get("response")
    if not isinstance(response, dict):
      raise BatchMetricsError(f"Batch output has no response object for {custom_id}")
    status_code = int(response.get("status_code") or 0)
    if status_code < 200 or status_code >= 300:
      raise BatchMetricsError(f"Batch request {custom_id} returned status {status_code}")

    body = response.get("body")
    if not isinstance(body, dict):
      raise BatchMetricsError(f"Batch response body is missing for {custom_id}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) < max_iter:
      raise BatchMetricsError(
        f"Response for {custom_id} returned {len(choices) if isinstance(choices, list) else 0} "
        f"choices; expected {max_iter}."
      )

    output_file = str(spec["output_file"])
    metric_group = str(spec["metric_group"])
    item_key = str(spec["item_key"])
    cost = response_usage_cost(body, input_cost_per_1m, output_cost_per_1m)
    payload = outputs.setdefault(output_file, {"cost": 0.0})
    payload["cost"] = float(payload.get("cost", 0.0)) + cost

    if metric_group in {"ctrs", "wai"}:
      values = [
        parse_score_prefix(
          get_field(get_field(choice, "message", {}), "content", "") or "",
          metric_group,
          custom_id,
        )
        for choice in choices[:max_iter]
      ]
      payload[item_key] = sum(values) / max_iter
    elif metric_group in {"panas_before", "panas_after"}:
      record = records_by_filename.get(str(spec["input_file"]))
      if record is None:
        raise BatchMetricsError(f"Unknown input file in request map for {custom_id}: {spec['input_file']}")
      panas_payload = parse_panas_choices(choices, max_iter, custom_id)
      panas_payload["cost"] = payload["cost"]
      panas_payload["attitude"] = record.attitude
      outputs[output_file] = panas_payload
    else:
      raise BatchMetricsError(f"Unknown metric group for {custom_id}: {metric_group}")

    parse_report["parsed_requests"] += 1
    output_entry = parse_report["output_files"].setdefault(
      output_file,
      {"metric_group": metric_group, "requests": 0},
    )
    output_entry["requests"] += 1

  validate_parsed_outputs(outputs, request_map)
  return outputs, parse_report


def validate_parsed_outputs(
  outputs: dict[str, dict[str, Any]],
  request_map: dict[str, dict[str, Any]],
) -> None:
  groups_by_output: dict[str, str] = {
    str(spec["output_file"]): str(spec["metric_group"])
    for spec in request_map.values()
  }
  for output_file, payload in outputs.items():
    metric_group = groups_by_output[output_file]
    if metric_group == "ctrs":
      missing = [key for key in CTRS_KEYS if _number(payload.get(key)) is None]
    elif metric_group == "wai":
      missing = [key for key in WAI_KEYS if _number(payload.get(key)) is None]
    elif metric_group in {"panas_before", "panas_after"}:
      missing = [key for key in FEELINGS if not _is_score_list(payload.get(key))]
      if not isinstance(payload.get("explanations"), list):
        missing.append("explanations")
      if not str(payload.get("attitude") or "").strip():
        missing.append("attitude")
    else:
      missing = [f"unknown metric group {metric_group}"]
    if _number(payload.get("cost")) is None:
      missing.append("cost")
    if missing:
      raise BatchMetricsError(f"Parsed output {output_file} is incomplete: {', '.join(missing)}")


def compute_signature(
  input_dir: Path,
  output_dir: Path,
  records: list[ConversationRecord],
  model: str,
  max_iter: int,
  excluded_metrics: set[str] | None = None,
  panas_v2: bool = False,
  azure: bool = False,
  limit: int = DEFAULT_ENQUEUED_TOKEN_LIMIT,
  run_from_json: Path | None = None,
  judge: str = "gpt",
) -> dict[str, Any]:
  files = [
    {
      "name": record.filename,
      "path": str(record.path.resolve()),
      "size": record.path.stat().st_size,
      "mtime_ns": record.path.stat().st_mtime_ns,
    }
    for record in records
  ]
  signature = {
    "input_dir": str(input_dir.resolve()),
    "output_dir": str(output_dir.resolve()),
    "model": model,
    "max_iter": max_iter,
    "excluded_metrics": sorted(excluded_metrics or set()),
    "panas_after_prompt": panas_after_prompt_filename(panas_v2),
    "provider": "azure_openai" if azure else "openai",
    "endpoint": BATCH_ENDPOINT,
    "batch_endpoint": AZURE_BATCH_ENDPOINT if azure else BATCH_ENDPOINT,
    "enqueued_token_limit": limit,
    "run_from_json": str(run_from_json.resolve()) if run_from_json is not None else "",
    "files": files,
  }
  if judge != "gpt":
    # Only non-default judges extend the signature, so existing GPT artifacts keep their hash.
    signature.update(
      {
        "judge": judge,
        "provider": "anthropic",
        "endpoint": ANTHROPIC_BATCH_ENDPOINT,
        "batch_endpoint": ANTHROPIC_BATCH_ENDPOINT,
        "anthropic_params": {
          "max_tokens": ANTHROPIC_MAX_TOKENS,
          "thinking": ANTHROPIC_THINKING,
          "effort": ANTHROPIC_EFFORT,
          "cache_control": ANTHROPIC_CACHE_CONTROL,
        },
      }
    )
  return signature


def signature_hash(signature: dict[str, Any]) -> str:
  rendered = json.dumps(signature, sort_keys=True, ensure_ascii=True)
  import hashlib

  return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def save_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
  manifest["updated_at"] = utc_now()
  atomic_write_json(artifact_dir / "manifest.json", manifest)


def load_manifest(path: Path) -> dict[str, Any] | None:
  try:
    payload = read_json(path)
  except (OSError, json.JSONDecodeError):
    return None
  return payload if isinstance(payload, dict) else None


def find_resume_manifest(artifact_root: Path, sig_hash: str) -> tuple[Path, dict[str, Any]] | None:
  candidates: list[tuple[float, Path, dict[str, Any]]] = []
  if not artifact_root.is_dir():
    return None
  for manifest_path in artifact_root.glob("*/manifest.json"):
    manifest = load_manifest(manifest_path)
    if not manifest:
      continue
    if manifest.get("signature_hash") != sig_hash:
      continue
    if str(manifest.get("status") or "") in DONE_MANIFEST_STATUSES:
      continue
    candidates.append((manifest_path.stat().st_mtime, manifest_path.parent, manifest))
  if not candidates:
    return None
  _, artifact_dir, manifest = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
  return artifact_dir, manifest


def create_manifest(
  artifact_root: Path,
  signature: dict[str, Any],
  sig_hash: str,
) -> tuple[Path, dict[str, Any]]:
  run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
  artifact_dir = artifact_root / run_id
  artifact_dir.mkdir(parents=True, exist_ok=False)
  manifest = {
    "version": 1,
    "run_id": run_id,
    "status": "running",
    "created_at": utc_now(),
    "updated_at": utc_now(),
    "signature_hash": sig_hash,
    "signature": signature,
    "model": signature["model"],
    "provider": signature["provider"],
    "endpoint": signature["endpoint"],
    "batch_endpoint": signature["batch_endpoint"],
    "enqueued_token_limit": signature["enqueued_token_limit"],
    "completion_window": COMPLETION_WINDOW,
    "phases": {},
    "skipped": {},
  }
  save_manifest(artifact_dir, manifest)
  return artifact_dir, manifest


def load_or_create_manifest(
  output_dir: Path,
  signature: dict[str, Any],
) -> tuple[Path, dict[str, Any], bool]:
  artifact_root = output_dir / "_batch_artifacts"
  artifact_root.mkdir(parents=True, exist_ok=True)
  sig_hash = signature_hash(signature)
  resumed = find_resume_manifest(artifact_root, sig_hash)
  if resumed is not None:
    artifact_dir, manifest = resumed
    return artifact_dir, manifest, True
  artifact_dir, manifest = create_manifest(artifact_root, signature, sig_hash)
  return artifact_dir, manifest, False


def staged_outputs_path(artifact_dir: Path) -> Path:
  return artifact_dir / "staged_outputs.json"


def load_staged_outputs(artifact_dir: Path) -> dict[str, dict[str, Any]]:
  path = staged_outputs_path(artifact_dir)
  if not path.is_file():
    return {}
  payload = read_json(path)
  if not isinstance(payload, dict):
    return {}
  return {
    str(key): value
    for key, value in payload.items()
    if isinstance(value, dict)
  }


def save_staged_outputs(artifact_dir: Path, staged_outputs: dict[str, dict[str, Any]]) -> None:
  atomic_write_json(staged_outputs_path(artifact_dir), staged_outputs)


def collect_existing_before_payloads(
  records: list[ConversationRecord],
  output_dir: Path,
  staged_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for record in records:
    rel = (Path("panas_before") / record.filename).as_posix()
    if rel in staged_outputs:
      payloads[record.filename] = staged_outputs[rel]
      continue
    path = output_dir / rel
    if is_panas_complete(path):
      payload = read_json(path)
      if isinstance(payload, dict):
        payloads[record.filename] = payload
  return payloads


def build_skipped_summary(
  records: list[ConversationRecord],
  output_dir: Path,
  excluded_metrics: set[str] | None = None,
) -> dict[str, list[str]]:
  skipped = {
    "ctrs": [],
    "wai": [],
    "panas_before": [],
    "panas_after": [],
  }
  excluded = excluded_metrics or set()
  for record in records:
    for group in skipped:
      if group in excluded or (group in {"panas_before", "panas_after"} and "panas" in excluded):
        continue
      if output_is_complete(group, output_dir / group / record.filename):
        skipped[group].append(record.filename)
  return skipped


def all_outputs_complete(
  records: list[ConversationRecord],
  output_dir: Path,
  metric_group: str,
) -> bool:
  return all(
    output_is_complete(metric_group, output_dir / metric_group / record.filename)
    for record in records
  )


def mark_phase_skipped_existing_outputs(
  manifest: dict[str, Any],
  phase_name: str,
  records: list[ConversationRecord],
) -> None:
  phase = manifest.setdefault("phases", {}).setdefault(phase_name, {})
  phase["status"] = "skipped_existing_outputs"
  phase["request_count"] = 0
  phase["skipped_existing_outputs"] = [record.filename for record in records]


def placeholder_panas_before_payload(record: ConversationRecord) -> dict[str, Any]:
  return {
    **{feeling: [3] for feeling in FEELINGS},
    "explanations": [],
    "attitude": record.attitude,
    "cost": 0.0,
  }


def build_dry_run_before_payloads(
  records: list[ConversationRecord],
  output_dir: Path,
  staged_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
  payloads = collect_existing_before_payloads(records, output_dir, staged_outputs)
  for record in records:
    payloads.setdefault(record.filename, placeholder_panas_before_payload(record))
  return payloads


def content_response_to_text(response: Any) -> str:
  if hasattr(response, "text"):
    text_value = response.text
    if callable(text_value):
      text_value = text_value()
    if isinstance(text_value, str):
      return text_value
  if hasattr(response, "read"):
    data = response.read()
  elif hasattr(response, "content"):
    data = response.content
  else:
    data = response
  if isinstance(data, bytes):
    return data.decode("utf-8")
  return str(data)


def download_file(client: Any, file_id: str, destination: Path) -> None:
  content = client.files.content(file_id)
  atomic_write_text(destination, content_response_to_text(content))


def phase_artifact_path(artifact_dir: Path, phase_name: str, filename: str) -> Path:
  return artifact_dir / phase_name / filename


def load_request_map(path: Path) -> dict[str, dict[str, Any]]:
  payload = read_json(path)
  if not isinstance(payload, dict):
    raise BatchMetricsError(f"Request map must be a JSON object: {path}")
  return {
    str(key): value
    for key, value in payload.items()
    if isinstance(value, dict)
  }


def prepare_phase_artifacts(
  artifact_dir: Path,
  phase_name: str,
  specs: list[BatchRequest],
  manifest: dict[str, Any],
  judge: str = "gpt",
  max_iter: int = 1,
) -> dict[str, Any]:
  phase_dir = artifact_dir / phase_name
  phase_dir.mkdir(parents=True, exist_ok=True)
  phases = manifest.setdefault("phases", {})
  phase = phases.setdefault(phase_name, {})

  if not specs and not phase.get("request_map"):
    phase.update({"status": "skipped", "request_count": 0})
    return phase

  request_map_path = phase_artifact_path(artifact_dir, phase_name, "request_map.json")
  input_jsonl_path = phase_artifact_path(artifact_dir, phase_name, "input.jsonl")

  if specs and not request_map_path.exists():
    request_map = request_map_from_specs(specs)
    if judge == "sonnet":
      for index, spec in enumerate(specs):
        request_map[spec.custom_id]["anthropic_custom_ids"] = anthropic_custom_ids(index, max_iter)
    atomic_write_json(request_map_path, request_map)
  if specs and not input_jsonl_path.exists():
    if judge == "sonnet":
      write_anthropic_jsonl(input_jsonl_path, specs, load_request_map(request_map_path))
    else:
      write_jsonl(input_jsonl_path, specs)

  request_count = len(load_request_map(request_map_path)) if request_map_path.exists() else len(specs)
  if judge == "sonnet":
    request_count *= max_iter
    validate_batch_input(input_jsonl_path, request_count, ANTHROPIC_MAX_BATCH_REQUESTS, ANTHROPIC_MAX_BATCH_BYTES)
  else:
    validate_batch_input(input_jsonl_path, request_count)
  phase.setdefault("status", "prepared")
  phase["request_count"] = request_count
  phase["request_map"] = str(request_map_path)
  phase["input_jsonl"] = str(input_jsonl_path)
  return phase


def upload_and_create_batch(
  client: Any,
  phase_name: str,
  phase: dict[str, Any],
  manifest: dict[str, Any],
  artifact_dir: Path,
  batch_endpoint: str = BATCH_ENDPOINT,
) -> None:
  if phase.get("batch_id"):
    return
  input_jsonl = Path(str(phase["input_jsonl"]))
  with input_jsonl.open("rb") as f:
    uploaded_file = client.files.create(file=f, purpose="batch")
  uploaded_file_id = get_field(uploaded_file, "id")
  if not uploaded_file_id:
    raise BatchMetricsError(f"OpenAI file upload did not return a file id for phase {phase_name}")

  phase["input_file_id"] = uploaded_file_id
  phase["uploaded_file"] = to_plain(uploaded_file)
  phase["status"] = "uploaded"
  save_manifest(artifact_dir, manifest)

  batch = client.batches.create(
    input_file_id=uploaded_file_id,
    endpoint=batch_endpoint,
    completion_window=COMPLETION_WINDOW,
    metadata={
      "description": "ReasonCBT batch metric evaluation",
      "phase": phase_name,
    },
  )
  batch_id = get_field(batch, "id")
  if not batch_id:
    raise BatchMetricsError(f"OpenAI batch creation did not return a batch id for phase {phase_name}")
  phase["batch_id"] = batch_id
  phase["status"] = get_field(batch, "status", "submitted")
  phase["batch"] = to_plain(batch)
  save_manifest(artifact_dir, manifest)


def download_phase_outputs(
  client: Any,
  phase_name: str,
  phase: dict[str, Any],
  artifact_dir: Path,
) -> None:
  output_file_id = phase.get("output_file_id")
  error_file_id = phase.get("error_file_id")
  if output_file_id and not phase.get("output_jsonl"):
    output_path = phase_artifact_path(artifact_dir, phase_name, "output.jsonl")
    download_file(client, str(output_file_id), output_path)
    phase["output_jsonl"] = str(output_path)
  if error_file_id and not phase.get("error_jsonl"):
    error_path = phase_artifact_path(artifact_dir, phase_name, "error.jsonl")
    download_file(client, str(error_file_id), error_path)
    phase["error_jsonl"] = str(error_path)


def poll_until_complete(
  client: Any,
  phase_name: str,
  phase: dict[str, Any],
  manifest: dict[str, Any],
  artifact_dir: Path,
  poll_interval: float,
  local_timeout_hours: float,
) -> None:
  if phase.get("output_jsonl"):
    return
  batch_id = str(phase.get("batch_id") or "")
  if not batch_id:
    raise BatchMetricsError(f"Phase {phase_name} cannot be polled before a batch is created.")

  deadline = time.monotonic() + (local_timeout_hours * 3600)
  while True:
    batch = client.batches.retrieve(batch_id)
    batch_plain = to_plain(batch)
    status = str(get_field(batch, "status", batch_plain.get("status", "")))
    phase["status"] = status
    phase["batch"] = batch_plain
    phase["output_file_id"] = get_field(batch, "output_file_id", batch_plain.get("output_file_id"))
    phase["error_file_id"] = get_field(batch, "error_file_id", batch_plain.get("error_file_id"))
    phase["request_counts"] = to_plain(get_field(batch, "request_counts", batch_plain.get("request_counts")))
    save_manifest(artifact_dir, manifest)

    if status in FINAL_STATUSES:
      download_phase_outputs(client, phase_name, phase, artifact_dir)
      save_manifest(artifact_dir, manifest)
      if status in FAILED_STATUSES:
        raise BatchMetricsError(f"Batch phase {phase_name} ended with status {status}.")
      counts = phase.get("request_counts") or {}
      failed_count = get_field(counts, "failed", 0) or 0
      if int(failed_count) > 0:
        raise BatchMetricsError(f"Batch phase {phase_name} completed with {failed_count} failed requests.")
      if not phase.get("output_jsonl"):
        raise BatchMetricsError(f"Batch phase {phase_name} completed without an output file.")
      return

    if time.monotonic() >= deadline:
      raise BatchMetricsError(
        f"Timed out waiting for batch phase {phase_name} after {local_timeout_hours} hours."
      )
    time.sleep(max(0.0, poll_interval))


def anthropic_create_batch(
  client: Any,
  phase_name: str,
  phase: dict[str, Any],
  manifest: dict[str, Any],
  artifact_dir: Path,
) -> None:
  if phase.get("batch_id"):
    return
  with Path(str(phase["input_jsonl"])).open("r", encoding="utf-8") as f:
    requests = [json.loads(line) for line in f if line.strip()]
  batch = client.messages.batches.create(requests=requests)
  if not batch.id:
    raise BatchMetricsError(f"Anthropic batch creation did not return a batch id for phase {phase_name}")
  phase["batch_id"] = batch.id
  phase["status"] = batch.processing_status
  phase["batch"] = batch.model_dump(mode="json")
  save_manifest(artifact_dir, manifest)


def convert_anthropic_results(
  results_by_id: dict[str, dict[str, Any]],
  request_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  # Merge the max_iter single-sample results of each item into one OpenAI-like output line
  # (choices + summed usage), so parse_phase_outputs and --resume read them unchanged.
  output_lines: list[dict[str, Any]] = []
  error_lines: list[dict[str, Any]] = []
  for custom_id, spec in request_map.items():
    choices = []
    usage = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
    }
    failed = {}
    for index, anthropic_id in enumerate(spec["anthropic_custom_ids"]):
      result = (results_by_id.get(anthropic_id) or {}).get("result") or {"type": "missing"}
      if result.get("type") != "succeeded":
        failed[anthropic_id] = result
        continue
      message = result["message"]
      text = "".join(block["text"] for block in message.get("content") or [] if block.get("type") == "text")
      choices.append(
        {
          "index": index,
          "message": {"role": "assistant", "content": text},
          "finish_reason": message.get("stop_reason"),
        }
      )
      for key in usage:
        usage[key] += (message.get("usage") or {}).get(key) or 0
    if failed:
      # Same as an OpenAI error-file line: the item counts as a failed request.
      error_lines.append(
        {"custom_id": custom_id, "response": None, "error": {"code": "anthropic_sample_failed", "samples": failed}}
      )
      continue
    output_lines.append(
      {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": {"model": message.get("model"), "choices": choices, "usage": usage}},
        "error": None,
      }
    )
  return output_lines, error_lines


def anthropic_download_phase_outputs(
  client: Any,
  phase_name: str,
  phase: dict[str, Any],
  artifact_dir: Path,
) -> None:
  results_path = phase_artifact_path(artifact_dir, phase_name, "results.jsonl")
  if not phase.get("results_jsonl"):
    lines = [
      json.dumps(result.model_dump(mode="json"), ensure_ascii=True)
      for result in client.messages.batches.results(phase["batch_id"])
    ]
    atomic_write_text(results_path, "\n".join(lines) + ("\n" if lines else ""))
    phase["results_jsonl"] = str(results_path)

  with results_path.open("r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f if line.strip()]
  results_by_id = {result["custom_id"]: result for result in results}
  request_map = load_request_map(Path(str(phase["request_map"])))
  output_lines, error_lines = convert_anthropic_results(results_by_id, request_map)
  output_path = phase_artifact_path(artifact_dir, phase_name, "output.jsonl")
  atomic_write_text(output_path, "".join(json.dumps(line, ensure_ascii=True) + "\n" for line in output_lines))
  phase["output_jsonl"] = str(output_path)
  if error_lines:
    error_path = phase_artifact_path(artifact_dir, phase_name, "error.jsonl")
    atomic_write_text(error_path, "".join(json.dumps(line, ensure_ascii=True) + "\n" for line in error_lines))
    phase["error_jsonl"] = str(error_path)
  phase["failed_item_count"] = len(error_lines)


def anthropic_poll_until_complete(
  client: Any,
  phase_name: str,
  phase: dict[str, Any],
  manifest: dict[str, Any],
  artifact_dir: Path,
  poll_interval: float,
  local_timeout_hours: float,
) -> None:
  if phase.get("output_jsonl"):
    return
  batch_id = str(phase.get("batch_id") or "")
  if not batch_id:
    raise BatchMetricsError(f"Phase {phase_name} cannot be polled before a batch is created.")

  deadline = time.monotonic() + (local_timeout_hours * 3600)
  while True:
    batch = client.messages.batches.retrieve(batch_id)
    phase["status"] = batch.processing_status
    phase["batch"] = batch.model_dump(mode="json")
    phase["request_counts"] = batch.request_counts.model_dump(mode="json")
    save_manifest(artifact_dir, manifest)

    if batch.processing_status == "ended":
      anthropic_download_phase_outputs(client, phase_name, phase, artifact_dir)
      save_manifest(artifact_dir, manifest)
      if phase["failed_item_count"] > 0:
        counts = phase["request_counts"]
        raise BatchMetricsError(
          f"Batch phase {phase_name} completed with {phase['failed_item_count']} failed requests "
          f"(samples errored={counts.get('errored')}, expired={counts.get('expired')}, "
          f"canceled={counts.get('canceled')})."
        )
      return

    if time.monotonic() >= deadline:
      raise BatchMetricsError(
        f"Timed out waiting for batch phase {phase_name} after {local_timeout_hours} hours."
      )
    time.sleep(max(0.0, poll_interval))


def run_phase(
  client: Any,
  phase_name: str,
  specs: list[BatchRequest],
  records_by_filename: dict[str, ConversationRecord],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
  poll_interval: float,
  local_timeout_hours: float,
  artifact_dir: Path,
  manifest: dict[str, Any],
  staged_outputs: dict[str, dict[str, Any]],
  batch_endpoint: str = BATCH_ENDPOINT,
) -> dict[str, dict[str, Any]]:
  phase = prepare_phase_artifacts(artifact_dir, phase_name, specs, manifest)
  save_manifest(artifact_dir, manifest)

  if phase.get("status") == "skipped":
    return {}
  if phase.get("status") == "parsed":
    return load_staged_outputs(artifact_dir)

  upload_and_create_batch(client, phase_name, phase, manifest, artifact_dir, batch_endpoint)
  poll_until_complete(client, phase_name, phase, manifest, artifact_dir, poll_interval, local_timeout_hours)

  request_map = load_request_map(Path(str(phase["request_map"])))
  output_jsonl = Path(str(phase["output_jsonl"]))
  parsed_outputs, parse_report = parse_phase_outputs(
    output_jsonl,
    request_map,
    records_by_filename,
    max_iter,
    input_cost_per_1m,
    output_cost_per_1m,
  )
  report_path = phase_artifact_path(artifact_dir, phase_name, "parse_report.json")
  atomic_write_json(report_path, parse_report)
  phase["parse_report"] = str(report_path)
  phase["status"] = "parsed"

  staged_outputs.update(parsed_outputs)
  save_staged_outputs(artifact_dir, staged_outputs)
  save_manifest(artifact_dir, manifest)
  return parsed_outputs


def parse_phase_output_jsonls(
  phase_name: str,
  output_jsonls: list[Path],
  specs: list[BatchRequest],
  records_by_filename: dict[str, ConversationRecord],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
  artifact_dir: Path,
  manifest: dict[str, Any],
  staged_outputs: dict[str, dict[str, Any]],
  *,
  report_filename: str = "parse_report.json",
) -> dict[str, dict[str, Any]]:
  phase_dir = artifact_dir / phase_name
  phase_dir.mkdir(parents=True, exist_ok=True)
  request_map = request_map_from_specs(specs)
  request_map_path = phase_artifact_path(artifact_dir, phase_name, "request_map.json")
  atomic_write_json(request_map_path, request_map)
  combined_output_jsonl = phase_artifact_path(artifact_dir, phase_name, "combined_output.jsonl")
  write_matching_batch_output_lines(output_jsonls, request_map, combined_output_jsonl)
  parsed_outputs, parse_report = parse_phase_outputs(
    combined_output_jsonl,
    request_map,
    records_by_filename,
    max_iter,
    input_cost_per_1m,
    output_cost_per_1m,
  )
  report_path = phase_artifact_path(artifact_dir, phase_name, report_filename)
  atomic_write_json(report_path, parse_report)
  phase = manifest.setdefault("phases", {}).setdefault(phase_name, {})
  phase["request_map"] = str(request_map_path)
  phase["output_jsonl"] = str(combined_output_jsonl)
  phase["parse_report"] = str(report_path)
  phase["status"] = "parsed"
  staged_outputs.update(parsed_outputs)
  save_staged_outputs(artifact_dir, staged_outputs)
  save_manifest(artifact_dir, manifest)
  return parsed_outputs


def run_phase_shards(
  client: Any,
  phase_name: str,
  specs: list[BatchRequest],
  records_by_filename: dict[str, ConversationRecord],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
  poll_interval: float,
  local_timeout_hours: float,
  artifact_dir: Path,
  manifest: dict[str, Any],
  staged_outputs: dict[str, dict[str, Any]],
  batch_endpoint: str = BATCH_ENDPOINT,
  token_limit: int = DEFAULT_ENQUEUED_TOKEN_LIMIT,
  resume_jsonls: list[Path] | None = None,
  resume_custom_ids: set[str] | None = None,
  judge: str = "gpt",
) -> dict[str, dict[str, Any]]:
  phase = manifest.setdefault("phases", {}).setdefault(phase_name, {})
  if phase.get("status") == "parsed":
    return load_staged_outputs(artifact_dir)
  if not specs:
    phase.update({"status": "skipped", "request_count": 0, "batch_count": 0})
    save_manifest(artifact_dir, manifest)
    return {}

  resumed_specs, pending_specs = filter_specs_missing_from_resume(specs, resume_custom_ids or set())
  pending_request_count = len(pending_specs)
  shard_token_limit, samples, max_requests = judge_shard_options(judge, max_iter, token_limit)
  shards = (
    shard_specs_by_enqueued_tokens(phase_name, pending_specs, shard_token_limit, samples, max_requests)
    if pending_specs
    else []
  )
  phase.update(
    {
      "status": "running",
      "request_count": len(specs),
      "resumed_request_count": len(resumed_specs),
      "submitted_request_count": pending_request_count,
      "batch_count": len(shards),
      "enqueued_tokens": sum(shard.enqueued_tokens for shard in shards),
      "resume_jsonls": [str(path) for path in (resume_jsonls or [])],
      "shards": [
        {
          "name": shard.name,
          "request_count": len(shard.specs),
          "enqueued_tokens": shard.enqueued_tokens,
        }
        for shard in shards
      ],
    }
  )
  save_manifest(artifact_dir, manifest)

  shard_phases: list[dict[str, Any]] = []
  progress_items = list(enumerate(shards, start=1))
  for index, shard in shard_progress(progress_items, f"Processing {phase_name}"):
    shard_name = phase_name if len(shards) == 1 else shard.name
    shard_specs = pending_specs if len(shards) == 1 else shard.specs
    phase["current_shard"] = {
      "index": index,
      "name": shard_name,
      "request_count": len(shard_specs),
      "enqueued_tokens": shard.enqueued_tokens,
    }
    save_manifest(artifact_dir, manifest)
    shard_phase = prepare_phase_artifacts(artifact_dir, shard_name, shard_specs, manifest, judge, max_iter)
    save_manifest(artifact_dir, manifest)
    if judge == "sonnet":
      if shard_phase.get("status") != "parsed" and not shard_phase.get("batch_id"):
        anthropic_create_batch(client, shard_name, shard_phase, manifest, artifact_dir)
      if not shard_phase.get("output_jsonl"):
        anthropic_poll_until_complete(
          client,
          shard_name,
          shard_phase,
          manifest,
          artifact_dir,
          poll_interval,
          local_timeout_hours,
        )
      shard_phases.append(shard_phase)
      continue
    if shard_phase.get("status") != "parsed" and not shard_phase.get("batch_id"):
      upload_and_create_batch(client, shard_name, shard_phase, manifest, artifact_dir, batch_endpoint)
    if not shard_phase.get("output_jsonl"):
      poll_until_complete(
        client,
        shard_name,
        shard_phase,
        manifest,
        artifact_dir,
        poll_interval,
        local_timeout_hours,
      )
    shard_phases.append(shard_phase)

  output_jsonls = [*(resume_jsonls or []), *[Path(str(shard_phase["output_jsonl"])) for shard_phase in shard_phases]]
  parsed_outputs = parse_phase_output_jsonls(
    phase_name,
    output_jsonls,
    specs,
    records_by_filename,
    max_iter,
    input_cost_per_1m,
    output_cost_per_1m,
    artifact_dir,
    manifest,
    staged_outputs,
  )
  phase["status"] = "parsed"
  phase["parsed_output_count"] = len(parsed_outputs)
  save_manifest(artifact_dir, manifest)
  return parsed_outputs


def parse_phase_from_resume_jsonls(
  phase_name: str,
  resume_jsonls: list[Path],
  specs: list[BatchRequest],
  records_by_filename: dict[str, ConversationRecord],
  max_iter: int,
  input_cost_per_1m: float,
  output_cost_per_1m: float,
  artifact_dir: Path,
  manifest: dict[str, Any],
  staged_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
  phase = manifest.setdefault("phases", {}).setdefault(phase_name, {})
  if not specs:
    phase.update({"status": "skipped", "request_count": 0, "batch_count": 0})
    save_manifest(artifact_dir, manifest)
    return {}
  phase.update(
    {
      "status": "resuming_from_output_jsonl",
      "request_count": len(specs),
      "resume_jsonls": [str(path) for path in resume_jsonls],
    }
  )
  save_manifest(artifact_dir, manifest)
  return parse_phase_output_jsonls(
    phase_name,
    resume_jsonls,
    specs,
    records_by_filename,
    max_iter,
    input_cost_per_1m,
    output_cost_per_1m,
    artifact_dir,
    manifest,
    staged_outputs,
    report_filename="resume_parse_report.json",
  )


def write_final_outputs(output_dir: Path, staged_outputs: dict[str, dict[str, Any]]) -> None:
  for rel_path, payload in sorted(staged_outputs.items()):
    atomic_write_json(output_dir / rel_path, payload)


def make_openai_client(*, azure: bool = False) -> Any:
  if load_dotenv is not None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
  if azure:
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = (
      os.getenv("AZURE_OPENAI_API_VERSION")
      or os.getenv("OPENAI_API_VERSION")
      or DEFAULT_AZURE_OPENAI_API_VERSION
    )
    if not api_key:
      raise BatchMetricsError("Missing Azure OpenAI API key. Set AZURE_OPENAI_KEY.")
    if not endpoint:
      raise BatchMetricsError("Missing Azure OpenAI endpoint. Set AZURE_OPENAI_ENDPOINT.")
    try:
      from openai import AzureOpenAI
    except ImportError as exc:
      raise BatchMetricsError("The openai Python package is required to run batch metrics.") from exc
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

  api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise BatchMetricsError("Missing OpenAI API key. Set GPT_API_KEY or OPENAI_API_KEY.")
  try:
    from openai import OpenAI
  except ImportError as exc:
    raise BatchMetricsError("The openai Python package is required to run batch metrics.") from exc
  return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


def make_anthropic_client() -> Any:
  if load_dotenv is not None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
  api_key = os.getenv("ANTHROPIC_API_KEY")
  if not api_key:
    raise BatchMetricsError("Missing Anthropic API key. Set ANTHROPIC_API_KEY.")
  try:
    import anthropic
  except ImportError as exc:
    raise BatchMetricsError("The anthropic Python package is required for --judge sonnet.") from exc
  return anthropic.Anthropic(api_key=api_key)


def make_judge_client(judge: str, azure: bool) -> Any:
  return make_anthropic_client() if judge == "sonnet" else make_openai_client(azure=azure)


def phase_requires_client(manifest: dict[str, Any], phase_name: str, specs: list[BatchRequest]) -> bool:
  phase = manifest.get("phases", {}).get(phase_name, {})
  if str(phase.get("status") or "") in {"skipped", "skipped_existing_outputs", "parsed"}:
    return False
  if specs:
    return True
  return bool(phase.get("batch_id") or phase.get("input_jsonl") or phase.get("request_map"))


def build_batch_plan(
  records: list[ConversationRecord],
  output_dir: Path,
  model: str,
  max_iter: int,
  excluded_metrics: set[str],
  panas_v2: bool,
  token_limit: int,
  resume_custom_ids: set[str] | None = None,
  force_recompute: bool = False,
  judge: str = "gpt",
) -> tuple[dict[str, Any], list[BatchRequest], list[BatchRequest]]:
  if token_limit < 1:
    raise BatchMetricsError(f"--limit must be a positive integer, got: {token_limit}")
  token_limit, samples, max_requests = judge_shard_options(judge, max_iter, token_limit)

  panas_before_complete = (
    True
    if "panas" in excluded_metrics
    else False if force_recompute else all_outputs_complete(records, output_dir, "panas_before")
  )
  phase1_specs = (
    []
    if "panas" in excluded_metrics or panas_before_complete
    else build_panas_before_requests(records, output_dir, model, max_iter, force_recompute=force_recompute)
  )
  before_payloads = (
    {}
    if "panas" in excluded_metrics
    else build_dry_run_before_payloads(records, output_dir, {})
  )
  phase2_specs = build_main_requests(
    records,
    output_dir,
    before_payloads,
    model,
    max_iter,
    excluded_metrics,
    panas_v2=panas_v2,
    force_recompute=force_recompute,
  )
  resume_ids = resume_custom_ids or set()
  phase1_resumed, phase1_pending = filter_specs_missing_from_resume(phase1_specs, resume_ids)
  phase2_resumed, phase2_pending = filter_specs_missing_from_resume(phase2_specs, resume_ids)

  phase1_status = "skipped_existing_outputs" if panas_before_complete and "panas" not in excluded_metrics else "planned"
  phase_summaries = [
    {
      **summarize_batch_plan(
        "panas_before", phase1_pending, token_limit, status=phase1_status, samples=samples, max_requests=max_requests
      ),
      "request_count": len(phase1_specs) * samples,
      "resumed_request_count": len(phase1_resumed) * samples,
      "pending_request_count": len(phase1_pending) * samples,
    },
    {
      **summarize_batch_plan("main", phase2_pending, token_limit, samples=samples, max_requests=max_requests),
      "request_count": len(phase2_specs) * samples,
      "resumed_request_count": len(phase2_resumed) * samples,
      "pending_request_count": len(phase2_pending) * samples,
    },
  ]
  plan = {
    "dry_run": True,
    "enqueued_token_limit": token_limit,
    "phase_count": len(phase_summaries),
    "batch_count": sum(int(phase["batch_count"]) for phase in phase_summaries),
    "request_count": sum(int(phase["request_count"]) for phase in phase_summaries),
    "enqueued_tokens": sum(int(phase["enqueued_tokens"]) for phase in phase_summaries),
    "phases": phase_summaries,
  }
  if judge != "gpt":
    plan.update({"judge": judge, "model": model, "requests_per_item": samples})
  return plan, phase1_specs, phase2_specs


def run_batch_metrics(
  input_dir: str | Path,
  output_dir: str | Path | None = None,
  max_iter: int = DEFAULT_MAX_ITER,
  *,
  client: Any | None = None,
  model: str | None = None,
  judge: str = "gpt",
  exc_wai: bool = False,
  exc_ctrs: bool = False,
  exc_panas: bool = False,
  panas_v2: bool = False,
  poll_interval: float = DEFAULT_POLL_INTERVAL,
  local_timeout_hours: float = DEFAULT_LOCAL_TIMEOUT_HOURS,
  input_cost_per_1m: float | None = None,
  output_cost_per_1m: float | None = None,
  azure: bool = False,
  limit: int = DEFAULT_ENQUEUED_TOKEN_LIMIT,
  dry_run: bool = False,
  resume_jsonls: list[str | Path] | None = None,
  run_from_json: str | Path | None = None,
) -> dict[str, Any]:
  if max_iter < 1:
    raise BatchMetricsError(f"MAX_ITER must be a positive integer, got: {max_iter}")
  if limit < 1:
    raise BatchMetricsError(f"--limit must be a positive integer, got: {limit}")
  if judge not in JUDGE_MODELS:
    raise BatchMetricsError(f"Unknown judge: {judge}")
  if azure and judge != "gpt":
    raise BatchMetricsError("--azure is only supported with --judge gpt.")
  model = model or JUDGE_MODELS[judge]
  if input_cost_per_1m is None:
    input_cost_per_1m = JUDGE_COSTS_PER_1M[judge][0]
  if output_cost_per_1m is None:
    output_cost_per_1m = JUDGE_COSTS_PER_1M[judge][1]
  resume_paths = [Path(path).resolve() for path in (resume_jsonls or [])]
  for resume_path in resume_paths:
    if not resume_path.is_file():
      raise BatchMetricsError(f"Resume batch output JSONL does not exist: {resume_path}")
  resume_custom_ids = collect_batch_output_custom_ids(resume_paths) if resume_paths else set()
  input_path = Path(input_dir).resolve()
  run_from_json_path = Path(run_from_json).resolve() if run_from_json is not None else None
  if run_from_json_path is not None and not run_from_json_path.is_file():
    raise BatchMetricsError(f"--run-from-json file does not exist: {run_from_json_path}")
  output_path = (
    Path(output_dir).resolve()
    if output_dir is not None
    else (EVALUATION_ROOT / "results").resolve()
  )
  excluded_metrics = {
    metric
    for metric, excluded in (
      ("wai", exc_wai),
      ("ctrs", exc_ctrs),
      ("panas", exc_panas),
    )
    if excluded
  }
  records = (
    discover_conversations_from_run_json(input_path, run_from_json_path)
    if run_from_json_path is not None
    else discover_conversations(input_path)
  )
  records_by_filename = {record.filename: record for record in records}
  if not dry_run:
    ensure_output_dirs(output_path)
    if run_from_json_path is not None:
      remove_metric_outputs_for_records(output_path, records, excluded_metrics)

  if dry_run:
    plan, _, _ = build_batch_plan(
      records,
      output_path,
      model,
      max_iter,
      excluded_metrics,
      panas_v2,
      limit,
      resume_custom_ids,
      force_recompute=run_from_json_path is not None,
      judge=judge,
    )
    plan["signature"] = compute_signature(
      input_path,
      output_path,
      records,
      model,
      max_iter,
      excluded_metrics,
      panas_v2=panas_v2,
      azure=azure,
      limit=limit,
      run_from_json=run_from_json_path,
      judge=judge,
    )
    return plan

  signature = compute_signature(
    input_path,
    output_path,
    records,
    model,
    max_iter,
    excluded_metrics,
    panas_v2=panas_v2,
    azure=azure,
    limit=limit,
    run_from_json=run_from_json_path,
    judge=judge,
  )
  artifact_dir, manifest, resumed = load_or_create_manifest(output_path, signature)
  if not resumed:
    manifest["excluded_metrics"] = sorted(excluded_metrics)
    manifest["panas_after_prompt"] = panas_after_prompt_filename(panas_v2)
    if run_from_json_path is not None:
      manifest["run_from_json"] = str(run_from_json_path)
      manifest["run_from_json_count"] = len(records)
    manifest["skipped"] = build_skipped_summary(records, output_path, excluded_metrics)
    save_manifest(artifact_dir, manifest)

  staged_outputs = load_staged_outputs(artifact_dir)
  active_client = client
  batch_endpoint = AZURE_BATCH_ENDPOINT if azure else BATCH_ENDPOINT

  try:
    panas_before_complete = (
      False
      if exc_panas
      else all_outputs_complete(records, output_path, "panas_before")
    )
    phase1_specs = (
      []
      if exc_panas or panas_before_complete
      else build_panas_before_requests(records, output_path, model, max_iter)
    )
    if exc_panas:
      manifest.setdefault("phases", {}).setdefault("panas_before", {}).update(
        {"status": "skipped", "request_count": 0}
      )
      save_manifest(artifact_dir, manifest)
    elif panas_before_complete:
      mark_phase_skipped_existing_outputs(manifest, "panas_before", records)
      save_manifest(artifact_dir, manifest)
    else:
      _, pending_phase1_specs = filter_specs_missing_from_resume(phase1_specs, resume_custom_ids)
      if active_client is None and phase_requires_client(manifest, "panas_before", pending_phase1_specs):
        active_client = make_judge_client(judge, azure)
      run_phase_shards(
        active_client,
        "panas_before",
        phase1_specs,
        records_by_filename,
        max_iter,
        input_cost_per_1m,
        output_cost_per_1m,
        poll_interval,
        local_timeout_hours,
        artifact_dir,
        manifest,
        staged_outputs,
        batch_endpoint,
        limit,
        resume_paths,
        resume_custom_ids,
        judge,
      )

    before_payloads = (
      {}
      if exc_panas
      else collect_existing_before_payloads(records, output_path, staged_outputs)
    )
    phase2_specs = build_main_requests(
      records,
      output_path,
      before_payloads,
      model,
      max_iter,
      excluded_metrics,
      panas_v2=panas_v2,
    )
    _, pending_phase2_specs = filter_specs_missing_from_resume(phase2_specs, resume_custom_ids)
    if active_client is None and phase_requires_client(manifest, "main", pending_phase2_specs):
      active_client = make_judge_client(judge, azure)
    run_phase_shards(
      active_client,
      "main",
      phase2_specs,
      records_by_filename,
      max_iter,
      input_cost_per_1m,
      output_cost_per_1m,
      poll_interval,
      local_timeout_hours,
      artifact_dir,
      manifest,
      staged_outputs,
      batch_endpoint,
      limit,
      resume_paths,
      resume_custom_ids,
      judge,
    )

    write_final_outputs(output_path, staged_outputs)
    manifest["status"] = "completed"
    manifest["final_outputs"] = sorted(staged_outputs)
    save_manifest(artifact_dir, manifest)
    return manifest
  except Exception as exc:
    manifest["status"] = "failed"
    manifest["error"] = str(exc)
    failure_report = {
      "error": str(exc),
      "type": type(exc).__name__,
      "updated_at": utc_now(),
    }
    atomic_write_json(artifact_dir / "failure_report.json", failure_report)
    manifest["failure_report"] = str(artifact_dir / "failure_report.json")
    save_manifest(artifact_dir, manifest)
    if isinstance(exc, BatchMetricsError):
      raise
    raise BatchMetricsError(str(exc)) from exc


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run CTRS, WAI, PANAS-before, and PANAS-after evaluations through the OpenAI or Anthropic Batch API."
  )
  parser.add_argument("input_dir", type=Path, help="Directory containing one JSON file per conversation.")
  parser.add_argument(
    "output_dir",
    type=Path,
    nargs="?",
    default=EVALUATION_ROOT / "results",
    help="Base directory for evaluation outputs. Default: evaluation/results",
  )
  parser.add_argument(
    "max_iter",
    type=int,
    nargs="?",
    default=DEFAULT_MAX_ITER,
    help="Number of evaluator samples per metric prompt. Default: 3",
  )
  parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
  parser.add_argument("--local-timeout-hours", type=float, default=DEFAULT_LOCAL_TIMEOUT_HOURS)
  parser.add_argument(
    "--judge",
    choices=sorted(JUDGE_MODELS),
    default="gpt",
    help="Judge: gpt (gpt-4o, OpenAI Batch API) or sonnet (claude-sonnet-5-5, Anthropic Message Batches API).",
  )
  parser.add_argument(
    "--input-cost-per-1m",
    type=float,
    default=None,
    help="Input price in USD per 1M tokens. Default: judge batch price (gpt 1.25, sonnet 1.0).",
  )
  parser.add_argument(
    "--output-cost-per-1m",
    type=float,
    default=None,
    help="Output price in USD per 1M tokens. Default: judge batch price (gpt 5.0, sonnet 5.0).",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=DEFAULT_ENQUEUED_TOKEN_LIMIT,
    help="Maximum estimated enqueued input tokens per batch. Default: 1350000",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print planned batch counts and enqueued tokens without requesting batches.",
  )
  parser.add_argument(
    "--resume",
    type=Path,
    action="append",
    default=[],
    help=(
      "Reuse completed requests from an existing batch output JSONL, only submitting missing requests. "
      "May be passed multiple times for sharded outputs."
    ),
  )
  parser.add_argument(
    "--run-from-json",
    type=Path,
    default=None,
    help="Recompute only simulation JSON paths listed in a failed-simulations JSON file.",
  )
  parser.add_argument("--exc-wai", action="store_true", help="Exclude WAI evaluation.")
  parser.add_argument("--exc-ctrs", action="store_true", help="Exclude CTRS evaluation.")
  parser.add_argument("--exc-panas", action="store_true", help="Exclude both PANAS-before and PANAS-after evaluation.")
  parser.add_argument(
    "--azure",
    action="store_true",
    help="Use AzureOpenAI with AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT.",
  )
  parser.add_argument(
    "--panas-v2",
    action="store_true",
    help="Use the PANAS-after v2 prompt with pre-session questionnaire context.",
  )
  return parser.parse_args(argv)


def print_dry_run_plan(plan: dict[str, Any]) -> None:
  print("Batch evaluation dry run.")
  if plan.get("judge"):
    print(f"Judge: {plan['judge']} ({plan['model']}), {plan['requests_per_item']} request(s) per metric item")
  print(f"Enqueued token limit: {plan['enqueued_token_limit']}")
  print(f"Total batches: {plan['batch_count']}")
  print(f"Total requests: {plan['request_count']}")
  print(f"Total estimated enqueued tokens: {plan['enqueued_tokens']}")
  for phase in plan.get("phases") or []:
    print(
      f"{phase['phase']}: {phase['batch_count']} batch(es), "
      f"{phase['request_count']} total request(s), "
      f"{phase.get('resumed_request_count', 0)} resumed, "
      f"{phase.get('pending_request_count', phase['request_count'])} pending, "
      f"{phase['enqueued_tokens']} enqueued tokens"
    )
    for batch in phase.get("batches") or []:
      print(
        f"  {batch['name']}: {batch['request_count']} request(s), "
        f"{batch['enqueued_tokens']} enqueued tokens"
      )


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    manifest = run_batch_metrics(
      args.input_dir,
      args.output_dir,
      args.max_iter,
      exc_wai=args.exc_wai,
      exc_ctrs=args.exc_ctrs,
      exc_panas=args.exc_panas,
      panas_v2=args.panas_v2,
      judge=args.judge,
      azure=args.azure,
      limit=args.limit,
      dry_run=args.dry_run,
      resume_jsonls=args.resume,
      poll_interval=args.poll_interval,
      local_timeout_hours=args.local_timeout_hours,
      input_cost_per_1m=args.input_cost_per_1m,
      output_cost_per_1m=args.output_cost_per_1m,
      run_from_json=args.run_from_json,
    )
  except BatchMetricsError as exc:
    print(f"Batch evaluation failed: {exc}", file=os.sys.stderr)
    return 1

  if args.dry_run:
    print_dry_run_plan(manifest)
    return 0

  print("Batch evaluation completed.")
  print(f"Run artifacts: {Path(manifest['signature']['output_dir']) / '_batch_artifacts' / manifest['run_id']}")
  excluded_metrics = set(manifest.get("excluded_metrics") or [])
  if "ctrs" not in excluded_metrics:
    print(f"CTRS results: {Path(manifest['signature']['output_dir']) / 'ctrs'}")
  if "wai" not in excluded_metrics:
    print(f"WAI results: {Path(manifest['signature']['output_dir']) / 'wai'}")
  if "panas" not in excluded_metrics:
    print(f"PANAS-before results: {Path(manifest['signature']['output_dir']) / 'panas_before'}")
    print(f"PANAS-after results: {Path(manifest['signature']['output_dir']) / 'panas_after'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
