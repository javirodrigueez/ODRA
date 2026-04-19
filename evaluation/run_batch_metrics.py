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
COMPLETION_WINDOW = "24h"
DEFAULT_MAX_ITER = 3
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_LOCAL_TIMEOUT_HOURS = 26.0
DEFAULT_INPUT_COST_PER_1M = 1.25
DEFAULT_OUTPUT_COST_PER_1M = 5.0
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_ROOT = Path(__file__).resolve().parent

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
) -> list[BatchRequest]:
  prompt_template = load_prompt(EVALUATION_ROOT / "PANAS" / "prompts" / "panas_before.txt")
  requests = []
  for index, record in enumerate(records):
    output_file = Path("panas_before") / record.filename
    if output_is_complete("panas_before", output_dir / output_file):
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


def build_main_requests(
  records: list[ConversationRecord],
  output_dir: Path,
  before_payloads: dict[str, dict[str, Any]],
  model: str,
  max_iter: int,
  excluded_metrics: set[str] | None = None,
) -> list[BatchRequest]:
  requests: list[BatchRequest] = []
  excluded = excluded_metrics or set()
  ctrs_templates = {
    key: load_prompt(EVALUATION_ROOT / "CTRS" / "prompts" / f"{key}.txt")
    for key in CTRS_KEYS
  } if "ctrs" not in excluded else {}
  wai_templates = {
    key: load_prompt(EVALUATION_ROOT / "WAI" / "prompts" / f"{key}.txt")
    for key in WAI_KEYS
  } if "wai" not in excluded else {}
  panas_after_template = (
    load_prompt(EVALUATION_ROOT / "PANAS" / "prompts" / "panas_after.txt")
    if "panas" not in excluded
    else ""
  )

  for index, record in enumerate(records):
    ctrs_output_file = Path("ctrs") / record.filename
    if "ctrs" not in excluded and not output_is_complete("ctrs", output_dir / ctrs_output_file):
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
    if "wai" not in excluded and not output_is_complete("wai", output_dir / wai_output_file):
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
    if "panas" not in excluded and not output_is_complete("panas_after", output_dir / panas_after_output_file):
      before_payload = before_payloads.get(record.filename)
      if before_payload is None:
        raise BatchMetricsError(
          f"PANAS-after requires a PANAS-before result for {record.filename}, but none is available."
        )
      prompt = render_template(
        panas_after_template,
        {
          "intake_form": record.intake_form,
          "conversation": record.history,
          "presession_questionnaire": format_presession_questionnaire(before_payload),
        },
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


def validate_batch_input(path: Path, request_count: int) -> None:
  if request_count > MAX_BATCH_REQUESTS:
    raise BatchMetricsError(
      f"Batch has {request_count} requests; maximum supported per batch is {MAX_BATCH_REQUESTS}."
    )
  size = path.stat().st_size
  if size > MAX_BATCH_FILE_BYTES:
    raise BatchMetricsError(
      f"Batch input file is {size} bytes; maximum supported size is {MAX_BATCH_FILE_BYTES} bytes."
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
    for line in str(content).splitlines():
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
  return (
    (float(prompt_tokens) * input_cost_per_1m)
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
  return {
    "input_dir": str(input_dir.resolve()),
    "output_dir": str(output_dir.resolve()),
    "model": model,
    "max_iter": max_iter,
    "excluded_metrics": sorted(excluded_metrics or set()),
    "endpoint": BATCH_ENDPOINT,
    "files": files,
  }


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
    "endpoint": BATCH_ENDPOINT,
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
    atomic_write_json(request_map_path, request_map_from_specs(specs))
  if specs and not input_jsonl_path.exists():
    write_jsonl(input_jsonl_path, specs)

  request_count = len(load_request_map(request_map_path)) if request_map_path.exists() else len(specs)
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
    endpoint=BATCH_ENDPOINT,
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
) -> dict[str, dict[str, Any]]:
  phase = prepare_phase_artifacts(artifact_dir, phase_name, specs, manifest)
  save_manifest(artifact_dir, manifest)

  if phase.get("status") == "skipped":
    return {}
  if phase.get("status") == "parsed":
    return load_staged_outputs(artifact_dir)

  upload_and_create_batch(client, phase_name, phase, manifest, artifact_dir)
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


def write_final_outputs(output_dir: Path, staged_outputs: dict[str, dict[str, Any]]) -> None:
  for rel_path, payload in sorted(staged_outputs.items()):
    atomic_write_json(output_dir / rel_path, payload)


def make_openai_client() -> Any:
  if load_dotenv is not None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
  api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise BatchMetricsError("Missing OpenAI API key. Set GPT_API_KEY or OPENAI_API_KEY.")
  try:
    from openai import OpenAI
  except ImportError as exc:
    raise BatchMetricsError("The openai Python package is required to run batch metrics.") from exc
  return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


def phase_requires_client(manifest: dict[str, Any], phase_name: str, specs: list[BatchRequest]) -> bool:
  phase = manifest.get("phases", {}).get(phase_name, {})
  if str(phase.get("status") or "") in {"skipped", "parsed"}:
    return False
  if specs:
    return True
  return bool(phase.get("batch_id") or phase.get("input_jsonl") or phase.get("request_map"))


def run_batch_metrics(
  input_dir: str | Path,
  output_dir: str | Path | None = None,
  max_iter: int = DEFAULT_MAX_ITER,
  *,
  client: Any | None = None,
  model: str = MODEL_NAME,
  exc_wai: bool = False,
  exc_ctrs: bool = False,
  exc_panas: bool = False,
  poll_interval: float = DEFAULT_POLL_INTERVAL,
  local_timeout_hours: float = DEFAULT_LOCAL_TIMEOUT_HOURS,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
) -> dict[str, Any]:
  if max_iter < 1:
    raise BatchMetricsError(f"MAX_ITER must be a positive integer, got: {max_iter}")
  input_path = Path(input_dir).resolve()
  output_path = (
    Path(output_dir).resolve()
    if output_dir is not None
    else (EVALUATION_ROOT / "results").resolve()
  )
  records = discover_conversations(input_path)
  records_by_filename = {record.filename: record for record in records}
  ensure_output_dirs(output_path)
  excluded_metrics = {
    metric
    for metric, excluded in (
      ("wai", exc_wai),
      ("ctrs", exc_ctrs),
      ("panas", exc_panas),
    )
    if excluded
  }

  signature = compute_signature(input_path, output_path, records, model, max_iter, excluded_metrics)
  artifact_dir, manifest, resumed = load_or_create_manifest(output_path, signature)
  if not resumed:
    manifest["excluded_metrics"] = sorted(excluded_metrics)
    manifest["skipped"] = build_skipped_summary(records, output_path, excluded_metrics)
    save_manifest(artifact_dir, manifest)

  staged_outputs = load_staged_outputs(artifact_dir)
  active_client = client

  try:
    phase1_specs = (
      []
      if exc_panas
      else build_panas_before_requests(records, output_path, model, max_iter)
    )
    if active_client is None and phase_requires_client(manifest, "panas_before", phase1_specs):
      active_client = make_openai_client()
    run_phase(
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
    )
    if active_client is None and phase_requires_client(manifest, "main", phase2_specs):
      active_client = make_openai_client()
    run_phase(
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
    description="Run CTRS, WAI, PANAS-before, and PANAS-after evaluations through the OpenAI Batch API."
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
  parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
  parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
  parser.add_argument("--exc-wai", action="store_true", help="Exclude WAI evaluation.")
  parser.add_argument("--exc-ctrs", action="store_true", help="Exclude CTRS evaluation.")
  parser.add_argument("--exc-panas", action="store_true", help="Exclude both PANAS-before and PANAS-after evaluation.")
  return parser.parse_args(argv)


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
      poll_interval=args.poll_interval,
      local_timeout_hours=args.local_timeout_hours,
      input_cost_per_1m=args.input_cost_per_1m,
      output_cost_per_1m=args.output_cost_per_1m,
    )
  except BatchMetricsError as exc:
    print(f"Batch evaluation failed: {exc}", file=os.sys.stderr)
    return 1

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
