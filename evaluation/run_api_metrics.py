#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_batch_metrics import (
  BatchMetricsError,
  BatchRequest,
  DEFAULT_INPUT_COST_PER_1M,
  DEFAULT_MAX_ITER,
  DEFAULT_OUTPUT_COST_PER_1M,
  EVALUATION_ROOT,
  MODEL_NAME,
  all_outputs_complete,
  atomic_write_json,
  atomic_write_text,
  build_main_requests,
  build_panas_before_requests,
  collect_existing_before_payloads,
  discover_conversations,
  ensure_output_dirs,
  estimate_request_enqueued_tokens,
  get_field,
  make_openai_client,
  parse_phase_outputs,
  panas_after_prompt_filename,
  request_map_from_specs,
  to_plain,
  utc_now,
  validate_conversation_file,
  write_final_outputs,
)


DEFAULT_N_PROCESSES = 1
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_WAIT_SECONDS = 30.0
DEFAULT_AZURE_REQUESTS_PER_MINUTE = 600
DEFAULT_AZURE_TOKENS_PER_MINUTE = 100_000
DEFAULT_RATE_LIMIT_OUTPUT_TOKEN_RESERVE = 100
RATE_LIMIT_WINDOW_SECONDS = 60.0
LOGGER = logging.getLogger(__name__)


def _load_tqdm() -> Any:
  if os.getenv("REASONCBT_DISABLE_TQDM"):
    return None
  try:
    from tqdm.auto import tqdm  # pyright: ignore[reportMissingImports]
  except ImportError:
    return None
  return tqdm


def configure_rate_limit_logging() -> None:
  if LOGGER.handlers:
    return
  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
  LOGGER.addHandler(handler)
  LOGGER.setLevel(logging.INFO)
  LOGGER.propagate = False


@dataclass(frozen=True)
class ApiWorkerPayload:
  worker_index: int
  input_dir: str
  filenames: list[str]
  output_dir: str
  artifact_dir: str
  max_iter: int
  model: str
  exc_wai: bool
  exc_ctrs: bool
  exc_panas: bool
  panas_v2: bool
  azure: bool
  input_cost_per_1m: float
  output_cost_per_1m: float
  max_retries: int
  retry_wait_seconds: float
  rate_limit_requests_per_minute: int
  rate_limit_tokens_per_minute: int
  rate_limit_output_token_reserve: int
  rate_limiter_state: Any | None = None
  rate_limiter_lock: Any | None = None


def atomic_append_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
  lines = [json.dumps(payload, ensure_ascii=True) for payload in payloads]
  atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl_line(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=True))
    f.write("\n")
    f.flush()


class ApiRateLimiter:
  def __init__(
    self,
    *,
    requests_per_minute: int,
    tokens_per_minute: int,
    state: Any | None = None,
    lock: Any | None = None,
    log_label: str = "api worker",
  ) -> None:
    self.requests_per_minute = int(requests_per_minute)
    self.tokens_per_minute = int(tokens_per_minute)
    self.state = state or {
      "window_start": time.monotonic(),
      "request_count": 0,
      "token_count": 0,
    }
    self.lock = lock
    self.log_label = log_label

  def enabled(self) -> bool:
    return self.requests_per_minute > 0 or self.tokens_per_minute > 0

  def wait_for_capacity(self, *, request_tokens: int) -> None:
    if not self.enabled():
      return
    request_tokens = max(0, int(request_tokens))
    if self.tokens_per_minute > 0 and request_tokens > self.tokens_per_minute:
      raise BatchMetricsError(
        f"Single request needs {request_tokens} estimated tokens, exceeding "
        f"the configured token/minute limit {self.tokens_per_minute}."
      )

    while True:
      wait_seconds = 0.0
      wait_reason = ""
      request_count = 0
      token_count = 0
      lock = self.lock
      if lock is None:
        wait_seconds, wait_reason, request_count, token_count = self._reserve_or_wait(request_tokens)
      else:
        with lock:
          wait_seconds, wait_reason, request_count, token_count = self._reserve_or_wait(request_tokens)
      if wait_seconds <= 0:
        return
      LOGGER.info(
        "%s sleeping %.2fs to stay within API rate limits (%s; current window: %s/%s requests, %s/%s estimated tokens; next request: %s estimated tokens)",
        self.log_label,
        wait_seconds,
        wait_reason,
        request_count,
        self.requests_per_minute or "unlimited",
        token_count,
        self.tokens_per_minute or "unlimited",
        request_tokens,
      )
      time.sleep(wait_seconds)

  def _reserve_or_wait(self, request_tokens: int) -> tuple[float, str, int, int]:
    now = time.monotonic()
    window_start = float(self.state.get("window_start", now))
    elapsed = now - window_start
    if elapsed >= RATE_LIMIT_WINDOW_SECONDS or elapsed < 0:
      self.state["window_start"] = now
      self.state["request_count"] = 0
      self.state["token_count"] = 0
      window_start = now

    request_count = int(self.state.get("request_count", 0))
    token_count = int(self.state.get("token_count", 0))
    request_limited = self.requests_per_minute > 0 and request_count + 1 > self.requests_per_minute
    token_limited = self.tokens_per_minute > 0 and token_count + request_tokens > self.tokens_per_minute
    if request_limited or token_limited:
      reasons = []
      if request_limited:
        reasons.append("request/minute budget")
      if token_limited:
        reasons.append("token/minute budget")
      return (
        max(0.0, RATE_LIMIT_WINDOW_SECONDS - (now - window_start)) + 0.05,
        " and ".join(reasons),
        request_count,
        token_count,
      )

    self.state["request_count"] = request_count + 1
    self.state["token_count"] = token_count + request_tokens
    return 0.0, "", request_count + 1, token_count + request_tokens


def estimate_api_request_tokens(spec: BatchRequest, output_token_reserve: int) -> int:
  n = spec.body.get("n", 1)
  try:
    response_count = max(1, int(n))
  except (TypeError, ValueError):
    response_count = 1
  return estimate_request_enqueued_tokens(spec) + max(0, int(output_token_reserve)) * response_count


def retry_after_seconds(exc: Exception) -> float | None:
  response = getattr(exc, "response", None)
  headers = getattr(response, "headers", None)
  if headers is None:
    return None
  for key in ("retry-after-ms", "Retry-After-Ms"):
    value = headers.get(key) if hasattr(headers, "get") else None
    if value is None:
      continue
    try:
      return max(0.0, float(value) / 1000.0)
    except (TypeError, ValueError):
      return None
  for key in ("retry-after", "Retry-After"):
    value = headers.get(key) if hasattr(headers, "get") else None
    if value is None:
      continue
    try:
      return max(0.0, float(value))
    except (TypeError, ValueError):
      try:
        retry_at = parsedate_to_datetime(str(value))
      except (TypeError, ValueError):
        return None
      if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
      return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
  return None


def chunk_filenames(filenames: list[str], n_processes: int) -> list[list[str]]:
  if n_processes < 1:
    raise BatchMetricsError(f"--n-processes must be a positive integer, got: {n_processes}")
  if not filenames:
    return []
  chunk_count = min(n_processes, len(filenames))
  base_size, remainder = divmod(len(filenames), chunk_count)
  chunks: list[list[str]] = []
  start = 0
  for index in range(chunk_count):
    size = base_size + (1 if index < remainder else 0)
    chunks.append(filenames[start : start + size])
    start += size
  return chunks


def excluded_metrics_from_flags(
  *,
  exc_wai: bool,
  exc_ctrs: bool,
  exc_panas: bool,
) -> set[str]:
  return {
    metric
    for metric, excluded in (
      ("wai", exc_wai),
      ("ctrs", exc_ctrs),
      ("panas", exc_panas),
    )
    if excluded
  }


def response_to_batch_line(spec: BatchRequest, response: Any) -> dict[str, Any]:
  body = to_plain(response)
  return {
    "id": get_field(response, "id", f"api-req-{spec.custom_id}"),
    "custom_id": spec.custom_id,
    "response": {
      "status_code": 200,
      "request_id": get_field(response, "id", ""),
      "body": body,
    },
    "error": None,
  }


def execute_api_request(
  client: Any,
  spec: BatchRequest,
  *,
  max_retries: int,
  retry_wait_seconds: float,
  rate_limiter: ApiRateLimiter | None = None,
  rate_limit_output_token_reserve: int = DEFAULT_RATE_LIMIT_OUTPUT_TOKEN_RESERVE,
) -> dict[str, Any]:
  attempt = 0
  while True:
    attempt += 1
    try:
      if rate_limiter is not None:
        rate_limiter.wait_for_capacity(
          request_tokens=estimate_api_request_tokens(spec, rate_limit_output_token_reserve),
        )
      response = client.chat.completions.create(**spec.body)
      return response_to_batch_line(spec, response)
    except Exception as exc:
      if attempt > max_retries:
        raise BatchMetricsError(
          f"API request failed for {spec.custom_id} after {attempt} attempt(s): {exc}"
        ) from exc
      wait_seconds = retry_after_seconds(exc)
      if wait_seconds is None:
        wait_seconds = max(0.0, retry_wait_seconds) * attempt
      time.sleep(wait_seconds)


def execute_specs_to_jsonl(
  specs: list[BatchRequest],
  output_jsonl: Path,
  *,
  azure: bool,
  max_retries: int,
  retry_wait_seconds: float,
  rate_limiter: ApiRateLimiter | None = None,
  rate_limit_output_token_reserve: int = DEFAULT_RATE_LIMIT_OUTPUT_TOKEN_RESERVE,
  progress: Any | None = None,
) -> None:
  if not specs:
    atomic_append_jsonl(output_jsonl, [])
    return
  client = make_openai_client(azure=azure)
  atomic_write_text(output_jsonl, "")
  for spec in specs:
    if progress is not None:
      progress.set_postfix_str(spec.custom_id[:80], refresh=False)
    line = execute_api_request(
      client,
      spec,
      max_retries=max_retries,
      retry_wait_seconds=retry_wait_seconds,
      rate_limiter=rate_limiter,
      rate_limit_output_token_reserve=rate_limit_output_token_reserve,
    )
    append_jsonl_line(output_jsonl, line)
    if progress is not None:
      progress.update(1)


def worker_artifact_path(artifact_dir: Path, worker_index: int, phase_name: str, filename: str) -> Path:
  return artifact_dir / f"worker_{worker_index:04d}" / phase_name / filename


def run_worker_phase(
  *,
  payload: ApiWorkerPayload,
  phase_name: str,
  specs: list[BatchRequest],
  records_by_filename: dict[str, Any],
  staged_outputs: dict[str, dict[str, Any]],
  rate_limiter: ApiRateLimiter | None = None,
  progress: Any | None = None,
) -> dict[str, Any]:
  artifact_dir = Path(payload.artifact_dir)
  phase_dir = artifact_dir / f"worker_{payload.worker_index:04d}" / phase_name
  phase_dir.mkdir(parents=True, exist_ok=True)
  request_map = request_map_from_specs(specs)
  request_map_path = worker_artifact_path(artifact_dir, payload.worker_index, phase_name, "request_map.json")
  output_jsonl = worker_artifact_path(artifact_dir, payload.worker_index, phase_name, "output.jsonl")
  parse_report_path = worker_artifact_path(artifact_dir, payload.worker_index, phase_name, "parse_report.json")

  atomic_write_json(request_map_path, request_map)
  if specs:
    execute_specs_to_jsonl(
      specs,
      output_jsonl,
      azure=payload.azure,
      max_retries=payload.max_retries,
      retry_wait_seconds=payload.retry_wait_seconds,
      rate_limiter=rate_limiter,
      rate_limit_output_token_reserve=payload.rate_limit_output_token_reserve,
      progress=progress,
    )
    parsed_outputs, parse_report = parse_phase_outputs(
      output_jsonl,
      request_map,
      records_by_filename,
      payload.max_iter,
      payload.input_cost_per_1m,
      payload.output_cost_per_1m,
    )
    atomic_write_json(parse_report_path, parse_report)
    staged_outputs.update(parsed_outputs)
  else:
    atomic_append_jsonl(output_jsonl, [])
    parse_report = {"parsed_requests": 0, "output_files": {}, "errors": []}
    atomic_write_json(parse_report_path, parse_report)

  return {
    "phase": phase_name,
    "request_count": len(specs),
    "request_map": str(request_map_path),
    "output_jsonl": str(output_jsonl),
    "parse_report": str(parse_report_path),
  }


def run_api_metrics_worker(payload: ApiWorkerPayload) -> dict[str, Any]:
  input_dir = Path(payload.input_dir)
  output_dir = Path(payload.output_dir)
  records = [validate_conversation_file(input_dir / filename) for filename in payload.filenames]
  records_by_filename = {record.filename: record for record in records}
  excluded_metrics = excluded_metrics_from_flags(
    exc_wai=payload.exc_wai,
    exc_ctrs=payload.exc_ctrs,
    exc_panas=payload.exc_panas,
  )
  staged_outputs: dict[str, dict[str, Any]] = {}
  phases: dict[str, Any] = {}
  progress = None
  tqdm = _load_tqdm()
  rate_limiter = ApiRateLimiter(
    requests_per_minute=payload.rate_limit_requests_per_minute,
    tokens_per_minute=payload.rate_limit_tokens_per_minute,
    state=payload.rate_limiter_state,
    lock=payload.rate_limiter_lock,
    log_label=f"worker {payload.worker_index}",
  )

  phase1_specs = (
    []
    if payload.exc_panas
    else build_panas_before_requests(records, output_dir, payload.model, payload.max_iter)
  )
  before_payloads_for_plan = (
    {}
    if payload.exc_panas
    else collect_existing_before_payloads(records, output_dir, staged_outputs)
  )
  phase2_plan_specs = build_main_requests(
    records,
    output_dir,
    before_payloads_for_plan,
    payload.model,
    payload.max_iter,
    excluded_metrics,
    panas_v2=payload.panas_v2,
  )

  if tqdm is not None:
    progress = tqdm(
      total=len(phase1_specs) + len(phase2_plan_specs),
      desc=f"worker {payload.worker_index}",
      position=payload.worker_index - 1,
      unit="req",
      leave=True,
      dynamic_ncols=True,
    )
  elif not os.getenv("REASONCBT_DISABLE_TQDM"):
    print(
      f"worker {payload.worker_index}: "
      f"{len(records)} conversation(s), {len(phase1_specs) + len(phase2_plan_specs)} planned request(s)",
      flush=True,
    )

  try:
    if not payload.exc_panas:
      phases["panas_before"] = run_worker_phase(
        payload=payload,
        phase_name="panas_before",
        specs=phase1_specs,
        records_by_filename=records_by_filename,
        staged_outputs=staged_outputs,
        rate_limiter=rate_limiter,
        progress=progress,
      )
    else:
      phases["panas_before"] = {"phase": "panas_before", "request_count": 0, "status": "skipped"}

    before_payloads = (
      {}
      if payload.exc_panas
      else collect_existing_before_payloads(records, output_dir, staged_outputs)
    )
    phase2_specs = build_main_requests(
      records,
      output_dir,
      before_payloads,
      payload.model,
      payload.max_iter,
      excluded_metrics,
      panas_v2=payload.panas_v2,
    )
    if progress is not None and len(phase1_specs) + len(phase2_specs) != progress.total:
      progress.total = len(phase1_specs) + len(phase2_specs)
      progress.refresh()
    phases["main"] = run_worker_phase(
      payload=payload,
      phase_name="main",
      specs=phase2_specs,
      records_by_filename=records_by_filename,
      staged_outputs=staged_outputs,
      rate_limiter=rate_limiter,
      progress=progress,
    )
  finally:
    if progress is not None:
      progress.close()

  write_final_outputs(output_dir, staged_outputs)
  return {
    "worker_index": payload.worker_index,
    "input_files": payload.filenames,
    "conversation_count": len(records),
    "request_count": len(phase1_specs) + len(phase2_specs),
    "output_count": len(staged_outputs),
    "phases": phases,
  }


def create_api_manifest(
  output_dir: Path,
  *,
  input_dir: Path,
  filenames: list[str],
  model: str,
  max_iter: int,
  azure: bool,
  n_processes: int,
  excluded_metrics: set[str],
  panas_v2: bool,
  rate_limit_requests_per_minute: int,
  rate_limit_tokens_per_minute: int,
  rate_limit_output_token_reserve: int,
) -> tuple[Path, dict[str, Any]]:
  run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
  artifact_dir = output_dir / "_api_artifacts" / run_id
  artifact_dir.mkdir(parents=True, exist_ok=False)
  manifest = {
    "version": 1,
    "run_id": run_id,
    "status": "running",
    "created_at": utc_now(),
    "updated_at": utc_now(),
    "provider": "azure_openai" if azure else "openai",
    "model": model,
    "max_iter": max_iter,
    "n_processes": n_processes,
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "conversation_count": len(filenames),
    "excluded_metrics": sorted(excluded_metrics),
    "panas_after_prompt": panas_after_prompt_filename(panas_v2),
    "rate_limits": {
      "requests_per_minute": rate_limit_requests_per_minute,
      "tokens_per_minute": rate_limit_tokens_per_minute,
      "output_token_reserve": rate_limit_output_token_reserve,
      "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
    },
    "workers": [],
  }
  atomic_write_json(artifact_dir / "manifest.json", manifest)
  return artifact_dir, manifest


def save_api_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
  manifest["updated_at"] = utc_now()
  atomic_write_json(artifact_dir / "manifest.json", manifest)


def build_api_dry_run_plan(
  records: list[Any],
  output_dir: Path,
  *,
  model: str,
  max_iter: int,
  exc_wai: bool,
  exc_ctrs: bool,
  exc_panas: bool,
  panas_v2: bool,
  n_processes: int,
  rate_limit_requests_per_minute: int,
  rate_limit_tokens_per_minute: int,
  rate_limit_output_token_reserve: int,
) -> dict[str, Any]:
  excluded_metrics = excluded_metrics_from_flags(exc_wai=exc_wai, exc_ctrs=exc_ctrs, exc_panas=exc_panas)
  chunks = chunk_filenames([record.filename for record in records], n_processes)
  phase1_specs = [] if exc_panas else build_panas_before_requests(records, output_dir, model, max_iter)
  before_payloads = {} if exc_panas else {
    record.filename: {
      "attitude": record.attitude,
      "cost": 0.0,
      "explanations": [],
    }
    for record in records
  }
  phase2_specs = build_main_requests(
    records,
    output_dir,
    before_payloads,
    model,
    max_iter,
    excluded_metrics,
    panas_v2=panas_v2,
  )
  return {
    "dry_run": True,
    "conversation_count": len(records),
    "n_processes": min(n_processes, len(records)) if records else 0,
    "chunks": [{"worker_index": index + 1, "conversation_count": len(chunk)} for index, chunk in enumerate(chunks)],
    "phases": {
      "panas_before": {"request_count": len(phase1_specs)},
      "main": {"request_count": len(phase2_specs)},
    },
    "rate_limits": {
      "requests_per_minute": rate_limit_requests_per_minute,
      "tokens_per_minute": rate_limit_tokens_per_minute,
      "output_token_reserve": rate_limit_output_token_reserve,
      "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
    },
    "request_count": len(phase1_specs) + len(phase2_specs),
  }


def resolve_rate_limit_settings(
  *,
  azure: bool,
  requests_per_minute: int | None,
  tokens_per_minute: int | None,
  output_token_reserve: int,
) -> tuple[int, int, int]:
  if output_token_reserve < 0:
    raise BatchMetricsError(
      f"--rate-limit-output-token-reserve must be non-negative, got: {output_token_reserve}"
    )
  resolved_requests = (
    DEFAULT_AZURE_REQUESTS_PER_MINUTE
    if azure and requests_per_minute is None
    else int(requests_per_minute or 0)
  )
  resolved_tokens = (
    DEFAULT_AZURE_TOKENS_PER_MINUTE
    if azure and tokens_per_minute is None
    else int(tokens_per_minute or 0)
  )
  if resolved_requests < 0:
    raise BatchMetricsError(
      f"--rate-limit-requests-per-minute must be non-negative, got: {resolved_requests}"
    )
  if resolved_tokens < 0:
    raise BatchMetricsError(
      f"--rate-limit-tokens-per-minute must be non-negative, got: {resolved_tokens}"
    )
  return resolved_requests, resolved_tokens, output_token_reserve


def run_api_metrics(
  input_dir: str | Path,
  output_dir: str | Path | None = None,
  max_iter: int = DEFAULT_MAX_ITER,
  *,
  model: str = MODEL_NAME,
  n_processes: int = DEFAULT_N_PROCESSES,
  exc_wai: bool = False,
  exc_ctrs: bool = False,
  exc_panas: bool = False,
  panas_v2: bool = False,
  input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
  output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
  azure: bool = False,
  dry_run: bool = False,
  max_retries: int = DEFAULT_MAX_RETRIES,
  retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
  rate_limit_requests_per_minute: int | None = None,
  rate_limit_tokens_per_minute: int | None = None,
  rate_limit_output_token_reserve: int = DEFAULT_RATE_LIMIT_OUTPUT_TOKEN_RESERVE,
) -> dict[str, Any]:
  if max_iter < 1:
    raise BatchMetricsError(f"MAX_ITER must be a positive integer, got: {max_iter}")
  if n_processes < 1:
    raise BatchMetricsError(f"--n-processes must be a positive integer, got: {n_processes}")
  if max_retries < 1:
    raise BatchMetricsError(f"--max-retries must be a positive integer, got: {max_retries}")

  input_path = Path(input_dir).resolve()
  output_path = (
    Path(output_dir).resolve()
    if output_dir is not None
    else (EVALUATION_ROOT / "results").resolve()
  )
  records = discover_conversations(input_path)
  resolved_rate_limit_requests_per_minute, resolved_rate_limit_tokens_per_minute, resolved_rate_limit_output_token_reserve = resolve_rate_limit_settings(
    azure=azure,
    requests_per_minute=rate_limit_requests_per_minute,
    tokens_per_minute=rate_limit_tokens_per_minute,
    output_token_reserve=rate_limit_output_token_reserve,
  )
  if dry_run:
    return build_api_dry_run_plan(
      records,
      output_path,
      model=model,
      max_iter=max_iter,
      exc_wai=exc_wai,
      exc_ctrs=exc_ctrs,
      exc_panas=exc_panas,
      panas_v2=panas_v2,
      n_processes=n_processes,
      rate_limit_requests_per_minute=resolved_rate_limit_requests_per_minute,
      rate_limit_tokens_per_minute=resolved_rate_limit_tokens_per_minute,
      rate_limit_output_token_reserve=resolved_rate_limit_output_token_reserve,
    )

  ensure_output_dirs(output_path)
  excluded_metrics = excluded_metrics_from_flags(exc_wai=exc_wai, exc_ctrs=exc_ctrs, exc_panas=exc_panas)
  filenames = [record.filename for record in records]
  chunks = chunk_filenames(filenames, n_processes)
  artifact_dir, manifest = create_api_manifest(
    output_path,
    input_dir=input_path,
    filenames=filenames,
    model=model,
    max_iter=max_iter,
    azure=azure,
    n_processes=len(chunks),
    excluded_metrics=excluded_metrics,
    panas_v2=panas_v2,
    rate_limit_requests_per_minute=resolved_rate_limit_requests_per_minute,
    rate_limit_tokens_per_minute=resolved_rate_limit_tokens_per_minute,
    rate_limit_output_token_reserve=resolved_rate_limit_output_token_reserve,
  )

  def make_payloads(rate_limiter_state: Any | None = None, rate_limiter_lock: Any | None = None) -> list[ApiWorkerPayload]:
    return [
      ApiWorkerPayload(
        worker_index=index + 1,
        input_dir=str(input_path),
        filenames=chunk,
        output_dir=str(output_path),
        artifact_dir=str(artifact_dir),
        max_iter=max_iter,
        model=model,
        exc_wai=exc_wai,
        exc_ctrs=exc_ctrs,
        exc_panas=exc_panas,
        panas_v2=panas_v2,
        azure=azure,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        max_retries=max_retries,
        retry_wait_seconds=retry_wait_seconds,
        rate_limit_requests_per_minute=resolved_rate_limit_requests_per_minute,
        rate_limit_tokens_per_minute=resolved_rate_limit_tokens_per_minute,
        rate_limit_output_token_reserve=resolved_rate_limit_output_token_reserve,
        rate_limiter_state=rate_limiter_state,
        rate_limiter_lock=rate_limiter_lock,
      )
      for index, chunk in enumerate(chunks)
    ]

  try:
    rate_limiter_enabled = (
      resolved_rate_limit_requests_per_minute > 0 or resolved_rate_limit_tokens_per_minute > 0
    )
    payloads = make_payloads()
    if len(payloads) == 1:
      worker_results = [run_api_metrics_worker(payloads[0])]
    elif rate_limiter_enabled:
      with multiprocessing.Manager() as manager:
        shared_state = manager.dict(
          {
            "window_start": time.monotonic(),
            "request_count": 0,
            "token_count": 0,
          }
        )
        shared_lock = manager.Lock()
        payloads = make_payloads(shared_state, shared_lock)
        with multiprocessing.Pool(processes=len(payloads)) as pool:
          worker_results = pool.map(run_api_metrics_worker, payloads)
    else:
      with multiprocessing.Pool(processes=len(payloads)) as pool:
        worker_results = pool.map(run_api_metrics_worker, payloads)

    manifest["workers"] = worker_results
    manifest["request_count"] = sum(int(result["request_count"]) for result in worker_results)
    manifest["output_count"] = sum(int(result["output_count"]) for result in worker_results)
    manifest["status"] = "completed"
    save_api_manifest(artifact_dir, manifest)
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
    save_api_manifest(artifact_dir, manifest)
    if isinstance(exc, BatchMetricsError):
      raise
    raise BatchMetricsError(str(exc)) from exc


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run CTRS, WAI, PANAS-before, and PANAS-after evaluations with normal API requests."
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
  parser.add_argument("--model", default=MODEL_NAME, help=f"Evaluator model. Default: {MODEL_NAME}")
  parser.add_argument("--n-processes", type=int, default=DEFAULT_N_PROCESSES)
  parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
  parser.add_argument("--retry-wait-seconds", type=float, default=DEFAULT_RETRY_WAIT_SECONDS)
  parser.add_argument(
    "--rate-limit-requests-per-minute",
    type=int,
    default=None,
    help=(
      "Client-side request limit. Default: 600 with --azure, disabled otherwise. "
      "Use 0 to disable."
    ),
  )
  parser.add_argument(
    "--rate-limit-tokens-per-minute",
    type=int,
    default=None,
    help=(
      "Client-side estimated token limit. Default: 100000 with --azure, disabled otherwise. "
      "Use 0 to disable."
    ),
  )
  parser.add_argument(
    "--rate-limit-output-token-reserve",
    type=int,
    default=DEFAULT_RATE_LIMIT_OUTPUT_TOKEN_RESERVE,
    help=(
      "Estimated output tokens reserved per requested completion for rate limiting. "
      f"Default: {DEFAULT_RATE_LIMIT_OUTPUT_TOKEN_RESERVE}"
    ),
  )
  parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
  parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
  parser.add_argument("--dry-run", action="store_true", help="Print planned API request counts without calling the API.")
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
  print("API evaluation dry run.")
  print(f"Conversations: {plan['conversation_count']}")
  print(f"Processes: {plan['n_processes']}")
  print(f"Total requests: {plan['request_count']}")
  rate_limits = plan.get("rate_limits") or {}
  print(
    "Rate limits: "
    f"{rate_limits.get('requests_per_minute', 0)} req/min, "
    f"{rate_limits.get('tokens_per_minute', 0)} estimated tokens/min, "
    f"{rate_limits.get('output_token_reserve', 0)} output-token reserve"
  )
  for phase_name, phase in plan.get("phases", {}).items():
    print(f"{phase_name}: {phase['request_count']} request(s)")
  for chunk in plan.get("chunks", []):
    print(f"worker {chunk['worker_index']}: {chunk['conversation_count']} conversation(s)")


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  configure_rate_limit_logging()
  try:
    manifest = run_api_metrics(
      args.input_dir,
      args.output_dir,
      args.max_iter,
      model=args.model,
      n_processes=args.n_processes,
      exc_wai=args.exc_wai,
      exc_ctrs=args.exc_ctrs,
      exc_panas=args.exc_panas,
      panas_v2=args.panas_v2,
      azure=args.azure,
      dry_run=args.dry_run,
      max_retries=args.max_retries,
      retry_wait_seconds=args.retry_wait_seconds,
      rate_limit_requests_per_minute=args.rate_limit_requests_per_minute,
      rate_limit_tokens_per_minute=args.rate_limit_tokens_per_minute,
      rate_limit_output_token_reserve=args.rate_limit_output_token_reserve,
      input_cost_per_1m=args.input_cost_per_1m,
      output_cost_per_1m=args.output_cost_per_1m,
    )
  except BatchMetricsError as exc:
    print(f"API evaluation failed: {exc}", file=os.sys.stderr)
    return 1

  if args.dry_run:
    print_dry_run_plan(manifest)
    return 0

  output_dir = Path(str(manifest["output_dir"]))
  print("API evaluation completed.")
  print(f"Run artifacts: {output_dir / '_api_artifacts' / manifest['run_id']}")
  excluded_metrics = set(manifest.get("excluded_metrics") or [])
  if "ctrs" not in excluded_metrics:
    print(f"CTRS results: {output_dir / 'ctrs'}")
  if "wai" not in excluded_metrics:
    print(f"WAI results: {output_dir / 'wai'}")
  if "panas" not in excluded_metrics:
    print(f"PANAS-before results: {output_dir / 'panas_before'}")
    print(f"PANAS-after results: {output_dir / 'panas_after'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
