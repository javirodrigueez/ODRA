#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
except ImportError:  # pragma: no cover - clearer CLI behavior if dotenv is absent.
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
RESISTANCE_ERROR_TOLERANCE_0_10 = 3.3

SCRIPT_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = SCRIPT_ROOT / "prompts"
REPO_ROOT = Path(__file__).resolve().parents[2]

FINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
FAILED_STATUSES = {"failed", "expired", "cancelled"}
FINAL_RATING_RE = re.compile(r"Rating:\s*\[?\[?\s*(-?\d+(?:\.\d+)?)\s*\]?\]?", re.IGNORECASE)


class SteeringEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    prompt_file: str
    min_score: float
    max_score: float


@dataclass(frozen=True)
class EvaluatedTurn:
    utterance_index: int
    stage: str | None
    substage: str | None
    previous_therapist_utterance: str
    patient_utterance: str
    actual_resistance_0_1: float
    actual_resistance_0_10: float
    resistance_update_reasoning: str


@dataclass(frozen=True)
class BatchRequest:
    custom_id: str
    metric: str
    input_file: str
    output_file: str
    item_key: str
    utterance_index: int
    min_score: float
    max_score: float
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
            "metric": self.metric,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "item_key": self.item_key,
            "utterance_index": self.utterance_index,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


METRIC_SPECS = {
    "resistance": MetricSpec("resistance", "resistance.txt", 0.0, 10.0),
    "context_relevance": MetricSpec("context_relevance", "context_relevance.txt", 0.0, 2.0),
    "patient_like_realism": MetricSpec("patient_like_realism", "patient_like_realism.txt", 0.0, 2.0),
    "prompt_leakage": MetricSpec("prompt_leakage", "prompt_leakage.txt", 0.0, 1.0),
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


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def harmonic_mean(scores: list[float]) -> float:
    if not scores or any(score <= 0 for score in scores):
        return 0.0
    return len(scores) / sum(1 / score for score in scores)


def resistance_interval(value_0_1: float) -> str:
    if value_0_1 <= 0.33:
        return "positive"
    if value_0_1 < 0.66:
        return "neutral"
    return "negative"


def parse_trace_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return {}
    text = output.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def trace_payload(trace: dict[str, Any], step_name: str) -> dict[str, Any]:
    steps = trace.get("traces")
    if not isinstance(steps, list):
        return {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("step") == step_name:
            return parse_trace_output(step.get("output"))
    return {}


def format_mapping(mapping: Any) -> str:
    if not isinstance(mapping, dict):
        return ""
    lines = []
    for key, value in mapping.items():
        label = str(key).replace("_", " ").title()
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            rendered = str(value)
        lines.append(f"{label}: {rendered}")
    return "\n".join(lines)


def format_patient_profile(data: dict[str, Any]) -> str:
    profile = data.get("patient_profile")
    if not isinstance(profile, dict):
        return ""

    sections = []
    personal = format_mapping(profile.get("personal_info"))
    if personal:
        sections.append("Personal information:\n" + personal)

    for key in ("presenting_problem", "reason_seeking_counseling", "patterns", "attitude"):
        value = profile.get(key)
        if value in (None, "", []):
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        sections.append(f"{label}:\n{rendered}")

    more_info = format_mapping(profile.get("more_information"))
    if more_info:
        sections.append("Additional background:\n" + more_info)

    return "\n\n".join(sections)


def indexed_messages(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return {}

    indexed: dict[int, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        index = coerce_int(message.get("utterance_index"))
        if index is not None:
            indexed[index] = message
    return indexed


def previous_therapist_message(messages: dict[int, dict[str, Any]], index: int) -> dict[str, Any] | None:
    for prior_index in sorted((item for item in messages if item < index), reverse=True):
        message = messages[prior_index]
        if str(message.get("role") or "").strip().lower() == "therapist":
            return message
    return None


def extract_evaluable_turns(data: dict[str, Any]) -> tuple[list[EvaluatedTurn], list[dict[str, Any]]]:
    turns: list[EvaluatedTurn] = []
    skipped: list[dict[str, Any]] = []
    messages = indexed_messages(data)
    traces = data.get("reasoning_traces")

    if not isinstance(traces, list):
        return [], [{"reason": "missing reasoning_traces list"}]

    for trace in traces:
        if not isinstance(trace, dict):
            skipped.append({"reason": "invalid trace object"})
            continue
        if str(trace.get("speaker") or "").strip().lower() != "patient":
            continue

        utterance_index = coerce_int(trace.get("utterance_index"))
        if utterance_index is None:
            skipped.append({"reason": "patient trace missing utterance_index"})
            continue

        update = trace_payload(trace, "resistance_update")
        actual = coerce_float(update.get("new_r"))
        if actual is None:
            skipped.append({"utterance_index": utterance_index, "reason": "missing resistance_update.new_r"})
            continue

        patient_message = messages.get(utterance_index)
        if not patient_message:
            skipped.append({"utterance_index": utterance_index, "reason": "missing matching patient message"})
            continue

        previous_message = previous_therapist_message(messages, utterance_index)
        if not previous_message:
            skipped.append({"utterance_index": utterance_index, "reason": "missing previous therapist utterance"})
            continue

        turns.append(
            EvaluatedTurn(
                utterance_index=utterance_index,
                stage=patient_message.get("stage"),
                substage=patient_message.get("substage"),
                previous_therapist_utterance=str(previous_message.get("content") or ""),
                patient_utterance=str(patient_message.get("content") or ""),
                actual_resistance_0_1=max(0.0, min(1.0, actual)),
                actual_resistance_0_10=max(0.0, min(10.0, actual * 10)),
                resistance_update_reasoning=str(update.get("reasoning") or ""),
            )
        )

    return turns, skipped


def load_prompt_template(prompt_file: str) -> str:
    path = PROMPT_ROOT / prompt_file
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def render_prompt(template: str, profile: str, turn: EvaluatedTurn) -> str:
    return render_template(
        template,
        {
            "patient_profile": profile or "(No patient profile available.)",
            "therapist_utterance": turn.previous_therapist_utterance,
            "patient_utterance": turn.patient_utterance,
        },
    )


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


def make_custom_id(index: int, filename: str, utterance_index: int, metric: str) -> str:
    name = safe_custom_id_part(Path(filename).stem)
    metric_part = safe_custom_id_part(metric)
    return f"steering:{index:06d}:{name}:u{utterance_index}:{metric_part}"


def make_item_key(filename: str, utterance_index: int) -> str:
    return f"{filename}#{utterance_index}"


def serialize_base_turn(turn: EvaluatedTurn) -> dict[str, Any]:
    return {
        "utterance_index": turn.utterance_index,
        "stage": turn.stage,
        "substage": turn.substage,
        "previous_therapist_utterance": turn.previous_therapist_utterance,
        "patient_utterance": turn.patient_utterance,
        "actual_resistance_0_1": turn.actual_resistance_0_1,
        "actual_resistance_0_10": turn.actual_resistance_0_10,
        "actual_resistance_interval": resistance_interval(turn.actual_resistance_0_1),
        "resistance_update_reasoning": turn.resistance_update_reasoning,
    }


def build_requests(
    files: list[Path],
    output_dir: Path,
    model: str,
    max_iter: int,
    n_turns: int | None = None,
    random_seed: int | None = None,
    metric_keys: Iterable[str] | None = None,
) -> tuple[list[BatchRequest], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    active_metric_keys = list(metric_keys) if metric_keys is not None else list(METRIC_SPECS)
    templates = {
        key: load_prompt_template(spec.prompt_file)
        for key, spec in METRIC_SPECS.items()
        if key in active_metric_keys
    }
    requests: list[BatchRequest] = []
    sessions: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    request_index = 0

    for path in files:
        try:
            payload = read_json(path)
        except Exception as exc:
            skipped_files.append({"file": str(path), "reason": f"could not read JSON: {exc}"})
            continue

        output_file = Path("sessions") / path.name
        if not isinstance(payload, dict):
            skipped_files.append({"file": str(path), "reason": "top-level JSON is not an object"})
            sessions.append(
                {
                    "input_file": str(path),
                    "input_filename": path.name,
                    "model": model,
                    "max_iter": max_iter,
                    "evaluated_metrics": active_metric_keys,
                    "evaluated_turns": [],
                    "metrics": session_metrics([]),
                    "skipped_turns": [{"reason": "top-level JSON is not an object"}],
                }
            )
            continue

        turns, skipped_turns = extract_evaluable_turns(payload)
        profile = format_patient_profile(payload)
        records.append(
            {
                "path": path,
                "output_file": output_file,
                "profile": profile,
                "turns": turns,
                "skipped_turns": skipped_turns,
            }
        )

        if not turns:
            skipped_files.append({"file": str(path), "reason": "no evaluable patient turns"})

    sampling = {
        "enabled": n_turns is not None,
        "requested_n_turns": n_turns,
        "random_seed": random_seed,
        "scope": "per_session",
        "total_evaluable_turns_before_sampling": 0,
        "selected_turns": 0,
        "sampled_out_turns": 0,
    }

    for record_index, record in enumerate(records):
        path = record["path"]
        output_file = record["output_file"]
        profile = record["profile"]
        turns = record["turns"]
        selected_turns: list[EvaluatedTurn] = []
        sampled_out_turns = []
        selected_turn_indices: set[int] | None = None

        if n_turns is not None:
            if n_turns >= len(turns):
                selected_turn_indices = set(range(len(turns)))
            else:
                rng = random.Random(None if random_seed is None else f"{random_seed}:{record_index}:{path.name}")
                selected_turn_indices = set(rng.sample(range(len(turns)), n_turns))

        for turn_index, turn in enumerate(turns):
            if selected_turn_indices is None or turn_index in selected_turn_indices:
                selected_turns.append(turn)
            else:
                sampled_out_turns.append(
                    {
                        "utterance_index": turn.utterance_index,
                        "stage": turn.stage,
                        "substage": turn.substage,
                        "reason": "not selected by per-session --n-turns sampling",
                    }
                )

        sampling["total_evaluable_turns_before_sampling"] += len(turns)
        sampling["selected_turns"] += len(selected_turns)
        sampling["sampled_out_turns"] += len(sampled_out_turns)

        session = {
            "input_file": str(path),
            "input_filename": path.name,
            "model": model,
            "max_iter": max_iter,
            "evaluated_metrics": active_metric_keys,
            "evaluated_turns": [],
            "metrics": session_metrics([]),
            "skipped_turns": record["skipped_turns"],
            "sampling": {
                "enabled": n_turns is not None,
                "requested_n_turns": n_turns,
                "random_seed": random_seed,
                "scope": "per_session",
                "source_evaluable_turn_count": len(turns),
                "selected_turn_count": len(selected_turns),
                "sampled_out_turn_count": len(sampled_out_turns),
            },
            "sampled_out_turns": sampled_out_turns,
        }

        for turn in selected_turns:
            turn_payload = serialize_base_turn(turn)
            turn_payload["item_key"] = make_item_key(path.name, turn.utterance_index)
            turn_payload["batch_custom_ids"] = {}

            for metric_key in active_metric_keys:
                spec = METRIC_SPECS[metric_key]
                prompt = render_prompt(templates[metric_key], profile, turn)
                custom_id = make_custom_id(request_index, path.name, turn.utterance_index, metric_key)
                request_index += 1
                turn_payload["batch_custom_ids"][metric_key] = custom_id
                requests.append(
                    BatchRequest(
                        custom_id=custom_id,
                        metric=metric_key,
                        input_file=path.name,
                        output_file=output_file.as_posix(),
                        item_key=turn_payload["item_key"],
                        utterance_index=turn.utterance_index,
                        min_score=spec.min_score,
                        max_score=spec.max_score,
                        body=make_chat_body(prompt, model, max_iter),
                    )
                )

            session["evaluated_turns"].append(turn_payload)

        sessions.append(session)

    return requests, sessions, skipped_files, sampling


def request_map_from_specs(specs: list[BatchRequest]) -> dict[str, dict[str, Any]]:
    return {spec.custom_id: spec.map_entry() for spec in specs}


def write_jsonl(path: Path, specs: list[BatchRequest]) -> None:
    lines = [json.dumps(spec.jsonl_line(), ensure_ascii=True) for spec in specs]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def validate_batch_input(path: Path, request_count: int) -> None:
    if request_count > MAX_BATCH_REQUESTS:
        raise SteeringEvaluationError(
            f"Batch has {request_count} requests; maximum supported per batch is {MAX_BATCH_REQUESTS}."
        )
    size = path.stat().st_size
    if size > MAX_BATCH_FILE_BYTES:
        raise SteeringEvaluationError(
            f"Batch input file is {size} bytes; maximum supported size is {MAX_BATCH_FILE_BYTES} bytes."
        )


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


def clamp_score(value: float, min_score: float, max_score: float) -> float | None:
    if min_score <= value <= max_score:
        return value
    return None


def parse_rating(text: str, min_score: float, max_score: float) -> tuple[float, str | None]:
    matches = FINAL_RATING_RE.findall(text)
    if not matches:
        return 0.0, "missing Rating line"
    raw_value = matches[-1]
    try:
        value = float(raw_value)
    except ValueError:
        return 0.0, f"invalid numeric rating: {raw_value!r}"
    clamped = clamp_score(value, min_score, max_score)
    if clamped is None:
        return 0.0, f"rating {value} outside [{min_score}, {max_score}]"
    return clamped, None


def parse_batch_output_lines(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SteeringEvaluationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            custom_id = payload.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise SteeringEvaluationError(f"Batch output line is missing custom_id at {path}:{line_number}")
            if custom_id in results:
                raise SteeringEvaluationError(f"Duplicate custom_id in batch output: {custom_id}")
            results[custom_id] = payload
    return results


def empty_judge_result(score: float = 0.0, *, error: str | None = None) -> dict[str, Any]:
    result = {
        "score": score,
        "cost": 0.0,
        "choices": [],
    }
    if error:
        result["error"] = error
    return result


def parse_batch_outputs(
    output_jsonl: Path,
    request_map: dict[str, dict[str, Any]],
    max_iter: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_outputs = parse_batch_output_lines(output_jsonl)
    parsed: dict[str, dict[str, Any]] = {}
    report = {
        "request_count": len(request_map),
        "response_count": len(raw_outputs),
        "missing_custom_ids": [],
        "unexpected_custom_ids": [],
        "parse_errors": [],
        "choice_count_warnings": [],
        "batch_errors": [],
    }

    for custom_id in sorted(raw_outputs):
        if custom_id not in request_map:
            report["unexpected_custom_ids"].append(custom_id)

    for custom_id, entry in sorted(request_map.items()):
        payload = raw_outputs.get(custom_id)
        if payload is None:
            report["missing_custom_ids"].append(custom_id)
            parsed[custom_id] = empty_judge_result(error="missing batch output")
            continue

        if payload.get("error"):
            report["batch_errors"].append({"custom_id": custom_id, "error": payload.get("error")})
            parsed[custom_id] = empty_judge_result(error=f"batch error: {payload.get('error')}")
            continue

        response = payload.get("response") or {}
        if not isinstance(response, dict):
            report["batch_errors"].append({"custom_id": custom_id, "error": "response is not an object"})
            parsed[custom_id] = empty_judge_result(error="response is not an object")
            continue

        body = response.get("body") or {}
        if not isinstance(body, dict):
            report["batch_errors"].append({"custom_id": custom_id, "error": "response body is not an object"})
            parsed[custom_id] = empty_judge_result(error="response body is not an object")
            continue

        body_error = body.get("error")
        if body_error:
            report["batch_errors"].append({"custom_id": custom_id, "error": body_error})
            parsed[custom_id] = empty_judge_result(error=f"response body error: {body_error}")
            continue

        choices = body.get("choices") or []
        if not isinstance(choices, list):
            choices = []
        if len(choices) != max_iter:
            report["choice_count_warnings"].append(
                {"custom_id": custom_id, "expected": max_iter, "received": len(choices)}
            )

        min_score = float(entry["min_score"])
        max_score = float(entry["max_score"])
        parsed_choices = []
        for choice_index, choice in enumerate(choices):
            message = get_field(choice, "message", {})
            raw_text = str(get_field(message, "content", "") or "")
            parsed_score, parse_error = parse_rating(raw_text, min_score, max_score)
            if parse_error:
                report["parse_errors"].append(
                    {"custom_id": custom_id, "choice_index": choice_index, "error": parse_error}
                )
            parsed_choices.append(
                {
                    "raw_text": raw_text,
                    "parsed_score": parsed_score,
                    "parse_error": parse_error,
                }
            )

        avg_score = mean(float(choice["parsed_score"]) for choice in parsed_choices)
        parsed[custom_id] = {
            "score": avg_score if avg_score is not None else 0.0,
            "cost": response_usage_cost(body, input_cost_per_1m, output_cost_per_1m),
            "choices": parsed_choices,
        }

    return parsed, report


def score_turn(
    turn: dict[str, Any],
    judge_scores: dict[str, dict[str, Any]],
    metric_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    active_metric_keys = set(metric_keys) if metric_keys is not None else set(METRIC_SPECS)
    resistance = judge_scores.get("resistance") or empty_judge_result(error="missing resistance result")
    context = (
        judge_scores.get("context_relevance") or empty_judge_result(error="missing context result")
        if "context_relevance" in active_metric_keys
        else None
    )
    realism = (
        judge_scores.get("patient_like_realism") or empty_judge_result(error="missing realism result")
        if "patient_like_realism" in active_metric_keys
        else None
    )
    leakage = (
        judge_scores.get("prompt_leakage") or empty_judge_result(error="missing leakage result")
        if "prompt_leakage" in active_metric_keys
        else None
    )

    actual_r_0_1 = float(turn["actual_resistance_0_1"])
    actual_r_0_10 = float(turn["actual_resistance_0_10"])
    judged_r_0_10 = float(resistance.get("score") or 0.0)
    judged_r_0_1 = max(0.0, min(1.0, judged_r_0_10 / 10.0))
    context_score = float(context.get("score") or 0.0) if context is not None else None
    realism_score = float(realism.get("score") or 0.0) if realism is not None else None
    leakage_score = float(leakage.get("score") or 0.0) if leakage is not None else None

    resistance_error_0_10 = abs(judged_r_0_10 - actual_r_0_10)
    resistance_error_0_1 = abs(judged_r_0_1 - actual_r_0_1)
    resistance_score_0_2 = 2 * max(0.0, 1 - resistance_error_0_10 / RESISTANCE_ERROR_TOLERANCE_0_10)
    harmonic = (
        harmonic_mean([resistance_score_0_2, context_score, realism_score])
        if context_score is not None and realism_score is not None
        else None
    )
    prompt_leakage_flag = leakage_score >= 0.5 if leakage_score is not None else None
    interval_match = resistance_interval(actual_r_0_1) == resistance_interval(judged_r_0_1)

    scored = dict(turn)
    scored.update(
        {
            "judged_resistance_0_10": judged_r_0_10,
            "judged_resistance_0_1": judged_r_0_1,
            "judged_resistance_interval": resistance_interval(judged_r_0_1),
            "context_relevance": context_score,
            "patient_like_realism": realism_score,
            "prompt_leakage_score": leakage_score,
            "prompt_leakage_flag": prompt_leakage_flag,
            "resistance_error_0_10": resistance_error_0_10,
            "resistance_error_0_1": resistance_error_0_1,
            "resistance_score_0_2": resistance_score_0_2,
            "harmonic_mean": harmonic,
            "interval_match": interval_match,
            "cost": sum(float(result.get("cost") or 0.0) for result in judge_scores.values()),
            "judge_scores": judge_scores,
            "evaluated_metrics": sorted(active_metric_keys),
        }
    )
    return scored


def attach_judgements(
    sessions: list[dict[str, Any]],
    parsed_results: dict[str, dict[str, Any]],
    artifact_dir: Path | None,
    metric_keys: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    active_metric_keys = list(metric_keys) if metric_keys is not None else list(METRIC_SPECS)
    scored_sessions = []
    for session in sessions:
        scored_turns = []
        for turn in session.get("evaluated_turns", []):
            custom_ids = turn.get("batch_custom_ids") or {}
            judge_scores = {
                metric: parsed_results.get(custom_id, empty_judge_result(error="missing parsed result"))
                for metric, custom_id in custom_ids.items()
                if metric in active_metric_keys
            }
            scored_turns.append(score_turn(turn, judge_scores, active_metric_keys))

        scored_session = dict(session)
        scored_session["evaluated_turns"] = scored_turns
        scored_session["metrics"] = session_metrics(scored_turns)
        if artifact_dir is not None:
            scored_session["batch_artifacts"] = str(artifact_dir)
        scored_sessions.append(scored_session)
    return scored_sessions


def mean_numeric_field(turns: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(turn[field])
        for turn in turns
        if turn.get(field) is not None
    ]
    return mean(values)


def rate_bool_field(turns: list[dict[str, Any]], field: str) -> float | None:
    values = [
        1.0 if turn[field] else 0.0
        for turn in turns
        if turn.get(field) is not None
    ]
    return mean(values)


def session_metrics(turns: list[dict[str, Any]]) -> dict[str, Any]:
    if not turns:
        return {
            "turn_count": 0,
            "mean_harmonic_score": None,
            "mean_resistance_score_0_2": None,
            "mean_judged_resistance_0_10": None,
            "mean_judged_resistance_0_1": None,
            "mean_context_relevance": None,
            "mean_patient_like_realism": None,
            "prompt_leakage_rate": None,
            "resistance_mae_0_1": None,
            "resistance_rmse_0_1": None,
            "interval_accuracy": None,
            "cost": 0.0,
        }

    errors = [
        float(turn["resistance_error_0_1"])
        for turn in turns
        if turn.get("resistance_error_0_1") is not None
    ]
    return {
        "turn_count": len(turns),
        "mean_harmonic_score": mean_numeric_field(turns, "harmonic_mean"),
        "mean_resistance_score_0_2": mean_numeric_field(turns, "resistance_score_0_2"),
        "mean_judged_resistance_0_10": mean_numeric_field(turns, "judged_resistance_0_10"),
        "mean_judged_resistance_0_1": mean_numeric_field(turns, "judged_resistance_0_1"),
        "mean_context_relevance": mean_numeric_field(turns, "context_relevance"),
        "mean_patient_like_realism": mean_numeric_field(turns, "patient_like_realism"),
        "prompt_leakage_rate": rate_bool_field(turns, "prompt_leakage_flag"),
        "resistance_mae_0_1": mean(errors),
        "resistance_rmse_0_1": math.sqrt(mean(error * error for error in errors) or 0.0) if errors else None,
        "interval_accuracy": rate_bool_field(turns, "interval_match"),
        "cost": sum(float(turn["cost"]) for turn in turns),
    }


def summary_from_sessions(
    input_dir: Path,
    output_dir: Path,
    files_seen: int,
    session_results: list[dict[str, Any]],
    skipped_files: list[dict[str, Any]],
    artifact_dir: Path | None,
    parse_report: dict[str, Any] | None,
) -> dict[str, Any]:
    all_turns = [
        turn
        for session in session_results
        for turn in session.get("evaluated_turns", [])
    ]
    errors = [
        float(turn["resistance_error_0_1"])
        for turn in all_turns
        if turn.get("resistance_error_0_1") is not None
    ]
    skipped_turns = [
        {"file": session.get("input_file"), **skipped}
        for session in session_results
        for skipped in session.get("skipped_turns", [])
    ]

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "batch_artifacts": str(artifact_dir) if artifact_dir is not None else None,
        "total_files_seen": files_seen,
        "total_files_evaluated": sum(1 for session in session_results if session.get("evaluated_turns")),
        "total_turns_evaluated": len(all_turns),
        "mean_judge_scores": {
            "judged_resistance_0_10": mean_numeric_field(all_turns, "judged_resistance_0_10"),
            "judged_resistance_0_1": mean_numeric_field(all_turns, "judged_resistance_0_1"),
            "resistance_score_0_2": mean_numeric_field(all_turns, "resistance_score_0_2"),
            "context_relevance": mean_numeric_field(all_turns, "context_relevance"),
            "patient_like_realism": mean_numeric_field(all_turns, "patient_like_realism"),
            "prompt_leakage": mean_numeric_field(all_turns, "prompt_leakage_score"),
        },
        "mean_harmonic_score": mean_numeric_field(all_turns, "harmonic_mean"),
        "resistance_mae_0_1": mean(errors),
        "resistance_rmse_0_1": math.sqrt(mean(error * error for error in errors) or 0.0) if errors else None,
        "interval_accuracy": rate_bool_field(all_turns, "interval_match"),
        "prompt_leakage_rate": rate_bool_field(all_turns, "prompt_leakage_flag"),
        "total_estimated_cost": sum(float(turn["cost"]) for turn in all_turns),
        "skipped_files": skipped_files,
        "skipped_turns": skipped_turns,
        "parse_report": parse_report,
    }


def iter_json_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def write_session_outputs(output_dir: Path, sessions: list[dict[str, Any]]) -> None:
    for session in sessions:
        filename = str(session.get("input_filename") or Path(str(session.get("input_file"))).name)
        atomic_write_json(output_dir / "sessions" / filename, session)


def create_run_artifacts(output_dir: Path, requests: list[BatchRequest], sessions: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    artifact_dir = output_dir / "_batch_artifacts" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    request_map = request_map_from_specs(requests)
    input_jsonl = artifact_dir / "input.jsonl"
    request_map_path = artifact_dir / "request_map.json"
    prepared_sessions_path = artifact_dir / "prepared_sessions.json"
    manifest_path = artifact_dir / "manifest.json"

    write_jsonl(input_jsonl, requests)
    atomic_write_json(request_map_path, request_map)
    atomic_write_json(prepared_sessions_path, sessions)
    validate_batch_input(input_jsonl, len(requests))

    manifest = {
        "version": 1,
        "run_id": run_id,
        "status": "prepared",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "endpoint": BATCH_ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "request_count": len(requests),
        "input_jsonl": str(input_jsonl),
        "request_map": str(request_map_path),
        "prepared_sessions": str(prepared_sessions_path),
    }
    atomic_write_json(manifest_path, manifest)
    return artifact_dir, manifest


def save_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_write_json(artifact_dir / "manifest.json", manifest)


def make_openai_client() -> Any:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=REPO_ROOT / ".env")
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise SteeringEvaluationError("Missing OpenAI API key. Set GPT_API_KEY or OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SteeringEvaluationError("The openai Python package is required to run steering batches.") from exc
    return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


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


def upload_and_create_batch(client: Any, artifact_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("batch_id"):
        return
    input_jsonl = Path(str(manifest["input_jsonl"]))
    with input_jsonl.open("rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    uploaded_file_id = get_field(uploaded_file, "id")
    if not uploaded_file_id:
        raise SteeringEvaluationError("OpenAI file upload did not return a file id.")

    manifest["input_file_id"] = uploaded_file_id
    manifest["uploaded_file"] = to_plain(uploaded_file)
    manifest["status"] = "uploaded"
    save_manifest(artifact_dir, manifest)

    batch = client.batches.create(
        input_file_id=uploaded_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={"description": "ReasonCBT steering judge evaluation"},
    )
    batch_id = get_field(batch, "id")
    if not batch_id:
        raise SteeringEvaluationError("OpenAI batch creation did not return a batch id.")
    manifest["batch_id"] = batch_id
    manifest["status"] = get_field(batch, "status", "submitted")
    manifest["batch"] = to_plain(batch)
    save_manifest(artifact_dir, manifest)


def download_batch_outputs(client: Any, artifact_dir: Path, manifest: dict[str, Any]) -> None:
    output_file_id = manifest.get("output_file_id")
    error_file_id = manifest.get("error_file_id")
    if output_file_id and not manifest.get("output_jsonl"):
        output_path = artifact_dir / "output.jsonl"
        download_file(client, str(output_file_id), output_path)
        manifest["output_jsonl"] = str(output_path)
    if error_file_id and not manifest.get("error_jsonl"):
        error_path = artifact_dir / "error.jsonl"
        download_file(client, str(error_file_id), error_path)
        manifest["error_jsonl"] = str(error_path)


def poll_until_complete(
    client: Any,
    artifact_dir: Path,
    manifest: dict[str, Any],
    poll_interval: float,
    local_timeout_hours: float,
) -> None:
    if manifest.get("output_jsonl"):
        return
    batch_id = str(manifest.get("batch_id") or "")
    if not batch_id:
        raise SteeringEvaluationError("Cannot poll before a batch is created.")

    deadline = time.monotonic() + (local_timeout_hours * 3600)
    while True:
        batch = client.batches.retrieve(batch_id)
        batch_plain = to_plain(batch)
        status = str(get_field(batch, "status", batch_plain.get("status", "")))
        manifest["status"] = status
        manifest["batch"] = batch_plain
        manifest["output_file_id"] = get_field(batch, "output_file_id", batch_plain.get("output_file_id"))
        manifest["error_file_id"] = get_field(batch, "error_file_id", batch_plain.get("error_file_id"))
        manifest["request_counts"] = to_plain(get_field(batch, "request_counts", batch_plain.get("request_counts")))
        save_manifest(artifact_dir, manifest)

        if status in FINAL_STATUSES:
            download_batch_outputs(client, artifact_dir, manifest)
            save_manifest(artifact_dir, manifest)
            if status in FAILED_STATUSES:
                raise SteeringEvaluationError(f"Batch ended with status {status}.")
            counts = manifest.get("request_counts") or {}
            failed_count = get_field(counts, "failed", 0) or 0
            if int(failed_count) > 0:
                raise SteeringEvaluationError(f"Batch completed with {failed_count} failed requests.")
            if not manifest.get("output_jsonl"):
                raise SteeringEvaluationError("Batch completed without an output file.")
            return

        if time.monotonic() >= deadline:
            raise SteeringEvaluationError(
                f"Timed out waiting for batch after {local_timeout_hours} hours."
            )
        time.sleep(max(0.0, poll_interval))


def run_openai_batch(
    artifact_dir: Path,
    manifest: dict[str, Any],
    poll_interval: float,
    local_timeout_hours: float,
    client: Any | None = None,
) -> dict[str, Any]:
    active_client = client or make_openai_client()
    upload_and_create_batch(active_client, artifact_dir, manifest)
    poll_until_complete(active_client, artifact_dir, manifest, poll_interval, local_timeout_hours)
    return manifest


def load_request_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SteeringEvaluationError(f"Request map must be a JSON object: {path}")
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def run_steering_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    max_iter: int = DEFAULT_MAX_ITER,
    *,
    model: str = MODEL_NAME,
    recursive: bool = False,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    local_timeout_hours: float = DEFAULT_LOCAL_TIMEOUT_HOURS,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
    n_turns: int | None = None,
    random_seed: int | None = None,
    only_resistance: bool = False,
    prepare_only: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    if max_iter < 1:
        raise SteeringEvaluationError(f"max_iter must be positive, got {max_iter}")
    if n_turns is not None and n_turns < 1:
        raise SteeringEvaluationError(f"n_turns must be positive when provided, got {n_turns}")
    active_metric_keys = ["resistance"] if only_resistance else list(METRIC_SPECS)

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    if not input_path.is_dir():
        raise SteeringEvaluationError(f"input_dir does not exist or is not a directory: {input_path}")

    files = iter_json_files(input_path, recursive)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "sessions").mkdir(parents=True, exist_ok=True)

    requests, prepared_sessions, skipped_files, sampling = build_requests(
        files,
        output_path,
        model,
        max_iter,
        n_turns=n_turns,
        random_seed=random_seed,
        metric_keys=active_metric_keys,
    )
    artifact_dir, manifest = create_run_artifacts(output_path, requests, prepared_sessions)
    manifest["model"] = model
    manifest["max_iter"] = max_iter
    manifest["evaluated_metrics"] = active_metric_keys
    manifest["only_resistance"] = only_resistance
    manifest["sampling"] = sampling
    manifest["input_dir"] = str(input_path)
    manifest["output_dir"] = str(output_path)
    manifest["files_seen"] = len(files)
    manifest["recursive"] = recursive
    save_manifest(artifact_dir, manifest)

    if prepare_only:
        return manifest

    parse_report: dict[str, Any] | None = None
    if requests:
        try:
            run_openai_batch(artifact_dir, manifest, poll_interval, local_timeout_hours, client=client)
            request_map = load_request_map(Path(str(manifest["request_map"])))
            output_jsonl = Path(str(manifest["output_jsonl"]))
            parsed_results, parse_report = parse_batch_outputs(
                output_jsonl,
                request_map,
                max_iter,
                input_cost_per_1m,
                output_cost_per_1m,
            )
            parse_report_path = artifact_dir / "parse_report.json"
            atomic_write_json(parse_report_path, parse_report)
            manifest["parse_report"] = str(parse_report_path)
            manifest["status"] = "parsed"
            save_manifest(artifact_dir, manifest)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            atomic_write_json(
                artifact_dir / "failure_report.json",
                {"error": str(exc), "type": type(exc).__name__, "updated_at": utc_now()},
            )
            save_manifest(artifact_dir, manifest)
            if isinstance(exc, SteeringEvaluationError):
                raise
            raise SteeringEvaluationError(str(exc)) from exc
    else:
        parsed_results = {}
        manifest["status"] = "no_requests"
        save_manifest(artifact_dir, manifest)

    scored_sessions = attach_judgements(prepared_sessions, parsed_results, artifact_dir, active_metric_keys)
    write_session_outputs(output_path, scored_sessions)
    summary = summary_from_sessions(
        input_path,
        output_path,
        len(files),
        scored_sessions,
        skipped_files,
        artifact_dir,
        parse_report,
    )
    summary["evaluated_metrics"] = active_metric_keys
    summary["only_resistance"] = only_resistance
    summary["sampling"] = sampling
    atomic_write_json(output_path / "summary.json", summary)
    manifest["status"] = "completed"
    manifest["summary"] = str(output_path / "summary.json")
    save_manifest(artifact_dir, manifest)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simulated patient prompt steering with AxBench-like judges through the OpenAI Batch API."
    )
    parser.add_argument("-i", "--input_dir", type=Path, default=Path("."), help="Directory with simulation JSON files.")
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("."), help="Directory for steering evaluation outputs.")
    parser.add_argument("-m_iter", "--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Judge samples per metric.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Judge model. Default: {MODEL_NAME}")
    parser.add_argument("--recursive", action="store_true", help="Scan JSON files recursively under input_dir.")
    parser.add_argument(
        "--n-turns",
        type=int,
        default=10,
        help="Randomly evaluate at most this many patient turns per session file.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible --n-turns sampling.",
    )
    parser.add_argument(
        "--only-resistance",
        action="store_true",
        help="Evaluate only the resistance judge and skip context, realism, and prompt leakage.",
    )
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--local-timeout-hours", type=float, default=DEFAULT_LOCAL_TIMEOUT_HOURS)
    parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
    parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write batch artifacts without uploading or polling an OpenAI batch.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_steering_batch(
            args.input_dir,
            args.output_dir,
            args.max_iter,
            model=args.model,
            recursive=args.recursive,
            poll_interval=args.poll_interval,
            local_timeout_hours=args.local_timeout_hours,
            input_cost_per_1m=args.input_cost_per_1m,
            output_cost_per_1m=args.output_cost_per_1m,
            n_turns=args.n_turns,
            random_seed=args.random_seed,
            only_resistance=args.only_resistance,
            prepare_only=args.prepare_only,
        )
    except SteeringEvaluationError as exc:
        print(f"Steering evaluation failed: {exc}", file=os.sys.stderr)
        return 1

    artifact_dir = Path(manifest["output_dir"]) / "_batch_artifacts" / manifest["run_id"]
    if args.prepare_only:
        print("Steering batch artifacts prepared.")
        print(f"Run artifacts: {artifact_dir}")
        print(f"Request count: {manifest['request_count']}")
        if manifest.get("only_resistance"):
            print("Evaluated metrics: resistance")
        sampling = manifest.get("sampling") or {}
        if sampling.get("enabled"):
            print(
                "Sampled turns: "
                f"{sampling.get('selected_turns')} / "
                f"{sampling.get('total_evaluable_turns_before_sampling')}"
            )
        return 0

    print("Steering evaluation completed.")
    print(f"Run artifacts: {artifact_dir}")
    print(f"Session outputs: {Path(manifest['output_dir']) / 'sessions'}")
    print(f"Summary: {Path(manifest['output_dir']) / 'summary.json'}")
    if manifest.get("only_resistance"):
        print("Evaluated metrics: resistance")
    sampling = manifest.get("sampling") or {}
    if sampling.get("enabled"):
        print(
            "Sampled turns: "
            f"{sampling.get('selected_turns')} / "
            f"{sampling.get('total_evaluable_turns_before_sampling')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
