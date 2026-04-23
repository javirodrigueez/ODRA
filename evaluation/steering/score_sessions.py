#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from .steering import (
        DEFAULT_INPUT_COST_PER_1M,
        DEFAULT_MAX_ITER,
        DEFAULT_OUTPUT_COST_PER_1M,
        METRIC_SPECS,
        SteeringEvaluationError,
        atomic_write_json,
        attach_judgements,
        empty_judge_result,
        load_request_map,
        mean,
        parse_batch_outputs,
        parse_rating,
        score_turn,
        session_metrics,
        summary_from_sessions,
        write_session_outputs,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from steering import (  # type: ignore[no-redef]
        DEFAULT_INPUT_COST_PER_1M,
        DEFAULT_MAX_ITER,
        DEFAULT_OUTPUT_COST_PER_1M,
        METRIC_SPECS,
        SteeringEvaluationError,
        atomic_write_json,
        attach_judgements,
        empty_judge_result,
        load_request_map,
        mean,
        parse_batch_outputs,
        parse_rating,
        score_turn,
        session_metrics,
        summary_from_sessions,
        write_session_outputs,
    )


METRIC_ALIASES = {
    "resistance": ("resistance", "resistance_adherence", "judged_resistance"),
    "context_relevance": ("context_relevance", "context", "instruct_score"),
    "patient_like_realism": ("patient_like_realism", "realism", "fluency", "fluency_score"),
    "prompt_leakage": ("prompt_leakage", "leakage", "prompt_leakage_rate"),
}


def coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_session_files(sessions_dir: Path) -> list[Path]:
    return sorted(path for path in sessions_dir.glob("*.json") if path.is_file())


def read_session_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_session_json(path)
    if not isinstance(payload, dict):
        raise SteeringEvaluationError(f"Expected JSON object at {path}")
    return payload


def read_prepared_sessions(path: Path) -> list[dict[str, Any]]:
    payload = read_session_json(path)
    if not isinstance(payload, list):
        raise SteeringEvaluationError(f"Prepared sessions must be a JSON list: {path}")

    sessions: list[dict[str, Any]] = []
    for index, session in enumerate(payload):
        if not isinstance(session, dict):
            raise SteeringEvaluationError(f"Prepared session at index {index} is not an object: {path}")
        sessions.append(session)
    return sessions


def resolve_batch_companion_path(output_jsonl: Path, explicit_path: str | Path | None, filename: str) -> Path:
    path = Path(explicit_path).resolve() if explicit_path is not None else output_jsonl.parent / filename
    if not path.is_file():
        raise SteeringEvaluationError(f"Missing batch companion file {filename}: {path}")
    return path


def default_batch_output_dir(output_jsonl: Path) -> Path:
    artifact_dir = output_jsonl.parent
    if artifact_dir.parent.name == "_batch_artifacts":
        return artifact_dir.parent.parent
    return artifact_dir


def positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SteeringEvaluationError(f"{field_name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise SteeringEvaluationError(f"{field_name} must be a positive integer, got {parsed}")
    return parsed


def files_seen_from_manifest(manifest: dict[str, Any], fallback: int) -> int:
    try:
        files_seen = int(manifest.get("files_seen", fallback))
    except (TypeError, ValueError):
        return fallback
    return max(0, files_seen)


def batch_metric_keys(manifest: dict[str, Any], request_map: dict[str, dict[str, Any]]) -> list[str]:
    manifest_metrics = manifest.get("evaluated_metrics")
    if isinstance(manifest_metrics, list):
        keys: list[str] = []
        for metric in manifest_metrics:
            metric_key = str(metric)
            if metric_key in METRIC_SPECS and metric_key not in keys:
                keys.append(metric_key)
        if keys:
            return keys

    metrics_in_requests = {
        str(entry.get("metric"))
        for entry in request_map.values()
        if isinstance(entry.get("metric"), str)
    }
    keys = [metric_key for metric_key in METRIC_SPECS if metric_key in metrics_in_requests]
    return keys or list(METRIC_SPECS)


def clamp(value: float, min_score: float, max_score: float) -> float | None:
    if min_score <= value <= max_score:
        return value
    return None


def choice_from_raw(raw_choice: Any, min_score: float, max_score: float) -> dict[str, Any]:
    if isinstance(raw_choice, str):
        raw_text = raw_choice
        parsed_score, parse_error = parse_rating(raw_text, min_score, max_score)
        return {
            "raw_text": raw_text,
            "parsed_score": parsed_score,
            "parse_error": parse_error,
        }

    if not isinstance(raw_choice, dict):
        return {
            "raw_text": "",
            "parsed_score": 0.0,
            "parse_error": f"unsupported choice type: {type(raw_choice).__name__}",
        }

    raw_text = str(
        raw_choice.get("raw_text")
        or raw_choice.get("content")
        or raw_choice.get("text")
        or ""
    )
    if raw_text:
        parsed_score, parse_error = parse_rating(raw_text, min_score, max_score)
        return {
            "raw_text": raw_text,
            "parsed_score": parsed_score,
            "parse_error": parse_error,
        }

    existing_score = coerce_float(raw_choice.get("parsed_score"))
    if existing_score is None:
        existing_score = coerce_float(raw_choice.get("score"))
    if existing_score is None:
        return {
            "raw_text": "",
            "parsed_score": 0.0,
            "parse_error": "choice has no raw_text or numeric parsed_score",
        }

    clamped = clamp(existing_score, min_score, max_score)
    if clamped is None:
        return {
            "raw_text": "",
            "parsed_score": 0.0,
            "parse_error": f"existing score {existing_score} outside [{min_score}, {max_score}]",
        }
    return {
        "raw_text": "",
        "parsed_score": clamped,
        "parse_error": None,
    }


def find_metric_payload(turn: dict[str, Any], metric_key: str) -> Any:
    aliases = METRIC_ALIASES.get(metric_key, (metric_key,))
    for container_name in ("judge_scores", "judge_outputs", "judge_results", "raw_judge_outputs"):
        container = turn.get(container_name)
        if not isinstance(container, dict):
            continue
        for alias in aliases:
            if alias in container:
                return container[alias]
    for alias in aliases:
        if alias in turn:
            return turn[alias]
    return None


def normalize_judge_result(metric_key: str, raw_result: Any) -> dict[str, Any]:
    spec = METRIC_SPECS[metric_key]
    min_score = spec.min_score
    max_score = spec.max_score

    if raw_result is None:
        return empty_judge_result(error=f"missing {metric_key} judge output")

    cost = 0.0
    choices_payload: list[Any] = []
    top_level_score: float | None = None

    if isinstance(raw_result, dict):
        cost = coerce_float(raw_result.get("cost")) or 0.0
        choices = raw_result.get("choices")
        if isinstance(choices, list):
            choices_payload = choices
        elif raw_result.get("raw_text") or raw_result.get("content") or raw_result.get("text"):
            choices_payload = [raw_result]
        else:
            top_level_score = coerce_float(raw_result.get("score"))
    elif isinstance(raw_result, list):
        choices_payload = raw_result
    elif isinstance(raw_result, str):
        choices_payload = [raw_result]
    else:
        return empty_judge_result(error=f"unsupported {metric_key} payload type: {type(raw_result).__name__}")

    if choices_payload:
        parsed_choices = [
            choice_from_raw(choice, min_score, max_score)
            for choice in choices_payload
        ]
        avg_score = mean(float(choice["parsed_score"]) for choice in parsed_choices)
        return {
            "score": avg_score if avg_score is not None else 0.0,
            "cost": cost,
            "choices": parsed_choices,
        }

    if top_level_score is not None:
        clamped = clamp(top_level_score, min_score, max_score)
        if clamped is None:
            return empty_judge_result(
                error=f"{metric_key} score {top_level_score} outside [{min_score}, {max_score}]"
            )
        return {
            "score": clamped,
            "cost": cost,
            "choices": [],
        }

    return empty_judge_result(error=f"{metric_key} judge output has no choices or score")


def normalize_turn_for_scoring(turn: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    actual_0_1 = coerce_float(turn.get("actual_resistance_0_1"))
    actual_0_10 = coerce_float(turn.get("actual_resistance_0_10"))
    if actual_0_1 is None and actual_0_10 is not None:
        actual_0_1 = actual_0_10 / 10.0
    if actual_0_10 is None and actual_0_1 is not None:
        actual_0_10 = actual_0_1 * 10.0
    if actual_0_1 is None or actual_0_10 is None:
        return None, {
            "utterance_index": turn.get("utterance_index"),
            "reason": "missing actual_resistance_0_1/actual_resistance_0_10",
        }

    normalized = dict(turn)
    normalized["actual_resistance_0_1"] = max(0.0, min(1.0, actual_0_1))
    normalized["actual_resistance_0_10"] = max(0.0, min(10.0, actual_0_10))
    return normalized, None


def score_session_payload(session: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_turns = []
    skipped_turns = list(session.get("skipped_turns") or [])

    turns = session.get("evaluated_turns")
    if not isinstance(turns, list):
        return {
            **session,
            "evaluated_turns": [],
            "metrics": session_metrics([]),
            "skipped_turns": skipped_turns + [{"reason": "missing evaluated_turns list"}],
        }, skipped_turns

    for turn in turns:
        if not isinstance(turn, dict):
            skipped_turns.append({"reason": "invalid turn object"})
            continue

        normalized_turn, skip_reason = normalize_turn_for_scoring(turn)
        if normalized_turn is None:
            skipped_turns.append(skip_reason or {"reason": "invalid turn"})
            continue

        judge_scores = {
            metric_key: normalize_judge_result(metric_key, find_metric_payload(normalized_turn, metric_key))
            for metric_key in METRIC_SPECS
        }
        scored_turns.append(score_turn(normalized_turn, judge_scores))

    scored_session = dict(session)
    scored_session["evaluated_turns"] = scored_turns
    scored_session["metrics"] = session_metrics(scored_turns)
    scored_session["skipped_turns"] = skipped_turns
    scored_session["rescored_from_judge_outputs"] = True
    return scored_session, skipped_turns


def score_sessions_folder(
    sessions_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    sessions_path = Path(sessions_dir).resolve()
    if not sessions_path.is_dir():
        raise SteeringEvaluationError(f"sessions_dir does not exist or is not a directory: {sessions_path}")

    output_path = Path(output_dir).resolve() if output_dir is not None else sessions_path.parent
    output_sessions_path = output_path / "sessions"
    output_sessions_path.mkdir(parents=True, exist_ok=True)

    files = iter_session_files(sessions_path)
    session_results: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []

    for path in files:
        try:
            payload = read_session_json(path)
        except Exception as exc:
            skipped_files.append({"file": str(path), "reason": f"could not read JSON: {exc}"})
            continue
        if not isinstance(payload, dict):
            skipped_files.append({"file": str(path), "reason": "top-level JSON is not an object"})
            continue

        scored_session, _ = score_session_payload(payload)
        scored_session.setdefault("input_filename", path.name)
        scored_session.setdefault("input_file", str(path))
        atomic_write_json(output_sessions_path / path.name, scored_session)
        session_results.append(scored_session)

    summary = summary_from_sessions(
        sessions_path,
        output_path,
        len(files),
        session_results,
        skipped_files,
        artifact_dir=None,
        parse_report=None,
    )
    summary["source_sessions_dir"] = str(sessions_path)
    summary["rescored_from_judge_outputs"] = True
    atomic_write_json(output_path / "summary.json", summary)
    return summary


def score_sessions_from_batch_jsonl(
    output_jsonl: str | Path,
    output_dir: str | Path | None = None,
    *,
    request_map_path: str | Path | None = None,
    prepared_sessions_path: str | Path | None = None,
    max_iter: int | None = None,
    input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
    output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
) -> dict[str, Any]:
    batch_output_path = Path(output_jsonl).resolve()
    if not batch_output_path.is_file():
        raise SteeringEvaluationError(f"batch output JSONL does not exist or is not a file: {batch_output_path}")

    request_map_file = resolve_batch_companion_path(batch_output_path, request_map_path, "request_map.json")
    prepared_sessions_file = resolve_batch_companion_path(
        batch_output_path,
        prepared_sessions_path,
        "prepared_sessions.json",
    )
    manifest = read_optional_json_object(batch_output_path.parent / "manifest.json")

    request_map = load_request_map(request_map_file)
    prepared_sessions = read_prepared_sessions(prepared_sessions_file)
    active_metric_keys = batch_metric_keys(manifest, request_map)
    expected_choices = (
        positive_int(max_iter, "--max-iter")
        if max_iter is not None
        else positive_int(manifest.get("max_iter") or DEFAULT_MAX_ITER, "manifest max_iter")
    )

    parsed_results, parse_report = parse_batch_outputs(
        batch_output_path,
        request_map,
        expected_choices,
        input_cost_per_1m,
        output_cost_per_1m,
    )
    parse_report["source_batch_jsonl"] = str(batch_output_path)
    parse_report["source_request_map"] = str(request_map_file)
    parse_report["source_prepared_sessions"] = str(prepared_sessions_file)

    output_path = Path(output_dir).resolve() if output_dir is not None else default_batch_output_dir(batch_output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scored_sessions = attach_judgements(prepared_sessions, parsed_results, batch_output_path.parent, active_metric_keys)
    write_session_outputs(output_path, scored_sessions)

    source_input_dir = Path(str(manifest.get("input_dir"))) if manifest.get("input_dir") else batch_output_path
    summary = summary_from_sessions(
        source_input_dir,
        output_path,
        files_seen_from_manifest(manifest, len(prepared_sessions)),
        scored_sessions,
        [],
        batch_output_path.parent,
        parse_report,
    )
    summary["source_batch_jsonl"] = str(batch_output_path)
    summary["source_request_map"] = str(request_map_file)
    summary["source_prepared_sessions"] = str(prepared_sessions_file)
    summary["rescored_from_batch_jsonl"] = True
    summary["evaluated_metrics"] = active_metric_keys
    if "only_resistance" in manifest:
        summary["only_resistance"] = bool(manifest["only_resistance"])
    if "sampling" in manifest:
        summary["sampling"] = manifest["sampling"]
    atomic_write_json(output_path / "summary.json", summary)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute steering metrics from existing session JSON files or raw OpenAI Batch output JSONL."
    )
    parser.add_argument(
        "-i",
        "--sessions_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing per-session JSON files with judge_scores or raw judge outputs. "
            "With --from-batch-jsonl, this is the raw batch output JSONL path."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: parent of sessions_dir in session mode; "
            "standard batch run output directory in --from-batch-jsonl mode."
        ),
    )
    parser.add_argument(
        "-b",
        "--from-batch-jsonl",
        action="store_true",
        help=(
            "Score from a raw OpenAI Batch output JSONL. Uses sibling request_map.json, "
            "prepared_sessions.json, and manifest.json unless overridden."
        ),
    )
    parser.add_argument(
        "--request-map",
        type=Path,
        default=None,
        help="Request map JSON to use with --from-batch-jsonl. Default: sibling request_map.json.",
    )
    parser.add_argument(
        "--prepared-sessions",
        type=Path,
        default=None,
        help="Prepared sessions JSON to use with --from-batch-jsonl. Default: sibling prepared_sessions.json.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=f"Expected choices per batch request. Default: manifest max_iter, or {DEFAULT_MAX_ITER}.",
    )
    parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
    parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.from_batch_jsonl:
            summary = score_sessions_from_batch_jsonl(
                args.sessions_dir,
                args.output_dir,
                request_map_path=args.request_map,
                prepared_sessions_path=args.prepared_sessions,
                max_iter=args.max_iter,
                input_cost_per_1m=args.input_cost_per_1m,
                output_cost_per_1m=args.output_cost_per_1m,
            )
        else:
            if args.request_map or args.prepared_sessions or args.max_iter is not None:
                raise SteeringEvaluationError(
                    "--request-map, --prepared-sessions, and --max-iter require --from-batch-jsonl"
                )
            summary = score_sessions_folder(args.sessions_dir, args.output_dir)
    except SteeringEvaluationError as exc:
        print(f"Session scoring failed: {exc}")
        return 1

    print("Batch JSONL scoring completed." if args.from_batch_jsonl else "Session scoring completed.")
    print(f"Session outputs: {Path(summary['output_dir']) / 'sessions'}")
    print(f"Summary: {Path(summary['output_dir']) / 'summary.json'}")
    print(f"Turns evaluated: {summary['total_turns_evaluated']}")
    if summary["total_turns_evaluated"] == 0:
        return 1
    if summary.get("resistance_rmse_0_1") is not None and math.isnan(float(summary["resistance_rmse_0_1"])):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
