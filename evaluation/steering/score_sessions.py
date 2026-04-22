#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from .steering import (
        METRIC_SPECS,
        SteeringEvaluationError,
        atomic_write_json,
        empty_judge_result,
        mean,
        parse_rating,
        score_turn,
        session_metrics,
        summary_from_sessions,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from steering import (  # type: ignore[no-redef]
        METRIC_SPECS,
        SteeringEvaluationError,
        atomic_write_json,
        empty_judge_result,
        mean,
        parse_rating,
        score_turn,
        session_metrics,
        summary_from_sessions,
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute steering metrics from existing session JSON files containing judge outputs."
    )
    parser.add_argument(
        "-i",
        "--sessions_dir",
        type=Path,
        required=True,
        help="Directory containing per-session JSON files with judge_scores or raw judge outputs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Default: parent of sessions_dir, which overwrites sessions/ and summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = score_sessions_folder(args.sessions_dir, args.output_dir)
    except SteeringEvaluationError as exc:
        print(f"Session scoring failed: {exc}")
        return 1

    print("Session scoring completed.")
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
