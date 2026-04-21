#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_expert_session_texts import (
    SessionSet,
    SessionTextExportError,
    build_session_sets,
    format_profile,
    load_json,
    write_text,
)


RCBT_REASONING_STAGES = (
    "opening",
    "address_tib",
    "id_distortions",
    "cbt_technique",
    "homework",
    "meta_conversation",
    "closing",
)

RCBT_ALLOWED_TRACES: dict[str, set[str]] = {
    "opening": {
        "opening_info_update",
        "opening_readiness_check",
        "resistance_prediction",
        "tib_check",
    },
    "id_distortions": {
        "cognitive_gap_planning",
        "cognitive_gap_response_review",
        "cognitive_refinement_planning",
        "cognitive_refinement_response_review",
        "core_belief_probability",
        "distortion_probability",
        "initial_cognitive_model_bootstrap",
        "opening_info_update",
        "opening_readiness_check",
        "resistance_prediction",
        "tib_check",
    },
    "cbt_technique": {
        "core_belief_probability",
        "distortion_probability",
        "id_distortions_completion_check",
        "resistance_prediction",
        "technique_auto_complete",
        "technique_completion_check",
        "technique_evaluation",
        "technique_fit_check",
        "technique_plan",
        "technique_selection",
        "tib_check",
    },
    "homework": {
        "apply_more_techniques_check",
        "homework_task_proposal",
        "resistance_prediction",
        "technique_completion_check",
        "technique_evaluation",
        "tib_check",
    },
    "meta_conversation": {
        "meta_conversation_end_session_redirect",
        "meta_conversation_turn_cap",
        "resistance_prediction",
        "tib_check",
    },
    "closing": {
        "closing_feedback_check",
        "closing_plan_agreement_check",
        "closing_summary_generation",
        "homework_agreement_check",
        "meta_conversation_end_session_repeat",
        "next_session_plan_generation",
        "resistance_prediction",
        "tib_check",
    },
    "address_tib": {
        "address_tib_plan_generation",
        "resistance_prediction",
        "tib_check",
    },
}

RCBT_TRACE_OUTPUT_FIELDS: dict[str, tuple[str, ...] | None] = {
    "address_tib_plan_generation": ("tib_plan",),
    "apply_more_techniques_check": ("apply_more", "reasoning"),
    "closing_feedback_check": (
        "improvement_requested",
        "improvement_description",
        "improvement_plan",
    ),
    "closing_plan_agreement_check": ("agreed", "reasoning"),
    "closing_summary_generation": ("session_summary",),
    "cognitive_gap_planning": ("plan",),
    "cognitive_gap_response_review": (
        "answered_satisfactorily",
        "identified_items",
        "reasoning",
    ),
    "cognitive_refinement_planning": ("target", "plan"),
    "cognitive_refinement_response_review": (
        "answered_satisfactorily",
        "identified_items",
        "reasoning",
    ),
    "core_belief_probability": None,
    "distortion_probability": None,
    "homework_agreement_check": ("agreement_reached", "agreed_tasks", "reasoning"),
    "homework_task_proposal": ("tasks", "reasoning"),
    "id_distortions_completion_check": ("needs_more", "reasoning"),
    "initial_cognitive_model_bootstrap": ("items", "reasoning"),
    "next_session_plan_generation": ("next_session_plan",),
    "opening_info_update": ("mood", "problem"),
    "opening_readiness_check": ("ready", "issue", "reasoning"),
    "technique_completion_check": ("completed", "reasoning"),
    "technique_fit_check": ("fits", "alternative_technique", "reasoning"),
    "technique_plan": ("plan_description", "expected_outcome", "current_progress"),
    "technique_selection": ("selected_techniques",),
    "tib_check": ("has_tib", "tib_name", "tib_description", "reasoning"),
}

MIN_ALLOWED_TRACES_PER_REASONING_SAMPLE = 3


def remove_previous_generated_files(set_dir: Path, stem: str) -> None:
    if not set_dir.exists():
        return
    if not set_dir.is_dir():
        raise SessionTextExportError(f"Output set path exists and is not a directory: {set_dir}")

    generated_pattern = re.compile(rf"{re.escape(stem)}_\d{{3}}\.(?:json|txt)\Z")
    for path in set_dir.iterdir():
        if path.name == "profile.txt" or generated_pattern.fullmatch(path.name):
            if path.is_file():
                path.unlink()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parsed_trace_output(raw_output: Any) -> Any:
    if not isinstance(raw_output, str):
        return raw_output
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output


def compact_trace_output(step: str, raw_output: Any) -> dict[str, Any]:
    output = parsed_trace_output(raw_output)

    if step == "resistance_prediction":
        if not isinstance(output, dict):
            return {"message": str(raw_output)}
        return {
            "previous_resistance": output.get("previous_p"),
            "predicted_resistance": output.get("new_p"),
            "reasoning": output.get("reasoning"),
        }

    if step == "technique_evaluation":
        if isinstance(output, str):
            return {"evaluation": output}
        if isinstance(output, dict):
            fields = RCBT_TRACE_OUTPUT_FIELDS.get(step)
            if fields:
                return {field: output[field] for field in fields if field in output}
        return {"evaluation": output}

    if step == "technique_auto_complete":
        return {"message": output if isinstance(output, str) else raw_output}

    if not isinstance(output, dict):
        return {"message": str(raw_output)}

    fields = RCBT_TRACE_OUTPUT_FIELDS.get(step)
    if fields is None:
        compact = dict(output)
    else:
        compact = {field: output[field] for field in fields if field in output}

    if "reasoning" in output and "reasoning" not in compact:
        compact["reasoning"] = output["reasoning"]

    return compact


def message_metadata(message: dict[str, Any] | None) -> dict[str, Any] | None:
    if message is None:
        return None

    utterance_index = message.get("utterance_index")
    role = message.get("role", message.get("speaker"))
    content = message.get("content", message.get("message"))
    return {
        "utterance_index": utterance_index,
        "role": role,
        "content": content,
    }


def previous_patient_message(
    messages: list[dict[str, Any]], utterance_index: int
) -> dict[str, Any] | None:
    previous_messages = (
        message
        for message in messages
        if isinstance(message.get("utterance_index"), int)
        and message["utterance_index"] < utterance_index
    )
    sorted_previous_messages = sorted(
        previous_messages,
        key=lambda item: item["utterance_index"],
        reverse=True,
    )
    for message in sorted_previous_messages:
        role = str(message.get("role", message.get("speaker", ""))).casefold()
        if role in {"patient", "client"}:
            return message
    return None


def build_rcbt_reasonings(
    rcbt_payload: dict[str, Any], n_traces: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if n_traces <= 0:
        return [], {}

    messages = rcbt_payload.get("messages")
    reasoning_traces = rcbt_payload.get("reasoning_traces")
    if not isinstance(messages, list) or not isinstance(reasoning_traces, list):
        return [], {stage: 0 for stage in RCBT_REASONING_STAGES}

    messages_by_index = {
        message.get("utterance_index"): message
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("utterance_index"), int)
    }
    candidates_by_stage: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in RCBT_REASONING_STAGES
    }

    for reasoning_trace in reasoning_traces:
        if not isinstance(reasoning_trace, dict):
            continue

        utterance_index = reasoning_trace.get("utterance_index")
        if not isinstance(utterance_index, int):
            continue

        message = messages_by_index.get(utterance_index)
        if not isinstance(message, dict):
            continue

        stage = message.get("stage")
        if not isinstance(stage, str) or stage not in RCBT_ALLOWED_TRACES:
            continue

        traces = reasoning_trace.get("traces")
        if not isinstance(traces, list):
            continue

        allowed_steps = RCBT_ALLOWED_TRACES[stage]
        filtered_traces = []
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            step = trace.get("step")
            if not isinstance(step, str) or step not in allowed_steps:
                continue
            filtered_traces.append(
                {
                    "step": step,
                    "output": compact_trace_output(step, trace.get("output")),
                }
            )

        if len(filtered_traces) < MIN_ALLOWED_TRACES_PER_REASONING_SAMPLE:
            continue

        candidates_by_stage[stage].append(
            {
                "stage": stage,
                "utterance_index": utterance_index,
                "generated_utterance": message_metadata(message),
                "prior_patient_utterance": message_metadata(
                    previous_patient_message(messages, utterance_index)
                ),
                "traces": filtered_traces,
            }
        )

    reasonings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for stage in RCBT_REASONING_STAGES:
        candidates = candidates_by_stage[stage]
        sample_size = min(n_traces, len(candidates))
        selected = random.sample(candidates, sample_size) if sample_size else []
        reasonings.extend(selected)
        counts[stage] = len(selected)

    return reasonings, counts


def write_session_set(
    output_dir: Path,
    session_set: SessionSet,
    *,
    rcbt_payload: dict[str, Any] | None = None,
    n_traces: int = 0,
) -> dict[str, int] | None:
    set_dir = output_dir / session_set.set_dir_name
    remove_previous_generated_files(set_dir, session_set.stem)
    set_dir.mkdir(parents=True, exist_ok=True)

    variants = list(session_set.variants)
    random.shuffle(variants)
    random_ids = random.sample(range(101), len(variants))
    reasonings: list[dict[str, Any]] = []
    reasoning_counts: dict[str, int] | None = None
    if n_traces > 0 and rcbt_payload is not None:
        reasonings, reasoning_counts = build_rcbt_reasonings(rcbt_payload, n_traces)

    for variant, random_id in zip(variants, random_ids):
        path = set_dir / f"{session_set.stem}_{random_id:03d}.json"
        payload: dict[str, Any] = {"conversation": variant.messages}
        if variant.method == "rcbt" and n_traces > 0:
            payload["reasonings"] = reasonings
        write_json(path, payload)

    write_text(set_dir / "profile.txt", format_profile(session_set.patient_profile))
    return reasoning_counts


def export_sessions(input_dir: Path, output_dir: Path, *, n_traces: int = 0) -> int:
    if not input_dir.is_dir():
        raise SessionTextExportError(f"Input path is not a directory: {input_dir}")
    if n_traces < 0:
        raise SessionTextExportError("--n-traces must be 0 or greater.")

    session_sets = build_session_sets(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for session_set in session_sets:
        rcbt_payload = None
        if n_traces > 0:
            source = input_dir / "rcbt" / session_set.file_name
            payload = load_json(source)
            if not isinstance(payload, dict):
                raise SessionTextExportError(f"Session file must contain an object: {source}")
            rcbt_payload = payload

        reasoning_counts = write_session_set(
            output_dir,
            session_set,
            rcbt_payload=rcbt_payload,
            n_traces=n_traces,
        )
        if reasoning_counts is not None:
            count_text = ", ".join(
                f"{stage}={reasoning_counts.get(stage, 0)}"
                for stage in RCBT_REASONING_STAGES
            )
            print(f"{session_set.file_name}: reasonings {count_text}")
    return len(session_sets)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export matched cactus/magnet/rcbt JSON sessions as anonymized JSON "
            "conversation variants plus profile.txt."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing cactus, magnet, and rcbt JSON subdirectories.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where set_<n> folders will be written.",
    )
    parser.add_argument(
        "--n-traces",
        default=0,
        type=int,
        help=(
            "For RCBT sessions, sample up to this many eligible reasoning bundles "
            "from each stage and include them in the exported RCBT JSON. Default: 0."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = export_sessions(args.input, args.output, n_traces=args.n_traces)
    except SessionTextExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Exported {count} session set(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
