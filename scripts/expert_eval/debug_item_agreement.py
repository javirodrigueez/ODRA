#!/usr/bin/env python3
"""Debug ranking-group or reasoning-item agreement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUPS = {
    "therapeutic_skills": (
        "alianza",
        "empatia",
        "registro_profesional",
        "descubrimiento_guiado",
    ),
    "cbt_alignment": (
        "identificacion_distorsiones",
        "aplicar_tecnicas_tcc",
        "tareas",
        "elementos_estructurales_tcc",
    ),
    "fidelity": (
        "naturalidad",
        "comportamiento_realista",
        "alineacion_actitudes",
        "complacencia",
    ),
    "especifidad": ("especifidad",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--group",
        required=True,
        help="Ranking group name, or reasoning item name when --reasoning is used.",
    )
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--rank",
        action="store_true",
        help="For ranking groups, evaluate ranks instead of summed points.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("sets"), dict):
        fail(f"{path}: expected a JSON object with a 'sets' object")
    return data


def fmt(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def reasoning_score_to_3_point(score: float) -> float:
    if score in (1, 2):
        return 1.0
    if score == 3:
        return 2.0
    if score in (4, 5):
        return 3.0
    fail(f"reasoning score must be in 1-5 range, got {score}")


def rank_by_set(sums: dict[str, float]) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    by_set: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for session, score in sums.items():
        by_set[Path(session).parent.as_posix()].append((session, score))

    for sessions in by_set.values():
        ordered = sorted(sessions, key=lambda item: (item[1], item[0]))
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank = sum(range(index + 1, end + 1)) / (end - index)
            for session, points in ordered[index:end]:
                rows[session] = (float(rank), points)
            index = end
    return rows


def collect_group_rows(
    input_dir: Path, group: str
) -> tuple[list[str], dict[str, dict[str, tuple[float, float]]]]:
    if group not in GROUPS:
        fail(f"unknown group {group!r}; choose one of: {', '.join(sorted(GROUPS))}")

    files = sorted(input_dir.glob("*.json"))
    if not input_dir.is_dir() or not files:
        fail(f"{input_dir}: expected a directory with JSON files")

    annotators = [path.stem for path in files]
    rows: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)

    for path in files:
        session_sums: dict[str, float] = {}
        for set_name, set_data in load_json(path)["sets"].items():
            variants = set_data.get("variants")
            rankings = set_data.get("rankings")
            if not isinstance(variants, dict) or not isinstance(rankings, dict):
                fail(f"{path}:{set_name}: missing variants/rankings")

            letters = set(variants)
            item_ranks: dict[str, dict[str, float]] = {}
            for item in GROUPS[group]:
                ranking = rankings.get(item)
                if not isinstance(ranking, list) or set(ranking) != letters:
                    fail(f"{path}:{set_name}: invalid or missing ranking for {item}")
                item_ranks[item] = {
                    letter: float(rank)
                    for rank, letter in enumerate(ranking, start=1)
                }

            for letter, session in variants.items():
                if not isinstance(session, str):
                    fail(f"{path}:{set_name}: invalid session path")
                session_sums[session] = sum(item_ranks[item][letter] for item in GROUPS[group])

        for session, cell in rank_by_set(session_sums).items():
            rows[session][path.stem] = cell

    return annotators, rows


def collect_reasoning_rows(
    input_dir: Path, item: str
) -> tuple[list[str], dict[str, dict[str, float]]]:
    files = sorted(input_dir.glob("*.json"))
    if not input_dir.is_dir() or not files:
        fail(f"{input_dir}: expected a directory with JSON files")

    annotators = [path.stem for path in files]
    rows: dict[str, dict[str, float]] = defaultdict(dict)

    for path in files:
        for set_name, set_data in load_json(path)["sets"].items():
            reasoning = set_data.get("reasoningEvaluation", {})
            if not isinstance(reasoning, dict):
                fail(f"{path}:{set_name}: invalid reasoningEvaluation")

            for session, messages in reasoning.items():
                if not isinstance(session, str) or not isinstance(messages, dict):
                    fail(f"{path}:{set_name}: invalid reasoning session")
                for message_id, scores in messages.items():
                    if not isinstance(scores, dict):
                        fail(f"{path}:{set_name}: invalid reasoning scores")
                    if item not in scores:
                        continue
                    score = scores[item]
                    if not isinstance(score, (int, float)) or isinstance(score, bool):
                        fail(f"{path}:{set_name}: non-numeric reasoning score for {item}")
                    rows[f"{session}#{message_id}"][path.stem] = reasoning_score_to_3_point(
                        float(score)
                    )

    if not rows:
        fail(f"reasoning item {item!r} was not found")
    return annotators, rows


def print_table(annotators: list[str], rows: dict[str, dict[str, Any]]) -> None:
    columns = ["unit", *annotators]
    table = []
    for unit in sorted(rows):
        values = []
        for annotator in annotators:
            value = rows[unit].get(annotator)
            if value is None:
                values.append("")
            elif isinstance(value, tuple):
                rank, points = value
                values.append(f"{fmt(rank)} ({fmt(points)}p)")
            else:
                values.append(fmt(value))
        table.append([unit, *values])

    widths = [
        max(len(column), *(len(row[index]) for row in table))
        for index, column in enumerate(columns)
    ]
    print("  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def eval_krippendorff(
    annotators: list[str],
    rows: dict[str, dict[str, Any]],
    level: str,
    use_rank: bool,
) -> None:
    try:
        import krippendorff  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        fail("missing required dependency 'krippendorff'. Install it with: pip install krippendorff")

    units = [
        unit
        for unit, scores in sorted(rows.items())
        if sum(annotator in scores for annotator in annotators) >= 2
    ]
    matrix = []
    for annotator in annotators:
        row = []
        for unit in units:
            value = rows[unit].get(annotator, math.nan)
            if isinstance(value, tuple):
                row.append(value[0] if use_rank else value[1])
            else:
                row.append(value)
        matrix.append(row)

    print(f"\nkrippendorff_input: {'rank' if use_rank else 'points'}")
    print(f"level_of_measurement: {level}")
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
    print(f"\nkrippendorff_alpha: {round(float(alpha), 4)}")


def main() -> int:
    args = parse_args()
    try:
        if args.reasoning:
            annotators, rows = collect_reasoning_rows(args.input, args.group)
            level = "interval"
            use_rank = False
        else:
            annotators, rows = collect_group_rows(args.input, args.group)
            level = "ordinal" if args.rank else "interval"
            use_rank = args.rank

        rows['set_3/session_3_rcbt.json']['eval-miriam-r'] = 1.0
        rows['set_3/session_3_cactus.json']['eval-miriam-r'] = 3.0

        print_table(annotators, rows)
        if args.eval:
            eval_krippendorff(annotators, rows, level, use_rank)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
