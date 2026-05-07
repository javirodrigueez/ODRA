#!/usr/bin/env python3
"""Aggregate renamed evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = ("rcbt", "magnet", "cactus")
SESSION_RE = re.compile(r"^session_\d+_(rcbt|magnet|cactus)\.json$")
AGREEMENT_GROUPS = {
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
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--individual",
        action="store_true",
        help="also compute Krippendorff agreement for each ranking item",
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


def method_from_session(session_path: str) -> str:
    match = SESSION_RE.match(Path(session_path).name)
    if not match:
        fail(f"invalid session filename: {session_path}")
    return match.group(1)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2)


def percent(part: int, total: int) -> float:
    return round(part * 100 / total, 2)


def reasoning_score_to_3_point(score: float) -> float:
    if score in (1, 2):
        return 1.0
    if score == 3:
        return 2.0
    if score in (4, 5):
        return 3.0
    fail(f"reasoning score must be in 1-5 range, got {score}")


def krippendorff_alpha(
    krippendorff: Any,
    values_by_annotator: dict[str, dict[str, float]],
    level: str,
) -> float | None:
    units = sorted(
        unit
        for unit in {unit for values in values_by_annotator.values() for unit in values}
        if sum(unit in values for values in values_by_annotator.values()) >= 2
    )
    if not units:
        return None

    matrix = [
        [values_by_annotator[annotator].get(unit, math.nan) for unit in units]
        for annotator in sorted(values_by_annotator)
    ]
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
    return None if alpha is None or math.isnan(float(alpha)) else round(float(alpha), 2)


def percent_agreement(values_by_annotator: dict[str, dict[str, float]]) -> float | None:
    overlapping = [
        [
            values[unit]
            for values in values_by_annotator.values()
            if unit in values
        ]
        for unit in sorted({unit for values in values_by_annotator.values() for unit in values})
        if sum(unit in values for values in values_by_annotator.values()) >= 2
    ]
    return (
        percent(sum(1 for values in overlapping if len(set(values)) == 1), len(overlapping))
        if overlapping
        else None
    )


def reasoning_agreement_result(
    krippendorff: Any,
    values_by_annotator: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    alpha = krippendorff_alpha(krippendorff, values_by_annotator, "interval")
    result = {"score_3point_interval": alpha}
    if alpha is not None and alpha < 0:
        result["score_3point_percent_agreement"] = percent_agreement(values_by_annotator)
    return result


def rank_group_scores_by_set(scores: dict[str, float]) -> dict[str, float]:
    ranked: dict[str, float] = {}
    by_set: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for session, score in scores.items():
        by_set[Path(session).parent.as_posix()].append((session, score))

    for sessions in by_set.values():
        ordered = sorted(sessions, key=lambda item: (item[1], item[0]))
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank = sum(range(index + 1, end + 1)) / (end - index)
            for session, _score in ordered[index:end]:
                ranked[session] = float(rank)
            index = end
    return ranked


def aggregate(input_dir: Path, individual: bool = False) -> dict[str, Any]:
    try:
        import krippendorff  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        fail("missing required dependency 'krippendorff'. Install it with: pip install krippendorff")

    files = sorted(input_dir.glob("*.json"))
    if not input_dir.is_dir() or not files:
        fail(f"{input_dir}: expected a directory with JSON files")

    rank_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    reasoning_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    reasoning_agreement: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    item_ranks: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    group_sums: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    safety_total = dict.fromkeys(METHODS, 0)
    safety_concerns = dict.fromkeys(METHODS, 0)
    safety_by_session: dict[str, dict[str, bool]] = defaultdict(dict)

    for path in files:
        annotator = path.stem
        for set_name, set_data in load_json(path)["sets"].items():
            variants = set_data.get("variants")
            rankings = set_data.get("rankings")
            safety = set_data.get("safety")
            reasoning = set_data.get("reasoningEvaluation", {})
            if not all(isinstance(obj, dict) for obj in (variants, rankings, safety, reasoning)):
                fail(f"{path}:{set_name}: missing variants/rankings/safety/reasoningEvaluation")

            letters = set(variants)
            letter_to_method = {letter: method_from_session(session) for letter, session in variants.items()}
            if set(letter_to_method.values()) != set(METHODS):
                fail(f"{path}:{set_name}: expected one variant for each method")

            ranks_by_item: dict[str, dict[str, float]] = {}
            for item, ranking in rankings.items():
                if not isinstance(ranking, list) or set(ranking) != letters:
                    fail(f"{path}:{set_name}: invalid ranking for {item}")
                ranks_by_item[item] = {}
                for index, letter in enumerate(ranking, start=1):
                    method = letter_to_method[letter]
                    session = variants[letter]
                    rank_values[item][method].append(float(index))
                    item_ranks[item][annotator][session] = float(index)
                    ranks_by_item[item][letter] = float(index)

            for group, items in AGREEMENT_GROUPS.items():
                missing = [item for item in items if item not in ranks_by_item]
                if missing:
                    fail(f"{path}:{set_name}: missing items for {group}: {', '.join(missing)}")
                for letter, session in variants.items():
                    group_sums[group][annotator][session] = sum(
                        ranks_by_item[item][letter] for item in items
                    )

            concerns = safety.get("concerns")
            if not isinstance(concerns, list) or not set(concerns).issubset(letters):
                fail(f"{path}:{set_name}: safety.concerns must be variant letters")
            for letter, session in variants.items():
                method = letter_to_method[letter]
                has_concern = letter in concerns
                safety_total[method] += 1
                safety_concerns[method] += int(has_concern)
                safety_by_session[session][annotator] = has_concern

            for session, messages in reasoning.items():
                if not isinstance(messages, dict):
                    fail(f"{path}:{set_name}: invalid reasoningEvaluation entry")
                method = method_from_session(session)
                for message_id, scores in messages.items():
                    if not isinstance(scores, dict):
                        fail(f"{path}:{set_name}: invalid reasoning scores")
                    for item, score in scores.items():
                        if not isinstance(score, (int, float)) or isinstance(score, bool):
                            fail(f"{path}:{set_name}: non-numeric reasoning score")
                        reasoning_values[item][method].append(float(score))
                        reasoning_agreement[item][annotator][f"{session}#{message_id}"] = (
                            reasoning_score_to_3_point(float(score))
                        )

    ranks = {
        item: {method: mean(values) for method, values in by_method.items()}
        for item, by_method in sorted(rank_values.items())
    }
    safety = {
        method: percent(safety_concerns[method], safety_total[method])
        for method in METHODS
        if safety_total[method]
    }
    reasoning = {
        item: {method: mean(values) for method, values in by_method.items()}
        for item, by_method in sorted(reasoning_values.items())
    }

    agreement: dict[str, Any] = {}
    for group, sums in group_sums.items():
        ranked_sums = {
            annotator: rank_group_scores_by_set(scores)
            for annotator, scores in sums.items()
        }
        agreement[group] = {
            "ranked_ordinal": krippendorff_alpha(krippendorff, ranked_sums, "ordinal"),
            "sum_interval": krippendorff_alpha(krippendorff, sums, "interval"),
        }

    if individual:
        agreement["individual"] = {
            item: {
                "ranked_ordinal": krippendorff_alpha(
                    krippendorff,
                    {
                        annotator: rank_group_scores_by_set(scores)
                        for annotator, scores in ranks.items()
                    },
                    "ordinal",
                ),
                "rank_interval": krippendorff_alpha(krippendorff, ranks, "interval"),
            }
            for item, ranks in sorted(item_ranks.items())
        }

    agreement["reasoning"] = {
        item: reasoning_agreement_result(krippendorff, values)
        for item, values in sorted(reasoning_agreement.items())
    }

    overlapping_safety = [scores for scores in safety_by_session.values() if len(scores) >= 2]
    agreement["safety"] = (
        percent(
            sum(1 for scores in overlapping_safety if len(set(scores.values())) == 1),
            len(overlapping_safety),
        )
        if overlapping_safety
        else None
    )

    return {
        "ranks": ranks,
        "safety": safety,
        "agreement": agreement,
        "reasoning": reasoning,
    }


def main() -> int:
    args = parse_args()
    try:
        results = aggregate(args.input, individual=args.individual)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
