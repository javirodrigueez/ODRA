#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
import re
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ANNOTATORS = ("miriam", "amaria", "mdolores")
ATTITUDES = ("positive", "neutral", "negative")
ANNOTATOR_PAIRS = tuple(itertools.combinations(range(len(ANNOTATORS)), 2))


class SeparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SetInfo:
    name: str
    path: Path
    attitude: str
    patient_name: str
    patient_age: str
    profile_id: tuple[str, str]


@dataclass(frozen=True)
class GroupCandidate:
    pairs: tuple[tuple[int, int], ...]
    additions: tuple[int, ...]
    repeat_penalty: int
    same_attitude_repeat_penalty: int

    @property
    def score(self) -> tuple[int, int]:
        return (self.same_attitude_repeat_penalty, self.repeat_penalty)


def natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def parse_profile(profile_path: Path) -> tuple[str, str, str]:
    try:
        lines = profile_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SeparationError(f"Missing profile file: {profile_path}") from exc

    if len(lines) < 9:
        raise SeparationError(f"Profile file has fewer than 9 lines: {profile_path}")

    attitude = parse_prefixed_line(lines[3], "ATTITUDE", profile_path, 4).casefold()
    patient_name = parse_prefixed_line(lines[7], "Name", profile_path, 8)
    patient_age = parse_prefixed_line(lines[8], "Age", profile_path, 9)

    if attitude not in ATTITUDES:
        expected = ", ".join(ATTITUDES)
        raise SeparationError(
            f"Invalid attitude {attitude!r} in {profile_path}; expected one of: {expected}"
        )

    return attitude, patient_name, patient_age


def parse_prefixed_line(line: str, prefix: str, path: Path, line_number: int) -> str:
    expected_prefix = f"{prefix}:"
    if not line.startswith(expected_prefix):
        raise SeparationError(
            f"Expected line {line_number} of {path} to start with {expected_prefix!r}; "
            f"got {line!r}"
        )
    value = line.split(":", 1)[1].strip()
    if not value:
        raise SeparationError(f"Line {line_number} of {path} has empty {prefix!r} value.")
    return value


def load_sets(input_dir: Path) -> list[SetInfo]:
    if not input_dir.is_dir():
        raise SeparationError(f"Input path is not a directory: {input_dir}")

    set_dirs = sorted(
        (path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("set_")),
        key=natural_sort_key,
    )
    if not set_dirs:
        raise SeparationError(f"No set_* directories found in {input_dir}")

    sets: list[SetInfo] = []
    for set_dir in set_dirs:
        attitude, patient_name, patient_age = parse_profile(set_dir / "profile.txt")
        sets.append(
            SetInfo(
                name=set_dir.name,
                path=set_dir,
                attitude=attitude,
                patient_name=patient_name,
                patient_age=patient_age,
                profile_id=(patient_name.casefold(), patient_age.casefold()),
            )
        )
    return sets


def grouped_sets(sets: list[SetInfo]) -> list[list[SetInfo]]:
    groups_by_profile: dict[tuple[str, str], list[SetInfo]] = {}
    for set_info in sets:
        groups_by_profile.setdefault(set_info.profile_id, []).append(set_info)

    groups = list(groups_by_profile.values())
    for group in groups:
        group.sort(key=lambda set_info: (ATTITUDES.index(set_info.attitude), natural_sort_key(set_info.path)))
    groups.sort(key=lambda group: (len(group), natural_sort_key(group[0].path)), reverse=True)
    return groups


def attitude_index(attitude: str) -> int:
    return ATTITUDES.index(attitude)


def count_index(annotator_index: int, attitude: str) -> int:
    return annotator_index * len(ATTITUDES) + attitude_index(attitude)


def balanced_total_targets(total_assignments: int) -> list[tuple[int, ...]]:
    base, remainder = divmod(total_assignments, len(ANNOTATORS))
    targets: list[tuple[int, ...]] = []
    for high_annotators in itertools.combinations(range(len(ANNOTATORS)), remainder):
        target = [base] * len(ANNOTATORS)
        for annotator_index in high_annotators:
            target[annotator_index] += 1
        targets.append(tuple(target))
    return targets


def balanced_attitude_distributions(total_for_attitude: int) -> list[tuple[int, ...]]:
    base, remainder = divmod(total_for_attitude, len(ANNOTATORS))
    distributions: list[tuple[int, ...]] = []
    for high_annotators in itertools.combinations(range(len(ANNOTATORS)), remainder):
        distribution = [base] * len(ANNOTATORS)
        for annotator_index in high_annotators:
            distribution[annotator_index] += 1
        distributions.append(tuple(distribution))
    return distributions


def target_matrices(sets: list[SetInfo], rng: random.Random) -> list[tuple[int, ...]]:
    attitude_counts = {
        attitude: sum(1 for set_info in sets if set_info.attitude == attitude)
        for attitude in ATTITUDES
    }
    attitude_distributions = [
        balanced_attitude_distributions(attitude_counts[attitude] * 2)
        for attitude in ATTITUDES
    ]
    total_targets = set(balanced_total_targets(len(sets) * 2))

    matrices: list[tuple[int, ...]] = []
    for distributions in itertools.product(*attitude_distributions):
        totals = tuple(
            sum(distribution[annotator_index] for distribution in distributions)
            for annotator_index in range(len(ANNOTATORS))
        )
        if totals not in total_targets:
            continue

        matrix = [0] * (len(ANNOTATORS) * len(ATTITUDES))
        for attitude, distribution in zip(ATTITUDES, distributions):
            for annotator_index, count in enumerate(distribution):
                matrix[count_index(annotator_index, attitude)] = count
        matrices.append(tuple(matrix))

    if not matrices:
        raise SeparationError("Could not build balanced attitude targets for this set count.")

    rng.shuffle(matrices)
    return matrices


def candidate_for_pairs(group: list[SetInfo], pairs: tuple[tuple[int, int], ...]) -> GroupCandidate:
    additions = [0] * (len(ANNOTATORS) * len(ATTITUDES))
    profile_counts = [0] * len(ANNOTATORS)
    profile_attitude_counts: dict[tuple[int, str], int] = {}

    for set_info, pair in zip(group, pairs):
        for annotator_index in pair:
            additions[count_index(annotator_index, set_info.attitude)] += 1
            profile_counts[annotator_index] += 1
            key = (annotator_index, set_info.attitude)
            profile_attitude_counts[key] = profile_attitude_counts.get(key, 0) + 1

    repeat_count = sum(max(0, count - 1) ** 2 for count in profile_counts)
    assignments_for_profile = len(group) * 2
    base, remainder = divmod(assignments_for_profile, len(ANNOTATORS))
    ideal_profile_counts = [base + (1 if index < remainder else 0) for index in range(len(ANNOTATORS))]
    min_repeat_count = sum(max(0, count - 1) ** 2 for count in ideal_profile_counts)
    same_attitude_repeat_penalty = sum(
        max(0, count - 1) for count in profile_attitude_counts.values()
    )

    return GroupCandidate(
        pairs=pairs,
        additions=tuple(additions),
        repeat_penalty=repeat_count - min_repeat_count,
        same_attitude_repeat_penalty=same_attitude_repeat_penalty,
    )


def group_candidates(group: list[SetInfo], rng: random.Random) -> list[GroupCandidate]:
    candidates = [
        candidate_for_pairs(group, pairs)
        for pairs in itertools.product(ANNOTATOR_PAIRS, repeat=len(group))
    ]
    rng.shuffle(candidates)
    candidates.sort(key=lambda candidate: candidate.score)
    return candidates


def solve_assignments(
    groups: list[list[SetInfo]],
    targets: tuple[int, ...],
    rng: random.Random,
) -> dict[str, tuple[str, str]] | None:
    candidates_by_group = [group_candidates(group, rng) for group in groups]

    @lru_cache(maxsize=None)
    def search(group_index: int, remaining: tuple[int, ...]) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...] | None:
        if group_index == len(groups):
            if all(count == 0 for count in remaining):
                return ()
            return None

        for candidate_index, candidate in enumerate(candidates_by_group[group_index]):
            if any(addition > count for addition, count in zip(candidate.additions, remaining)):
                continue

            next_remaining = tuple(
                count - addition for count, addition in zip(remaining, candidate.additions)
            )
            result = search(group_index + 1, next_remaining)
            if result is not None:
                return ((candidate_index, candidate.pairs),) + result

        return None

    result = search(0, targets)
    if result is None:
        return None

    assignments: dict[str, tuple[str, str]] = {}
    for group, (_, pairs) in zip(groups, result):
        for set_info, pair in zip(group, pairs):
            assignments[set_info.name] = tuple(ANNOTATORS[index] for index in pair)

    return assignments


def generate_assignments(sets: list[SetInfo], seed: int) -> dict[str, tuple[str, str]]:
    rng = random.Random(seed)
    groups = grouped_sets(sets)

    for targets in target_matrices(sets, rng):
        assignments = solve_assignments(groups, targets, rng)
        if assignments is not None:
            return assignments

    raise SeparationError("Could not find a valid separation assignment.")


def table_line(values: Iterable[object], widths: list[int]) -> str:
    return " | ".join(str(value).ljust(width) for value, width in zip(values, widths))


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)
    lines = [table_line(headers, widths), separator]
    lines.extend(table_line(row, widths) for row in rows)
    return "\n".join(lines)


def assignment_report(sets: list[SetInfo], assignments: dict[str, tuple[str, str]]) -> str:
    rows: list[list[object]] = []
    for set_info in sets:
        assigned = set(assignments[set_info.name])
        rows.append(
            [
                set_info.name,
                set_info.attitude,
                f"{set_info.patient_name} ({set_info.patient_age})",
                "X" if "miriam" in assigned else "-",
                "X" if "amaria" in assigned else "-",
                "X" if "mdolores" in assigned else "-",
            ]
        )

    count_rows: list[list[object]] = []
    for annotator in ANNOTATORS:
        assigned_sets = [
            set_info for set_info in sets if annotator in assignments[set_info.name]
        ]
        profile_count: dict[tuple[str, str], int] = {}
        for set_info in assigned_sets:
            profile_count[set_info.profile_id] = profile_count.get(set_info.profile_id, 0) + 1
        repeated_profiles = sum(1 for count in profile_count.values() if count > 1)
        count_rows.append(
            [
                annotator,
                len(assigned_sets),
                sum(1 for set_info in assigned_sets if set_info.attitude == "positive"),
                sum(1 for set_info in assigned_sets if set_info.attitude == "neutral"),
                sum(1 for set_info in assigned_sets if set_info.attitude == "negative"),
                repeated_profiles,
            ]
        )

    return "\n\n".join(
        [
            "ASSIGNMENTS",
            format_table(
                ["set", "attitude", "profile", "miriam", "amaria", "mdolores"],
                rows,
            ),
            "SUMMARY",
            format_table(
                ["annotator", "total", "positive", "neutral", "negative", "repeated_profiles"],
                count_rows,
            ),
        ]
    )


def recreate_output(output_dir: Path, sets: list[SetInfo], assignments: dict[str, tuple[str, str]]) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SeparationError(f"Output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)

    for annotator in ANNOTATORS:
        (output_dir / annotator).mkdir(parents=True, exist_ok=True)

    sets_by_name = {set_info.name: set_info for set_info in sets}
    for set_name, annotators in assignments.items():
        set_info = sets_by_name[set_name]
        for annotator in annotators:
            shutil.copytree(set_info.path, output_dir / annotator / set_name)


def export_report(report: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "assignment_table.txt"
    report_path.write_text(report + "\n", encoding="utf-8", newline="\n")
    return report_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a two-annotator separation plan for expert evaluation sets. "
            "Dry-runs by default; use --apply to recreate and copy the output folders."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing set_<n> folders.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where miriam/amaria/mdolores folders will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to break assignment ties.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Recreate the output directory and copy assigned set folders.",
    )
    parser.add_argument(
        "--export-txt",
        action="store_true",
        help="Write the assignment table to assignment_table.txt in the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sets = load_sets(args.input)
        assignments = generate_assignments(sets, args.seed)
        report = assignment_report(sets, assignments)

        print(report)
        if args.apply:
            recreate_output(args.output, sets, assignments)
            print(f"\nCreated separation folders under {args.output}")
        else:
            print("\nDry run only; pass --apply to recreate and copy separation folders.")

        if args.export_txt:
            report_path = export_report(report, args.output)
            print(f"Exported assignment table to {report_path}")
    except SeparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
