import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


NO_RESISTANCE_LABEL = "no resistance"
EVAL_ROWS = 150
EXPECTED_EVAL_SEEDS = 50


def load_json_list(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: Could not decode JSON from {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: Expected {path} to contain a JSON list.", file=sys.stderr)
        sys.exit(1)

    return data


def normalize_text(value):
    return str(value or "").strip()


def get_personal_info(profile):
    return profile.get("personal_info") or profile.get("personal_information") or {}


def get_eval_seed_key(profile):
    personal_info = get_personal_info(profile)
    name = normalize_text(personal_info.get("name"))
    age = normalize_text(personal_info.get("age"))
    return name, age


def get_unique_eval_seeds(eval_data):
    unique_seeds = []
    seen = set()

    for profile in eval_data[:EVAL_ROWS]:
        seed_key = get_eval_seed_key(profile)
        if not all(seed_key):
            print(f"Error: Eval profile is missing name or age: {profile.get('idx')}", file=sys.stderr)
            sys.exit(1)
        if seed_key in seen:
            continue
        seen.add(seed_key)
        unique_seeds.append(profile)

    if len(unique_seeds) != EXPECTED_EVAL_SEEDS:
        print(
            f"Error: Expected {EXPECTED_EVAL_SEEDS} unique eval seeds from the first "
            f"{EVAL_ROWS} rows, found {len(unique_seeds)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    return unique_seeds


def stringify_profile_field(value):
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = stringify_profile_field(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "\n".join(parts)

    if isinstance(value, list):
        parts = []
        for item in value:
            item_text = stringify_profile_field(item)
            if item_text:
                parts.append(item_text)
        return "\n".join(parts)

    return normalize_text(value)


def profile_to_text(profile):
    fields = [
        ("personal_info", get_personal_info(profile)),
        ("presenting_problem", profile.get("presenting_problem")),
        ("reason_seeking_counseling", profile.get("reason_seeking_counseling")),
        ("more_information", profile.get("more_information")),
    ]

    text_parts = []
    for label, value in fields:
        text = stringify_profile_field(value)
        if text:
            text_parts.append(f"{label}:\n{text}")
    return "\n\n".join(text_parts)


def build_profile_lookup(cactus_uniques, cactus_all):
    unique_profiles = {profile["idx"]: profile for profile in cactus_uniques if "idx" in profile}
    all_profiles = {profile["idx"]: profile for profile in cactus_all if "idx" in profile}

    def lookup(cactus_idx):
        return unique_profiles.get(cactus_idx) or all_profiles.get(cactus_idx)

    return lookup


def group_mirror_by_cactus_idx(mirror_data):
    grouped = defaultdict(list)
    for item in mirror_data:
        if "cactus_idx" not in item:
            continue
        grouped[item["cactus_idx"]].append(item)
    return grouped


def has_resistance(item):
    resistance_type = normalize_text(item.get("resistance_type")).lower()
    return bool(resistance_type) and resistance_type != NO_RESISTANCE_LABEL


def get_eligible_mirror_candidates(mirror_data, profile_lookup):
    grouped = group_mirror_by_cactus_idx(mirror_data)
    candidates = []
    skipped_missing_profile = 0

    for cactus_idx, items in grouped.items():
        no_resistance_items = [
            item for item in items if normalize_text(item.get("resistance_type")).lower() == NO_RESISTANCE_LABEL
        ]
        resistance_items = [item for item in items if has_resistance(item)]

        if not no_resistance_items or not resistance_items:
            continue

        profile = profile_lookup(cactus_idx)
        if profile is None:
            skipped_missing_profile += 1
            continue

        candidates.append(
            {
                "cactus_idx": cactus_idx,
                "profile": profile,
                "no_resistance_items": sorted(no_resistance_items, key=lambda item: str(item.get("idx", ""))),
                "resistance_items": resistance_items,
            }
        )

    if skipped_missing_profile:
        print(f"Warning: skipped {skipped_missing_profile} eligible mirror seeds with missing profiles.")

    if not candidates:
        print("Error: No eligible mirror seeds found.", file=sys.stderr)
        sys.exit(1)

    return candidates


def load_sentence_transformer(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(
            "Error: sentence-transformers is required. Install project requirements "
            "or run `pip install sentence-transformers`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return SentenceTransformer(model_name)


def compute_best_matches(eval_seeds, mirror_candidates, model_name):
    import torch

    model = load_sentence_transformer(model_name)
    eval_texts = [profile_to_text(profile) for profile in eval_seeds]
    mirror_texts = [profile_to_text(candidate["profile"]) for candidate in mirror_candidates]

    eval_embeddings = model.encode(eval_texts, convert_to_tensor=True, normalize_embeddings=True)
    mirror_embeddings = model.encode(mirror_texts, convert_to_tensor=True, normalize_embeddings=True)

    similarities = torch.matmul(eval_embeddings, mirror_embeddings.T)
    best_scores, best_indices = torch.max(similarities, dim=1)

    matches = []
    for eval_profile, score, mirror_index in zip(eval_seeds, best_scores, best_indices):
        matches.append(
            {
                "eval_profile": eval_profile,
                "mirror_candidate": mirror_candidates[int(mirror_index.item())],
                "similarity": float(score.item()),
            }
        )

    return matches


def make_output_item(item, match, selected_role):
    eval_profile = match["eval_profile"]
    eval_name, eval_age = get_eval_seed_key(eval_profile)
    candidate = match["mirror_candidate"]

    output_item = dict(item)
    output_item["match_metadata"] = {
        "eval_idx": eval_profile.get("idx"),
        "eval_name": eval_name,
        "eval_age": eval_age,
        "matched_cactus_idx": candidate["cactus_idx"],
        "similarity": match["similarity"],
        "selected_role": selected_role,
        "resistance_source_label": item.get("resistance_type"),
    }
    return output_item


def build_output_items(matches, seed):
    rng = random.Random(seed)
    output_items = []

    for match in matches:
        candidate = match["mirror_candidate"]
        no_resistance_item = candidate["no_resistance_items"][0]
        resistance_item = rng.choice(candidate["resistance_items"])

        output_items.append(make_output_item(no_resistance_item, match, "no_resistance"))
        output_items.append(make_output_item(resistance_item, match, "resistance"))

    return output_items


def normalize_message_role(role):
    normalized = normalize_text(role).lower()
    if normalized.startswith("therapist") or normalized.startswith("counselor"):
        return "therapist"
    if normalized.startswith("client") or normalized.startswith("patient"):
        return "patient"
    return normalized or "unknown"


def mirror_dialogue_to_messages(item):
    dialogue = item.get("proc_dialogue")
    if not isinstance(dialogue, list):
        print(f"Error: Mirror item is missing proc_dialogue list: {item.get('idx')}", file=sys.stderr)
        sys.exit(1)

    messages = []
    for index, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            print(f"Error: Invalid dialogue turn {index} in mirror item {item.get('idx')}", file=sys.stderr)
            sys.exit(1)
        content = normalize_text(turn.get("content"))
        if not content:
            continue
        messages.append(
            {
                "role": normalize_message_role(turn.get("role")),
                "content": content,
            }
        )

    if not messages:
        print(f"Error: Mirror item has no usable dialogue turns: {item.get('idx')}", file=sys.stderr)
        sys.exit(1)

    return messages


def build_evaluation_patient_profile(match, item, selected_role):
    profile = copy.deepcopy(match["mirror_candidate"]["profile"])
    profile["attitude"] = "positive" if selected_role == "no_resistance" else "negative"
    profile["mirror_resistance_type"] = item.get("resistance_type")
    profile["mirror_item_idx"] = item.get("idx")
    profile["matched_eval_idx"] = match["eval_profile"].get("idx")
    profile["matched_similarity"] = match["similarity"]
    return profile


def build_evaluation_payload(match, item, selected_role):
    return {
        "messages": mirror_dialogue_to_messages(item),
        "patient_profile": build_evaluation_patient_profile(match, item, selected_role),
        "mirror_metadata": make_output_item(item, match, selected_role)["match_metadata"],
    }


def evaluation_attitude(selected_role):
    return "positive" if selected_role == "no_resistance" else "negative"


def write_evaluation_files(matches, seed, output_dir):
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    stale_files = list(output_dir.glob("simulation_*.json"))
    for path in stale_files:
        path.unlink()

    written_files = []
    for index, match in enumerate(matches):
        candidate = match["mirror_candidate"]
        no_resistance_item = candidate["no_resistance_items"][0]
        resistance_item = rng.choice(candidate["resistance_items"])

        selections = [
            ("no_resistance", no_resistance_item),
            ("resistance", resistance_item),
        ]
        for selected_role, item in selections:
            attitude = evaluation_attitude(selected_role)
            filename = f"simulation_{index:03d}_{attitude}.json"
            path = output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(build_evaluation_payload(match, item, selected_role), f, indent=2, ensure_ascii=False)
                f.write("\n")
            written_files.append(filename)

    return {
        "output_dir": str(output_dir),
        "written_files": len(written_files),
        "removed_files": len(stale_files),
        "files": written_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Match eval cactus seeds to similar mirror seeds and extract paired mirror samples."
    )
    parser.add_argument("--input", required=True, help="Path to data/baselines/mirror_data.json.")
    parser.add_argument("--output", required=True, help="Path to write the selected mirror JSON list.")
    parser.add_argument("--eval", default="data/eval_full_cactus/eval_full_cactus.json")
    parser.add_argument("--cactus-uniques", default="data/processed_cactus_uniques.json")
    parser.add_argument("--cactus-all", default="data/processed_cactus.json")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        help="Optional directory for run_batch_metrics-compatible per-session JSON files.",
    )
    args = parser.parse_args()

    eval_data = load_json_list(args.eval)
    mirror_data = load_json_list(args.input)
    cactus_uniques = load_json_list(args.cactus_uniques)
    cactus_all = load_json_list(args.cactus_all)

    eval_seeds = get_unique_eval_seeds(eval_data)
    profile_lookup = build_profile_lookup(cactus_uniques, cactus_all)
    mirror_candidates = get_eligible_mirror_candidates(mirror_data, profile_lookup)
    matches = compute_best_matches(eval_seeds, mirror_candidates, args.model)
    output_items = build_output_items(matches, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(eval_seeds)} eval seeds.")
    print(f"Matched against {len(mirror_candidates)} eligible mirror seeds.")
    print(f"Wrote {len(output_items)} mirror samples to {output_path}.")

    if args.eval_output_dir:
        summary = write_evaluation_files(matches, args.seed, args.eval_output_dir)
        print(
            f"Wrote {summary['written_files']} run_batch_metrics-compatible files to "
            f"{summary['output_dir']}."
        )


if __name__ == "__main__":
    main()
