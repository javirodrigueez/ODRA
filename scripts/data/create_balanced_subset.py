import json
import random
import argparse
import sys
from collections import defaultdict

ATTITUDES = ("positive", "neutral", "negative")


def ensure_emotional_recovery(profile):
    normalized = dict(profile)
    normalized["emotional_recovery"] = str(normalized.get("emotional_recovery", "moderate")).strip().lower() or "moderate"
    if normalized["emotional_recovery"] not in {"high", "moderate", "low"}:
        normalized["emotional_recovery"] = "moderate"
    return normalized


def normalize_attitude(attitude):
    return str(attitude or "").strip().lower()


def load_json_entries(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'.")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: Expected '{input_file}' to contain a JSON array.")
        sys.exit(1)

    return data


def get_seed_id(entry):
    personal_info = entry.get('personal_info') or {}
    name = str(personal_info.get('name', '')).strip()
    if name:
        return ("name", name)
    if 'idx' in entry:
        return ("idx", entry['idx'])
    return None


def format_seed_id(seed_id):
    if seed_id is None:
        return "<unknown>"
    return str(seed_id[1])


def group_entries_by_seed(entries):
    seed_groups = defaultdict(list)
    for entry in entries:
        seed_id = get_seed_id(entry)
        if seed_id is not None:
            seed_groups[seed_id].append(entry)
    return seed_groups


def sample_entries(items, n_samples, label):
    if len(items) < n_samples:
        print(f"Note: Only found {len(items)} samples for '{label}' (requested {n_samples}).")
    return random.sample(items, min(len(items), n_samples))


def pick_seed_entries(seed_entries, attitudes):
    entries_by_attitude = defaultdict(list)
    for entry in seed_entries:
        entries_by_attitude[normalize_attitude(entry.get('attitude'))].append(entry)

    selected_entries = []
    missing_attitudes = []

    for attitude in attitudes:
        matches = entries_by_attitude.get(attitude, [])
        if matches:
            selected_entries.append(random.choice(matches))
        else:
            missing_attitudes.append(attitude)

    return selected_entries, missing_attitudes


def get_unique_seed_ids(entries):
    ordered_seed_ids = []
    seen_seed_ids = set()

    for entry in entries:
        seed_id = get_seed_id(entry)
        if seed_id is None or seed_id in seen_seed_ids:
            continue
        seen_seed_ids.add(seed_id)
        ordered_seed_ids.append(seed_id)

    return ordered_seed_ids


def group_entries_by_seed_attitude(entries):
    entries_by_seed_attitude = defaultdict(list)
    entries_by_exact_request = defaultdict(list)

    for entry in entries:
        seed_id = get_seed_id(entry)
        entry_attitude = normalize_attitude(entry.get('attitude'))
        if seed_id is None or not entry_attitude:
            continue

        entries_by_seed_attitude[(seed_id, entry_attitude)].append(entry)
        if 'idx' in entry:
            exact_key = (seed_id, entry_attitude, entry['idx'])
            entries_by_exact_request[exact_key].append(entry)

    return entries_by_seed_attitude, entries_by_exact_request


def get_requested_attitudes(seed_entry, attitude):
    seed_attitude = normalize_attitude(seed_entry.get('attitude'))

    if attitude:
        if seed_attitude and seed_attitude != attitude:
            return []
        return [attitude]

    if seed_attitude in ATTITUDES:
        return [seed_attitude]
    if not seed_attitude:
        return list(ATTITUDES)
    return []


def pick_duplicate_seed_entry(
    seed_entry,
    target_attitude,
    entries_by_seed_attitude,
    entries_by_exact_request,
    next_offsets,
):
    seed_id = get_seed_id(seed_entry)
    if seed_id is None:
        return None

    seed_attitude = normalize_attitude(seed_entry.get('attitude'))
    if seed_attitude == target_attitude and 'idx' in seed_entry:
        exact_key = (seed_id, target_attitude, seed_entry['idx'])
        exact_matches = entries_by_exact_request.get(exact_key, [])
        if exact_matches:
            return exact_matches[0]

    key = (seed_id, target_attitude)
    matches = entries_by_seed_attitude.get(key, [])
    if not matches:
        return None

    offset = next_offsets[key]
    next_offsets[key] += 1
    return matches[offset % len(matches)]


def extract_duplicate_seed_entries(data, seeds_data, attitude, input_file, seeds_file):
    entries_by_seed_attitude, entries_by_exact_request = group_entries_by_seed_attitude(
        data
    )
    next_offsets = defaultdict(int)
    final_selection = []
    requested_count = 0

    for seed_entry in seeds_data:
        seed_id = get_seed_id(seed_entry)
        if seed_id is None:
            continue

        requested_attitudes = get_requested_attitudes(seed_entry, attitude)
        if not requested_attitudes:
            continue

        requested_count += len(requested_attitudes)

        for requested_attitude in requested_attitudes:
            selected_entry = pick_duplicate_seed_entry(
                seed_entry,
                requested_attitude,
                entries_by_seed_attitude,
                entries_by_exact_request,
                next_offsets,
            )
            if selected_entry is not None:
                final_selection.append(selected_entry)
            else:
                print(
                    f"Note: Seed '{format_seed_id(seed_id)}' with attitude '{requested_attitude}' "
                    f"was not found in '{input_file}'."
                )

    if not requested_count:
        print(f"Note: No valid seed entries were found in '{seeds_file}'.")

    return final_selection


def extract_from_seed_file(
    data,
    seeds_file,
    attitude,
    input_file,
    allow_duplicates=False,
):
    seeds_data = load_json_entries(seeds_file)
    if allow_duplicates:
        return extract_duplicate_seed_entries(
            data,
            seeds_data,
            attitude,
            input_file,
            seeds_file,
        )

    requested_seed_ids = get_unique_seed_ids(seeds_data)
    seed_groups = group_entries_by_seed(data)
    requested_attitudes = [attitude] if attitude else list(ATTITUDES)

    final_selection = []

    if not requested_seed_ids:
        print(f"Note: No valid seeds were found in '{seeds_file}'.")
        return final_selection

    for seed_id in requested_seed_ids:
        if seed_id not in seed_groups:
            print(f"Note: Seed '{format_seed_id(seed_id)}' was not found in '{input_file}'.")
            continue

        selected_entries, missing_attitudes = pick_seed_entries(seed_groups[seed_id], requested_attitudes)
        final_selection.extend(selected_entries)

        if missing_attitudes:
            missing_str = ", ".join(missing_attitudes)
            print(f"Note: Seed '{format_seed_id(seed_id)}' is missing attitudes: {missing_str}.")

    return final_selection


def extract_equal_seed_samples(data, n_samples):
    seed_groups = group_entries_by_seed(data)
    valid_seed_ids = []

    for seed_id, entries in seed_groups.items():
        _, missing_attitudes = pick_seed_entries(entries, ATTITUDES)
        if not missing_attitudes:
            valid_seed_ids.append(seed_id)

    if len(valid_seed_ids) < n_samples:
        print(f"Note: Only found {len(valid_seed_ids)} valid complete seeds (requested {n_samples}).")

    selected_seed_ids = random.sample(valid_seed_ids, min(len(valid_seed_ids), n_samples))
    final_selection = []

    for seed_id in selected_seed_ids:
        selected_entries, _ = pick_seed_entries(seed_groups[seed_id], ATTITUDES)
        final_selection.extend(selected_entries)

    return final_selection


def extract_balanced_samples(
    input_file,
    output_file,
    n_samples,
    equal_seeds,
    attitude=None,
    seeds_file=None,
    allow_duplicates=False,
):
    data = load_json_entries(input_file)

    if seeds_file:
        final_selection = extract_from_seed_file(
            data,
            seeds_file,
            attitude,
            input_file,
            allow_duplicates=allow_duplicates,
        )
    elif equal_seeds:
        final_selection = extract_equal_seed_samples(data, n_samples)
    elif attitude:
        matching_entries = [entry for entry in data if normalize_attitude(entry.get('attitude')) == attitude]
        final_selection = sample_entries(matching_entries, n_samples, attitude)
    else:
        final_selection = []
        categories = {attitude_name: [] for attitude_name in ATTITUDES}

        for entry in data:
            entry_attitude = normalize_attitude(entry.get('attitude'))
            if entry_attitude in categories:
                categories[entry_attitude].append(entry)

        for attitude_name, items in categories.items():
            final_selection.extend(sample_entries(items, n_samples, attitude_name))

    final_selection = [ensure_emotional_recovery(entry) for entry in final_selection]

    # Shuffle for a non-ordered final output dataset to avoid introducing batch biases
    random.shuffle(final_selection)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_selection, f, indent=2)
    
    print(f"Success: Extracted {len(final_selection)} total samples to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract balanced attitude samples from a JSON file.")
    
    parser.add_argument('--input', type=str, required=True, help="Path to the source JSON file.")
    parser.add_argument('--output', type=str, required=True, help="Path where the sampled JSON will be saved.")
    parser.add_argument('--n-samples', type=int, default=10, 
                        help="Number of samples to extract PER category, or number of unique seeds if --equal is used (default: 10). Ignored when --seeds is used.")
    parser.add_argument('--equal', action='store_true', 
                        help="If flagged, ensures extraction of N identical seeds containing the positive/neutral/negative set based on personal_info.name, with idx as fallback.")
    parser.add_argument('--attitude', type=str,
                        help="If provided, extract only N samples for this single attitude.")
    parser.add_argument('--seeds', type=str,
                        help="Path to a JSON file with entries whose unique seeds should be found in --input. If --attitude is set, only that attitude is extracted for each seed; otherwise the positive/neutral/negative set is extracted.")
    parser.add_argument(
        '--allow-duplicates',
        '---allow-duplicates',
        '-ad',
        action='store_true',
        dest='allow_duplicates',
        help="When used with --seeds, preserve every matching seed entry instead of deduplicating by seed name.",
    )

    args = parser.parse_args()

    if args.equal and args.attitude:
        parser.error("--equal cannot be combined with --attitude.")

    if args.equal and args.seeds:
        parser.error("--equal cannot be combined with --seeds.")

    extract_balanced_samples(
        args.input,
        args.output,
        args.n_samples,
        args.equal,
        attitude=normalize_attitude(args.attitude) if args.attitude else None,
        seeds_file=args.seeds,
        allow_duplicates=args.allow_duplicates,
    )
