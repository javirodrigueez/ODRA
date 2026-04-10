import json
import random
import argparse
import sys
from collections import defaultdict

def extract_balanced_samples(input_file, output_file, n_samples, equal_seeds):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'.")
        sys.exit(1)

    final_selection = []

    if equal_seeds:
        seed_groups = defaultdict(list)
        for entry in data:
            # Primary grouping by unique name.
            # Fallback to sequential grouping (idx // 3) if name is missing but idx exists.
            personal_info = entry.get('personal_info', {})
            seed_id = personal_info.get('name')
            
            if not seed_id and 'idx' in entry:
                seed_id = entry['idx'] // 3 
                
            if seed_id is not None:
                seed_groups[seed_id].append(entry)
        
        # Enforce that valid seeds contain all three distinct attitudes
        valid_seeds = {
            seed_id: entries for seed_id, entries in seed_groups.items() 
            if len({str(e.get('attitude', '')).lower() for e in entries}) >= 3
        }
        
        seed_keys = list(valid_seeds.keys())
        if len(seed_keys) < n_samples:
            print(f"Note: Only found {len(seed_keys)} valid complete seeds (requested {n_samples}).")
        
        count_to_take = min(len(seed_keys), n_samples)
        selected_seeds = random.sample(seed_keys, count_to_take)
        
        for seed in selected_seeds:
            final_selection.extend(valid_seeds[seed])
            
    else:
        categories = {'positive': [], 'neutral': [], 'negative': []}
        
        for entry in data:
            attitude = str(entry.get('attitude', '')).lower()
            if attitude in categories:
                categories[attitude].append(entry)

        for attitude, items in categories.items():
            if len(items) < n_samples:
                print(f"Note: Only found {len(items)} samples for '{attitude}' (requested {n_samples}).")
            
            count_to_take = min(len(items), n_samples)
            final_selection.extend(random.sample(items, count_to_take))

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
                        help="Number of samples to extract PER category, or number of unique seeds if --equal is used (default: 10).")
    parser.add_argument('--equal', action='store_true', 
                        help="If flagged, ensures extraction of N identical seeds containing all 3 attitudes based on personal_info.name.")

    args = parser.parse_args()

    extract_balanced_samples(args.input, args.output, args.n_samples, args.equal)