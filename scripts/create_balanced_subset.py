import json
import random
import argparse
import sys

def extract_balanced_samples(input_file, output_file, n_per_category):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'.")
        sys.exit(1)

    # Initialize buckets for each attitude
    categories = {
        'positive': [],
        'neutral': [],
        'negative': []
    }

    # Sort the data into buckets
    for entry in data:
        # Using .get() with a fallback and .lower() for robust matching
        attitude = str(entry.get('attitude', '')).lower()
        if attitude in categories:
            categories[attitude].append(entry)

    final_selection = []

    # Sample from each category
    for attitude, items in categories.items():
        if len(items) < n_per_category:
            print(f"Note: Only found {len(items)} samples for '{attitude}' (requested {n_per_category}).")
        
        # Take the requested amount, or everything if the bucket is too small
        count_to_take = min(len(items), n_per_category)
        selection = random.sample(items, count_to_take)
        final_selection.extend(selection)

    # Shuffle for a non-ordered final output file
    random.shuffle(final_selection)

    # Save the result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_selection, f, indent=2)
    
    print(f"Success: Extracted {len(final_selection)} total samples to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract balanced attitude samples from a JSON file.")
    
    # Required File Arguments
    parser.add_argument('--input', type=str, required=True, help="Path to the source JSON file.")
    parser.add_argument('--output', type=str, required=True, help="Path where the sampled JSON will be saved.")
    
    # Optional Sample Size Argument
    parser.add_argument('--n-samples', type=int, default=10, 
                        help="Number of samples to extract PER attitude category (default: 10).")

    args = parser.parse_args()

    extract_balanced_samples(args.input, args.output, args.n_samples)