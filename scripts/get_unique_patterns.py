import json
import argparse
from typing import List, Set

def extract_unique_patterns(input_path: str) -> Set[str]:
    """
    Parses a JSON dataset to extract unique unhelpful thinking patterns.
    """
    unique_patterns = set()
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for entry in data:
            # Assumes 'patterns' is a list of strings or a dictionary
            patterns = entry.get("patterns", [])
            if isinstance(patterns, list):
                unique_patterns.update(patterns)
            elif isinstance(patterns, dict):
                unique_patterns.update(patterns.keys())
                
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing file: {e}")
        
    return unique_patterns

def main():
    parser = argparse.ArgumentParser(description="Extract unique thinking style keys from clinical JSON data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, help="Path to save unique keys (optional).")
    
    args = parser.parse_args()
    
    results = extract_unique_patterns(args.input)
    
    print(f"Detected {len(results)} unique patterns:")
    for pattern in sorted(results):
        print(f" - {pattern}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(list(results), f, indent=4)

if __name__ == "__main__":
    main()