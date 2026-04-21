import json
import argparse
import sys
from itertools import islice

def extract_first_n_records(input_path: str, output_path: str, limit: int = 18):
    """
    Extracts the first N elements from a JSON list and saves them to a new file.
    Designed for scalability in NLP data preprocessing.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON root element must be a list to slice records.")

        # Using islice for consistent slicing performance
        subset = list(islice(data, limit))

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subset, f, indent=4)
        
        print(f"Successfully extracted {len(subset)} records to {output_path}")

    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_path}'.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Dataset Subsampling Utility")
    parser.add_argument("--input", type=str, required=True, help="Path to source JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save sliced JSON")
    parser.add_argument("--limit", type=int, default=18, help="Number of records to retrieve (default: 18)")
    
    args = parser.parse_args()
    
    extract_first_n_records(args.input, args.output, args.limit)