import os
import json
import re
import argparse

def find_missing_simulations(json_path: str, directory: str):
    """
    Identifies files in the directory whose 'idx' is not present in the master JSON.
    """
    # 1. Extract valid idxs from master JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Handle both single object and list of objects
            if isinstance(data, dict):
                data = [data]
            valid_idxs = {str(item['idx']) for item in data if 'idx' in item}
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # 2. Pattern to match simulation_*.txt or simulation_*.json
    # Regex captures the 'idx' part of the filename
    pattern = re.compile(r"simulation_(\d+)\.(txt|json)")

    missing_files = []

    # 3. Scan directory
    try:
        files = os.listdir(directory)
        for filename in files:
            match = pattern.match(filename)
            if match:
                file_idx = match.group(1)
                # If idx from filename is NOT in the master JSON set
                if file_idx not in valid_idxs:
                    missing_files.append(filename)
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return

    # 4. Output results
    if missing_files:
        print(f"--- Files found in '{directory}' not present in {json_path} ---")
        for f in sorted(missing_files):
            print(f)
    else:
        print("No orphan files found. All files match the JSON records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find simulation files not indexed in a master JSON file."
    )
    parser.add_argument(
        "--json", 
        type=str, 
        required=True, 
        help="Path to the master JSON file containing valid 'idx' fields."
    )
    parser.add_argument(
        "--dir", 
        type=str, 
        default=".", 
        help="Directory to scan for simulation files (default: current)."
    )

    args = parser.parse_args()

    find_missing_simulations(args.json, args.dir)
