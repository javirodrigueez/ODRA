import json
import os
import shutil
import argparse

def move_files_by_attitude():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Move files associated with a specific attitude from a JSON manifest."
    )
    
    parser.add_argument("--json", required=True, help="Path to the master JSON file.")
    parser.add_argument("--source", default="./files", help="Directory where the IDX files are located.")
    parser.add_argument("--target", default="./negative_files", help="Subdirectory to move files into.")
    parser.add_argument("--attitude", default="negative", help="The attitude string to filter for.")
    parser.add_argument("--ext", default=".json", help="File extension of the IDX files (e.g., .json, .txt).")

    args = parser.parse_args()

    # 2. Validate paths
    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.")
        return

    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    # 3. Create target directory if it doesn't exist
    if not os.path.exists(args.target):
        os.makedirs(args.target)
        print(f"Created target directory: {args.target}")

    # 4. Load and process the JSON data
    try:
        with open(args.json, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Check the file format.")
        return

    moved_count = 0
    skipped_count = 0

    print(f"Filtering for attitude: '{args.attitude}'...")

    for item in data:
        idx = item.get('idx')
        # Use .strip().lower() to handle whitespace or casing inconsistencies
        current_attitude = str(item.get('attitude', '')).strip().lower()

        if current_attitude == args.attitude.lower():
            filename = f"simulation_{idx}{args.ext}"
            src_path = os.path.join(args.source, filename)
            dest_path = os.path.join(args.target, filename)

            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
                moved_count += 1
            else:
                print(f"Skipping: {filename} (File not found in source)")
                skipped_count += 1

    # 5. Summary Report
    print("-" * 30)
    print(f"Task Completed.")
    print(f"Successfully moved: {moved_count}")
    if skipped_count > 0:
        print(f"Files missing from source: {skipped_count}")
    print("-" * 30)

if __name__ == "__main__":
    move_files_by_attitude()