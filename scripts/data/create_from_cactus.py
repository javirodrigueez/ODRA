import re
import json
import argparse
import glob
from pathlib import Path

ATTITUDES = ["positive", "neutral", "negative"]

def extract_section(header, full_text):
    pattern = rf"{header}\s*(.*?)(?=\n\n\d\.|\n\n[A-Z]|$)"
    match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).lstrip(': ').strip() if match else ""

def extract_line(key, full_text):
    pattern = rf"{key}[:\n\s]*(.*?)(?=\n[A-Z]|\n\n|$)"
    match = re.search(pattern, full_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def load_huggingface_rows():
    from datasets import load_dataset

    print("Loading dataset...")
    ds = load_dataset("LangAGI-Lab/cactus")
    return ds["train"]

def sort_json_path(path):
    match = re.search(r"(\d+)(?=\.json$)", path.name)
    return (path.parent.as_posix(), 0 if match else 1, int(match.group(1)) if match else path.name)

def expand_json_sources(json_source):
    sources = []
    source_text = str(json_source)
    if any(char in source_text for char in "*?[]"):
        sources = [Path(path) for path in glob.glob(source_text)]
    else:
        sources = [Path(json_source)]

    json_files = []
    for source in sources:
        if source.is_dir():
            json_files.extend(path for path in source.iterdir() if path.suffix.lower() == ".json")
        else:
            json_files.append(source)
    return sorted(json_files, key=sort_json_path)

def load_json_rows(json_source):
    print(f"Loading JSON source: {json_source}")
    json_files = expand_json_sources(json_source)
    rows = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.append(data)
        else:
            raise ValueError(f"JSON source must contain a list or object record: {json_file}")
    return rows

def normalize_item(item):
    if not isinstance(item, dict):
        return item

    ai_client = item.get("AI_client")
    if not isinstance(ai_client, dict) and isinstance(item.get("example"), dict):
        ai_client = item["example"].get("AI_client")

    if isinstance(ai_client, dict):
        return {
            "intake_form": ai_client.get("intake_form", ""),
            "patterns": item.get("patterns", []),
            "attitude": ai_client.get("attitude"),
            "cbt_plan": item.get("cbt_plan", ""),
        }
    return item

def build_entry(idx, item, attitude, include_cbt_plan=False):
    text = item["intake_form"]
    name = extract_line("Name", text)

    personal_info = {
        "name": name,
        "age": extract_line("Age", text),
        "gender": extract_line("Gender", text),
        "occupation": extract_line("Occupation", text),
        "education": extract_line("Education", text),
        "marital_status": extract_line("Marital Status", text),
        "family_details": extract_line("Family Details", text)
    }
    
    more_info = {
        "past_history": extract_section("4. Past History", text),
        "academic_occupational_functioning": extract_section("5. Academic/occupational functioning level", text),
        "social_support_system": extract_section("6. Social Support System", text)
    }
    
    entry = {
        "idx": idx,
        "personal_info": personal_info,
        "presenting_problem": extract_section("2. Presenting Problem", text),
        "reason_seeking_counseling": extract_section("3. Reason for Seeking Counseling", text),
        "patterns": item.get("patterns", []),
        "attitude": attitude,
        "emotional_recovery": "moderate",
        "more_information": more_info,
    }
    if include_cbt_plan:
        entry["cbt_plan"] = item.get("cbt_plan", "")
    return entry

def normalize_attitude(attitude):
    return str(attitude or "").strip().lower()

def process_dataset(output_file, json_source=None, include_cbt_plan=False):
    source_rows = load_json_rows(json_source) if json_source else load_huggingface_rows()
    processed_data = []

    print(f"Processing {len(source_rows)} items...")
    added_seeds = set()
    for idx, raw_item in enumerate(source_rows):
        item = normalize_item(raw_item)
        text = item["intake_form"]
        name = extract_line("Name", text)

        if not json_source and not include_cbt_plan and name in added_seeds:
            continue

        if not json_source and not include_cbt_plan:
            added_seeds.add(name)

        source_attitude = normalize_attitude(item.get("attitude"))
        attitudes = [source_attitude] if (json_source or include_cbt_plan) and source_attitude else ATTITUDES
        for attitude in attitudes:
            processed_data.append(build_entry(idx, item, attitude, include_cbt_plan))

    # 3. Save the results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Success! Saved to: {output_file}")

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Process the Cactus dataset into structured JSON.")
    
    # Add the output-file argument
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="data/processed_cactus.json", 
        help="The file where the processed JSON will be saved (default: data/processed_cactus.json)"
    )
    parser.add_argument(
        "--json-source",
        type=str,
        help="Optional local JSON source instead of loading LangAGI-Lab/cactus from Hugging Face."
    )
    parser.add_argument(
        "--include-cbt-plan",
        action="store_true",
        help="Include cbt_plan from top-level Cactus records. Missing plans are written as empty strings."
    )

    args = parser.parse_args()
    
    # Run the main processing function
    process_dataset(args.output_file, args.json_source, args.include_cbt_plan)
