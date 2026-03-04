import json
import argparse

def process_data(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    seen_profiles = set()  # Using a set for O(1) lookups
    counter = 0

    for entry in data:
        client_data = entry.get("AI_client", {})
        # Correct path: AI_counselor is a sibling of AI_client
        counselor_data = entry.get("AI_counselor", {})
        profile = counselor_data.get("Response", {}).get("client_information")

        # Deduplication logic
        if profile and profile not in seen_profiles:
            seen_profiles.add(profile)
            
            transformed_entry = {
                "idx": counter,
                "intake_form": client_data.get("intake_form"),
                "dialogue_history_init": client_data.get("dialogue_history_init"),
                "profile": profile
            }
            processed_data.append(transformed_entry)
            counter += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate counseling data.")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    process_data(args.input, args.output)