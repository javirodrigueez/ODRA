import re
import json
import os
import argparse
from datasets import load_dataset

def extract_section(header, full_text):
    pattern = rf"{header}\s*(.*?)(?=\n\n\d\.|\n\n[A-Z]|$)"
    match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).lstrip(': ').strip() if match else ""

def extract_line(key, full_text):
    pattern = rf"{key}[:\n\s]*(.*?)(?=\n[A-Z]|\n\n|$)"
    match = re.search(pattern, full_text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def process_dataset(output_file):
    # 2. Load the dataset
    print("Loading dataset...")
    ds = load_dataset("LangAGI-Lab/cactus")
    
    processed_data = []

    print(f"Processing {len(ds['train'])} items...")
    for idx, item in enumerate(ds["train"]):
        text = item["intake_form"]
        
        personal_info = {
            "name": extract_line("Name", text),
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
        
        new_entry = {
            "idx": idx,
            "personal_info": personal_info,
            "presenting_problem": extract_section("2. Presenting Problem", text),
            "reason_seeking_counseling": extract_section("3. Reason for Seeking Counseling", text),
            "patterns": item["patterns"],
            "attitude": item["attitude"],
            "more_information": more_info,
        }
        
        processed_data.append(new_entry)

    # 3. Save the results
    with open(output_file, "w") as f:
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

    args = parser.parse_args()
    
    # Run the main processing function
    process_dataset(args.output_file)