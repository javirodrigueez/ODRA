import os
import json
import glob
import argparse
from collections import defaultdict

def analyze_sessions(input_dir: str) -> None:
    """
    Parses JSON session files to compute turn metrics and analyze session termination reasons,
    stratified by user attitude.
    """
    total_sessions = 0
    total_turns = 0
    
    # Aggregation dictionaries
    attitude_metrics = defaultdict(lambda: {"sessions": 0, "turns": 0})
    overall_reasons = defaultdict(int)
    attitude_reasons = defaultdict(lambda: defaultdict(int))

    # Safely iterate over all JSON files
    file_pattern = os.path.join(input_dir, "*.json")
    for filepath in glob.glob(file_pattern):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract schema fields with strict fallbacks
            turns = len(data.get("messages", []))
            attitude = data.get("patient_profile", {}).get("attitude", "unspecified")
            end_reason = data.get("session_ended_reason", "unspecified")

            # Aggregate overarching state
            total_sessions += 1
            total_turns += turns
            overall_reasons[end_reason] += 1

            # Aggregate state conditioned on attitude
            attitude_metrics[attitude]["sessions"] += 1
            attitude_metrics[attitude]["turns"] += turns
            attitude_reasons[attitude][end_reason] += 1

        except Exception as e:
            print(f"Skipping {filepath} due to parsing error: {e}")

    # Compute and evaluate results
    if total_sessions == 0:
        print("No valid JSON sessions found in the provided directory.")
        return

    # 1. Evaluate Turn Metrics
    avg_turns_overall = total_turns / total_sessions
    print(f"--- Turn Metrics ---")
    print(f"Total Sessions: {total_sessions}")
    print(f"Overall Average Turns per Session: {avg_turns_overall:.2f}\n")

    print("Average Turns Grouped by Attitude:")
    for att, stats in sorted(attitude_metrics.items()):
        avg_turns = stats["turns"] / stats["sessions"]
        print(f" - [{att.upper()}]: {avg_turns:.2f} turns/session (n={stats['sessions']})")
    
    # 2. Evaluate Termination Distributions
    print("\n--- Session Termination Analysis ---")
    print("Overall Session Ended Reasons:")
    for reason, count in sorted(overall_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f" - {reason}: {count}")

    print("\nSession Ended Reasons Stratified by Attitude:")
    for att, reasons in sorted(attitude_reasons.items()):
        print(f" [{att.upper()}]:")
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {reason}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze NLP session datasets for conversational metrics and termination distributions."
    )
    parser.add_argument(
        "-d", "--dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing the .json session files."
    )
    args = parser.parse_args()

    analyze_sessions(args.dir)