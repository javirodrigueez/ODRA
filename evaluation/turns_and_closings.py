import os
import json
import glob
import argparse
import re
from collections import defaultdict

WORD_RE = re.compile(r"\b[\w']+\b")


def count_words(text: object) -> int:
    return len(WORD_RE.findall(str(text or "")))


def _normalized_messages(data: dict) -> list[dict]:
    """
    Return utterances as role/content dicts for both ReasonCBT and Magnet sessions.
    Magnet stores the conversation under dialogue with role/message fields.
    """
    if isinstance(data.get("messages"), list):
        messages = data.get("messages", [])
        return [
            {
                "role": str(message.get("role", "")).strip().lower(),
                "content": message.get("content", ""),
            }
            for message in messages
            if isinstance(message, dict)
        ]

    if isinstance(data.get("dialogue"), list):
        utterance_roles = {"agent", "user", "counselor", "client", "therapist", "patient"}
        messages = []
        for message in data.get("dialogue", []):
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip().lower()
            if role not in utterance_roles:
                continue

            messages.append(
                {
                    "role": role,
                    "content": message.get("message", message.get("content", "")),
                }
            )
        return messages

    return []


def _session_attitude(data: dict) -> str:
    patient_profile = data.get("patient_profile", {})
    if isinstance(patient_profile, dict) and patient_profile.get("attitude"):
        return patient_profile.get("attitude")

    ai_client = data.get("AI_client", {})
    if isinstance(ai_client, dict) and ai_client.get("attitude"):
        return ai_client.get("attitude")

    return "unspecified"


def _session_end_reason(data: dict) -> str:
    return data.get("session_ended_reason", "unspecified")


def analyze_sessions(input_dir: str) -> dict:
    """
    Parses JSON session files to compute turn metrics and analyze session termination reasons,
    stratified by user attitude.
    """
    total_sessions = 0
    total_turns = 0
    total_therapist_words = 0
    total_therapist_messages = 0
    
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
            messages = _normalized_messages(data)
            turns = len(messages)
            attitude = _session_attitude(data)
            end_reason = _session_end_reason(data)
            therapist_messages = [
                message
                for message in messages
                if isinstance(message, dict)
                and message.get("role") in {"therapist", "counselor", "agent"}
            ]
            therapist_words = sum(
                count_words(message.get("content", "")) for message in therapist_messages
            )

            # Aggregate overarching state
            total_sessions += 1
            total_turns += turns
            total_therapist_words += therapist_words
            total_therapist_messages += len(therapist_messages)
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
        return {
            "input_dir": input_dir,
            "total_sessions": 0,
            "avg_turns_overall": 0.0,
            "avg_therapist_words_per_session": 0.0,
            "avg_therapist_words_per_therapist_turn": 0.0,
            "attitudes": {},
            "overall_reasons": {},
            "attitude_reasons": {},
        }

    # 1. Evaluate Turn Metrics
    avg_turns_overall = total_turns / total_sessions
    avg_therapist_words_per_session = total_therapist_words / total_sessions
    avg_therapist_words_per_therapist_turn = (
        total_therapist_words / total_therapist_messages
        if total_therapist_messages
        else 0.0
    )
    print(f"--- Turn Metrics ---")
    print(f"Total Sessions: {total_sessions}")
    print(f"Overall Average Turns per Session: {avg_turns_overall:.2f}\n")
    print("Therapist Word Metrics:")
    print(f"Average Therapist Words per Session: {avg_therapist_words_per_session:.2f}")
    print(
        "Average Therapist Words per Therapist Turn: "
        f"{avg_therapist_words_per_therapist_turn:.2f}\n"
    )

    print("Average Turns Grouped by Attitude:")
    attitude_summary = {}
    for att, stats in sorted(attitude_metrics.items()):
        avg_turns = stats["turns"] / stats["sessions"]
        attitude_summary[att] = {
            "sessions": stats["sessions"],
            "turns": stats["turns"],
            "avg_turns": avg_turns,
        }
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

    return {
        "input_dir": input_dir,
        "total_sessions": total_sessions,
        "total_turns": total_turns,
        "avg_turns_overall": avg_turns_overall,
        "total_therapist_words": total_therapist_words,
        "total_therapist_messages": total_therapist_messages,
        "avg_therapist_words_per_session": avg_therapist_words_per_session,
        "avg_therapist_words_per_therapist_turn": avg_therapist_words_per_therapist_turn,
        "attitudes": attitude_summary,
        "overall_reasons": dict(overall_reasons),
        "attitude_reasons": {
            attitude: dict(reasons) for attitude, reasons in attitude_reasons.items()
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze NLP session datasets for conversational metrics and termination distributions."
    )
    parser.add_argument(
        "-d", "--dir", "--eval-dir",
        type=str, 
        required=True, 
        help="Path to the directory containing the .json session files."
    )
    args = parser.parse_args()

    analyze_sessions(args.dir)
