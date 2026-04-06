import json
import random

def ensure_emotional_recovery(profile):
    normalized = dict(profile)
    normalized["emotional_recovery"] = str(normalized.get("emotional_recovery", "moderate")).strip().lower() or "moderate"
    if normalized["emotional_recovery"] not in {"high", "moderate", "low"}:
        normalized["emotional_recovery"] = "moderate"
    return normalized

def get_cactus_sample():
    with open('data/processed_cactus.json', 'r') as f:
        data = json.load(f)
    return ensure_emotional_recovery(data[random.randint(0, len(data) - 1)])

if __name__ == "__main__":
    sample = get_cactus_sample()
    profiles = [sample]
    with open('data/sample_cactus.json', 'w') as f:
        json.dump(profiles, f, indent=2)
