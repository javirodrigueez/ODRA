import json
import random

def get_cactus_sample():
    with open('data/processed_cactus.json', 'r') as f:
        data = json.load(f)
    return data[random.randint(0, len(data) - 1)]

if __name__ == "__main__":
    sample = get_cactus_sample()
    profiles = [sample]
    with open('data/sample_cactus.json', 'w') as f:
        json.dump(profiles, f, indent=2)