from datasets import load_dataset
import json
import os

# 1. Load the dataset
ds = load_dataset("LangAGI-Lab/cactus")

# 2. Ensure the directory exists
os.makedirs("data", exist_ok=True)

# 3. Convert the 'train' split to a list of dictionaries and dump
with open("data/cactus.json", "w") as f:
    # Use .to_list() to make it JSON serializable
    json.dump(ds["train"].to_list(), f, indent=2)