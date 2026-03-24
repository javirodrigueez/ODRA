import json
import random
import argparse

def get_cactus_samples(input_file, num_samples):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Safety check: ensure we don't request more samples than available
    if num_samples > len(data):
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(data)}. Extracting all {len(data)}.")
        num_samples = len(data)
        
    # random.sample extracts multiple unique items at once
    return random.sample(data, num_samples)

if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Extract random samples from a JSON dataset.")
    parser.add_argument("-i", "--input", required=True, help="Path to the source dataset JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path for the output JSON file")
    parser.add_argument("-n", "--num_samples", type=int, required=True, help="Number of random samples to extract")
    
    # 2. Parse the arguments
    args = parser.parse_args()

    # 3. Extract the samples using the provided arguments
    samples = get_cactus_samples(args.input, args.num_samples)
    
    # 4. Write the output to the specified file
    with open(args.output, 'w') as f:
        json.dump(samples, f, indent=2)
        
    print(f"Successfully extracted {len(samples)} sample(s) from {args.input} into {args.output}.")