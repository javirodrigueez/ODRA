import argparse
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Initiating download for {repo_id}...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        # Optimization: ignore unnecessary files like .bin if safetensors exist
        ignore_patterns=["*.bin", "*.pth", "*.msgpack"]
    )
    print(f"Model successfully located at: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Model Downloader")
    parser.add_argument("--repo", type=str, required=True, help="Repo ID (e.g., 'bert-base-uncased')")
    parser.add_argument("--out", type=str, default="./model_weights", help="Local destination")
    
    args = parser.parse_args()
    download_model(args.repo, args.out)