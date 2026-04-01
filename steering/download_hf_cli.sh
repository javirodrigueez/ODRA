export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=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

# Download only the BF16 folder
hf download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir ./steering/models/Qwen3.5-35B-FP8
#hf download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir ./steering/models/Qwen3.5-122B-FP8