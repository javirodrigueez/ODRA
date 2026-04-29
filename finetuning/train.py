from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse
import wandb
import os
from pathlib import Path
import time

MODEL_REPO_FALLBACKS = {
    "qwen3-high": "unsloth/Qwen3-32B",
    "qwen3-medium": "unsloth/Qwen3-32B-FP8",
    "qwen3-low": "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    "qwen3-moe": "unsloth/Qwen3-30B-A3B",
    "qwen3-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
}


def local_models_dir():
    return Path(__file__).resolve().parent.parent / "models"


def is_local_model_available(model_dir):
    return model_dir.is_dir() and any(model_dir.iterdir())


def resolve_hf_repo_id(model_name, model_repo=None):
    return model_repo or MODEL_REPO_FALLBACKS.get(model_name, model_name)


def ensure_local_model(model_name, model_repo=None, models_dir=None):
    models_dir = Path(models_dir) if models_dir is not None else local_models_dir()
    model_dir = models_dir / model_name
    if is_local_model_available(model_dir):
        print(f"Using local model from: {model_dir}")
        return model_dir

    repo_id = resolve_hf_repo_id(model_name, model_repo)
    print(f"Local model not found at {model_dir}. Downloading {repo_id} from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download missing models. "
            "Install it or place the model under models/<model-name>."
        ) from exc

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        ignore_patterns=["*.bin", "*.pth", "*.msgpack"],
    )
    print(f"Model available at: {path}")
    return Path(path)

def create_model(model, max_seq_length, load_in_4bit = False, model_repo=None):
    # 1. Configuration
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 2. Load Model & Tokenizer
    model_path = ensure_local_model(model, model_repo=model_repo)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(model_path),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,  # param to control qlora
        device_map = {"": 0}
    )

    # 3. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Suggested rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",   
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )
    return model, tokenizer

# Formatting Function (ChatML Style for Qwen)
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Qwen ChatML Format
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n" \
               f"<|im_start|>user\n{input}<|im_end|>\n" \
               f"<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

def create_dataset(input_file):
    dataset = load_dataset("json", data_files=input_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    return dataset

def train_model(model, tokenizer, dataset, eval_dataset, max_seq_length):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        #eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can speed up training for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Increase this if you run OOM
            gradient_checkpointing = True,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "wandb",
            run_name = f"{model}_{int(time.time())}",
            # For saving best model, not latest
            # output_dir = "tmp_trainer",
            # eval_strategy = "steps",
            # eval_steps = 10,
            # save_strategy = "steps",
            # save_steps = 10,
            # save_total_limit = 2,
            # metric_for_best_model = "eval_loss",
            # greater_is_better = False,
            # load_best_model_at_end = True,
        ),
    )
    trainer.train()

def save_model(model, tokenizer, save_directory, save_merged=False):
  print("Saving LoRA adapters in 16-bit format...")
  output_dir = os.path.abspath(save_directory+'_lora')
  model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
  os.sync()

  #print("Exporting to GGUF format for llama.cpp/Ollama...")
  #output_dir = os.path.abspath(save_directory+'_gguf')
  #model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q4_k_m")
  #os.sync()
  
  if save_merged:
    print("Saving merged model in 16-bit format...")
    output_dir = os.path.abspath(save_directory)
    model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")
    os.sync()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive therapy parser using readlines() and conditional logic.")
    parser.add_argument("--model", type=str, required=True, help="Local model directory name under models/. If missing, it is downloaded from Hugging Face.")
    parser.add_argument("--model-repo", type=str, help="Optional Hugging Face repo ID to use when models/<model> is missing. Defaults to known aliases or --model.")
    parser.add_argument("--input-data", type=str, required=True, help="JSONL file containing training data with fields: instruction, input, output")
    parser.add_argument("--eval-data", type=str, required=True, help="JSONL file containing eval data with fields: instruction, input, output")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length for training")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--qlora", action="store_true", help="Load model in 4-bit mode (for QLoRA training)")
    parser.add_argument("--save-merged", action="store_true", help="Save merged model in 16-bit format")
    args = parser.parse_args()
    
    if not args.disable_wandb:
      os.environ["WANDB_PROJECT"] = "qwen3-finetuning"
      wandb.init()
    else:
      os.environ["WANDB_MODE"] = "disabled"
    
    dataset = create_dataset(args.input_data)
    eval_dataset = create_dataset(args.eval_data)
    print(f"Dataset loaded with {len(dataset)} samples.")
    model, tokenizer = create_model(args.model, args.max_seq_length, load_in_4bit=args.qlora, model_repo=args.model_repo)
    print("Model and tokenizer loaded successfully.")
    start_time = time.time()
    train_model(model, tokenizer, dataset, eval_dataset, args.max_seq_length)
    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds. Saving model...")
    save_model(model, tokenizer, f"models/{args.model}_{int(time.time())}", args.save_merged)
    
    if not args.disable_wandb:
      wandb.finish()
