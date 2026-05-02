import os
import builtins

# Torch Inductor can fail on Unsloth's dynamic compiled kernels for long,
# padded sequences. Prefer stable eager/Xformers kernels unless overridden.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1") # TODO: Quitar y reparar con transformers<5 o stack que funcione como lo hacia con CAMI


def _identity_decorator(*args, **kwargs):
    if args and len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda value: value


def _interval_validator(**_kwargs):
    return lambda **kwargs: kwargs.get("default")


# Unsloth 2026.2.x re-executes Transformers 5 config classes without importing
# their decorators. Supplying these names keeps the import path working.
builtins.auto_docstring = _identity_decorator
builtins.strict = _identity_decorator
builtins.interval = _interval_validator

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import argparse
import wandb
from pathlib import Path
import time
import json

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

def apply_training_chat_template(tokenizer, chat_template):
    normalized = str(chat_template or "").strip().lower()
    if normalized in {"", "none"}:
        return tokenizer

    if normalized in {"tokenizer", "model", "native", "qwen3.6-thinking", "qwen3.6"}:
        if getattr(tokenizer, "chat_template", None):
            return get_chat_template(
                tokenizer,
                chat_template = (tokenizer.chat_template, "<|im_end|>"),
                map_eos_token = False,
            )
        return get_chat_template(
            tokenizer,
            chat_template = "qwen3-thinking",
        )

    return get_chat_template(
        tokenizer,
        chat_template = chat_template,
    )


def create_model(model, max_seq_length, load_in_4bit = False, model_repo=None, chat_template="qwen3.6-thinking"):
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

    tokenizer = apply_training_chat_template(tokenizer, chat_template)

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

def load_chatml_session(path):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{path}: failed to read file: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a JSON object.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}: expected non-empty 'messages' list.")

    normalized_messages = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"{path}: messages[{index}] must be a JSON object.")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{path}: messages[{index}] has unsupported role {role!r}.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}: messages[{index}] must have non-empty string content.")
        normalized_messages.append({"role": role, "content": content})

    return {
        "messages": normalized_messages,
        "source_file": Path(path).name,
    }


def load_chatml_sessions(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"--input-data must be a directory containing ChatML session JSON files: {input_dir}")

    input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    if not input_files:
        raise ValueError(f"No .json session files found in {input_dir}")

    return [load_chatml_session(path) for path in input_files]


def formatting_prompts_func(examples, tokenizer):
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
            preserve_thinking = True,
        )
        for messages in examples["messages"]
    ]
    return {"text": texts}


def create_dataset(input_dir, tokenizer):
    dataset = Dataset.from_list(load_chatml_sessions(input_dir))
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched = True,
        remove_columns = dataset.column_names,
    )
    return dataset

def train_model(model, tokenizer, dataset, eval_dataset, max_seq_length, trainer_output_dir):
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
            num_train_epochs = 2,
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
            output_dir = str(trainer_output_dir),
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
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    trainer.train()

def save_model(model, tokenizer, output_dir, save_merged=False):
  output_dir = Path(output_dir)
  print("Saving LoRA adapters in 16-bit format...")
  lora_output_dir = output_dir / "lora_adapters"
  model.save_pretrained(str(lora_output_dir)); tokenizer.save_pretrained(str(lora_output_dir))
  os.sync()

  #print("Exporting to GGUF format for llama.cpp/Ollama...")
  #gguf_output_dir = output_dir / "gguf"
  #model.save_pretrained_gguf(gguf_output_dir, tokenizer, quantization_method = "q4_k_m")
  #os.sync()
  
  if save_merged:
    print("Saving merged model in 16-bit format...")
    merged_output_dir = output_dir / "merged_model"
    model.save_pretrained_merged(str(merged_output_dir), tokenizer, save_method = "merged_16bit")
    os.sync()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen thinking model from ChatML session JSON files.")
    parser.add_argument("--model", type=str, required=True, help="Local model directory name under models/. If missing, it is downloaded from Hugging Face.")
    parser.add_argument("--model-repo", type=str, help="Optional Hugging Face repo ID to use when models/<model> is missing. Defaults to known aliases or --model.")
    parser.add_argument("--input-data", "--input-dir", type=str, required=True, help="Directory containing ChatML training session JSON files.")
    parser.add_argument("--eval-data", "--eval-dir", type=str, required=False, help="Optional directory containing ChatML eval session JSON files.")
    parser.add_argument("--max-seq-length", type=int, default=20000, help="Maximum sequence length for training")
    parser.add_argument("--chat-template", type=str, default="qwen3.6-thinking", help="Unsloth chat template name to apply to the tokenizer. Default uses the model/tokenizer native Qwen3.6 thinking template via get_chat_template; pass qwen3-thinking to force Unsloth's named Qwen3 template.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--qlora", action="store_true", help="Load model in 4-bit mode (for QLoRA training)")
    parser.add_argument("--save-merged", action="store_true", help="Save merged model in 16-bit format")
    parser.add_argument("--output-dir", "--output_dir", type=str, help="Directory for all training artifacts. Defaults to models/<model>_<timestamp>.")
    args = parser.parse_args()

    run_timestamp = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else Path("models") / f"{args.model}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_output_dir = output_dir / "trainer_output"
    print(f"Training artifacts will be saved under: {output_dir.resolve()}")
    
    if not args.disable_wandb:
      os.environ["WANDB_PROJECT"] = "qwen3-finetuning"
      wandb.init()
    else:
      os.environ["WANDB_MODE"] = "disabled"
    
    model, tokenizer = create_model(
        args.model,
        args.max_seq_length,
        load_in_4bit=args.qlora,
        model_repo=args.model_repo,
        chat_template=args.chat_template,
    )
    print("Model and tokenizer loaded successfully.")
    dataset = create_dataset(args.input_data, tokenizer)
    eval_dataset = create_dataset(args.eval_data, tokenizer) if args.eval_data else None
    print(f"Dataset loaded with {len(dataset)} training session(s).")
    if eval_dataset is not None:
      print(f"Eval dataset loaded with {len(eval_dataset)} session(s).")
    start_time = time.time()
    train_model(model, tokenizer, dataset, eval_dataset, args.max_seq_length, trainer_output_dir)
    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds. Saving model...")
    save_model(model, tokenizer, output_dir, args.save_merged)
    
    if not args.disable_wandb:
      wandb.finish()
