import os
import argparse
import json
from pathlib import Path
import time
from typing import Any

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
import wandb
from datasets import Dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


def local_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def resolve_local_model_path(model_name: str) -> Path:
    direct_path = Path(model_name).expanduser()
    candidates = [direct_path]
    if not direct_path.is_absolute():
        candidates.append(local_models_dir() / model_name)

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Local model not found. Checked: {searched}")


def apply_training_chat_template(tokenizer: Any, chat_template: str) -> Any:
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
        return get_chat_template(tokenizer, chat_template = "qwen3-thinking")

    return get_chat_template(tokenizer, chat_template = chat_template)


def create_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool = False,
    chat_template: str = "qwen3.6-thinking",
) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_path = resolve_local_model_path(model_name)
    print(f"Using local model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(model_path),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,  # param to control qlora
        device_map = "cuda:0"
    )

    tokenizer = apply_training_chat_template(tokenizer, chat_template)

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0, 
        bias = "none",   
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )
    return model, tokenizer


def load_chatml_session(path: Path) -> dict[str, Any]:
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


def load_chatml_sessions(input_dir: str | Path) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"--input-data must be a directory containing ChatML session JSON files: {input_dir}")

    input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    if not input_files:
        raise ValueError(f"No .json session files found in {input_dir}")

    return [load_chatml_session(path) for path in input_files]


def formatting_prompts_func(examples: dict[str, list[Any]], tokenizer: Any) -> dict[str, list[str]]:
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


def create_dataset(input_dir: str | Path, tokenizer: Any) -> Dataset:
    dataset = Dataset.from_list(load_chatml_sessions(input_dir))
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched = True,
        remove_columns = dataset.column_names,
    )
    return dataset


def train_model(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    max_seq_length: int,
    trainer_output_dir: Path,
    report_to: str,
    run_name: str,
) -> None:
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_length = max_seq_length,
            dataset_num_proc = 2,
            eos_token = tokenizer.eos_token,
            pad_token = tokenizer.pad_token or tokenizer.eos_token,
            packing = False,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
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
            report_to = report_to,
            run_name = run_name,
            output_dir = str(trainer_output_dir),
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    trainer.train()


def save_model(model: Any, tokenizer: Any, output_dir: str | Path, save_merged: bool = False) -> None:
    output_dir = Path(output_dir)
    print("Saving LoRA adapters in 16-bit format...")
    lora_output_dir = output_dir / "lora_adapters"
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))
    os.sync()

    if save_merged:
        print("Saving merged model in 16-bit format...")
        merged_output_dir = output_dir / "merged_model"
        model.save_pretrained_merged(str(merged_output_dir), tokenizer, save_method = "merged_16bit")
        os.sync()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen thinking model from ChatML session JSON files.")
    parser.add_argument("--model", type=str, required=True, help="Local model path, or a directory name under ./models.")
    parser.add_argument("--input-data", "--input-dir", type=str, required=True, help="Directory containing ChatML training session JSON files.")
    parser.add_argument("--max-seq-length", type=int, default=256000, help="Maximum sequence length for training")
    parser.add_argument("--chat-template", type=str, default="qwen3.6-thinking", help="Unsloth chat template name to apply to the tokenizer. Default uses the model/tokenizer native Qwen3.6 thinking template via get_chat_template; pass qwen3-thinking to force Unsloth's named Qwen3 template.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--qlora", action="store_true", help="Load model in 4-bit mode (for QLoRA training)")
    parser.add_argument("--save-merged", action="store_true", help="Save merged model in 16-bit format")
    parser.add_argument("--output-dir", "--output_dir", type=str, help="Directory for all training artifacts. Defaults to models/<model>_<timestamp>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_timestamp = int(time.time())
    output_name = Path(args.model).name
    output_dir = Path(args.output_dir) if args.output_dir else Path("models") / f"{output_name}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_output_dir = output_dir / "trainer_output"
    print(f"Training artifacts will be saved under: {output_dir.resolve()}")

    run_name = f"{args.model}-{int(time.time())}"
    if not args.disable_wandb:
        os.environ["WANDB_PROJECT"] = "qwen3-finetuning"
        wandb.init(name=run_name)
        report_to = "wandb"
    else:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = "none"

    model, tokenizer = create_model(
        args.model,
        args.max_seq_length,
        load_in_4bit=args.qlora,
        chat_template=args.chat_template,
    )
    print("Model and tokenizer loaded successfully.")
    dataset = create_dataset(args.input_data, tokenizer)
    print(f"Dataset loaded with {len(dataset)} training session(s).")
    start_time = time.time()
    train_model(model, tokenizer, dataset, args.max_seq_length, trainer_output_dir, report_to, run_name)
    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds. Saving model...")
    save_model(model, tokenizer, output_dir, args.save_merged)

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
