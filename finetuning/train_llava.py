import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100

USER_ROLES = {"human", "user", "patient", "client"}
ASSISTANT_ROLES = {"gpt", "assistant", "therapist", "counselor", "counsellor"}
SYSTEM_ROLES = {"system"}

LORA_TARGET_MODULES = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def default_run_dir(output: str | Path, run_name: str) -> Path:
    output = resolve_path(output)
    if output.name == run_name:
        return output
    return output / run_name


def load_chatml_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or "messages" not in payload:
        raise ValueError(f"{path}: expected a ChatML JSON object with a 'messages' list.")
    messages = payload["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}: 'messages' must be a non-empty list.")
    return messages


def normalize_content(path: Path, message_index: int, content: Any) -> str:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")).strip())
        text = "\n".join(part for part in text_parts if part)
    else:
        text = ""

    if not text:
        raise ValueError(f"{path}: messages[{message_index}] has empty text content.")
    if "<image>" in text:
        raise ValueError(f"{path}: messages[{message_index}] contains an image token, but this script is text-only.")
    return text


def normalize_role(path: Path, message_index: int, role: Any) -> str:
    role = str(role or "").strip().lower()
    if role in SYSTEM_ROLES:
        return "system"
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    raise ValueError(f"{path}: messages[{message_index}] has unsupported role {role!r}.")


def to_llava_messages(path: Path, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llava_messages = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{path}: messages[{index}] must be a JSON object.")
        role = normalize_role(path, index, message.get("role"))
        content = normalize_content(path, index, message.get("content"))
        llava_messages.append(
            {
                "role": role,
                "content": [{"type": "text", "text": content}],
            }
        )

    if not any(message["role"] == "assistant" for message in llava_messages):
        raise ValueError(f"{path}: no assistant message found to train on.")
    return llava_messages


class ChatMLLlavaTextDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        tokenizer: Any,
        max_length: int,
        max_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        input_dir = resolve_path(input_dir)
        if not input_dir.is_dir():
            raise ValueError(f"--input must be a directory containing per-turn ChatML JSON files: {input_dir}")

        paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
        if max_samples is not None:
            paths = paths[:max_samples]
        if not paths:
            raise ValueError(f"No .json files found in {input_dir}.")

        self.samples = [to_llava_messages(path, load_chatml_file(path)) for path in paths]
        self.source_files = [path.name for path in paths]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer.apply_chat_template(
            self.samples[index],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        assistant_mask = list(encoded["assistant_masks"])

        if self.tokenizer.eos_token_id is not None and (
            not input_ids or input_ids[-1] != self.tokenizer.eos_token_id
        ):
            eos_is_assistant = 1 if self.samples[index][-1]["role"] == "assistant" else 0
            if len(input_ids) >= self.max_length:
                input_ids[-1] = self.tokenizer.eos_token_id
                attention_mask[-1] = 1
                assistant_mask[-1] = eos_is_assistant
            else:
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)
                assistant_mask.append(eos_is_assistant)

        labels = [
            token_id if is_assistant else IGNORE_INDEX
            for token_id, is_assistant in zip(input_ids, assistant_mask)
        ]
        if all(label == IGNORE_INDEX for label in labels):
            raise ValueError(f"{self.source_files[index]}: assistant text was fully truncated.")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [instance["attention_mask"] for instance in instances],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class LlavaTextTrainer(Trainer):
    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum" if num_items_in_batch is not None else "mean",
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch.to(loss.device)
        return (loss, outputs) if return_outputs else loss


def load_processor_and_tokenizer(model_path: str | Path, max_length: int) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "chat_template", None) is None and getattr(processor, "chat_template", None):
        tokenizer.chat_template = processor.chat_template
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(f"{model_path} does not provide a LLaVA chat template.")
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer


def create_model(args: argparse.Namespace) -> Any:
    dtype = training_dtype()
    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if args.qlora else None,
    )
    model.config.use_cache = False

    for parameter in model.vision_tower.parameters():
        parameter.requires_grad = False
    for parameter in model.multi_modal_projector.parameters():
        parameter.requires_grad = False

    if args.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def training_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def save_model(
    model: Any,
    tokenizer: Any,
    processor: Any,
    output_dir: Path,
    base_model_path: str,
    save_merged: bool,
) -> None:
    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.tokenizer = tokenizer
    processor.save_pretrained(output_dir)

    if not save_merged:
        return

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_output_dir = output_dir / "merged_model"
    print(f"Saving merged model to {merged_output_dir}")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=training_dtype(),
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base_model, output_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model.config.use_cache = True
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    processor.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)


def train(args: argparse.Namespace) -> None:
    run_timestamp = int(time.time())
    run_name = args.exp_name or f"{args.model}-{run_timestamp}"
    output_dir = default_run_dir(args.output, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = "none"
    else:
        os.environ["WANDB_PROJECT"] = "EMNLP26 - ODRA"
        wandb.init(name=run_name, group=args.group)
        report_to = "wandb"

    processor, tokenizer = load_processor_and_tokenizer(args.model, args.max_length)
    dataset = ChatMLLlavaTextDataset(args.input, tokenizer, args.max_length, args.max_samples)
    print(f"Loaded {len(dataset)} text-only LLaVA training sample(s) from {resolve_path(args.input)}.")
    if args.inspect_sample is not None:
        inspect_training_sample(dataset, tokenizer, args.inspect_sample, args.breakpoint_after_inspect)
        if not args.disable_wandb:
            wandb.finish()
        return

    model = create_model(args)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = LlavaTextTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_state()

    if args.save_merged:
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_model(model, tokenizer, processor, output_dir, args.model, args.save_merged)

    if not args.disable_wandb:
        wandb.finish()


def inspect_training_sample(
    dataset: ChatMLLlavaTextDataset,
    tokenizer: Any,
    sample_index: int,
    stop_in_debugger: bool,
) -> None:
    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError(f"--inspect-sample must be between 0 and {len(dataset) - 1}.")

    sample = dataset[sample_index]
    labels = sample["labels"]
    trainable_token_ids = labels[labels != IGNORE_INDEX].tolist()

    print(f"Source file: {dataset.source_files[sample_index]}")
    print("\nRaw normalized messages:")
    for message in dataset.samples[sample_index]:
        text = message["content"][0]["text"]
        preview = text[:1000] + ("..." if len(text) > 1000 else "")
        print(f"\n[{message['role']}]\n{preview}")

    print("\nFull formatted text:")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))

    print("\nAssistant tokens used as labels:")
    print(tokenizer.decode(trainable_token_ids, skip_special_tokens=False))
    print(
        "\nTensor shapes: "
        f"input_ids={tuple(sample['input_ids'].shape)}, "
        f"attention_mask={tuple(sample['attention_mask'].shape)}, "
        f"labels={tuple(sample['labels'].shape)}"
    )
    print(f"Trainable label tokens: {len(trainable_token_ids)}")

    if stop_in_debugger:
        breakpoint()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-only QLoRA fine-tuning for LLaVA from ChatML JSON files.")
    parser.add_argument("--input", type=str, default="mirror/it_turns", help="Directory containing per-turn ChatML JSON files.")
    parser.add_argument("--output", type=str, default="ft_models", help="Output folder for fine-tuned adapters.")
    parser.add_argument("--model", type=str, default="models/Llava-1.5v-7B", help="Local LLaVA model path.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum training sequence length.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on loaded JSON files.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional max optimizer steps for smoke tests.")
    parser.add_argument("--inspect-sample", type=int, default=None, help="Print one processed dataset sample and exit before model loading.")
    parser.add_argument("--breakpoint-after-inspect", action="store_true", help="Enter pdb after --inspect-sample prints the processed sample.")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", type=str, default="steps")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="Disable 4-bit QLoRA loading.")
    parser.add_argument("--save-merged", action="store_true", help="Also save a standalone merged model under <run_dir>/merged_model.")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--group", type=str, default=None, help="Weights & Biases run group.")
    parser.add_argument("--exp-name", type=str, default=None, help="Run name and output subdirectory.")
    parser.set_defaults(qlora=True, gradient_checkpointing=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
