import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset


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
    from unsloth.chat_templates import get_chat_template

    normalized = str(chat_template or "").strip().lower()
    if normalized in {"", "none"}:
        return tokenizer

    if normalized in {"tokenizer", "model", "native", "qwen3.6-thinking", "qwen3.6"}:
        if getattr(tokenizer, "chat_template", None):
            return get_chat_template(
                tokenizer,
                chat_template=(tokenizer.chat_template, "<|im_end|>"),
                map_eos_token=False,
            )
        return get_chat_template(tokenizer, chat_template="qwen3-thinking")

    return get_chat_template(tokenizer, chat_template=chat_template)


def parse_role_map(role_map_entries: list[str]) -> dict[str, str]:
    role_map = {}
    for entry in role_map_entries:
        if "=" not in entry:
            raise ValueError(f"Role map entries must be formatted as source=target, got: {entry!r}")
        source, target = (part.strip() for part in entry.split("=", 1))
        if not source or not target:
            raise ValueError(f"Role map entries must include source and target roles, got: {entry!r}")
        role_map[source] = target
    return role_map


def load_chatml_session(path: Path, role_map: dict[str, str] | None = None) -> dict[str, Any]:
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
        role = (role_map or {}).get(role, role)
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


def load_chatml_sessions(input_dir: str | Path, role_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"--input-data must be a directory containing ChatML session JSON files: {input_dir}")

    input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    if not input_files:
        raise ValueError(f"No .json session files found in {input_dir}")

    return [load_chatml_session(path, role_map) for path in input_files]


def formatting_prompts_func(examples: dict[str, list[Any]], tokenizer: Any) -> dict[str, list[str]]:
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            preserve_thinking=True,
        )
        for messages in examples["messages"]
    ]
    return {"text": texts}


def prepare_dataset(input_dir: str | Path, tokenizer: Any, role_map: dict[str, str] | None = None) -> Dataset:
    dataset = Dataset.from_list(load_chatml_sessions(input_dir, role_map))
    return dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )


def create_debug_tokenizer(model_name: str, chat_template: str) -> Any:
    from unsloth.chat_templates import get_chat_template as _
    from transformers import AutoTokenizer

    model_path = resolve_local_model_path(model_name)
    print(f"Using tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    return apply_training_chat_template(tokenizer, chat_template)


def preview_dataset(dataset: Dataset, limit: int, print_full_text: bool) -> None:
    print(f"Prepared dataset with {len(dataset)} row(s).")
    for index, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        text = row["text"]
        print(f"\n--- Rendered sample {index} ({len(text)} chars) ---")
        print(text if print_full_text else text[:2000])
        if not print_full_text and len(text) > 2000:
            print(f"\n... truncated {len(text) - 2000} chars; pass --full-text to print everything.")


def write_jsonl(dataset: Dataset, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
    print(f"Wrote rendered dataset JSONL to: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug ChatML session loading and tokenizer.apply_chat_template output."
    )
    parser.add_argument("--model", type=str, required=True, help="Local model path, or a directory name under ./models.")
    parser.add_argument(
        "--input-data",
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing ChatML session JSON files.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="qwen3.6-thinking",
        help="Template to pass through apply_training_chat_template.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of rendered samples to preview.")
    parser.add_argument("--full-text", action="store_true", help="Print complete rendered sample text.")
    parser.add_argument("--output-jsonl", type=str, help="Optional path to save rendered {'text': ...} rows.")
    parser.add_argument(
        "--role-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Optional role mapping before validation, for example --role-map therapist=assistant --role-map patient=user.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print source file names and message counts before applying the chat template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = create_debug_tokenizer(args.model, args.chat_template)
    role_map = parse_role_map(args.role_map)

    if args.show_raw:
        sessions = load_chatml_sessions(args.input_data, role_map)
        print(f"Loaded {len(sessions)} raw ChatML session(s).")
        for index, session in enumerate(sessions[: args.limit]):
            print(f"{index}: {session['source_file']} ({len(session['messages'])} messages)")

    dataset = prepare_dataset(args.input_data, tokenizer, role_map)
    breakpoint()
    preview_dataset(dataset, args.limit, args.full_text)

    if args.output_jsonl:
        write_jsonl(dataset, args.output_jsonl)


if __name__ == "__main__":
    main()
