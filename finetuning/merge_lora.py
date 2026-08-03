import argparse
import builtins
import json
import os
import shutil
import tempfile
from pathlib import Path

# Keep this aligned with train.py: Unsloth's dynamic compiled kernels can be
# brittle on some stacks, and eager kernels are more reliable for export.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


def _identity_decorator(*args, **kwargs):
    if args and len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda value: value


def _interval_validator(**_kwargs):
    return lambda **kwargs: kwargs.get("default")


# Unsloth 2026.2.x can re-execute Transformers 5 config classes without these
# decorators in scope. Supplying them keeps the import path working.
builtins.auto_docstring = _identity_decorator
builtins.strict = _identity_decorator
builtins.interval = _interval_validator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Unsloth LoRA adapters into a base model."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Path or Hugging Face repo ID for the base model.",
    )
    parser.add_argument(
        "--output",
        help="Directory for the merged model. Defaults to <adapter-parent>/merged_model.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=20000,
        help="Maximum sequence length used when loading the model.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit mode before merging.",
    )
    parser.add_argument(
        "--save-method",
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "merged_4bit_forced", "mxfp4"],
        help="Unsloth save method for the merged output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    parser.add_argument(
        "--backend",
        default="unsloth",
        choices=["unsloth", "peft"],
        help=(
            "Merge backend. Use 'peft' for standard Transformers/PEFT adapters "
            "such as LLaVA; use 'unsloth' for Unsloth text-model adapters."
        ),
    )
    return parser.parse_args()


def resolve_model_ref(model_ref):
    path = Path(model_ref).expanduser()
    if path.exists():
        return str(path.resolve())
    return model_ref


def prepare_adapter_with_base(adapter_dir, base_model_ref, temp_dir):
    adapter_dir = Path(adapter_dir).expanduser()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json in adapter directory: {adapter_dir}"
        )

    temp_adapter_dir = Path(temp_dir) / adapter_dir.name
    shutil.copytree(adapter_dir, temp_adapter_dir)

    temp_config_path = temp_adapter_dir / "adapter_config.json"
    with temp_config_path.open("r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    adapter_config["base_model_name_or_path"] = base_model_ref

    with temp_config_path.open("w", encoding="utf-8") as file:
        json.dump(adapter_config, file, indent=2)
        file.write("\n")

    return temp_adapter_dir


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_output_dir(output_dir, overwrite, protected_paths):
    output_dir = Path(output_dir).expanduser()
    output_resolved = output_dir.resolve()
    for protected_path in protected_paths:
        if protected_path is None:
            continue
        protected_resolved = Path(protected_path).expanduser().resolve()
        if output_resolved == protected_resolved or is_relative_to(
            protected_resolved, output_resolved
        ):
            raise ValueError(
                f"Refusing to use protected path as output directory: {output_dir}"
            )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_base_rope_theta(base_model_ref):
    base_path = Path(base_model_ref).expanduser()
    if not base_path.is_dir():
        return None

    config_path = base_path / "config.json"
    if not config_path.is_file():
        return None

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("rope_theta")


def normalize_rope_config_for_save(model_config, base_model_ref):
    """Fill fields required by Transformers 5 RoPE validation after Unsloth patching."""
    rope_parameters = getattr(model_config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return

    rope_type = rope_parameters.get("rope_type") or rope_parameters.get("type")
    if rope_type != "linear" or rope_parameters.get("rope_theta") is not None:
        return

    rope_theta = getattr(model_config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = load_base_rope_theta(base_model_ref)
    if rope_theta is None:
        return

    rope_parameters["rope_theta"] = rope_theta
    setattr(model_config, "rope_parameters", rope_parameters)
    if getattr(model_config, "rope_theta", None) is None:
        setattr(model_config, "rope_theta", rope_theta)


def merge_with_unsloth(args, adapter_dir, output_dir, base_model_ref, base_model_path):
    with tempfile.TemporaryDirectory(prefix="unsloth_merge_") as temp_dir:
        temp_adapter_dir = prepare_adapter_with_base(
            adapter_dir=adapter_dir,
            base_model_ref=base_model_ref,
            temp_dir=temp_dir,
        )
        output_dir = ensure_output_dir(
            output_dir,
            args.overwrite,
            protected_paths=[adapter_dir, base_model_path],
        )

        import torch
        from unsloth import FastLanguageModel

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(f"Loading base model: {base_model_ref}")
        print(f"Loading LoRA adapter: {adapter_dir.resolve()}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(temp_adapter_dir),
            max_seq_length=args.max_seq_length,
            dtype=dtype,
            load_in_4bit=args.load_in_4bit,
            device_map={"": 0},
        )
        normalize_rope_config_for_save(model.config, base_model_ref)

        print(f"Saving merged model to: {output_dir.resolve()}")
        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method=args.save_method,
        )
        os.sync()

    print("Merge complete.")


def load_peft_base_model(base_model_ref, dtype):
    from transformers import AutoModelForCausalLM, LlavaForConditionalGeneration

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        return LlavaForConditionalGeneration.from_pretrained(base_model_ref, **load_kwargs)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(base_model_ref, **load_kwargs)


def save_processor_or_tokenizer(adapter_dir, base_model_ref, output_dir):
    from transformers import AutoProcessor, AutoTokenizer

    for loader in (AutoProcessor, AutoTokenizer):
        for source in (adapter_dir, base_model_ref):
            try:
                item = loader.from_pretrained(source)
            except Exception:
                continue
            item.save_pretrained(output_dir)
            return


def merge_with_peft(args, adapter_dir, output_dir, base_model_ref, base_model_path):
    output_dir = ensure_output_dir(
        output_dir,
        args.overwrite,
        protected_paths=[adapter_dir, base_model_path],
    )

    import torch
    from peft import PeftModel

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(f"Loading base model with Transformers: {base_model_ref}")
    base_model = load_peft_base_model(base_model_ref, dtype)
    print(f"Loading PEFT LoRA adapter: {adapter_dir.resolve()}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    print("Merging LoRA weights into the base model.")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir.resolve()}")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    save_processor_or_tokenizer(adapter_dir, base_model_ref, output_dir)
    os.sync()
    print("Merge complete.")


def main():
    args = parse_args()

    adapter_dir = Path(args.adapter).expanduser()
    output_dir = (
        Path(args.output).expanduser()
        if args.output
        else adapter_dir.parent / "merged_model"
    )
    base_model_ref = resolve_model_ref(args.base)
    base_path_candidate = Path(args.base).expanduser()
    base_model_path = base_path_candidate if base_path_candidate.exists() else None

    if args.backend == "peft":
        merge_with_peft(args, adapter_dir, output_dir, base_model_ref, base_model_path)
        return

    merge_with_unsloth(args, adapter_dir, output_dir, base_model_ref, base_model_path)


if __name__ == "__main__":
    main()
