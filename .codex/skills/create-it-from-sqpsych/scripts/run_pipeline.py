#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
  return Path(__file__).resolve().parents[4]


def run(command: list[str], dry_run: bool) -> None:
  print(f"$ {shlex.join(command)}", flush=True)
  if not dry_run:
    subprocess.run(command, check=True)


def replace_directory(path: Path) -> None:
  if path.exists():
    shutil.rmtree(path)


def create_it_from_sqpsych(
  input_root: Path,
  output_root: Path,
  system_prompt: Path,
  n_samples: int,
  seed: int,
  dry_run: bool,
) -> None:
  root = project_root()
  input_root = input_root.resolve()
  output_root = output_root.resolve()
  system_prompt = system_prompt.resolve()

  if not input_root.is_dir():
    raise ValueError(f"--input must be a dataset directory: {input_root}")
  if not list(input_root.rglob("*.parquet")):
    raise ValueError(f"No Parquet files found under: {input_root}")
  if not system_prompt.is_file():
    raise ValueError(f"--system-prompt must be a file: {system_prompt}")
  if n_samples < 1:
    raise ValueError("--n-samples must be at least 1.")

  json_dir = output_root / "json"
  sample_dir = output_root / "json_150"
  turns_dir = output_root / "it_turns"
  targets = (json_dir, sample_dir, turns_dir)

  print("Output directories to replace:", flush=True)
  for target in targets:
    print(f"  {target}", flush=True)

  commands = [
    [
      sys.executable,
      str(root / "scripts/data/sqpsych_to_cactus.py"),
      "--input",
      str(input_root),
      "--output",
      str(json_dir),
    ],
    [
      sys.executable,
      str(root / "scripts/data/random_from_sqpsych.py"),
      "--input",
      str(json_dir),
      "--output",
      str(sample_dir),
      "--n-samples",
      str(n_samples),
      "--seed",
      str(seed),
    ],
    [
      sys.executable,
      str(root / "scripts/ft/create_it_dataset_from_raw.py"),
      "--input-dir",
      str(sample_dir),
      "--output-dir",
      str(turns_dir),
      "--system-prompt",
      str(system_prompt),
      "--mode",
      "turns",
    ],
  ]

  for target, command in zip(targets, commands):
    if not dry_run:
      replace_directory(target)
    run(command, dry_run)

  if dry_run:
    return

  converted_count = len(list(json_dir.glob("*.json")))
  sample_files = list(sample_dir.glob("*.json"))
  turn_count = len(list(turns_dir.glob("*.json")))

  if converted_count < 1:
    raise RuntimeError("Conversion produced no JSON files.")
  if len(sample_files) != n_samples:
    raise RuntimeError(f"Expected {n_samples} sampled files, found {len(sample_files)}.")
  if any(not path.name.endswith("_train.json") for path in sample_files):
    raise RuntimeError("The sampled directory contains a non-training conversation.")
  if turn_count < 1:
    raise RuntimeError("Instruction-tuning generation produced no turn files.")

  print("\nPipeline complete:")
  print(f"  Converted conversations: {converted_count} ({json_dir})")
  print(f"  Sampled training conversations: {len(sample_files)} ({sample_dir})")
  print(f"  Instruction-tuning examples: {turn_count} ({turns_dir})")
  print(f"  Seed: {seed}")
  print(f"  System prompt: {system_prompt}")


def main() -> int:
  root = project_root()
  parser = argparse.ArgumentParser(
    description="Convert SQPsychConv Parquet data and create a sampled instruction-tuning dataset."
  )
  parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="Dataset root containing Parquet files recursively.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    help="Root for json, json_150, and it_turns. Defaults to --input.",
  )
  parser.add_argument("--n-samples", type=int, default=150)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--system-prompt",
    type=Path,
    default=root / "ft_prompts/sqpsych.txt",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print replacement paths and commands without changing files.",
  )
  args = parser.parse_args()

  try:
    create_it_from_sqpsych(
      input_root=args.input,
      output_root=args.output or args.input,
      system_prompt=args.system_prompt,
      n_samples=args.n_samples,
      seed=args.seed,
      dry_run=args.dry_run,
    )
  except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
