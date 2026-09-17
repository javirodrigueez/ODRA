#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
from typing import Any
from urllib.parse import urlparse
import urllib.request

import yaml


CTRS_KEYS = [
  "general_1_understanding",
  "general_2_interpersonal_effectiveness",
  "general_3_collaboration",
  "CBT_1_guided_discovery",
  "CBT_2_focus",
  "CBT_3_strategy",
]


def repo_root() -> Path:
  for path in Path(__file__).resolve().parents:
    if (path / "agent_simulation.py").is_file():
      return path
  raise RuntimeError("Could not find the ReasonCBT repository root.")


def resolve_path(path: str | Path, root: Path) -> Path:
  path = Path(path)
  return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_gpus(value: str | None, evaluations: int) -> list[str]:
  if evaluations == 1 and value:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
      raise ValueError("At least one GPU ID is required.")
    return [",".join(gpus)]
  if value:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
  else:
    gpus = [str(index) for index in range(evaluations)]
  if len(gpus) != evaluations:
    raise ValueError(f"Expected {evaluations} GPU IDs, received {len(gpus)}: {gpus}")
  return gpus


def port_is_open(port: int) -> bool:
  with socket.socket() as connection:
    connection.settimeout(1)
    return connection.connect_ex(("127.0.0.1", port)) == 0


def write_config(path: Path, model: Path, profiles: Path, output: Path, port: int) -> None:
  quote = lambda value: json.dumps(str(value))
  content = f"""therapist:
  type: local
  base_url: http://localhost:{port}/v1
  model_id: {quote(model)}
  max_model_len: 2000
  temperature: 0.7

patient:
  type: deepseek
  base_url: https://openrouter.ai/api/v1
  model_id: deepseek/deepseek-v3.2
  profiles: {quote(profiles)}
  prompt_variant: base
  temperature: 0.7

topic_exploration:
  patient_classifier_path: BAAI/bge-reranker-v2-m3
  patient_wikipedia_dir: data/wikipedias

env:
  max_turns: 100
  min_turns: 60
  output_dir: {quote(output)}
  verbose: true
  resume_file:
  resistance: false
  resistance_handling: none
  long_session_turn_threshold: 70
"""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(content, encoding="utf-8")


def load_config_run(path: Path, root: Path, gpu: str) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as file:
    payload = yaml.safe_load(file)
  if not isinstance(payload, dict):
    raise ValueError(f"Config must contain a YAML mapping: {path}")

  therapist = payload.get("therapist")
  environment = payload.get("env")
  if not isinstance(therapist, dict) or not isinstance(environment, dict):
    raise ValueError(f"Config must contain therapist and env mappings: {path}")

  model_id = therapist.get("model_id")
  base_url = therapist.get("base_url")
  output_dir = environment.get("output_dir")
  if not all(isinstance(value, str) and value.strip() for value in (model_id, base_url, output_dir)):
    raise ValueError(f"Config requires therapist.model_id, therapist.base_url, and env.output_dir: {path}")

  parsed_url = urlparse(base_url)
  if parsed_url.hostname not in {"localhost", "127.0.0.1"} or parsed_url.port is None:
    raise ValueError(f"Config therapist.base_url must use localhost with an explicit port: {path}")

  configured_model = resolve_path(model_id, root)
  model = configured_model
  if not (model / "config.json").is_file() and (model / "merged_model" / "config.json").is_file():
    model = model / "merged_model"

  return {
    "label": path.stem,
    "output": resolve_path(output_dir, root),
    "config": path,
    "model": model,
    "served_model": model_id,
    "port": parsed_url.port,
    "gpu": gpu,
    "retried_profiles": 0,
  }


def manifest_path(output: Path) -> Path:
  prefix = output.as_posix().strip("/").replace("/", "_") or "output"
  return output / f"{prefix}.json"


def run_logged(command: list[str], log_path: Path, root: Path) -> tuple[int, str]:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  started = time.monotonic()
  with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
  elapsed = time.monotonic() - started
  return process.returncode, f"{elapsed:.1f}s"


def launch_server(
  root: Path,
  model: Path,
  served_model: str,
  tokenizer: Path | None,
  gpu: str,
  gpu_memory_utilization: float,
  port: int,
  log_path: Path,
) -> subprocess.Popen[str]:
  command = [
    "vllm",
    "serve",
    str(model),
    "--served-model-name",
    served_model,
    "--port",
    str(port),
    "--quantization",
    "fp8",
    "--kv-cache-dtype",
    "fp8",
    "--attention-backend",
    "TRITON_ATTN",
    "--gpu-memory-utilization",
    str(gpu_memory_utilization),
    "--language-model-only",
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--max-num-seqs",
    "150",
  ]
  tensor_parallel_size = len(gpu.split(","))
  if tensor_parallel_size > 1:
    command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
  if tokenizer is not None:
    command[3:3] = ["--tokenizer", str(tokenizer)]
  environment = os.environ.copy()
  environment["CUDA_VISIBLE_DEVICES"] = gpu
  environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  log = log_path.open("w", encoding="utf-8")
  process = subprocess.Popen(
    command,
    cwd=root,
    env=environment,
    stdout=log,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
  )
  log.close()
  return process


def wait_for_server(process: subprocess.Popen[str], served_model: str, port: int, timeout: int) -> None:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if process.poll() is not None:
      raise RuntimeError(f"vLLM on port {port} exited with status {process.returncode}.")
    try:
      with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
        healthy = response.status == 200
      with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
        payload = json.load(response)
      model_ids = [item.get("id") for item in payload.get("data", [])]
      if healthy and served_model in model_ids:
        return
    except Exception:
      pass
    time.sleep(2)
  raise TimeoutError(f"vLLM on port {port} did not become ready within {timeout}s.")


def stop_servers(servers: list[tuple[subprocess.Popen[str], int]]) -> None:
  for process, _ in servers:
    if process.poll() is None:
      os.killpg(process.pid, signal.SIGTERM)
  for process, _ in servers:
    if process.poll() is None:
      try:
        process.wait(timeout=30)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
  deadline = time.monotonic() + 30
  while time.monotonic() < deadline and any(port_is_open(port) for _, port in servers):
    time.sleep(1)
  open_ports = [port for _, port in servers if port_is_open(port)]
  if open_ports:
    raise RuntimeError(f"vLLM ports remained open after shutdown: {open_ports}")


def simulation_command(
  root: Path,
  config: Path,
  profiles: Path,
  n_profiles: int,
  n_processes: int,
  simulation_script: str = "agent_simulation.py",
) -> list[str]:
  return [
    "python",
    str(resolve_path(simulation_script, root)),
    "--cfg",
    str(config),
    "--profiles-path",
    str(profiles),
    "--n-profiles",
    str(n_profiles),
    "--multiprocessing",
    "--n-processes",
    str(n_processes),
  ]


def load_simulation_status(output: Path) -> tuple[dict[str, Any], list[str]]:
  path = manifest_path(output)
  with path.open("r", encoding="utf-8") as file:
    manifest = json.load(file)
  failed = [
    index
    for index, status in manifest.get("profiles", {}).items()
    if status.get("executed") and not status.get("success")
  ]
  return manifest, failed


def validate_simulation(output: Path, expected: int) -> dict[str, Any]:
  manifest, failed = load_simulation_status(output)
  metadata = manifest.get("metadata", {})
  json_count = len(list((output / "json").glob("*.json")))
  if metadata.get("executed_count") != expected:
    raise RuntimeError(f"{output}: expected {expected} executed profiles, got {metadata.get('executed_count')}.")
  if metadata.get("success_count") != expected or failed:
    raise RuntimeError(f"{output}: expected {expected} successful profiles; failures: {failed}.")
  if json_count != expected:
    raise RuntimeError(f"{output}: expected {expected} conversation JSONs, got {json_count}.")
  return manifest


def dry_run_request_count(log_path: Path) -> tuple[int, int, int]:
  text = log_path.read_text(encoding="utf-8")
  total_match = re.search(r"^Total requests: (\d+)$", text, flags=re.MULTILINE)
  main_match = re.search(r"^main: .*? (\d+) total request\(s\)", text, flags=re.MULTILINE)
  panas_match = re.search(r"^panas_before: .*? (\d+) total request\(s\)", text, flags=re.MULTILINE)
  if not total_match or not main_match or not panas_match:
    raise RuntimeError(f"Could not parse batch dry-run output: {log_path}")
  return int(total_match.group(1)), int(main_match.group(1)), int(panas_match.group(1))


def turn_metrics(log_path: Path) -> tuple[float, float]:
  text = log_path.read_text(encoding="utf-8")
  turns_match = re.search(r"^Overall Average Turns per Session: ([0-9.]+)$", text, flags=re.MULTILINE)
  words_match = re.search(
    r"^Average Therapist Words per Therapist Turn: ([0-9.]+)$",
    text,
    flags=re.MULTILINE,
  )
  if not turns_match or not words_match:
    raise RuntimeError(f"Could not parse turn metrics output: {log_path}")
  return float(turns_match.group(1)), float(words_match.group(1))


def validate_ctrs(eval_dir: Path, expected: int) -> tuple[int, float]:
  paths = sorted((eval_dir / "ctrs").glob("*.json"))
  if len(paths) != expected:
    raise RuntimeError(f"{eval_dir}: expected {expected} CTRS files, got {len(paths)}.")
  total_cost = 0.0
  for path in paths:
    with path.open("r", encoding="utf-8") as file:
      payload = json.load(file)
    values = [payload.get(key) for key in CTRS_KEYS]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
      raise RuntimeError(f"Incomplete CTRS fields: {path}")
    cost = payload.get("cost")
    if not isinstance(cost, (int, float)) or not math.isfinite(float(cost)):
      raise RuntimeError(f"Missing numeric CTRS cost: {path}")
    total_cost += float(cost)
  return len(paths), total_cost


def print_plan(
  tokenizer: Path | None,
  output: Path,
  runs: list[dict[str, Any]],
  n_profiles: int,
  n_processes: int,
  eval_max_iter: int,
  simulation_script: str,
) -> None:
  print(f"Tokenizer: {tokenizer or 'checkpoint default'}")
  print(f"Summary output: {output}")
  print(f"Parallel evaluations: {len(runs)}")
  for run in runs:
    print(
      f"{run['label']}: checkpoint {run['model']}, GPU {run['gpu']}, "
      f"served as {run['served_model']}, port {run['port']}, "
      f"output {run['output']}, config {run['config']}"
    )
  print(f"Simulation: {n_profiles} profiles, {n_processes} processes per run")
  print(f"Simulation script: {simulation_script}")
  print(f"Evaluation: CTRS only, max_iter={eval_max_iter}, expected requests/run={n_profiles * len(CTRS_KEYS)}")


def run_pipeline(args: argparse.Namespace) -> Path | None:
  root = repo_root()
  if args.resume and args.overwrite:
    raise ValueError("--resume and --overwrite cannot be combined.")
  tokenizer = resolve_path(args.tokenizer, root) if args.tokenizer else None
  output = resolve_path(args.output, root)
  profiles = resolve_path(args.profiles_path, root)

  config_paths = [resolve_path(path, root) for path in (args.configs or [])]
  if config_paths:
    evaluations = len(config_paths)
    if args.evaluations is not None and args.evaluations != evaluations:
      raise ValueError(f"--evaluations must match the {evaluations} supplied configs.")
  else:
    evaluations = args.evaluations or 1
    if not args.input:
      raise ValueError("Provide --input, or use --configs for distinct YAML configurations.")

  if evaluations < 1 or args.n_profiles < 1 or args.n_processes < 1:
    raise ValueError("--evaluations, --n-profiles, and --n-processes must be positive.")
  gpus = parse_gpus(args.gpus, evaluations)
  if tokenizer is not None and not tokenizer.exists():
    raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer}")
  if not profiles.is_file():
    raise FileNotFoundError(f"Profiles file does not exist: {profiles}")

  configs_dir = output / "configs"
  runs: list[dict[str, Any]] = []
  if config_paths:
    for config, gpu in zip(config_paths, gpus):
      if not config.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config}")
      runs.append(load_config_run(config, root, gpu))
  else:
    model = resolve_path(args.input, root)
    for index in range(evaluations):
      label = f"run-{index + 1:03d}"
      runs.append(
        {
          "label": label,
          "output": output / label,
          "config": configs_dir / f"{label}.yaml",
          "model": model,
          "served_model": str(model),
          "port": args.base_port + index,
          "gpu": gpus[index],
          "retried_profiles": 0,
        }
      )

  if len({run["label"] for run in runs}) != evaluations:
    raise ValueError("Config filenames must have distinct stems.")
  if len({run["port"] for run in runs}) != evaluations:
    raise ValueError("Every evaluation config must use a distinct therapist port.")
  if len({run["output"] for run in runs}) != evaluations:
    raise ValueError("Every evaluation config must use a distinct env.output_dir.")
  for run in runs:
    if not run["model"].is_dir():
      raise FileNotFoundError(f"Checkpoint directory does not exist: {run['model']}")

  print_plan(
    tokenizer,
    output,
    runs,
    args.n_profiles,
    args.n_processes,
    args.eval_max_iter,
    args.simulation_script,
  )
  if args.dry_run:
    return None

  if output.exists() and not output.is_dir():
    raise FileExistsError(f"Output exists and is not a directory: {output}")
  if output.exists() and any(output.iterdir()):
    if not args.overwrite and not args.resume:
      raise FileExistsError(f"Output is not empty: {output}. Pass --overwrite only with explicit authorization.")
    if args.overwrite:
      shutil.rmtree(output)
  if config_paths:
    for run in runs:
      run_output = run["output"]
      if run_output.exists() and not run_output.is_dir():
        raise FileExistsError(f"Run output exists and is not a directory: {run_output}")
      if run_output.exists() and any(run_output.iterdir()):
        if not args.overwrite and not args.resume:
          raise FileExistsError(
            f"Run output is not empty: {run_output}. Pass --overwrite only with explicit authorization."
          )
        if args.overwrite:
          shutil.rmtree(run_output)
      if args.resume and not manifest_path(run_output).is_file():
        raise FileNotFoundError(f"Resume manifest does not exist: {manifest_path(run_output)}")
  output.mkdir(parents=True, exist_ok=True)
  logs_dir = output / "logs"
  if not config_paths:
    for run in runs:
      write_config(run["config"], run["model"], profiles, run["output"], run["port"])
  for run in runs:
    if port_is_open(run["port"]):
      raise RuntimeError(f"Port {run['port']} is already open.")

  servers: list[tuple[subprocess.Popen[str], int]] = []
  gpu_run_counts = {gpu: gpus.count(gpu) for gpu in set(gpus)}
  try:
    print("Launching vLLM servers...")
    for run in runs:
      process = launch_server(
        root,
        run["model"],
        run["served_model"],
        tokenizer,
        run["gpu"],
        0.95 / gpu_run_counts[run["gpu"]],
        run["port"],
        logs_dir / f"{run['label']}-vllm.log",
      )
      servers.append((process, run["port"]))
    with ThreadPoolExecutor(max_workers=evaluations) as pool:
      futures = [
        pool.submit(wait_for_server, process, run["served_model"], run["port"], args.server_timeout)
        for (process, _), run in zip(servers, runs)
      ]
      for future in futures:
        future.result()
    print("All server gates passed. Running simulations...")

    with ThreadPoolExecutor(max_workers=evaluations) as pool:
      futures = []
      for run in runs:
        command = simulation_command(root, run["config"], profiles, args.n_profiles, args.n_processes, args.simulation_script)
        if args.resume:
          _, failed = load_simulation_status(run["output"])
          run["retried_profiles"] = len(failed)
          command.extend(["--resume-from-json", str(manifest_path(run["output"])), "--retry-failed"])
        futures.append(pool.submit(run_logged, command, logs_dir / f"{run['label']}-simulation.log", root))
      simulation_results = [future.result() for future in futures]
    for run, (returncode, elapsed) in zip(runs, simulation_results):
      if returncode:
        raise RuntimeError(f"{run['label']} simulation exited {returncode}; see logs ({elapsed}).")

    retry_runs = []
    for run in runs:
      _, failed = load_simulation_status(run["output"])
      if failed:
        retry_runs.append((run, len(failed)))
    if args.stop_on_conversation_error and retry_runs:
      failures = {run["label"]: count for run, count in retry_runs}
      raise RuntimeError(f"Conversation errors detected; stopping before retry or CTRS evaluation: {failures}")
    if retry_runs:
      print(f"Retrying failed profiles once in {len(retry_runs)} run(s)...")
      with ThreadPoolExecutor(max_workers=len(retry_runs)) as pool:
        futures = []
        for run, failure_count in retry_runs:
          run["retried_profiles"] = failure_count
          command = simulation_command(root, run["config"], profiles, failure_count, args.n_processes, args.simulation_script)
          command.extend(["--resume-from-json", str(manifest_path(run["output"])), "--retry-failed"])
          futures.append(pool.submit(run_logged, command, logs_dir / f"{run['label']}-simulation-retry.log", root))
        retry_results = [future.result() for future in futures]
      for (run, _), (returncode, elapsed) in zip(retry_runs, retry_results):
        if returncode:
          raise RuntimeError(f"{run['label']} retry exited {returncode}; see logs ({elapsed}).")

    for run in runs:
      validate_simulation(run["output"], args.n_profiles)
    print("All simulation gates passed.")
  finally:
    if servers:
      print("Stopping vLLM servers...")
      stop_servers(servers)

  print("Running CTRS batch dry-run gates...")
  expected_requests = args.n_profiles * len(CTRS_KEYS)
  with ThreadPoolExecutor(max_workers=evaluations) as pool:
    futures = []
    for run in runs:
      eval_dir = run["output"] / "evaluation"
      command = [
        "python",
        str(root / "evaluation" / "run_batch_metrics.py"),
        "--dry-run",
        "--exc-panas",
        "--exc-wai",
        str(run["output"] / "json"),
        str(eval_dir),
        str(args.eval_max_iter),
      ]
      futures.append(pool.submit(run_logged, command, logs_dir / f"{run['label']}-ctrs-dry-run.log", root))
    dry_results = [future.result() for future in futures]
  for run, (returncode, _) in zip(runs, dry_results):
    if returncode:
      raise RuntimeError(f"{run['label']} CTRS dry run exited {returncode}.")
    total, main, panas = dry_run_request_count(logs_dir / f"{run['label']}-ctrs-dry-run.log")
    if (total, main, panas) != (expected_requests, expected_requests, 0):
      raise RuntimeError(
        f"{run['label']} CTRS gate mismatch: total={total}, main={main}, panas={panas}, expected={expected_requests}."
      )

  print("All dry-run gates passed. Submitting CTRS evaluations...")
  with ThreadPoolExecutor(max_workers=evaluations) as pool:
    futures = []
    for run in runs:
      eval_dir = run["output"] / "evaluation"
      command = [
        "python",
        str(root / "evaluation" / "run_batch_metrics.py"),
        "--exc-panas",
        "--exc-wai",
        str(run["output"] / "json"),
        str(eval_dir),
        str(args.eval_max_iter),
      ]
      futures.append(pool.submit(run_logged, command, logs_dir / f"{run['label']}-ctrs.log", root))
    eval_results = [future.result() for future in futures]
  for run, (returncode, elapsed) in zip(runs, eval_results):
    if returncode:
      raise RuntimeError(f"{run['label']} CTRS evaluation exited {returncode}; see logs ({elapsed}).")

  summaries = []
  for run in runs:
    eval_dir = run["output"] / "evaluation"
    count, direct_cost = validate_ctrs(eval_dir, args.n_profiles)
    excluded_files = [
      path
      for group in ("wai", "panas_before", "panas_after")
      for path in (eval_dir / group).glob("*")
    ]
    if excluded_files:
      raise RuntimeError(f"Excluded metric outputs were created: {excluded_files[:5]}")
    metrics_path = eval_dir / "ctrs_metrics.json"
    command = [
      "python",
      str(root / "evaluation" / "rating_metrics.py"),
      "--eval-dir",
      str(eval_dir),
      "--output-json",
      str(metrics_path),
      "--exc-wai",
      "--exc-panas",
    ]
    returncode, _ = run_logged(command, logs_dir / f"{run['label']}-aggregate.log", root)
    if returncode:
      raise RuntimeError(f"{run['label']} aggregation failed.")
    turns_log = logs_dir / f"{run['label']}-turns.log"
    returncode, _ = run_logged(
      [
        "python",
        str(root / "evaluation" / "turns_and_closings.py"),
        "--eval-dir",
        str(run["output"] / "json"),
      ],
      turns_log,
      root,
    )
    if returncode:
      raise RuntimeError(f"{run['label']} turn analysis failed.")
    average_turns, average_therapist_words = turn_metrics(turns_log)
    with metrics_path.open("r", encoding="utf-8") as file:
      metrics = json.load(file)
    if metrics.get("ctrs", {}).get("count") != args.n_profiles:
      raise RuntimeError(f"{run['label']} aggregate count does not match {args.n_profiles}.")
    aggregate_cost = float(metrics.get("cost", {}).get("total", -1))
    if not math.isclose(aggregate_cost, direct_cost, rel_tol=1e-9, abs_tol=1e-9):
      raise RuntimeError(f"{run['label']} aggregate cost {aggregate_cost} does not match {direct_cost}.")
    summaries.append(
      {
        "run": run["label"],
        "config": str(run["config"]),
        "checkpoint": str(run["model"]),
        "served_model": run["served_model"],
        "output_dir": str(run["output"]),
        "evaluation_dir": str(eval_dir),
        "metrics_path": str(metrics_path),
        "conversation_count": args.n_profiles,
        "ctrs_file_count": count,
        "retried_profiles": run["retried_profiles"],
        "average_turns": average_turns,
        "average_therapist_words_per_turn": average_therapist_words,
        "ctrs": {key: metrics["ctrs"][key] for key in CTRS_KEYS},
        "cost": direct_cost,
      }
    )

  aggregate = {}
  for key in CTRS_KEYS:
    values = [float(run["ctrs"][key]) for run in summaries]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    aggregate[key] = {"mean": mean, "population_std": std}
  aggregate_turns = sum(run["average_turns"] for run in summaries) / len(summaries)
  aggregate_therapist_words = (
    sum(run["average_therapist_words_per_turn"] for run in summaries) / len(summaries)
  )
  summary = {
    "checkpoints": sorted({str(run["model"]) for run in runs}),
    "tokenizer": str(tokenizer) if tokenizer else None,
    "evaluations": evaluations,
    "profiles_per_evaluation": args.n_profiles,
    "runs": summaries,
    "aggregate_ctrs": aggregate,
    "aggregate_conversation_metrics": {
      "average_turns": aggregate_turns,
      "average_therapist_words_per_turn": aggregate_therapist_words,
    },
    "total_cost": sum(float(run["cost"]) for run in summaries),
  }
  summary_path = output / "cbt_evaluation_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(f"Evaluation complete: {summary_path}")
  return summary_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run parallel ReasonCBT simulations and CTRS evaluations.")
  parser.add_argument("--input", help="Local therapist checkpoint directory.")
  parser.add_argument(
    "--configs",
    nargs="+",
    help="Distinct simulation YAML configs; model, port, and run output are read from each config.",
  )
  parser.add_argument("--output", required=True, help="Output root for logs and the combined summary.")
  parser.add_argument("--evaluations", type=int, help="Number of independent parallel pipelines.")
  parser.add_argument("--tokenizer", help="Optional tokenizer path; defaults to the checkpoint tokenizer.")
  parser.add_argument("--gpus", help="Comma-separated GPU IDs; defaults to 0..evaluations-1.")
  parser.add_argument("--base-port", type=int, default=8000)
  parser.add_argument("--profiles-path", default="data/odra.json")
  parser.add_argument("--n-profiles", type=int, default=150)
  parser.add_argument("--n-processes", type=int, default=50)
  parser.add_argument("--eval-max-iter", type=int, default=3)
  parser.add_argument("--server-timeout", type=int, default=900)
  parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
  parser.add_argument("--resume", action="store_true", help="Resume pending and failed profiles from existing manifests.")
  parser.add_argument(
    "--stop-on-conversation-error",
    action="store_true",
    help="Stop before retries or CTRS evaluation if any simulation ends with conversation_error.",
  )
  parser.add_argument(
    "--simulation-script",
    default="agent_simulation.py",
    help="Simulation entry point to run, path relative to the repo root (default: agent_simulation.py).",
  )
  parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching or writing outputs.")
  arguments = parser.parse_args()
  run_pipeline(arguments)
