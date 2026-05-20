#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: PyYAML. Install the project requirements or run "
        "`pip install PyYAML`."
    ) from exc


DEFAULT_VLLM_ARGS = {
    "quantization": "fp8",
    "kv-cache-dtype": "fp8",
    "gpu-memory-utilization": 0.95,
    "language-model-only": True,
    "enable-prefix-caching": True,
    "enable-chunked-prefill": True,
    "max-num-seqs": 150,
}
VLLM_MANAGER_ARGS = {"davinci"}
DAVINCI_NCCL_ENV = "NCCL_P2P_DISABLE=1"
DAVINCI_VLLM_ARG = "disable-custom-all-reduce"

DEFAULT_EVAL_SETTINGS = {
    "enabled": True,
    "metrics": ["ctrs", "wai", "panas"],
    "model": "gpt-4o",
    "max_iter": 3,
    "n_processes": 1,
    "azure": False,
    "batch": False,
    "panas_v2": False,
}
ALLOWED_EVAL_METRICS = {"ctrs", "wai", "panas"}


def load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return tqdm


@dataclass(frozen=True)
class EvalSettings:
    enabled: bool
    metrics: list[str]
    model: str
    max_iter: int
    n_processes: int
    azure: bool
    batch: bool
    panas_v2: bool
    input_dir: Path | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class Job:
    id: str
    cfg: Path
    profiles_path: Path
    n_profiles: int
    multiprocessing: bool
    n_processes: int | None
    gpu_count: int
    python: str
    script: str
    env: dict[str, str]
    model_id: str
    base_url: str
    output_dir: Path
    vllm_args: dict[str, Any] = field(default_factory=dict)
    eval_settings: EvalSettings | None = None


@dataclass
class RunningJob:
    job: Job
    server_process: subprocess.Popen[Any]
    simulation_process: subprocess.Popen[Any]
    eval_process: subprocess.Popen[Any] | None
    log_handle: Any
    log_path: Path
    started_at: float
    gpus: list[str]
    progress_position: int = 0
    simulation_progress: Any | None = None
    eval_progress: Any | None = None
    eval_progress_total: int = 0
    eval_progress_started: bool = False


class QueueConfigError(ValueError):
    pass


class JobStartError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def output_dir_manifest_prefix(output_dir: str | Path) -> str:
    normalized = Path(output_dir).as_posix().strip("/")
    return normalized.replace("/", "_") or "output"


def simulation_manifest_path(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    return output_path / f"{output_dir_manifest_prefix(output_path)}.json"


def read_json_silent(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def simulation_executed_count(job: Job) -> int:
    manifest = read_json_silent(simulation_manifest_path(job.output_dir))
    if not isinstance(manifest, dict):
        return 0
    metadata = manifest.get("metadata")
    if isinstance(metadata, dict):
        try:
            return max(0, int(metadata.get("executed_count") or 0))
        except (TypeError, ValueError):
            pass
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict):
        return 0
    return sum(1 for status in profiles.values() if isinstance(status, dict) and status.get("executed"))


def eval_metric_dirs(settings: EvalSettings) -> list[str]:
    dirs: list[str] = []
    if "ctrs" in settings.metrics:
        dirs.append("ctrs")
    if "wai" in settings.metrics:
        dirs.append("wai")
    if "panas" in settings.metrics:
        dirs.extend(["panas_before", "panas_after"])
    return dirs


def eval_expected_outputs(job: Job) -> int:
    if job.eval_settings is None or not job.eval_settings.enabled:
        return 0
    return job.n_profiles * len(eval_metric_dirs(job.eval_settings))


def eval_output_dir(job: Job) -> Path:
    if job.eval_settings is not None and job.eval_settings.output_dir is not None:
        return job.eval_settings.output_dir
    return job.output_dir / "eval"


def eval_completed_outputs(job: Job) -> int:
    if job.eval_settings is None or not job.eval_settings.enabled:
        return 0
    base = eval_output_dir(job)
    total = 0
    for dirname in eval_metric_dirs(job.eval_settings):
        metric_dir = base / dirname
        if metric_dir.is_dir():
            total += sum(1 for path in metric_dir.glob("*.json") if path.is_file())
    return total


def create_progress_bar(description: str, total: int, position: int) -> Any | None:
    tqdm = load_tqdm()
    if tqdm is None or total <= 0:
        return None
    return tqdm(
        total=total,
        desc=description,
        position=position,
        unit="item",
        leave=True,
        dynamic_ncols=True,
    )


def set_progress_value(progress: Any | None, value: int, total: int | None = None) -> None:
    if progress is None:
        return
    if total is not None and total > 0 and total != progress.total:
        progress.total = total
    target = max(0, min(int(value), int(progress.total or value)))
    delta = target - int(progress.n)
    if delta > 0:
        progress.update(delta)
    elif delta < 0:
        progress.n = target
        progress.refresh()


def close_progress(progress: Any | None) -> None:
    if progress is not None:
        progress.close()


def update_running_progress(running_jobs: list[RunningJob]) -> None:
    for running in running_jobs:
        set_progress_value(
            running.simulation_progress,
            simulation_executed_count(running.job),
            running.job.n_profiles,
        )
        if running.eval_progress_started:
            set_progress_value(
                running.eval_progress,
                eval_completed_outputs(running.job),
                running.eval_progress_total,
            )


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise SystemExit(f"Failed to read config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config {path} must contain a YAML mapping.")
    return data


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "jobs": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read state file {path}: {exc}") from exc

    if not isinstance(state, dict):
        raise SystemExit(f"State file {path} must contain a JSON object.")
    jobs = state.setdefault("jobs", {})
    if not isinstance(jobs, dict):
        raise SystemExit(f"State file {path} has invalid 'jobs' value.")
    state.setdefault("version", 1)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def config_value(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise QueueConfigError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise QueueConfigError(f"{name} must be a positive integer.")
    return parsed


def optional_positive_int(value: Any, name: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return positive_int(value, name)


def bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise QueueConfigError(f"{name} must be a boolean.")


def merge_mapping(defaults: dict[str, Any], job_data: dict[str, Any], key: str) -> dict[str, Any]:
    default_value = defaults.get(key, {})
    if default_value is None:
        default_value = {}
    if not isinstance(default_value, dict):
        raise QueueConfigError(f"defaults.{key} must be a mapping.")

    job_value = job_data.get(key, {})
    if job_value is None:
        job_value = {}
    if not isinstance(job_value, dict):
        raise QueueConfigError(f"Job {job_data.get('id')!r} has non-mapping {key}.")
    return {**default_value, **job_value}


def optional_path(value: Any, name: str) -> Path | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return Path(str(value)).expanduser()


def parse_eval_metrics(value: Any, name: str) -> list[str]:
    if value is None:
        return list(DEFAULT_EVAL_SETTINGS["metrics"])
    if not isinstance(value, list) or not value:
        raise QueueConfigError(f"{name} must be a non-empty list.")
    metrics = [str(item).strip().lower() for item in value if str(item).strip()]
    unknown = sorted(set(metrics) - ALLOWED_EVAL_METRICS)
    if unknown:
        raise QueueConfigError(f"{name} has unknown metric(s): {', '.join(unknown)}")
    return metrics


def parse_eval_settings(defaults: dict[str, Any], job_data: dict[str, Any]) -> EvalSettings:
    eval_data = merge_mapping(defaults, job_data, "eval")
    merged = {**DEFAULT_EVAL_SETTINGS, **eval_data}
    job_id = str(job_data.get("id") or "<unknown>")
    enabled = bool_value(merged.get("enabled"), f"Job {job_id!r}.eval.enabled")
    metrics = parse_eval_metrics(merged.get("metrics"), f"Job {job_id!r}.eval.metrics")
    azure = bool_value(merged.get("azure"), f"Job {job_id!r}.eval.azure")
    batch = bool_value(merged.get("batch"), f"Job {job_id!r}.eval.batch")
    if azure and batch:
        raise QueueConfigError(f"Job {job_id!r}.eval.batch is incompatible with eval.azure.")
    return EvalSettings(
        enabled=enabled,
        metrics=metrics,
        model=str(merged.get("model") or DEFAULT_EVAL_SETTINGS["model"]),
        max_iter=positive_int(merged.get("max_iter"), f"Job {job_id!r}.eval.max_iter"),
        n_processes=positive_int(merged.get("n_processes"), f"Job {job_id!r}.eval.n_processes"),
        azure=azure,
        batch=batch,
        panas_v2=bool_value(merged.get("panas_v2"), f"Job {job_id!r}.eval.panas_v2"),
        input_dir=optional_path(merged.get("input_dir"), f"Job {job_id!r}.eval.input_dir"),
        output_dir=optional_path(merged.get("output_dir"), f"Job {job_id!r}.eval.output_dir"),
    )


def host_port_from_base_url(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise QueueConfigError(f"Invalid therapist.base_url: {base_url!r}")
    return parsed.hostname or "localhost", parsed.port or 8000


def readiness_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def load_job_runtime_values(cfg_path: Path) -> tuple[str, str, Path]:
    try:
        cfg = load_yaml(cfg_path)
    except SystemExit as exc:
        raise QueueConfigError(str(exc)) from exc

    therapist = config_value(cfg, "therapist", {})
    therapist_type = str(config_value(therapist, "type", "") or "").strip().lower()
    if therapist_type != "local":
        raise QueueConfigError(
            f"{cfg_path}: therapist.type must be 'local' for vLLM-managed evaluation."
        )

    model_id = str(config_value(therapist, "model_id", "") or "").strip()
    base_url = str(config_value(therapist, "base_url", "") or "").strip()
    output_dir = str(config_value(config_value(cfg, "env", {}), "output_dir", "") or "").strip()
    if not model_id:
        raise QueueConfigError(f"{cfg_path}: therapist.model_id is required.")
    if not base_url:
        raise QueueConfigError(f"{cfg_path}: therapist.base_url is required.")
    if not output_dir:
        raise QueueConfigError(f"{cfg_path}: env.output_dir is required.")
    host_port_from_base_url(base_url)
    return model_id, base_url, Path(output_dir)


def normalize_vllm_args(raw_args: dict[str, Any]) -> list[str]:
    command_args: list[str] = []
    for key, value in raw_args.items():
        normalized_key = str(key).replace("_", "-")
        if normalized_key in VLLM_MANAGER_ARGS:
            continue
        if value is None or value is False:
            continue

        flag = f"--{normalized_key}"
        if value is True:
            command_args.append(flag)
            continue

        if isinstance(value, list):
            for item in value:
                command_args.extend([flag, str(item)])
            continue

        command_args.extend([flag, str(value)])
    return command_args


def vllm_arg_names(raw_args: dict[str, Any]) -> set[str]:
    return {
        str(key).replace("_", "-")
        for key, value in raw_args.items()
        if value is not None and value is not False
    }


def davinci_vllm_enabled(raw_args: dict[str, Any], name: str) -> bool:
    if "davinci" not in raw_args:
        return False
    return bool_value(raw_args["davinci"], name)


def split_env_prefix(command: list[str]) -> tuple[list[str], dict[str, str]]:
    env: dict[str, str] = {}
    index = 0
    for token in command:
        if "=" not in token:
            break
        key, value = token.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            break
        env[key] = value
        index += 1
    return command[index:], env


def load_jobs(config_path: Path) -> list[Job]:
    config = load_yaml(config_path)
    defaults = config.get("defaults", {})
    jobs_data = config.get("jobs")

    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise SystemExit("defaults must be a mapping.")
    if not isinstance(jobs_data, list) or not jobs_data:
        raise SystemExit("Config must contain a non-empty jobs list.")

    jobs: list[Job] = []
    seen_ids: set[str] = set()
    for index, raw_job in enumerate(jobs_data, start=1):
        if not isinstance(raw_job, dict):
            raise SystemExit(f"jobs[{index}] must be a mapping.")
        job_id = raw_job.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise SystemExit(f"jobs[{index}] must have a non-empty string id.")
        if job_id in seen_ids:
            raise SystemExit(f"Duplicate job id: {job_id}")
        seen_ids.add(job_id)

        cfg_value = raw_job.get("cfg", defaults.get("cfg"))
        profiles_value = raw_job.get("profiles_path", defaults.get("profiles_path"))
        n_profiles_value = raw_job.get("n_profiles", defaults.get("n_profiles"))
        multiprocessing_value = raw_job.get("multiprocessing")
        if multiprocessing_value is None:
            raise SystemExit(f"Job {job_id!r} must set multiprocessing true/false.")
        if cfg_value is None:
            raise SystemExit(f"Job {job_id!r} must set cfg.")
        if profiles_value is None:
            raise SystemExit(f"Job {job_id!r} must set profiles_path.")
        if n_profiles_value is None:
            raise SystemExit(f"Job {job_id!r} must set n_profiles.")

        try:
            cfg_path = Path(str(cfg_value)).expanduser()
            model_id, base_url, output_dir = load_job_runtime_values(cfg_path)
            multiprocessing_enabled = bool_value(multiprocessing_value, f"Job {job_id!r}.multiprocessing")
            n_processes = optional_positive_int(
                raw_job.get("n_processes", defaults.get("n_processes")),
                f"Job {job_id!r}.n_processes",
            )
            if multiprocessing_enabled and n_processes is None:
                raise QueueConfigError(f"Job {job_id!r} must set n_processes when multiprocessing is true.")
            gpu_count = positive_int(raw_job.get("gpu_count", defaults.get("gpu_count", 1)), f"Job {job_id!r}.gpu_count")
            n_profiles = positive_int(n_profiles_value, f"Job {job_id!r}.n_profiles")
            eval_settings = parse_eval_settings(defaults, raw_job)
        except QueueConfigError as exc:
            raise SystemExit(str(exc)) from exc

        env = merge_mapping(defaults, raw_job, "env")
        merged_env = {str(key): str(value) for key, value in env.items()}

        default_vllm_args = dict(DEFAULT_VLLM_ARGS)
        configured_vllm_args = merge_mapping(defaults, raw_job, "vllm_args")
        vllm_args = {**default_vllm_args, **configured_vllm_args}
        try:
            if "davinci" in vllm_args:
                vllm_args["davinci"] = bool_value(
                    vllm_args["davinci"],
                    f"Job {job_id!r}.vllm_args.davinci",
                )
        except QueueConfigError as exc:
            raise SystemExit(str(exc)) from exc

        jobs.append(
            Job(
                id=job_id,
                cfg=cfg_path,
                profiles_path=Path(str(profiles_value)).expanduser(),
                n_profiles=n_profiles,
                multiprocessing=multiprocessing_enabled,
                n_processes=n_processes,
                gpu_count=gpu_count,
                python=str(raw_job.get("python", defaults.get("python", "python"))),
                script=str(raw_job.get("script", defaults.get("script", "agent_simulation.py"))),
                env=merged_env,
                model_id=model_id,
                base_url=base_url,
                output_dir=output_dir,
                vllm_args=vllm_args,
                eval_settings=eval_settings,
            )
        )
    return jobs


def build_vllm_command(job: Job) -> list[str]:
    host, port = host_port_from_base_url(job.base_url)
    davinci_enabled = davinci_vllm_enabled(
        job.vllm_args,
        f"Job {job.id!r}.vllm_args.davinci",
    )
    command = [
        "vllm",
        "serve",
        job.model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        job.model_id,
    ]
    command.extend(normalize_vllm_args(job.vllm_args))
    configured_vllm_arg_names = vllm_arg_names(job.vllm_args)
    if davinci_enabled and DAVINCI_VLLM_ARG not in configured_vllm_arg_names:
        command.append(f"--{DAVINCI_VLLM_ARG}")
    if job.gpu_count > 1 and "tensor-parallel-size" not in configured_vllm_arg_names:
        command.extend(["--tensor-parallel-size", str(job.gpu_count)])
    if davinci_enabled:
        command = [DAVINCI_NCCL_ENV, *command]
    return command


def build_simulation_command(job: Job) -> list[str]:
    command = [
        job.python,
        job.script,
        "--cfg",
        str(job.cfg),
        "--profiles-path",
        str(job.profiles_path),
        "--n-profiles",
        str(job.n_profiles),
    ]
    if job.multiprocessing:
        command.append("--multiprocessing")
        command.extend(["--n-processes", str(job.n_processes)])
    return command


def build_eval_command(job: Job) -> list[str]:
    settings = job.eval_settings or EvalSettings(**DEFAULT_EVAL_SETTINGS)  # type: ignore[arg-type]
    if settings.azure and settings.batch:
        raise QueueConfigError(f"Job {job.id!r}.eval.batch is incompatible with eval.azure.")
    input_dir = settings.input_dir or (job.output_dir / "json")
    output_dir = settings.output_dir or (job.output_dir / "eval")
    command = [
        job.python,
        "evaluation/run_batch_metrics.py" if settings.batch else "evaluation/run_api_metrics.py",
        str(input_dir),
        str(output_dir),
        str(settings.max_iter),
    ]
    if not settings.batch:
        command.extend(["--model", settings.model])
        command.extend(["--n-processes", str(settings.n_processes)])
    if "wai" not in settings.metrics:
        command.append("--exc-wai")
    if "ctrs" not in settings.metrics:
        command.append("--exc-ctrs")
    if "panas" not in settings.metrics:
        command.append("--exc-panas")
    if settings.azure:
        command.append("--azure")
    if settings.panas_v2:
        command.append("--panas-v2")
    return command


def eval_script_name(settings: EvalSettings) -> str:
    return "run_batch_metrics.py" if settings.batch else "run_api_metrics.py"


def parse_gpus(raw_gpus: str | None) -> list[str]:
    if raw_gpus is None:
        return []
    gpus = [gpu.strip() for gpu in raw_gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus was provided but no GPU ids were found.")
    return gpus


def mark_stale_running_jobs(state: dict[str, Any]) -> bool:
    changed = False
    for job_state in state.get("jobs", {}).values():
        if isinstance(job_state, dict) and job_state.get("status") == "running":
            job_state["status"] = "interrupted"
            job_state["ended_at"] = utc_now()
            job_state["note"] = "Marked interrupted by queue manager startup."
            changed = True
    return changed


def attempt_number(state: dict[str, Any], job_id: str) -> int:
    previous = state.get("jobs", {}).get(job_id, {})
    if not isinstance(previous, dict):
        return 1
    return int(previous.get("attempts", 0)) + 1


def selected_gpu_group(available: list[str], gpu_count: int) -> list[str] | None:
    if len(available) < gpu_count:
        return None
    return available[:gpu_count]


def available_gpus(all_gpus: list[str], running_jobs: list[RunningJob]) -> list[str]:
    used = {gpu for running in running_jobs for gpu in running.gpus}
    return [gpu for gpu in all_gpus if gpu not in used]


def wait_for_server_ready(
    url: str,
    process: subprocess.Popen[Any],
    timeout_seconds: float,
    poll_interval: float,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise JobStartError(f"vLLM exited before readiness check passed, exit={exit_code}.")
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, URLError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval)
    raise JobStartError(f"Timed out waiting for vLLM readiness at {url}: {last_error}")


def terminate_process(process: subprocess.Popen[Any], timeout_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def start_job(
    job: Job,
    gpus: list[str],
    run_eval: bool,
    progress_position: int,
    logs_dir: Path,
    state: dict[str, Any],
    state_file: Path,
    readiness_timeout: float,
    readiness_poll_interval: float,
) -> RunningJob:
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempt = attempt_number(state, job.id)
    log_path = logs_dir / f"{job.id}_{slug_timestamp()}_attempt{attempt}.log"
    log_handle = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env.update(job.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    vllm_command = build_vllm_command(job)
    vllm_process_command, vllm_env = split_env_prefix(vllm_command)
    server_env = {**env, **vllm_env}
    simulation_command = build_simulation_command(job)
    eval_command = build_eval_command(job) if run_eval and job.eval_settings and job.eval_settings.enabled else None
    started_at = time.time()

    log_handle.write(f"job_id: {job.id}\n")
    log_handle.write(f"attempt: {attempt}\n")
    log_handle.write(f"started_at: {utc_now()}\n")
    log_handle.write(f"gpus: {','.join(gpus)}\n")
    log_handle.write(f"cfg: {job.cfg}\n")
    log_handle.write(f"output_dir: {job.output_dir}\n")
    log_handle.write(f"vllm_command: {shlex.join(vllm_command)}\n")
    log_handle.write(f"simulation_command: {shlex.join(simulation_command)}\n")
    if eval_command is not None:
        log_handle.write(f"eval_command: {shlex.join(eval_command)}\n")
    log_handle.write("\n--- vLLM startup ---\n")
    log_handle.flush()

    server_process = subprocess.Popen(
        vllm_process_command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=server_env,
    )

    state.setdefault("jobs", {})[job.id] = {
        "attempts": attempt,
        "cfg": str(job.cfg),
        "gpus": gpus,
        "log_path": str(log_path),
        "model_id": job.model_id,
        "output_dir": str(job.output_dir),
        "profiles_path": str(job.profiles_path),
        "server_pid": server_process.pid,
        "started_at": utc_now(),
        "status": "running",
        "vllm_command": vllm_command,
        "simulation_command": simulation_command,
    }
    if eval_command is not None:
        state["jobs"][job.id]["eval_command"] = eval_command
    save_state(state_file, state)
    print(f"[start] {job.id} server_pid={server_process.pid} gpus={','.join(gpus) or '-'}")
    print(f"        log={log_path}")

    try:
        wait_for_server_ready(
            readiness_url(job.base_url),
            server_process,
            timeout_seconds=readiness_timeout,
            poll_interval=readiness_poll_interval,
        )
    except Exception:
        terminate_process(server_process)
        log_handle.write("\n--- vLLM readiness failed ---\n")
        log_handle.flush()
        log_handle.close()
        raise

    log_handle.write("\n--- agent_simulation.py ---\n")
    log_handle.flush()
    simulation_process = subprocess.Popen(
        simulation_command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    state["jobs"][job.id]["simulation_pid"] = simulation_process.pid
    save_state(state_file, state)
    print(f"        simulation_pid={simulation_process.pid}")

    return RunningJob(
        job=job,
        server_process=server_process,
        simulation_process=simulation_process,
        eval_process=None,
        log_handle=log_handle,
        log_path=log_path,
        started_at=started_at,
        gpus=gpus,
        progress_position=progress_position,
        simulation_progress=create_progress_bar(f"{job.id} conversations", job.n_profiles, progress_position),
    )


def finish_job(
    running: RunningJob,
    state: dict[str, Any],
    state_file: Path,
    status: str | None = None,
    note: str = "",
) -> str:
    simulation_exit = running.simulation_process.returncode
    eval_exit = running.eval_process.returncode if running.eval_process is not None else None
    server_exit_before_stop = running.server_process.poll()
    terminate_process(running.server_process)
    server_exit = running.server_process.returncode
    duration_seconds = round(time.time() - running.started_at, 2)
    final_status = status or ("succeeded" if simulation_exit == 0 and (eval_exit is None or eval_exit == 0) else "failed")
    set_progress_value(running.simulation_progress, running.job.n_profiles if simulation_exit == 0 else simulation_executed_count(running.job), running.job.n_profiles)
    if running.eval_progress_started:
        set_progress_value(running.eval_progress, eval_completed_outputs(running.job), running.eval_progress_total)

    running.log_handle.write("\n")
    running.log_handle.write(f"ended_at: {utc_now()}\n")
    running.log_handle.write(f"duration_seconds: {duration_seconds}\n")
    running.log_handle.write(f"simulation_exit_code: {simulation_exit}\n")
    if eval_exit is not None:
        running.log_handle.write(f"eval_exit_code: {eval_exit}\n")
    running.log_handle.write(f"server_exit_before_stop: {server_exit_before_stop}\n")
    running.log_handle.write(f"server_exit_code: {server_exit}\n")
    running.log_handle.write(f"status: {final_status}\n")
    if note:
        running.log_handle.write(f"note: {note}\n")
    running.log_handle.close()

    job_state = state.setdefault("jobs", {}).setdefault(running.job.id, {})
    job_state.update(
        {
            "duration_seconds": duration_seconds,
            "ended_at": utc_now(),
            "log_path": str(running.log_path),
            "server_exit_before_stop": server_exit_before_stop,
            "server_exit_code": server_exit,
            "server_pid": running.server_process.pid,
            "eval_exit_code": eval_exit,
            "eval_pid": running.eval_process.pid if running.eval_process is not None else None,
            "simulation_exit_code": simulation_exit,
            "simulation_pid": running.simulation_process.pid,
            "status": final_status,
        }
    )
    if note:
        job_state["note"] = note
    save_state(state_file, state)

    label = "done" if final_status == "succeeded" else final_status
    print(f"[{label}] {running.job.id} sim_exit={simulation_exit} duration={duration_seconds}s")
    close_progress(running.simulation_progress)
    close_progress(running.eval_progress)
    return final_status


def fail_start_job(
    job: Job,
    state: dict[str, Any],
    state_file: Path,
    error: Exception,
) -> None:
    job_state = state.setdefault("jobs", {}).setdefault(job.id, {})
    job_state.update(
        {
            "ended_at": utc_now(),
            "status": "failed",
            "note": str(error),
        }
    )
    save_state(state_file, state)
    print(f"[failed] {job.id} start failed: {error}")


def terminate_running(running_jobs: list[RunningJob]) -> None:
    for running in running_jobs:
        if running.eval_process is not None:
            terminate_process(running.eval_process, timeout_seconds=10)
        terminate_process(running.simulation_process, timeout_seconds=10)
        terminate_process(running.server_process, timeout_seconds=30)


def select_jobs(
    jobs: list[Job],
    state: dict[str, Any],
    only: list[str] | None,
    rerun_succeeded: bool,
) -> tuple[list[Job], list[str]]:
    job_by_id = {job.id: job for job in jobs}
    if only:
        missing = sorted(set(only) - set(job_by_id))
        if missing:
            raise SystemExit(f"Unknown job id(s) in --only: {', '.join(missing)}")
        selected = [job_by_id[job_id] for job_id in only]
    else:
        selected = jobs

    skipped: list[str] = []
    if not rerun_succeeded:
        filtered: list[Job] = []
        for job in selected:
            job_state = state.get("jobs", {}).get(job.id, {})
            if isinstance(job_state, dict) and job_state.get("status") == "succeeded":
                skipped.append(job.id)
            else:
                filtered.append(job)
        selected = filtered
    return selected, skipped


def print_summary(selected_jobs: list[Job], skipped: list[str], state: dict[str, Any]) -> None:
    selected_ids = {job.id for job in selected_jobs}
    succeeded = []
    failed = []
    interrupted = []
    for job_id, job_state in state.get("jobs", {}).items():
        if job_id not in selected_ids:
            continue
        if not isinstance(job_state, dict):
            continue
        status = job_state.get("status")
        if status == "succeeded":
            succeeded.append(job_id)
        elif status == "failed":
            failed.append(job_id)
        elif status == "interrupted":
            interrupted.append(job_id)

    print("\nSummary")
    print(f"  selected: {len(selected_jobs)}")
    print(f"  skipped succeeded: {len(skipped)}")
    print(f"  succeeded: {len(succeeded)}")
    print(f"  failed: {len(failed)}")
    print(f"  interrupted: {len(interrupted)}")
    if skipped:
        print(f"  skipped ids: {', '.join(skipped)}")
    if failed:
        print(f"  failed ids: {', '.join(failed)}")
    if interrupted:
        print(f"  interrupted ids: {', '.join(interrupted)}")


def dry_run_jobs(jobs: list[Job], gpus: list[str]) -> None:
    for job in jobs:
        print(f"{job.id}:")
        print(f"  gpu_count: {job.gpu_count}")
        if gpus:
            print(f"  needs GPUs from: {','.join(gpus)}")
        print(f"  vllm: {shlex.join(build_vllm_command(job))}")
        print(f"  simulation: {shlex.join(build_simulation_command(job))}")
        if job.eval_settings and job.eval_settings.enabled:
            print(f"  eval: {shlex.join(build_eval_command(job))}")
        elif job.eval_settings:
            print("  eval: disabled")


def find_startable_job(
    pending: list[Job],
    gpus: list[str],
    running_jobs: list[RunningJob],
    max_parallel: int,
) -> tuple[int, list[str]] | None:
    if len(running_jobs) >= max_parallel:
        return None
    if not gpus:
        return (0, [])

    free = available_gpus(gpus, running_jobs)
    for index, job in enumerate(pending):
        group = selected_gpu_group(free, job.gpu_count)
        if group is not None:
            return index, group
    return None


def run_queue(args: argparse.Namespace) -> int:
    jobs = load_jobs(args.config)
    gpus = parse_gpus(args.gpus)
    if gpus and any(job.gpu_count > len(gpus) for job in jobs):
        oversized = [job.id for job in jobs if job.gpu_count > len(gpus)]
        raise SystemExit(
            f"Job gpu_count exceeds available --gpus for: {', '.join(oversized)}"
        )

    state = load_state(args.state_file)
    if mark_stale_running_jobs(state):
        save_state(args.state_file, state)

    selected_jobs, skipped = select_jobs(
        jobs,
        state,
        only=args.only,
        rerun_succeeded=args.rerun_succeeded,
    )

    if args.dry_run:
        dry_run_jobs(selected_jobs, gpus)
        if skipped:
            print(f"Skipped succeeded: {', '.join(skipped)}")
        return 0

    if not selected_jobs:
        print("No jobs to run.")
        if skipped:
            print(f"Skipped succeeded: {', '.join(skipped)}")
        return 0

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"\nReceived signal {signum}; stopping running jobs...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pending = list(selected_jobs)
    running_jobs: list[RunningJob] = []

    while pending or running_jobs:
        update_running_progress(running_jobs)
        if stop_requested:
            terminate_running(running_jobs)
            for running in list(running_jobs):
                running.simulation_process.poll()
                finish_job(running, state, args.state_file, status="interrupted")
                running_jobs.remove(running)
            print_summary(selected_jobs, skipped, state)
            return 130

        while pending:
            startable = find_startable_job(pending, gpus, running_jobs, args.max_parallel)
            if startable is None:
                break
            pending_index, gpu_group = startable
            job = pending.pop(pending_index)
            try:
                running_jobs.append(
                    start_job(
                        job,
                        gpu_group,
                        args.eval,
                        len(running_jobs) * 2,
                        args.logs_dir,
                        state,
                        args.state_file,
                        readiness_timeout=args.readiness_timeout,
                        readiness_poll_interval=args.readiness_poll_interval,
                    )
                )
            except Exception as exc:
                fail_start_job(job, state, args.state_file, exc)

        for running in list(running_jobs):
            server_exit = running.server_process.poll()
            simulation_exit = running.simulation_process.poll()
            eval_exit = running.eval_process.poll() if running.eval_process is not None else None
            if server_exit is not None and simulation_exit is None:
                terminate_process(running.simulation_process, timeout_seconds=10)
                if running.eval_process is not None:
                    terminate_process(running.eval_process, timeout_seconds=10)
                running.simulation_process.poll()
                finish_job(
                    running,
                    state,
                    args.state_file,
                    status="failed",
                    note=f"vLLM exited before simulation completed, exit={server_exit}.",
                )
                running_jobs.remove(running)
                continue
            if simulation_exit is None:
                continue
            if (
                simulation_exit == 0
                and args.eval
                and running.eval_process is None
                and running.job.eval_settings is not None
                and running.job.eval_settings.enabled
            ):
                eval_command = build_eval_command(running.job)
                running.log_handle.write(f"\n--- {eval_script_name(running.job.eval_settings)} ---\n")
                running.log_handle.flush()
                set_progress_value(running.simulation_progress, running.job.n_profiles, running.job.n_profiles)
                terminate_process(running.server_process)
                running.gpus = []
                running.eval_progress_total = eval_expected_outputs(running.job)
                running.eval_progress_started = True
                running.eval_progress = create_progress_bar(
                    f"{running.job.id} evaluation",
                    running.eval_progress_total,
                    running.progress_position + 1,
                )
                set_progress_value(running.eval_progress, eval_completed_outputs(running.job), running.eval_progress_total)
                running.eval_process = subprocess.Popen(
                    eval_command,
                    stdout=running.log_handle,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        **running.job.env,
                        "PYTHONUNBUFFERED": "1",
                    },
                )
                state["jobs"][running.job.id]["eval_pid"] = running.eval_process.pid
                save_state(args.state_file, state)
                print(f"        eval_pid={running.eval_process.pid}")
                continue
            if running.eval_process is not None and eval_exit is None:
                continue
            finish_job(running, state, args.state_file)
            running_jobs.remove(running)

        if running_jobs:
            time.sleep(args.poll_interval)

    print_summary(selected_jobs, skipped, state)
    failed = [
        job.id
        for job in selected_jobs
        if state.get("jobs", {}).get(job.id, {}).get("status") != "succeeded"
    ]
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM-backed agent simulation jobs from a YAML queue."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evaluation/eval_queue.yaml"),
        help="YAML queue config.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of jobs to run at once.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids available to the queue manager.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("evaluation/eval_queue_state.json"),
        help="Persistent queue state JSON.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("evaluation/eval_queue_logs"),
        help="Directory for per-job logs.",
    )
    parser.add_argument("--only", nargs="+", default=None, help="Run only these job ids.")
    parser.add_argument(
        "--rerun-succeeded",
        action="store_true",
        help="Run jobs even if the state file marks them succeeded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without launching jobs.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation/run_api_metrics.py after successful simulations unless job eval.enabled is false.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between process polling checks.",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for vLLM readiness before failing a job.",
    )
    parser.add_argument(
        "--readiness-poll-interval",
        type=float,
        default=5.0,
        help="Seconds between vLLM readiness checks.",
    )
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1.")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be greater than 0.")
    if args.readiness_timeout <= 0:
        raise SystemExit("--readiness-timeout must be greater than 0.")
    if args.readiness_poll_interval <= 0:
        raise SystemExit("--readiness-poll-interval must be greater than 0.")
    return args


def main() -> int:
    return run_queue(parse_args())


if __name__ == "__main__":
    sys.exit(main())
