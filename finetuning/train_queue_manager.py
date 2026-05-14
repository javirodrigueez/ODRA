#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: PyYAML. Install the finetuning requirements or run "
        "`pip install PyYAML`."
    ) from exc


RESERVED_JOB_KEYS = {"id", "python", "script", "args", "env", "common_args"}


@dataclass(frozen=True)
class Job:
    id: str
    command: list[str]
    env: dict[str, str]


@dataclass
class RunningJob:
    job: Job
    process: subprocess.Popen[Any]
    log_handle: Any
    log_path: Path
    started_at: float
    gpu: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def normalize_args(raw_args: dict[str, Any]) -> list[str]:
    command_args: list[str] = []
    for key, value in raw_args.items():
        if value is None or value is False:
            continue

        flag = f"--{str(key).replace('_', '-')}"
        if value is True:
            command_args.append(flag)
            continue

        if isinstance(value, list):
            for item in value:
                command_args.extend([flag, str(item)])
            continue

        command_args.extend([flag, str(value)])
    return command_args


def merge_args(defaults: dict[str, Any], job_data: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    common_args = defaults.get("common_args", {})
    if common_args is not None:
        if not isinstance(common_args, dict):
            raise SystemExit("defaults.common_args must be a mapping.")
        args.update(common_args)

    if defaults.get("group") is not None:
        args["group"] = defaults["group"]

    direct_job_args = {
        key: value for key, value in job_data.items() if key not in RESERVED_JOB_KEYS
    }
    args.update(direct_job_args)

    explicit_job_args = job_data.get("args", {})
    if explicit_job_args is not None:
        if not isinstance(explicit_job_args, dict):
            raise SystemExit(f"Job {job_data.get('id')!r} has non-mapping 'args'.")
        args.update(explicit_job_args)

    return args


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

        python_bin = str(raw_job.get("python", defaults.get("python", "python")))
        script = str(raw_job.get("script", defaults.get("script", "finetuning/train.py")))
        args = merge_args(defaults, raw_job)
        command = [python_bin, script, *normalize_args(args)]

        env = defaults.get("env", {})
        if env is None:
            env = {}
        if not isinstance(env, dict):
            raise SystemExit("defaults.env must be a mapping.")
        job_env = raw_job.get("env", {})
        if job_env is None:
            job_env = {}
        if not isinstance(job_env, dict):
            raise SystemExit(f"Job {job_id!r} has non-mapping env.")
        merged_env = {str(key): str(value) for key, value in {**env, **job_env}.items()}

        jobs.append(Job(id=job_id, command=command, env=merged_env))
    return jobs


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


def start_job(
    job: Job,
    slot: int,
    gpu: str | None,
    logs_dir: Path,
    state: dict[str, Any],
    state_file: Path,
) -> RunningJob:
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempt = attempt_number(state, job.id)
    log_path = logs_dir / f"{job.id}_{slug_timestamp()}_attempt{attempt}.log"
    log_handle = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env.update(job.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    started_at = time.time()
    log_handle.write(f"job_id: {job.id}\n")
    log_handle.write(f"attempt: {attempt}\n")
    log_handle.write(f"started_at: {utc_now()}\n")
    log_handle.write(f"slot: {slot}\n")
    log_handle.write(f"gpu: {gpu if gpu is not None else ''}\n")
    log_handle.write(f"command: {shlex.join(job.command)}\n")
    log_handle.write("\n")
    log_handle.flush()

    process = subprocess.Popen(
        job.command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )

    state.setdefault("jobs", {})[job.id] = {
        "attempts": attempt,
        "command": job.command,
        "gpu": gpu,
        "log_path": str(log_path),
        "pid": process.pid,
        "started_at": utc_now(),
        "status": "running",
    }
    save_state(state_file, state)
    print(f"[start] {job.id} pid={process.pid} slot={slot} gpu={gpu or '-'}")
    print(f"        log={log_path}")

    return RunningJob(
        job=job,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        started_at=started_at,
        gpu=gpu,
    )


def finish_job(
    running: RunningJob,
    state: dict[str, Any],
    state_file: Path,
    status: str | None = None,
) -> str:
    exit_code = running.process.returncode
    duration_seconds = round(time.time() - running.started_at, 2)
    final_status = status or ("succeeded" if exit_code == 0 else "failed")

    running.log_handle.write("\n")
    running.log_handle.write(f"ended_at: {utc_now()}\n")
    running.log_handle.write(f"duration_seconds: {duration_seconds}\n")
    running.log_handle.write(f"exit_code: {exit_code}\n")
    running.log_handle.write(f"status: {final_status}\n")
    running.log_handle.close()

    job_state = state.setdefault("jobs", {}).setdefault(running.job.id, {})
    job_state.update(
        {
            "duration_seconds": duration_seconds,
            "ended_at": utc_now(),
            "exit_code": exit_code,
            "log_path": str(running.log_path),
            "pid": running.process.pid,
            "status": final_status,
        }
    )
    save_state(state_file, state)

    label = "done" if final_status == "succeeded" else final_status
    print(f"[{label}] {running.job.id} exit={exit_code} duration={duration_seconds}s")
    return final_status


def terminate_running(running_by_slot: dict[int, RunningJob]) -> None:
    for running in running_by_slot.values():
        if running.process.poll() is None:
            running.process.terminate()

    deadline = time.time() + 30
    while time.time() < deadline:
        if all(running.process.poll() is not None for running in running_by_slot.values()):
            return
        time.sleep(0.5)

    for running in running_by_slot.values():
        if running.process.poll() is None:
            running.process.kill()

    for running in running_by_slot.values():
        if running.process.poll() is None:
            try:
                running.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


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


def print_summary(
    selected_jobs: list[Job],
    skipped: list[str],
    state: dict[str, Any],
) -> None:
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


def run_queue(args: argparse.Namespace) -> int:
    jobs = load_jobs(args.config)
    gpus = parse_gpus(args.gpus)
    if gpus and args.max_parallel > len(gpus):
        raise SystemExit(
            f"--max-parallel={args.max_parallel} requires at least that many --gpus entries."
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
        for job in selected_jobs:
            print(f"{job.id}: {shlex.join(job.command)}")
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
    running_by_slot: dict[int, RunningJob] = {}
    slot_gpus = gpus if gpus else [None] * args.max_parallel

    while pending or running_by_slot:
        if stop_requested:
            terminate_running(running_by_slot)
            for slot, running in list(running_by_slot.items()):
                running.process.poll()
                finish_job(running, state, args.state_file, status="interrupted")
                del running_by_slot[slot]
            print_summary(selected_jobs, skipped, state)
            return 130

        while pending and len(running_by_slot) < args.max_parallel:
            free_slots = [
                slot for slot in range(args.max_parallel) if slot not in running_by_slot
            ]
            if not free_slots:
                break
            slot = free_slots[0]
            job = pending.pop(0)
            gpu = slot_gpus[slot] if gpus else None
            running_by_slot[slot] = start_job(
                job,
                slot,
                gpu,
                args.logs_dir,
                state,
                args.state_file,
            )

        for slot, running in list(running_by_slot.items()):
            exit_code = running.process.poll()
            if exit_code is None:
                continue
            finish_job(running, state, args.state_file)
            del running_by_slot[slot]

        if running_by_slot:
            time.sleep(args.poll_interval)

    print_summary(selected_jobs, skipped, state)
    failed = [
        job.id
        for job in selected_jobs
        if state.get("jobs", {}).get(job.id, {}).get("status") != "succeeded"
    ]
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finetuning jobs from a YAML queue.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("finetuning/train_queue.yaml"),
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
        help="Comma-separated GPU ids. Each parallel slot gets one CUDA_VISIBLE_DEVICES value.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("finetuning/train_queue_state.json"),
        help="Persistent queue state JSON.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("finetuning/train_queue_logs"),
        help="Directory for per-job logs.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these job ids.",
    )
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
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between process polling checks.",
    )
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1.")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be greater than 0.")
    return args


def main() -> int:
    return run_queue(parse_args())


if __name__ == "__main__":
    sys.exit(main())
