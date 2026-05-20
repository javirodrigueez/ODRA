from pathlib import Path
import tempfile
import unittest

from evaluation import eval_queue_manager as manager


def _write_yaml(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_runtime_config(path: Path, *, base_url: str = "http://localhost:8000/v1") -> None:
  _write_yaml(
    path,
    f"""
therapist:
  type: local
  base_url: {base_url}
  model_id: ft_models/example/merged_model

patient:
  type: deepseek
  base_url: https://openrouter.ai/api/v1
  model_id: deepseek/deepseek-v3.2
  prompt_variant: base

env:
  max_turns: 10
  min_turns: 0
  output_dir: /tmp/eval-out
  resistance: false
""",
  )


class _Running:
  def __init__(self, gpus: list[str]):
    self.gpus = gpus


class EvalQueueManagerTests(unittest.TestCase):
  def test_load_jobs_reads_runtime_config_and_builds_commands(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg, base_url="http://127.0.0.1:8123/v1")
      _write_yaml(
        queue,
        f"""
defaults:
  python: python
  gpu_count: 1
  eval:
    metrics: [ctrs, panas]
    model: gpt-4o-mini
    max_iter: 2
    n_processes: 4
    azure: true
    panas_v2: true
  vllm_args:
    max-num-seqs: 64

jobs:
  - id: eval-a
    cfg: {cfg}
    profiles_path: data/eval_full_cactus/eval_full_cactus.json
    n_profiles: 5
    multiprocessing: true
    n_processes: 2
""",
      )

      jobs = manager.load_jobs(queue)

    self.assertEqual(len(jobs), 1)
    job = jobs[0]
    self.assertEqual(job.model_id, "ft_models/example/merged_model")
    self.assertEqual(job.base_url, "http://127.0.0.1:8123/v1")
    self.assertEqual(job.output_dir, Path("/tmp/eval-out"))

    vllm = manager.build_vllm_command(job)
    self.assertEqual(vllm[:2], ["vllm", "serve"])
    self.assertIn("vllm", vllm)
    self.assertIn("--host", vllm)
    self.assertIn("127.0.0.1", vllm)
    self.assertIn("--port", vllm)
    self.assertIn("8123", vllm)
    self.assertIn("--quantization", vllm)
    self.assertIn("fp8", vllm)
    self.assertNotIn("NCCL_P2P_DISABLE=1", vllm)
    self.assertNotIn("--davinci", vllm)
    self.assertNotIn("--disable-custom-all-reduce", vllm)
    self.assertNotIn("--tensor-parallel-size", vllm)

    simulation = manager.build_simulation_command(job)
    self.assertEqual(simulation[:4], ["python", "agent_simulation.py", "--cfg", str(job.cfg)])
    self.assertIn("--multiprocessing", simulation)
    self.assertIn("--n-processes", simulation)

    eval_command = manager.build_eval_command(job)
    self.assertEqual(eval_command[:2], ["python", "evaluation/run_api_metrics.py"])
    self.assertIn(str(job.output_dir / "json"), eval_command)
    self.assertIn(str(job.output_dir / "eval"), eval_command)
    self.assertIn("--exc-wai", eval_command)
    self.assertNotIn("--exc-ctrs", eval_command)
    self.assertNotIn("--exc-panas", eval_command)
    self.assertIn("--azure", eval_command)
    self.assertIn("--panas-v2", eval_command)
    self.assertEqual(eval_command[eval_command.index("--n-processes") + 1], "4")

  def test_davinci_vllm_args_add_nccl_prefix_and_all_reduce_flag(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg)
      _write_yaml(
        queue,
        f"""
jobs:
  - id: davinci-eval
    cfg: {cfg}
    profiles_path: profiles.json
    n_profiles: 1
    multiprocessing: false
    vllm_args:
      davinci: true
      max-num-seqs: 32
""",
      )

      jobs = manager.load_jobs(queue)

    command = manager.build_vllm_command(jobs[0])
    self.assertEqual(command[:3], ["NCCL_P2P_DISABLE=1", "vllm", "serve"])
    self.assertIn("--disable-custom-all-reduce", command)
    self.assertNotIn("--davinci", command)
    self.assertIn("--max-num-seqs", command)
    self.assertEqual(command[command.index("--max-num-seqs") + 1], "32")
    process_command, env = manager.split_env_prefix(command)
    self.assertEqual(process_command[:2], ["vllm", "serve"])
    self.assertEqual(env, {"NCCL_P2P_DISABLE": "1"})

  def test_davinci_vllm_arg_must_be_boolean(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg)
      _write_yaml(
        queue,
        f"""
jobs:
  - id: davinci-invalid
    cfg: {cfg}
    profiles_path: profiles.json
    n_profiles: 1
    multiprocessing: false
    vllm_args:
      davinci: maybe
""",
      )

      with self.assertRaises(SystemExit) as error:
        manager.load_jobs(queue)

    self.assertIn("vllm_args.davinci", str(error.exception))

  def test_eval_batch_uses_run_batch_metrics_command(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg)
      _write_yaml(
        queue,
        f"""
jobs:
  - id: batch-eval
    cfg: {cfg}
    profiles_path: profiles.json
    n_profiles: 1
    multiprocessing: false
    eval:
      batch: true
      metrics: [wai]
      model: gpt-4o-mini
      max_iter: 2
      n_processes: 3
      panas_v2: true
""",
      )

      jobs = manager.load_jobs(queue)

    job = jobs[0]
    self.assertTrue(job.eval_settings.batch)
    eval_command = manager.build_eval_command(job)
    self.assertEqual(eval_command[:2], ["python", "evaluation/run_batch_metrics.py"])
    self.assertIn("--exc-ctrs", eval_command)
    self.assertIn("--exc-panas", eval_command)
    self.assertNotIn("--exc-wai", eval_command)
    self.assertNotIn("--azure", eval_command)
    self.assertIn("--panas-v2", eval_command)
    self.assertEqual(eval_command[eval_command.index("--n-processes") + 1], "3")
    self.assertEqual(manager.eval_script_name(job.eval_settings), "run_batch_metrics.py")

  def test_eval_batch_rejects_azure(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg)
      _write_yaml(
        queue,
        f"""
jobs:
  - id: batch-azure
    cfg: {cfg}
    profiles_path: profiles.json
    n_profiles: 1
    multiprocessing: false
    eval:
      batch: true
      azure: true
""",
      )

      with self.assertRaises(SystemExit) as error:
        manager.load_jobs(queue)

    self.assertIn("incompatible", str(error.exception))

  def test_multi_gpu_adds_tensor_parallel_size(self) -> None:
    job = manager.Job(
      id="multi",
      cfg=Path("configs/local.yaml"),
      profiles_path=Path("profiles.json"),
      n_profiles=1,
      multiprocessing=False,
      n_processes=None,
      gpu_count=2,
      python="python",
      script="agent_simulation.py",
      env={},
      model_id="model",
      base_url="http://localhost:8000/v1",
      output_dir=Path("/tmp/out"),
      vllm_args=dict(manager.DEFAULT_VLLM_ARGS),
    )

    command = manager.build_vllm_command(job)

    self.assertIn("--tensor-parallel-size", command)
    self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")

  def test_multiprocessing_requires_n_processes(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg)
      _write_yaml(
        queue,
        f"""
jobs:
  - id: missing-processes
    cfg: {cfg}
    profiles_path: profiles.json
    n_profiles: 1
    multiprocessing: true
""",
      )

      with self.assertRaises(SystemExit):
        manager.load_jobs(queue)

  def test_eval_can_be_disabled_per_job(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      base = Path(tmp)
      cfg = base / "local.yaml"
      queue = base / "queue.yaml"
      _write_runtime_config(cfg)
      _write_yaml(
        queue,
        f"""
jobs:
  - id: no-eval
    cfg: {cfg}
    profiles_path: profiles.json
    n_profiles: 1
    multiprocessing: false
    eval:
      enabled: false
""",
      )

      jobs = manager.load_jobs(queue)

    self.assertIsNotNone(jobs[0].eval_settings)
    self.assertFalse(jobs[0].eval_settings.enabled)

  def test_find_startable_job_waits_until_enough_gpus_are_free(self) -> None:
    jobs = [
      manager.Job(
        id="needs-two",
        cfg=Path("cfg.yaml"),
        profiles_path=Path("profiles.json"),
        n_profiles=1,
        multiprocessing=False,
        n_processes=None,
        gpu_count=2,
        python="python",
        script="agent_simulation.py",
        env={},
        model_id="model",
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/out"),
      )
    ]

    result = manager.find_startable_job(jobs, ["0", "1"], [_Running(["0"])], max_parallel=2)  # type: ignore[list-item]

    self.assertIsNone(result)


if __name__ == "__main__":
  unittest.main()
