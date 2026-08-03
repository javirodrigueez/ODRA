---
name: cbt-evaluation
description: Run reproducible CBT therapist-checkpoint evaluations inside the ReasonCBT repository, including one or more parallel vLLM-backed agent_simulation.py runs, retry and completeness gates, CTRS-only OpenAI Batch evaluation, aggregation, cost accounting, and comparative reporting. Use when a user asks to evaluate a local therapist checkpoint, run multiple independent CBT evaluation replicates, score simulated conversations with CTRS, or compare repeated evaluation scores and costs.
---

# CBT Evaluation

Evaluate one local therapist checkpoint through independent parallel simulation and CTRS pipelines with `scripts/run_cbt_evaluation.py`.

## Collect inputs

Confirm these values before running:

- Local checkpoint path passed as `--input`.
- Output root passed as `--output`.
- Number of independent parallel pipelines passed as `--evaluations`.
- Optional tokenizer override. Omit `--tokenizer` to use the checkpoint's tokenizer.
- GPU IDs when they should differ from `0..N-1`.

Interpret the evaluation count as the number of complete independent pipelines, not the number of patient profiles. Require one available GPU per parallel pipeline.

Use these defaults unless the user requests changes:

- 150 profiles from `data/odra.json`.
- 50 simulation worker processes per pipeline.
- FP8 model quantization and FP8 KV cache.
- Ports starting at 8000.
- CTRS only, six dimensions, evaluator `max_iter=3`.
- One retry for unsuccessful simulations.

## Run the pipeline

From the ReasonCBT repository root, run:

```bash
python .codex/skills/cbt-evaluation/scripts/run_cbt_evaluation.py \
  --input models/THERAPIST_CHECKPOINT \
  --output /outs/local/THERAPIST_CHECKPOINT-evaluation \
  --evaluations 2
```

Add `--tokenizer PATH` only when the checkpoint needs a different tokenizer. Add `--gpus 2,3` to select explicit GPUs.

For distinct simulation conditions, supply one YAML per parallel pipeline. The runner reads each config's checkpoint, localhost port, and output directory without modifying the source YAML:

```bash
python .codex/skills/cbt-evaluation/scripts/run_cbt_evaluation.py \
  --configs evaluation/configs/condition-a.yaml evaluation/configs/condition-b.yaml \
  --output /outs/local/combined-evaluation \
  --gpus 0,1
```

If the output root already exists and contains files, do not pass `--overwrite` until the user explicitly authorizes replacing that tree. Use `--dry-run` to inspect the planned GPU, port, output, simulation, and evaluation settings without starting servers, simulations, or paid evaluations.

The script performs these gates in order:

1. Launch one vLLM server per evaluation and require HTTP 200 health plus the exact served model ID.
2. Run all `agent_simulation.py` jobs concurrently.
3. Require the requested executed, successful, and JSON conversation counts; retry failed profiles once.
4. Stop all vLLM process groups and verify their ports close.
5. Dry-run every CTRS batch and require exactly six requests per successful conversation with no WAI/PANAS requests.
6. Submit all CTRS-only evaluations concurrently only after every dry-run gate passes.
7. Require one complete CTRS file per conversation with all six numeric fields and a numeric cost.
8. Run `rating_metrics.py` and `turns_and_closings.py` and report CTRS independent metrics, average turns, and average length of therapist utterances to `cbt_evaluation_summary.json` in the output root.

Keep the user updated during long simulations and batch polling. Record the known scale-1.0 warning when a checkpoint lacks calibrated FP8 attention scales.

## Report results

Report a Markdown table with each run's means for:

- `general_1_understanding`
- `general_2_interpersonal_effectiveness`
- `general_3_collaboration`
- `CBT_1_guided_discovery`
- `CBT_2_focus`
- `CBT_3_strategy`
- `Average number of turns`
- `Average length of therapist utterances`

Include the cross-run mean, CTRS file counts, per-run and total costs, retry usage, and links to each run directory.

If any simulation remains unsuccessful after its retry, or any batch dry run has a request-count/metric mismatch, stop before paid evaluation and ask the user for direction.
