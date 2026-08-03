---
name: create-it-from-sqpsych
description: Convert an SQPsychConv dataset from Parquet to CACTUS-style JSON, select a reproducible random training subset, and create one ChatML instruction-tuning example per therapist turn. Use when Codex needs to prepare SQPsych or SQPsychConv Parquet exports for instruction tuning inside the ReasonCBT repository.
---

# Create IT from SQPsych

Run the complete pipeline with `scripts/run_pipeline.py`. The script delegates each stage to the existing ReasonCBT data scripts.

## Collect inputs

Use these defaults unless the user requests different values:

- Dataset root containing Parquet files recursively: `--input`.
- Generated-data root: `--output`; omit it to use the input root.
- Training conversations sampled uniformly without replacement: `--n-samples 150`.
- Reproducible sampling seed: `--seed 42`.
- System prompt: `ft_prompts/sqpsych.txt`.

Tell the user before execution that the pipeline replaces the output root's `json`, `json_150`, and `it_turns` directories.

## Run the pipeline

From the ReasonCBT repository root, run:

```bash
.venv/bin/python .codex/skills/create-it-from-sqpsych/scripts/run_pipeline.py \
  --input /outs/SQPsychConv_DATASET
```

Add `--output ROOT`, `--n-samples N`, `--seed SEED`, or `--system-prompt PATH` when requested. Use `--dry-run` to inspect paths and commands without deleting or creating files.

The orchestrator runs these repository scripts in order:

1. `scripts/data/sqpsych_to_cactus.py` to create `json`.
2. `scripts/data/random_from_sqpsych.py` to create `json_150` from training files only.
3. `scripts/ft/create_it_dataset_from_raw.py --mode turns` to create `it_turns`.

Use an interpreter that can import `pyarrow`; prefer `/ReasonCBT/.venv/bin/python`.

## Verify results

Require the pipeline summary to report:

- At least one converted JSON file.
- Exactly the requested number of sampled training JSON files.
- At least one instruction-tuning turn file.

Report the three output directories, converted conversation count, sampled conversation count, instruction-tuning example count, seed, and system prompt.
