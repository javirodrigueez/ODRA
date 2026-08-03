# ODRA: Synthesizing Cognitive Behavioral Therapy Sessions with Structured Chain-Of-Thought and Dynamic Patient Resistance

## Structure

- `data/odra.json`: evaluation profiles.
- `agents/`, `graph/`, `prompts/`: simulation logic and prompts.
- `evaluation/`: CTRS batch scoring, summaries, and queue runner.
- `finetuning/`: training and queue runner.
- `.codex/skills/cbt-evaluation/`: reproducible end-to-end evaluation workflow.

## Evaluation

Run the complete simulation, CTRS evaluation, and summary pipeline:

```bash
python .codex/skills/cbt-evaluation/scripts/run_cbt_evaluation.py \
  --input models/CHECKPOINT --output /outs/local/evaluation --evaluations 1
```

For existing results, generate CTRS and conversation summaries with:

```bash
python evaluation/rating_metrics.py --eval-dir RUN/evaluation \
  --output-json RUN/ctrs_metrics.json --exc-wai --exc-panas
python evaluation/turns_and_closings.py --dir RUN/json
```

You can also use the `cbt-evaluation` skill in Codex or another AI Coding assistant to generate a simulation dataset and evaluate it using CTRS. Specify the local therapist model you want to deploy using vLLM when using this skill. 

## Fine-tuning

Edit model and ChatML data paths in `finetuning/configs/`, then preview or run:

```bash
python finetuning/train_queue_manager.py \
  --config finetuning/configs/train_queue_example.yaml --gpus 0 --dry-run
```
