# ReasonCBT

ReasonCBT contains the ODRA profiles and code for simulating, evaluating, and fine-tuning CBT therapist agents.

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

Preview the evaluation queue with:

```bash
python evaluation/eval_queue_manager.py \
  --config evaluation/configs/eval_queue_example.yaml --gpus 0 --dry-run
```

## Fine-tuning

Edit model and ChatML data paths in `finetuning/configs/`, then preview or run:

```bash
python finetuning/train_queue_manager.py \
  --config finetuning/configs/train_queue_example.yaml --gpus 0 --dry-run
```
