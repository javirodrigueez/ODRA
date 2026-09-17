# ODRA: Synthesizing Cognitive Behavioral Therapy Sessions with Structured Chain-Of-Thought and Dynamic Patient Resistance

## Structure

- `data/odra.json`: evaluation profiles.
- `agents/`, `graph/`, `prompts/`: simulation logic and prompts.
- `evaluation/`: CTRS batch scoring, summaries, and queue runner.
- `finetuning/`: training and queue runner.
- `.codex/skills/cbt-evaluation/`: reproducible end-to-end evaluation workflow.

## Evaluation

To generate sessions with OpenAI GPT-4o-mini for both therapist and patient,
set `GPT_API_KEY` in your environment or `.env`, then run from the repository root:

```bash
python simulation.py --cfg configs/gpt4o_mini.yaml --n-profiles 3
```

Edit `configs/gpt4o_mini.yaml` to configure the models, temperature, patient
profiles, output folder, and turn limits. Omit `--n-profiles` to run the full
dataset; add `--multiprocessing --n-processes N` for parallel simulations.
The default input is `data/eval_full_cactus/eval_full_cactus.json` and the output
is `/outs/eval_full_cactus_gpt4o_mini`. The private CACTUS profile file must exist
locally; for a public checkout, supply `--profiles-path data/odra.json`.
Resume using `--resume-from-json PATH_TO_MANIFEST` with the same config.

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

## Chinese therapist baselines (SoulChat2.0 / Psyche-R1)

`agent_simulation_zh.py` runs the same simulation with a Chinese therapist model behind a
translation layer: patient utterances are translated EN→ZH before reaching the therapist and
therapist utterances ZH→EN before reaching the patient, the closing moderator and CTRS. The English
text is stored in `content`, the Chinese original in `content_zh`; Psyche-R1 reasoning traces are kept
in Chinese. The therapist follows each repo's conventions (SoulChat2.0: PsyDT REBT system prompt, full
history, temperature 0.75 / top_p 0.9; Psyche-R1: Chinese `ft_prompts/standard_zh.txt` system prompt,
`<think>` prefix from its `run.py`, temperature 0.7 / top_p 0.9). Configs live in `configs/zh_baselines/`.

Weights: `ft_models/Psyche-R1` from HuggingFace (`MindIntLab/Psyche-R1`); SoulChat2.0 is only on ModelScope:

```bash
scripts/download_modelscope.sh YIRONGCHEN/SoulChat2.0-Qwen2-7B ft_models/SoulChat2.0-Qwen2-7B
```

Serve the therapist on one GPU and the translator on the other (`--served-model-name` must match
`therapist.model_id`; `google/translategemma-12b-it` is a gated Gemma-license repo, so `HF_TOKEN`
must have accepted its terms. It is a Gemma 3 multimodal checkpoint with a 2K-token input context;
at bf16 it needs ~25 GB, so on a 24 GB card add `--quantization fp8` or use `translategemma-4b-it`):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ft_models/SoulChat2.0-Qwen2-7B --served-model-name ft_models/SoulChat2.0-Qwen2-7B --port 8000 --max-model-len 20000
# or: CUDA_VISIBLE_DEVICES=0 vllm serve ft_models/Psyche-R1 --served-model-name ft_models/Psyche-R1 --port 8000 --max-model-len 32768
CUDA_VISIBLE_DEVICES=1 vllm serve google/translategemma-12b-it --port 8001 --max-model-len 4096 --language-model-only
python agent_simulation_zh.py --cfg configs/zh_baselines/soulchat2_wor.yaml --n-profiles 1   # smoke test
python agent_simulation_zh.py --cfg configs/zh_baselines/soulchat2_wor.yaml --multiprocessing --n-processes 8 --eval
```

TranslateGemma takes the languages from `source_lang_code` / `target_lang_code` fields inside the
content part, but vLLM's chat endpoint strips those keys before the model's chat template runs (HTTP
400 `'dict object' has no attribute ''`). `agents/translator.py` therefore renders the model's own
chat template client-side and posts it to `/v1/completions` with `add_special_tokens: false`; this is
token-identical to the chat path and needs no special server flags. Full-history requests can exceed
`--max-model-len` late in long sessions; set `therapist.history_limit` (e.g. 20, as the other local
baselines use) or raise `--max-model-len` if that happens.
