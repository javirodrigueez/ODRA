# Handoff: run the Chinese therapist baselines (SoulChat2.0 / Psyche-R1)

Instructions for a coding agent on the deployment server. Everything below is already implemented
and unit-tested; your job is to deploy the servers, validate one session, run the 4 experiments,
score them with CTRS, and report. Do not change the experiment design (prompts, sampling
parameters, turn limits, patient model) unless a step below says so.

## 1. What this pipeline is

`agent_simulation_zh.py` runs the repo's CBT patient–therapist simulation with a Chinese therapist
model behind a translation layer:

- therapist = `ft_models/SoulChat2.0-Qwen2-7B` or `ft_models/Psyche-R1`, served by vLLM on GPU 0
- translator = `google/translategemma-12b-it`, served by vLLM on GPU 1
- patient = DeepSeek V3.2 via OpenRouter (API); the closing moderator uses the same API
- each therapist turn: patient EN→ZH (cached as `content_zh`), Chinese ChatML request per the
  model's repo conventions, generation, ZH→EN stored in `content` (what the patient, moderator and
  CTRS see). Psyche-R1 `<think>` reasoning is kept in Chinese in `reasoning_traces`.

Relevant files (all on `main`, commit `45c56bb` or later):

| File | Role |
|---|---|
| `agent_simulation_zh.py` | CLI, same flags as `agent_simulation.py` |
| `graph/zh_therapist.py` | `ZhTherapistGraphBuilder` (therapist turn, translation caching) |
| `agents/translator.py` | TranslateGemma client (`prompt_mode: chat_template` or `manual`) |
| `configs/zh_baselines/{soulchat2,psyche_r1}_{wor,wr}.yaml` | the 4 experiments (wor = no resistance, wr = resistance) |
| `agent_prompts/soulchat2_system_zh.txt`, `ft_prompts/standard_zh.txt`, `prompts/opening/initial_greeting_zh.txt` | Chinese prompts |
| `scripts/download_modelscope.sh` | curl downloader for ModelScope repos |
| `tests/test_zh_therapist.py` | unit tests (no network) |
| README section "Chinese therapist baselines" | short version of this document |

## 2. Prerequisites (check every one before launching anything)

1. Code: repository at commit `45c56bb` or later (`git log --oneline -1`). If the server does not
   share the NAS checkout, pull it; the branch was merged locally and may not be on GitHub yet.
2. Run inside the project container (`.launch_docker.sh`, image `jrodriguez/synther`, which has
   vLLM). Nothing new needs to be installed for this pipeline.
3. Weights:
   - `ft_models/SoulChat2.0-Qwen2-7B/` (15.2 GB, 4 safetensors shards; `config.json` says
     `Qwen2ForCausalLM`). If missing:
     `scripts/download_modelscope.sh YIRONGCHEN/SoulChat2.0-Qwen2-7B ft_models/SoulChat2.0-Qwen2-7B`
   - `ft_models/Psyche-R1/` (15.2 GB, HF `MindIntLab/Psyche-R1`).
   - `google/translategemma-12b-it` is downloaded by vLLM from HuggingFace; it is a **gated**
     Gemma-license repo: `HF_TOKEN` must belong to an account that accepted the terms.
4. `.env` in the repo root with `OR_API_KEY` (OpenRouter: patient + moderator) and
   `AZURE_OPENAI_KEY` / `AZURE_OPENAI_ENDPOINT` (CTRS batch scoring in `evaluation/run_batch_metrics.py`).
5. Profiles file `data/eval_full_cactus/eval_full_cactus.json` (450 profiles, private) exists.
6. `/outs` is mounted and writable (outputs go to `/outs/local/<model>-{wor,wr}`).
7. Unit tests pass: `python -m unittest tests.test_zh_therapist` → `Ran 12 tests ... OK`.

## 3. Launch the servers

Therapist on GPU 0 (`--served-model-name` MUST equal `therapist.model_id` in the config):

```bash
# SoulChat2.0 (repo convention: --max-model-len 20000)
CUDA_VISIBLE_DEVICES=0 vllm serve ft_models/SoulChat2.0-Qwen2-7B \
  --served-model-name ft_models/SoulChat2.0-Qwen2-7B --port 8000 --max-model-len 20000 \
  --enable-prefix-caching --max-num-seqs 64 > /outs/local/vllm-soulchat2.log 2>&1 &

# Psyche-R1 (later, after the two SoulChat2.0 runs; stop the SoulChat2.0 server first)
CUDA_VISIBLE_DEVICES=0 vllm serve ft_models/Psyche-R1 \
  --served-model-name ft_models/Psyche-R1 --port 8000 --max-model-len 32768 \
  --enable-prefix-caching --max-num-seqs 64 > /outs/local/vllm-psyche.log 2>&1 &
```

Translator on GPU 1 (Gemma 3 multimodal checkpoint → `--language-model-only`; bf16 needs ~25 GB,
so on a 24 GB card add `--quantization fp8`; if it still does not fit, use
`google/translategemma-4b-it` and update `translator.model_id` in the 4 configs):

```bash
CUDA_VISIBLE_DEVICES=1 HF_TOKEN=$HF_TOKEN vllm serve google/translategemma-12b-it \
  --port 8001 --max-model-len 4096 --language-model-only --max-num-seqs 64 \
  > /outs/local/vllm-translategemma.log 2>&1 &
```

Readiness: `curl -s localhost:8000/v1/models` and `curl -s localhost:8001/v1/models` list the
served names. Both servers can take several minutes to load from NFS.

## 4. Validate the translator request format (mandatory, once)

TranslateGemma reads the languages from `source_lang_code` / `target_lang_code` inside the content
part. Some vLLM versions strip those extra keys. Test:

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "google/translategemma-12b-it", "temperature": 0, "max_tokens": 200,
  "messages": [{"role": "user", "content": [{"type": "text", "source_lang_code": "en",
    "target_lang_code": "zh", "text": "I have been feeling anxious about work lately."}]}]}'
```

Expected: a Chinese translation only (e.g. 我最近对工作感到很焦虑。). If the reply is English, not a
translation, empty, or the server returns HTTP 400, set `translator.prompt_mode: manual` in all 4
configs and re-test with the manual rendering (a plain string user message); `agents/translator.py`
`MANUAL_PROMPT_TEMPLATE` shows the exact text. Also test the reverse direction
(`source_lang_code: zh`, `target_lang_code: en`) with a Chinese sentence.

## 5. Validate the therapist models (once each)

```bash
python - <<'EOF'
from openai import OpenAI
from graph.zh_therapist import PSYCHE_R1_THINK_PREFIX
c = OpenAI(api_key="test", base_url="http://localhost:8000/v1")
model = c.models.list().data[0].id
system = open("ft_prompts/standard_zh.txt" if "Psyche" in model else "agent_prompts/soulchat2_system_zh.txt", encoding="utf-8").read().strip()
user = "我最近工作压力很大，晚上总是睡不着。"
if "Psyche" in model:
  user = PSYCHE_R1_THINK_PREFIX + user
r = c.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.7, top_p=0.9, max_tokens=2048)
print(r.choices[0].message.content)
EOF
```

Expected: SoulChat2.0 → a short empathetic Chinese counselor reply. Psyche-R1 → `<think>…</think>`
followed by the Chinese answer (the parser tolerates a missing block, but most replies should have it).

## 6. Smoke test one full session per model

```bash
python agent_simulation_zh.py --cfg configs/zh_baselines/soulchat2_wor.yaml --n-profiles 1
```

Check `/outs/local/SoulChat2.0-Qwen2-7B-wor/json/simulation_<idx>_<attitude>.json`:

- `messages[*]` alternate `therapist` / `patient`, start with the therapist greeting, every message
  has English `content` and a `content_zh`
- `reasoning_traces` entries for the therapist have `step: agent_reasoning` (Chinese for Psyche-R1,
  empty for SoulChat2.0) and `input` = the last Chinese user message sent
- `session_ended_reason` is `moderator_detected_closing`, `therapist_requested_end`,
  `patient_requested_end` or `max_turns` — NOT `conversation_error` (if it is, read
  `conversation_error`; typical causes: translator returned empty text, therapist server 400 on
  context length, OpenRouter auth)
- `transcripts/*.txt` is readable English
- the manifest is `/outs/local/SoulChat2.0-Qwen2-7B-wor/outs_local_SoulChat2.0-Qwen2-7B-wor.json`

Delete the smoke-test output directory before the full run (or keep it: the manifest marks the profile
as executed and the full run will simply skip it). Repeat with `psyche_r1_wor.yaml` after switching
the therapist server.

## 7. Full runs (4 experiments)

Order (one therapist server at a time on GPU 0; the translator stays up):

```bash
# SoulChat2.0 server up
python agent_simulation_zh.py --cfg configs/zh_baselines/soulchat2_wor.yaml --multiprocessing --n-processes 20
python agent_simulation_zh.py --cfg configs/zh_baselines/soulchat2_wr.yaml  --multiprocessing --n-processes 20
# stop SoulChat2.0 server, start Psyche-R1 server
python agent_simulation_zh.py --cfg configs/zh_baselines/psyche_r1_wor.yaml --multiprocessing --n-processes 20
python agent_simulation_zh.py --cfg configs/zh_baselines/psyche_r1_wr.yaml  --multiprocessing --n-processes 20
```

- Each config uses all 450 profiles from `data/eval_full_cactus/eval_full_cactus.json`. If the
  experiment owner wants the 150-profile set used by the `cbt-evaluation` skill instead, add
  `--profiles-path data/odra.json --n-profiles 150` to every command (same for all 4).
- Interrupted or partially failed run: re-run the same command with
  `--resume-from-json <output_dir>/<manifest>.json --retry-failed`.
- Sessions ending with `max_turns` can be extended later with `--resume-input-dir <output_dir>/json`
  and a config with a larger `env.max_turns` (only if the owner asks).
- Expected failure modes and fixes:
  - therapist HTTP 400 "maximum context length": full history overflowed `--max-model-len`. Set
    `therapist.history_limit: 20` in the config (the other local baselines use a 20-message window)
    and resume with `--retry-failed`. Report that you did this.
  - translator errors/timeouts: reduce `--n-processes`; the translator gets ~2 requests per therapist turn.
  - OpenRouter 401/429: check `OR_API_KEY`, lower `--n-processes`.

Completeness gate before scoring: in each manifest, every profile has `success: true`; count JSON
files in `<output_dir>/json` (450, or 150 with the odra set) and `session_ended_reason` values.

## 8. CTRS scoring (same procedure as the `cbt-evaluation` skill: CTRS only, 3 evaluator samples)

For each of the 4 output dirs `RUN`:

```bash
python evaluation/run_batch_metrics.py --dry-run --exc-panas --exc-wai RUN/json RUN/evaluation 3   # gate: 6 CTRS requests per session
python evaluation/run_batch_metrics.py           --exc-panas --exc-wai RUN/json RUN/evaluation 3   # Azure OpenAI batch
python evaluation/rating_metrics.py --eval-dir RUN/evaluation --output-json RUN/evaluation/ctrs_metrics.json --exc-wai --exc-panas
python evaluation/turns_and_closings.py --eval-dir RUN/json
```

## 9. Report back

For each of the 4 runs: number of sessions, distribution of `session_ended_reason`, number of
`conversation_error` sessions (and the error messages), mean utterances per session, CTRS means from
`ctrs_metrics.json`, and any config change you had to make (`prompt_mode`, `history_limit`,
translator model, profile set). Include the exact vLLM commands and versions used
(`vllm --version`, `python -c "import transformers; print(transformers.__version__)"`).

## 10. Do not

- change prompts, sampling parameters, `max_turns`/`min_turns`, the patient model, or the
  translator temperature (0) — these define the experiment
- run the therapist and translator on the same GPU
- push to `main`; commit config edits on a branch and mention them in the report
