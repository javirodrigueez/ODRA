#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/outs/eval_full_catus_wo_r"
WANDB_GROUP="wo_r_wo_think"

# LLAMA
python finetuning/train.py \
  --model models/Llama3-8B-Instruct/ \
  --input-data "${DATA_ROOT}/it_turns/" \
  --chat-template none \
  --qlora \
  --save-merged \
  --group "${WANDB_GROUP}" \
  --exp-name "Llama3-8B-Instruct-turns-ODRA" \
  --output-dir "ft_models/Llama3-8B-Instruct-turns-ODRA/"

python finetuning/train.py \
  --model models/Llama3-8B-Instruct/ \
  --input-data "/outs/cactus_150/it_turns/" \
  --chat-template none \
  --qlora \
  --save-merged \
  --group "${WANDB_GROUP}" \
  --exp-name "Llama3-8B-Instruct-turns-CACTUS" \
  --output-dir "ft_models/Llama3-8B-Instruct-turns-CACTUS/"

python finetuning/train.py \
  --model models/Llama3-8B-Instruct/ \
  --input-data "/outs/magnet/magnet_150/it_turns/" \
  --chat-template none \
  --qlora \
  --save-merged \
  --group "${WANDB_GROUP}" \
  --exp-name "Llama3-8B-Instruct-turns-MAGNET" \
  --output-dir "ft_models/Llama3-8B-Instruct-turns-MAGNET/"

# QWEN
python finetuning/train.py \
  --model models/Qwen3.5-9B-Unsloth/ \
  --input-data "${DATA_ROOT}/it_turns/" \
  --chat-template none \
  --qlora \
  --save-merged \
  --group "${WANDB_GROUP}" \
  --exp-name "Qwen3.5-9B-Unsloth-turns-ODRA" \
  --output-dir "ft_models/Qwen3.5-9B-Unsloth-turns-ODRA/"

python finetuning/train.py \
  --model models/Qwen3.5-9B-Unsloth/ \
  --input-data "/outs/cactus_150/it_turns/" \
  --chat-template none \
  --qlora \
  --save-merged \
  --group "${WANDB_GROUP}" \
  --exp-name "Qwen3.5-9B-Unsloth-turns-CACTUS" \
  --output-dir "ft_models/Qwen3.5-9B-Unsloth-turns-CACTUS/"