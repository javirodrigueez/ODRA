# python finetuning/train.py \
#   --model models/Qwen3.5-9B-Unsloth/ \
#   --input-data /outs/eval_full_catus_wo_r/it_think_json_turns/ \
#   --qlora \
#   --save-merged \
#   --output-dir ft_models/Qwen3.5-9B-json-turns-wo_r-bs3/

python finetuning/train.py \
  --model models/Qwen3.5-9B-Unsloth/ \
  --input-data /outs/eval_full_catus_wo_r/it_think_text_turns/ \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.5-9B-text-turns-wo_r-bs3/