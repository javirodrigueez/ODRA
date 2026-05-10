python finetuning/train.py \
  --model models/Qwen3.6-27B-Unsloth/ \
  --input-data /outs/eval_full_catus_wo_r/it_multiturn/ \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.6-27B-text-wo_r/

python finetuning/train.py \
  --model models/Qwen3.5-9B-Unsloth/ \
  --input-data /outs/eval_full_catus_wo_r/it_multiturn/ \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.5-9B-text-wo_r/