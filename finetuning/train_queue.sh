python finetuning/train.py \
  --model models/Qwen3.5-9B-Unsloth \
  --input-dir /outs/eval_full_catus/it_multiturn/ \
  --max-seq-length 65536 \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.5-9B-chatml

python finetuning/train.py \
  --model models/Qwen3.5-4B-Unsloth \
  --input-dir /outs/eval_full_catus/it_multiturn/ \
  --max-seq-length 65536 \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.5-4B-chatml