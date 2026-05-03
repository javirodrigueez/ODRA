python finetuning/train.py \
  --model models/Qwen3.6-27B-Unsloth \
  --input-dir /outs/eval_full_catus/it_multiturn_v2/ \
  --max-seq-length 65536 \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.6-27B-chatml-v2 \

python finetuning/train.py \
  --model models/Qwen3.5-9B-Unsloth \
  --input-dir /outs/eval_full_catus/it_multiturn/ \
  --max-seq-length 65536 \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.5-9B-chatml-v2 \

python finetuning/train.py \
  --model models/Qwen3.5-4B-Unsloth \
  --input-dir /outs/eval_full_catus/it_multiturn/ \
  --max-seq-length 65536 \
  --qlora \
  --save-merged \
  --output-dir ft_models/Qwen3.5-4B-chatml-v2 \