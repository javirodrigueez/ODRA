#!/usr/bin/env bash

python simulation.py --cfg configs/deepseek_or.yaml
bash evaluation/run_all_metrics.sh out/deepseek_or evaluation/deepseek_or-eval 1