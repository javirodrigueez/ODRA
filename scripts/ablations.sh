#!/usr/bin/env bash

## no calibration
python simulation.py \
    --cfg configs/eval_no_calibration.yaml \
    --profiles-path data/eval_full_cactus/eval_full_cactus.json \
    --n-profiles 121 \
    --resume-from-json /outs/eval_full_catus_no_calib/outs_eval_full_catus_no_calib.json \
    --multiprocessing \
    --n-processes 30

## no few-shot
python simulation.py \
    --cfg configs/eval_no_fewshot.yaml \
    --profiles-path data/eval_full_cactus/eval_full_cactus.json \
    --n-profiles 150 \
    --multiprocessing \
    --n-processes 30

## no should not
python simulation.py \
    --cfg configs/eval_no_shouldnot.yaml \
    --profiles-path data/eval_full_cactus/eval_full_cactus.json \
    --n-profiles 150 \
    --multiprocessing \
    --n-processes 30