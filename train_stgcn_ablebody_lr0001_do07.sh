#!/usr/bin/env bash
# STGCN / Able-bodied (AbleBody2) / Exact Windowing, 100 epochs, LR 0.001 + dropout 0.7 (regularization experiment).
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/stgcn/ablebody2_wattkg/exact_100ep_lr0001_do07.py 1 \
    --validate --test-last --test-best --seed 0
