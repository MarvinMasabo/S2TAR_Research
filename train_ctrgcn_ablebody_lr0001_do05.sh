#!/usr/bin/env bash
# CTR-GCN / Able-bodied (AbleBody2) / Exact Windowing, 100 epochs, LR 0.001 + dropout 0.5 (regularization experiment).
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/ctrgcn/ablebody2_wattkg/exact_100ep_lr0001_do05.py 1 \
    --validate --test-last --test-best --seed 0
