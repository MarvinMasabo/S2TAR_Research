#!/usr/bin/env bash
# CTR-GCN / Prosthetic (TAMUSA + Thailand) / Exact Windowing, 100 epochs, LR 0.001 + dropout 0.7 (regularization experiment).
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/ctrgcn/prosthetic_wattkg/merged_100ep_lr0001_do07.py 1 \
    --validate --test-last --test-best --seed 0
