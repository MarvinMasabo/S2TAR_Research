#!/usr/bin/env bash
# STGCN / Exact Windowing, combined able-bodied + prosthetic dataset, 100 epochs.
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/stgcn/combined_wattkg/exact_100ep.py 1 \
    --validate --test-last --test-best --seed 0
