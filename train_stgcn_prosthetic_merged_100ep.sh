#!/usr/bin/env bash
# STGCN / Exact Windowing, merged prosthetic cohort (TAMUSA + Thailand), 100 epochs.
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/stgcn/prosthetic_wattkg/merged_100ep.py 1 \
    --validate --test-last --test-best --seed 0
