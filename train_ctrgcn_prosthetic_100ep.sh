#!/usr/bin/env bash
# CTR-GCN / Exact Windowing, prosthetic-only cohort (TAMUSA / PRO_A&M), 100 epochs.
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/ctrgcn/prosthetic_wattkg/exact_100ep.py 1 \
    --validate --test-last --test-best --seed 0
