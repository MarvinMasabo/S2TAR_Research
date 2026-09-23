#!/usr/bin/env bash
# STGCN / Exact Windowing, 100 epochs (was 24) - per Ricardo's 2026-09-11 request to see
# the true train-vs-val convergence curve for the agreed winning method.
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/stgcn/ablebody2_wattkg/exact_100ep.py 1 \
    --validate --test-last --test-best --seed 0
