#!/usr/bin/env bash
# CTR-GCN / able-bodied vs prosthetic classification on the combined dataset, 100 epochs.
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/ctrgcn/population_cls/able_vs_prosthetic_100ep.py 1 \
    --validate --test-last --test-best --seed 0
