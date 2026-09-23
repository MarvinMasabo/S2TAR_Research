#!/usr/bin/env bash
# CTR-GCN / Exact Windowing, 100 epochs - same "train longer, plot train vs val" treatment
# given to STGCN (see train_stgcn_exact_100ep.sh).
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
bash tools/dist_train.sh configs/ctrgcn/ablebody2_wattkg/exact_100ep.py 1 \
    --validate --test-last --test-best --seed 0
