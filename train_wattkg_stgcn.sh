#!/usr/bin/env bash
# Train STGCN joint on all three W/kg method targets (nearest / exact / windowing).
# Single GPU. --validate runs per-epoch val eval; --test-best/-last test at the end.
set -e
cd /home/students/mmasabo1/summer26Research

export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
GPUS=1
METHODS=${*:-"nearest exact windowing"}

for m in $METHODS; do
    echo "=================  training: $m  ================="
    bash tools/dist_train.sh configs/stgcn/ablebody2_wattkg/$m.py $GPUS \
        --validate --test-last --test-best --seed 0
done
