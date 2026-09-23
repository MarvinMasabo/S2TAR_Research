#!/usr/bin/env bash
# PoseC3D retrain with tamed LR (0.05 -> 0.0125 + warmup) to fix the overfitting
# seen in overfitting_check.png (val MSE diverges after epoch ~5-8 at lr=0.05).
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}   # GPU 1 is free
for m in ${*:-nearest exact windowing}; do
    echo "=========  posec3d lr0125: $m  (GPU $CUDA_VISIBLE_DEVICES)  ========="
    bash tools/dist_train.sh configs/posec3d/ablebody2_wattkg/${m}_lr0125.py 1 \
        --validate --test-last --test-best --seed 0
done
