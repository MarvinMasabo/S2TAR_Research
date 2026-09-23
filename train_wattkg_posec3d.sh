#!/usr/bin/env bash
# Train PoseC3D (SlowOnly-R50) joint on the three W/kg method targets.
set -e
cd /home/students/mmasabo1/summer26Research
export PATH=/home/students/mmasabo1/miniconda3/envs/pyskl_310/bin:$PATH
GPUS=1
METHODS=${*:-"nearest exact windowing"}
for m in $METHODS; do
    echo "=================  posec3d training: $m  ================="
    bash tools/dist_train.sh configs/posec3d/ablebody2_wattkg/$m.py $GPUS \
        --validate --test-last --test-best --seed 0
done
