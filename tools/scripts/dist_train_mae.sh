#!/bin/bash
# scripts/dist_train_mae.sh - R-MAE Distributed Training Script

set -x
NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train_ssl.py --launcher pytorch ${PY_ARGS}