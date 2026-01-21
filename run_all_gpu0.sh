#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

LOSSES=(ce weighted_ce focal)
DATASETS=(rest14 rest15 rest16)

LOG=logs/queue_gpu0.log
mkdir -p logs

nohup bash -c '
for DATASET in rest14 rest15 rest16; do
  for LOSS in ce weighted_ce focal; do
    echo "[GPU0] base_model $DATASET $LOSS"
    bash -lc "scripts/run_base_model.sh \"$LOSS\" \"$DATASET\""

    echo "[GPU0] hagmoe $DATASET $LOSS"
    bash -lc "scripts/run_hagmoe_model.sh \"$LOSS\" \"$DATASET\""

    echo "[GPU0] moe_head $DATASET $LOSS"
    bash -lc "scripts/run_moe_head.sh \"$LOSS\" \"$DATASET\""
  done
done
' > "$LOG" 2>&1 &
