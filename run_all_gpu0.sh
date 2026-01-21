#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export USE_NOHUP=1

LOSSES=(ce weighted_ce focal)
DATASETS=(rest14 rest15 rest16)

echo "▶ GPU 0: base_model, hagmoe, moe_head"

for DATASET in "${DATASETS[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    echo "[GPU0] base_model | $DATASET | $LOSS"
    bash scripts/run_base_model.sh "$LOSS" "$DATASET"

    echo "[GPU0] hagmoe | $DATASET | $LOSS"
    bash scripts/run_hagmoe_model.sh "$LOSS" "$DATASET"

    echo "[GPU0] moe_head | $DATASET | $LOSS"
    bash scripts/run_moe_head.sh "$LOSS" "$DATASET"
  done
done

echo "✅ GPU 0 jobs submitted (nohup)"
