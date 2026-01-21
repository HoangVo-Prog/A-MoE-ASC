#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1
export USE_NOHUP=1

LOSSES=(ce weighted_ce focal)
DATASETS=(rest14 rest15 rest16)

echo "▶ GPU 1: bert_spc_model, moe_ffn, moe_skconnection"

for DATASET in "${DATASETS[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    echo "[GPU1] bert_spc_model | $DATASET | $LOSS"
    bash scripts/run_bert_spc_model.sh "$LOSS" "$DATASET"

    echo "[GPU1] moe_ffn | $DATASET | $LOSS"
    bash scripts/run_moe_ffn.sh "$LOSS" "$DATASET"

    echo "[GPU1] moe_skconnection | $DATASET | $LOSS"
    bash scripts/run_moe_skconnection.sh "$LOSS" "$DATASET"
  done
done

echo "✅ GPU 1 jobs submitted (nohup)"
