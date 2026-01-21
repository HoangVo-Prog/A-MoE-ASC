#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

LOG=logs/queue_gpu1.log
mkdir -p logs

nohup bash -c '
for DATASET in rest14 rest15 rest16; do
  for LOSS in ce weighted_ce focal; do
    echo "[GPU1] bert_spc $DATASET $LOSS"
    bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate hoang; scripts/run_bert_spc_model.sh \"$LOSS\" \"$DATASET\""

    echo "[GPU1] moe_ffn $DATASET $LOSS"
    bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate hoang; scripts/run_moe_ffn.sh \"$LOSS\" \"$DATASET\""

    echo "[GPU1] moe_skconnection $DATASET $LOSS"
    bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate hoang; scripts/run_moe_skconnection.sh \"$LOSS\" \"$DATASET\""
  done
done
' > "$LOG" 2>&1 &
