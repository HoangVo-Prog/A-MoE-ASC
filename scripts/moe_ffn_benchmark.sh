#!/usr/bin/env bash
set -e

# ===== Nhận input =====
EPOCHS="${1:-10}"   

echo "Running with epochs = $EPOCHS"

# ===== Đi về project root =====
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ===== Để python thấy src/ là package root =====
export PYTHONPATH="$ROOT_DIR/src"

python -m moe_ffn.runner \
  --locked_baseline \
  --benchmark_fusions \
  --num_seeds 3 \
  --epochs "$EPOCHS" \
  --output_dir "$ROOT_DIR/saved_model" \
  --output_name phase1_locked_baseline \
  --train_path "$ROOT_DIR/dataset/atsa/laptop14/train.json" \
  --val_path   "$ROOT_DIR/dataset/atsa/laptop14/val.json" \
  --test_path  "$ROOT_DIR/dataset/atsa/laptop14/test.json" \
  --use_moe \
  --route_mask_pad_tokens \
  --aux_loss_weight 0.01 \
  --step_print_moe 100 \
  --amp_dtype fp16
