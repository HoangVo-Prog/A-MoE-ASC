#!/usr/bin/env bash
set -e

EPOCHS="${1:-10}"   

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR/src"

python -m baseline.runner \
  --locked_baseline \
  --benchmark_fusions \
  --num_seeds 3 \
  --epochs "$EPOCHS" \
  --output_dir "$ROOT_DIR/saved_model" \
  --output_name phase1_locked_baseline \
  --train_path "$ROOT_DIR/dataset/atsa/laptop14/train.json" \
  --val_path   "$ROOT_DIR/dataset/atsa/laptop14/val.json" \
  --test_path  "$ROOT_DIR/dataset/atsa/laptop14/test.json" \
  --benchmark_methods concat 
