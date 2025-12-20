#!/usr/bin/env bash
set -e

# đi về project root (cha của scripts/)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# để python thấy src/ là package root
export PYTHONPATH="$ROOT_DIR/src"

python -m baseline.runner \
  --locked_baseline \
  --benchmark_fusions \
  --num_seeds 3 \
  --output_dir "$ROOT_DIR/saved_model" \
  --output_name phase1_locked_baseline \
  --train_path "$ROOT_DIR/dataset/atsa/laptop14/train.json" \
  --val_path   "$ROOT_DIR/dataset/atsa/laptop14/val.json" \
  --test_path  "$ROOT_DIR/dataset/atsa/laptop14/test.json"
