#!/usr/bin/env bash
set -e

# =========================
# Inputs
# =========================
EPOCHS="${1:-10}"
LOSS_TYPE="${2:-ce}"

# =========================
# Project root
# =========================
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

# =========================
# Loss-specific flags
# =========================
LOSS_FLAGS="--loss_type ${LOSS_TYPE}"

case "${LOSS_TYPE}" in
  ce)
    # CE 
    ;;
  weighted_ce)
    # Neutral ~20% → boost neutral
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights 1.0,2.5,1.0"
    ;;
  focal)
    # Focal + class weight
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights 1.0,2.5,1.0 --focal_gamma 2.0"
    ;;
  *)
    echo "❌ Unsupported loss_type: ${LOSS_TYPE}"
    echo "Supported: ce | weighted_ce | focal"
    exit 1
    ;;
esac

echo "▶ Running benchmark baseline with:"
echo "  epochs     = ${EPOCHS}"
echo "  loss_type  = ${LOSS_TYPE}"
echo "  loss_flags = ${LOSS_FLAGS}"
echo

# =========================
# Run
# =========================
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
  ${LOSS_FLAGS}
