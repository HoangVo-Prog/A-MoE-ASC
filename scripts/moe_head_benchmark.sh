#!/usr/bin/env bash
set -e

# =========================
# Inputs
# =========================
EPOCHS="${1:-10}"
TOP_K="${2:-1}"
LOSS_TYPE="${3:-ce}"

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
    # Cross-Entropy
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

echo "▶ Running moe head with:"
echo "  epochs     = ${EPOCHS}"
echo "  top_k      = ${TOP_K}"
echo "  loss_type  = ${LOSS_TYPE}"
echo "  loss_flags = ${LOSS_FLAGS}"
echo

# =========================
# Run
# =========================
python -m moe_head.runner \
  --locked_baseline \
  --benchmark_fusions \
  --num_seeds 3 \
  --epochs "$EPOCHS" \
  --moe_top_k "${TOP_K}" \
  --output_dir "$ROOT_DIR/saved_model" \
  --output_name phase1_locked_baseline \
  --train_path "$ROOT_DIR/dataset/atsa/laptop14/train.json" \
  --val_path   "$ROOT_DIR/dataset/atsa/laptop14/val.json" \
  --test_path  "$ROOT_DIR/dataset/atsa/laptop14/test.json" \
  --use_moe \
  --route_mask_pad_tokens \
  --aux_loss_weight 0.01 \
  --step_print_moe 100 \
  --amp_dtype fp16 \
  ${LOSS_FLAGS}