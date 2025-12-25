#!/usr/bin/env bash
set -e

# =========================
# Inputs
# =========================
LOSS_TYPE="${1:-ce}"
DATASET_TYPE="${2:-laptop14}"
EXPERTS="${3:-sent,term,concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction}"

# Only support these dataset types
case "${DATASET_TYPE}" in
  laptop14|rest14)
    ;;
  *)
    echo "❌ Unsupported dataset_type: ${DATASET_TYPE}"
    echo "Supported: laptop14 | rest14"
    exit 1
    ;;
esac

# =========================
# Project root
# =========================
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

# =========================
# Dataset-specific weights/gamma (ABSA 3-class)
# Order: (Positive, Negative, Neutral)
# =========================
CLASS_WEIGHTS=""
FOCAL_GAMMA=""

case "${DATASET_TYPE}" in
  laptop14)
    CLASS_WEIGHTS="1.0,1.6,1.2"
    FOCAL_GAMMA="2.0"
    ;;
  rest14)
    CLASS_WEIGHTS="1.4,2.2,0.8"
    FOCAL_GAMMA="2.0"
    ;;
esac

# =========================
# Loss-specific flags
# =========================
LOSS_FLAGS="--loss_type ${LOSS_TYPE}"

case "${LOSS_TYPE}" in
  ce)
    # Plain CE
    ;;
  weighted_ce)
    # Weighted CE uses dataset-specific alpha
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights ${CLASS_WEIGHTS}"
    ;;
  focal)
    # Focal uses dataset-specific alpha and gamma
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights ${CLASS_WEIGHTS} --focal_gamma ${FOCAL_GAMMA}"
    ;;
  *)
    echo "❌ Unsupported loss_type: ${LOSS_TYPE}"
    echo "Supported: ce | weighted_ce | focal"
    exit 1
    ;;
esac

# =========================
# MoF include sent/term flag (store_true)
# =========================

echo "▶ Running benchmark baseline with:"
echo "  dataset_type           = ${DATASET_TYPE}"
echo "  loss_type              = ${LOSS_TYPE}"
echo "  loss_flags             = ${LOSS_FLAGS}"
echo "  moe_experts            = ${EXPERTS}"
echo

# =========================
# Run
# =========================
python -m mof.runner \
  --locked_baseline \
  --benchmark_fusions \
  --num_seeds 3 \
  --output_dir "$ROOT_DIR/saved_model" \
  --output_name phase1_locked_baseline \
  --train_path "$ROOT_DIR/dataset/atsa/${DATASET_TYPE}/train.json" \
  --val_path   "$ROOT_DIR/dataset/atsa/${DATASET_TYPE}/val.json" \
  --test_path  "$ROOT_DIR/dataset/atsa/${DATASET_TYPE}/test.json" \
  --head_type mof \
  --mof_experts "$EXPERTS"\
  --mof_lb_coef 0.001 \
  --mof_expert_norm_clamp 5.0 \
  --mof_debug \
  ${LOSS_FLAGS} \
