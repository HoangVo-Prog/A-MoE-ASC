#!/usr/bin/env bash
set -e

# =========================
# Inputs
# =========================
LOSS_TYPE="${1:-ce}"
DATASET_TYPE="${2:-laptop14}"

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
    CLASS_WEIGHTS="0.74,0.84,1.42"
    FOCAL_GAMMA="2.0"
    ;;
  rest14)
    CLASS_WEIGHTS="0.4,1.1,1.5"
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

METHOD_FLAGS=""
if [[ -n "${benchmark_methods:-}" ]]; then
  METHOD_FLAGS="--benchmark_methods ${benchmark_methods}"
fi

echo "▶ Running moe ffn with:"
echo "  dataset_type   = ${DATASET_TYPE}"
echo "  loss_type      = ${LOSS_TYPE}"
echo "  loss_flags     = ${LOSS_FLAGS}"
echo "  method_flags   = ${METHOD_FLAGS}"
echo
# =========================
# Run
# =========================
python -m main \
  --mode MoEFFN \
  --model_name bert-base-uncased \
  --benchmark_fusions \
  --num_seeds 3 \
  --output_dir "$ROOT_DIR/results" \
  --output_name moe_ffn_${LOSS_TYPE}.json \
  --train_path "$ROOT_DIR/dataset/atsa/${DATASET_TYPE}/train.json" \
  --test_path  "$ROOT_DIR/dataset/atsa/${DATASET_TYPE}/test.json" \
  --route_mask_pad_tokens \
  ${LOSS_FLAGS} \
  ${METHOD_FLAGS} 
