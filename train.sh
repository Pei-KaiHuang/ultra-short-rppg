#!/usr/bin/env bash
set -euo pipefail

# Default cross-domain reproduction: train on PURE+UBFC and test on VIPL-HR.
# Override these paths when running on your machine:
#   RPPG_DATA_ROOT=/path/to/data RPPG_OUTPUT_DIR=/path/to/results bash train.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${RPPG_DATA_ROOT:-./data}"
OUTPUT_DIR="${RPPG_OUTPUT_DIR:-./results}"
LOG_DIR="${RPPG_LOG_DIR:-./logs}"
CACHE_DIR="${RPPG_CACHE_DIR:-./cache/preprocessed}"

CONV="${CONV:-LDC_M}"
MODEL_S="${MODEL_S:-2}"
EPOCH_TRAIN="${EPOCH_TRAIN:-30}"
EPOCH_CGAN="${EPOCH_CGAN:-2500}"
EPOCH_ADAPT="${EPOCH_ADAPT:-100}"
TRAIN_DATASET="${TRAIN_DATASET:-P,U}"
TEST_DATASET="${TEST_DATASET:-V}"
TRAIN_FOLD="${TRAIN_FOLD:-1}"
TEST_FOLD="${TEST_FOLD:-5}"
TEST_SEQ="${TEST_SEQ:-60}"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --log_dir "$LOG_DIR"
  --cache_dir "$CACHE_DIR"
  --conv "$CONV"
  --model_S "$MODEL_S"
)

echo "=== Training periodicity-guided rPPG model ==="
"$PYTHON_BIN" scripts/train_label.py \
  "${COMMON_ARGS[@]}" \
  --train_dataset "$TRAIN_DATASET" \
  --epoch "$EPOCH_TRAIN" \
  --bs 6 \
  --lr 0.0001 \
  --testFold "$TRAIN_FOLD"

echo "=== Selecting the best rPPG checkpoint ==="
"$PYTHON_BIN" scripts/test_label.py \
  "${COMMON_ARGS[@]}" \
  --train_dataset "$TRAIN_DATASET" \
  --test_dataset "$TEST_DATASET" \
  --epoch "$EPOCH_TRAIN" \
  --bs 1 \
  --testFold "$TEST_FOLD" \
  --test_seq "$TEST_SEQ"

echo "=== Training base signal reconstruction generator ==="
"$PYTHON_BIN" scripts/train_cGAN.py \
  "${COMMON_ARGS[@]}" \
  --train_dataset "$TRAIN_DATASET" \
  --epoch "$EPOCH_CGAN" \
  --bs 100 \
  --lr 0.001

echo "=== Adapting generator with real training clips ==="
"$PYTHON_BIN" scripts/train_adapt_G.py \
  "${COMMON_ARGS[@]}" \
  --train_dataset "$TRAIN_DATASET" \
  --epoch "$EPOCH_ADAPT" \
  --bs 6 \
  --lr 0.00005 \
  --testFold "$TRAIN_FOLD"

echo "=== Evaluating ultra-short HR estimation on ${TEST_DATASET} ==="
"$PYTHON_BIN" scripts/test_cGAN_G.py \
  "${COMMON_ARGS[@]}" \
  --train_dataset "$TRAIN_DATASET" \
  --test_dataset "$TEST_DATASET" \
  --epoch "$EPOCH_ADAPT" \
  --bs 1 \
  --testFold "$TEST_FOLD" \
  --test_seq "$TEST_SEQ"
