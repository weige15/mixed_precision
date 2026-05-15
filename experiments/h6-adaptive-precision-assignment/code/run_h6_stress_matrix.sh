#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${ROOT_DIR}/experiments/h1-selective-fp32-norms/code/run_lora_precision.py"
RESULTS_DIR="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/results"

SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SEQ_LEN="${SEQ_LEN:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

run_policy() {
  local seed="$1"
  local policy="$2"
  local label="$3"
  shift 3

  python "${RUNNER}" \
    --precision-policy "${policy}" \
    --seed "${seed}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --eval-every "${EVAL_EVERY}" \
    --train-size "${TRAIN_SIZE}" \
    --eval-size "${EVAL_SIZE}" \
    --eval-max-batches "${EVAL_MAX_BATCHES}" \
    --seq-len "${SEQ_LEN}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    "$@" \
    --output-dir "${RESULTS_DIR}/stress_lr${LEARNING_RATE}_${label}_seed${seed}_${MAX_STEPS}"
}

for seed in ${SEEDS}; do
  run_policy "${seed}" "bf16_baseline" "bf16"

  run_policy "${seed}" "h6_late_mlp_int8_candidate" "h6_late_mlp_int8"

  run_policy "${seed}" "h6_custom_int8" "h6_highrisk_int8" \
    --fake-int8-modules \
    layers.2.mlp.down_proj \
    layers.3.mlp.down_proj \
    layers.21.mlp.down_proj
done
