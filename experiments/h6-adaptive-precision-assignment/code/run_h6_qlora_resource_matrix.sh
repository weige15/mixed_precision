#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${ROOT_DIR}/experiments/h1-selective-fp32-norms/code/run_lora_precision.py"
RESULTS_DIR="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/results"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

HARDWARE_LABEL="${HARDWARE_LABEL:-unknown_gpu}"
RUN_TAG="${RUN_TAG:-${HARDWARE_LABEL}}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SEQ_LEN="${SEQ_LEN:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

run_policy() {
  local seed="$1"
  local policy="$2"
  local label="$3"

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
    --hardware-label "${HARDWARE_LABEL}" \
    --output-dir "${RESULTS_DIR}/resource_${RUN_TAG}_lr${LEARNING_RATE}_${label}_seed${seed}_${MAX_STEPS}"
}

for seed in ${SEEDS}; do
  run_policy "${seed}" "bf16_baseline" "bf16"
  run_policy "${seed}" "qlora_4bit_nf4" "qlora_4bit_nf4"
done
