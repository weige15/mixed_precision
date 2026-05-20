#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${ROOT_DIR}/experiments/h1-selective-fp32-norms/code/run_lora_precision.py"
POLICY_BUILDER="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/code/build_snip_style_policy.py"
RESULTS_DIR="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/results"
POLICY_DIR="${RESULTS_DIR}/snip_style_policy"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

SEEDS="${SEEDS:-42}"
BUDGETS="${BUDGETS:-4 8 16 24}"
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SEQ_LEN="${SEQ_LEN:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
HARDWARE_LABEL="${HARDWARE_LABEL:-unknown_gpu}"
RUN_TAG="${RUN_TAG:-default}"

python "${POLICY_BUILDER}" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${POLICY_DIR}" \
  --budgets ${BUDGETS}

run_policy() {
  local seed="$1"
  local budget="$2"
  local modules_file="${POLICY_DIR}/snip_style_k${budget}_modules.txt"
  local output_dir="${RESULTS_DIR}/train_${RUN_TAG}_h6_snip_style_k${budget}_seed${seed}_${MAX_STEPS}"

  python "${RUNNER}" \
    --precision-policy h6_custom_int8 \
    --fake-int8-modules $(tr '\n' ' ' < "${modules_file}") \
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
    --output-dir "${output_dir}"
}

run_bf16_baseline() {
  local seed="$1"
  local output_dir="${RESULTS_DIR}/train_${RUN_TAG}_bf16_seed${seed}_${MAX_STEPS}"

  python "${RUNNER}" \
    --precision-policy bf16_baseline \
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
    --output-dir "${output_dir}"
}

for seed in ${SEEDS}; do
  run_bf16_baseline "${seed}"
  for budget in ${BUDGETS}; do
    run_policy "${seed}" "${budget}"
  done
done
