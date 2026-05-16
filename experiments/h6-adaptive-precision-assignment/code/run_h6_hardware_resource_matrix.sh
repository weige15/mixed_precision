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

HARDWARE_LABEL="${HARDWARE_LABEL:-rtx4050-local}"
RUN_TAG="${RUN_TAG:-h6_2_hw}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EVAL_EVERY="${EVAL_EVERY:-50}"
TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-100}"
SEQ_LEN="${SEQ_LEN:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
POLICIES="${POLICIES:-bf16 fake_k24 qlora_4bit_nf4 lora_8bit_int8}"

python "${POLICY_BUILDER}" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${POLICY_DIR}" \
  --budgets 24

run_policy() {
  local seed="$1"
  local policy_label="$2"
  local output_dir="${RESULTS_DIR}/resource_${RUN_TAG}_${policy_label}_seed${seed}_${MAX_STEPS}"

  case "${policy_label}" in
    bf16)
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
      ;;
    fake_k24)
      python "${RUNNER}" \
        --precision-policy h6_custom_int8 \
        --fake-int8-modules $(tr '\n' ' ' < "${POLICY_DIR}/snip_style_k24_modules.txt") \
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
      ;;
    qlora_4bit_nf4)
      python "${RUNNER}" \
        --precision-policy qlora_4bit_nf4 \
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
      ;;
    lora_8bit_int8)
      python "${RUNNER}" \
        --precision-policy lora_8bit_int8 \
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
      ;;
    *)
      echo "Unknown policy label: ${policy_label}" >&2
      exit 2
      ;;
  esac
}

for seed in ${SEEDS}; do
  for policy in ${POLICIES}; do
    run_policy "${seed}" "${policy}"
  done
done
