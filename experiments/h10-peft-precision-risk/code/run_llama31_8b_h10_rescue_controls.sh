#!/usr/bin/env bash
set -euo pipefail

# H10 Llama selective-rescue launcher.
# Runs H10 selector policies through the existing H8 QLoRA/NF4 rescue runner.

GPU_ID="${GPU_ID:-0}"
HARDWARE_LABEL="${HARDWARE_LABEL:-rtx3090-lab}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-100}"
SEQ_LEN="${SEQ_LEN:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-2e-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/h10-peft-precision-risk/results/training}"
RUNNER="${RUNNER:-experiments/h1-selective-fp32-norms/code/run_lora_precision.py}"
H10_CANDIDATES="${H10_CANDIDATES:-experiments/h10-peft-precision-risk/results/h10_h8_runner_candidates_llama31_8b.json}"
H10_POLICIES="${H10_POLICIES:-h10_activation_outlier_top4}"
H10_RESCUE_PRECISION="${H10_RESCUE_PRECISION:-bf16}"
SETUP_ONLY="${SETUP_ONLY:-0}"

mkdir -p "${OUTPUT_ROOT}"

common_args=(
  --model-name "${MODEL_NAME}"
  --max-steps "${MAX_STEPS}"
  --seq-len "${SEQ_LEN}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
  --learning-rate "${LR}"
  --eval-max-batches "${EVAL_MAX_BATCHES}"
  --hardware-label "${HARDWARE_LABEL}"
)

setup_flag=()
if [[ "${SETUP_ONLY}" == "1" ]]; then
  setup_flag=(--setup-only)
fi

for seed in ${SEEDS}; do
  for policy_name in ${H10_POLICIES}; do
    suffix="${MAX_STEPS}_${HARDWARE_LABEL}"
    if [[ "${SETUP_ONLY}" == "1" ]]; then
      suffix="setup_${HARDWARE_LABEL}"
    fi
    out_dir="${OUTPUT_ROOT}/llama31_8b_${policy_name}_${H10_RESCUE_PRECISION}_seed${seed}_${suffix}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${RUNNER}" \
      "${common_args[@]}" \
      --seed "${seed}" \
      --precision-policy h8_qlora_nf4_rescue_projection_top4 \
      --h8-candidates "${H10_CANDIDATES}" \
      --h8-policy-name "${policy_name}" \
      --h8-rescue-precision "${H10_RESCUE_PRECISION}" \
      --output-dir "${out_dir}" \
      "${setup_flag[@]}"
  done
done

