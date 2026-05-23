#!/usr/bin/env bash
set -euo pipefail

# H8 Llama metrics launcher.
# Starts with matched bf16 and blanket QLoRA/NF4 resource runs.

GPU_ID="${GPU_ID:-0}"
HARDWARE_LABEL="${HARDWARE_LABEL:-unknown-hardware}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-512}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-2e-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/h8-hardware-aware-precision-search/results}"
RUN_BF16="${RUN_BF16:-1}"
RUN_QLORA="${RUN_QLORA:-1}"
RUN_SELECTIVE_RESCUE="${RUN_SELECTIVE_RESCUE:-0}"
RUNNER="${RUNNER:-experiments/h1-selective-fp32-norms/code/run_lora_precision.py}"

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

for seed in ${SEEDS}; do
  if [[ "${RUN_BF16}" == "1" ]]; then
    out_dir="${OUTPUT_ROOT}/llama31_8b_bf16_seed${seed}_${MAX_STEPS}_${HARDWARE_LABEL}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${RUNNER}" \
      "${common_args[@]}" \
      --seed "${seed}" \
      --precision-policy bf16_baseline \
      --output-dir "${out_dir}"
  fi

  if [[ "${RUN_QLORA}" == "1" ]]; then
    out_dir="${OUTPUT_ROOT}/llama31_8b_qlora_nf4_seed${seed}_${MAX_STEPS}_${HARDWARE_LABEL}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${RUNNER}" \
      "${common_args[@]}" \
      --seed "${seed}" \
      --precision-policy qlora_4bit_nf4 \
      --output-dir "${out_dir}"
  fi

  if [[ "${RUN_SELECTIVE_RESCUE}" == "1" ]]; then
    echo "Selective rescue is not wired yet. First verify backend support for mixed QLoRA + rescued modules." >&2
    exit 2
  fi
done

