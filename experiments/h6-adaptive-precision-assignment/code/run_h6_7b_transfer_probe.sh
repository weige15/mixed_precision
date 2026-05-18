#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIGNAL_PROBE="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/code/probe_stability_signals.py"
PERTURB_PROBE="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/code/probe_precision_perturbations.py"
RESULTS_DIR="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/results"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
SEEDS="${SEEDS:-42}"
RUN_TAG="${RUN_TAG:-h6_4_qwen7b_transfer}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-4}"
DATASET_SIZE="${DATASET_SIZE:-128}"
DTYPE="${DTYPE:-bf16}"
BITS="${BITS:-8}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

MODULES=(
  base_model.model.model.layers.2.mlp.down_proj
  base_model.model.model.layers.3.mlp.down_proj
  base_model.model.model.layers.24.mlp.down_proj
  base_model.model.model.layers.26.mlp.gate_proj
  base_model.model.model.layers.26.mlp.up_proj
  base_model.model.model.layers.27.mlp.gate_proj
  base_model.model.model.layers.27.mlp.up_proj
  base_model.model.model.layers.4.input_layernorm
  base_model.model.model.layers.4.post_attention_layernorm
  base_model.model.model.layers.2.self_attn.o_proj
  base_model.model.model.layers.26.self_attn.q_proj
  base_model.model.model.layers.26.self_attn.o_proj
  base_model.model.model.norm
  base_model.model.lm_head
)

LOCAL_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_ARGS=(--local-files-only)
fi

for seed in ${SEEDS}; do
  signal_dir="${RESULTS_DIR}/${RUN_TAG}_signals_seed${seed}"
  perturb_dir="${RESULTS_DIR}/${RUN_TAG}_perturb_seed${seed}"

  python "${SIGNAL_PROBE}" \
    --model-name "${MODEL_NAME}" \
    --seed "${seed}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --calibration-batches "${CALIBRATION_BATCHES}" \
    --dataset-size "${DATASET_SIZE}" \
    --dtype "${DTYPE}" \
    --policy-name "${RUN_TAG}_seed${seed}" \
    --modules "${MODULES[@]}" \
    --output-dir "${signal_dir}" \
    "${LOCAL_ARGS[@]}"

  python "${PERTURB_PROBE}" \
    --model-name "${MODEL_NAME}" \
    --seed "${seed}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --calibration-batches "${CALIBRATION_BATCHES}" \
    --dataset-size "${DATASET_SIZE}" \
    --dtype "${DTYPE}" \
    --bits ${BITS} \
    --candidate-policy "${signal_dir}/policy_trace.json" \
    --modules "${MODULES[@]}" \
    --output-dir "${perturb_dir}" \
    "${LOCAL_ARGS[@]}"
done
