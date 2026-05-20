#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIGNAL_PROBE="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/code/probe_stability_signals.py"
PERTURB_PROBE="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/code/probe_precision_perturbations.py"
TRAIN_RUNNER="${ROOT_DIR}/experiments/h1-selective-fp32-norms/code/run_lora_precision.py"
DATASET_BUILDER="${ROOT_DIR}/experiments/h7-precision-predictor/code/build_precision_dataset.py"
PREDICTOR_TRAINER="${ROOT_DIR}/experiments/h7-precision-predictor/code/train_precision_predictor.py"
POLICY_SELECTOR="${ROOT_DIR}/experiments/h7-precision-predictor/code/select_policy_from_predictions.py"

H6_RESULTS_DIR="${ROOT_DIR}/experiments/h6-adaptive-precision-assignment/results"
H7_RESULTS_DIR="${ROOT_DIR}/experiments/h7-precision-predictor/results"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
RUN_TAG="${RUN_TAG:-llama31_8b_h7_rich}"
SEEDS="${SEEDS:-42 43 44}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-4}"
DATASET_SIZE="${DATASET_SIZE:-128}"
DTYPE="${DTYPE:-bf16}"
BITS="${BITS:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-100}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
HARDWARE_LABEL="${HARDWARE_LABEL:-rtx3090-lab}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

RUN_PROBES="${RUN_PROBES:-1}"
RUN_PREDICTOR="${RUN_PREDICTOR:-1}"
RUN_TRAINING="${RUN_TRAINING:-0}"
RUN_RESOURCE_BASELINES="${RUN_RESOURCE_BASELINES:-0}"
MODULE_FILE="${MODULE_FILE:-}"

if [[ -n "${MODULE_FILE}" ]]; then
  mapfile -t MODULES < "${MODULE_FILE}"
else
  MODULES=(
    base_model.model.model.layers.2.mlp.down_proj
    base_model.model.model.layers.3.mlp.down_proj
    base_model.model.model.layers.28.mlp.down_proj
    base_model.model.model.layers.30.mlp.gate_proj
    base_model.model.model.layers.30.mlp.up_proj
    base_model.model.model.layers.31.mlp.gate_proj
    base_model.model.model.layers.31.mlp.up_proj
    base_model.model.model.layers.4.input_layernorm
    base_model.model.model.layers.4.post_attention_layernorm
    base_model.model.model.layers.2.self_attn.o_proj
    base_model.model.model.layers.30.self_attn.q_proj
    base_model.model.model.layers.30.self_attn.o_proj
    base_model.model.model.norm
    base_model.model.lm_head
  )
fi

LOCAL_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_ARGS=(--local-files-only)
fi

mkdir -p "${H7_RESULTS_DIR}"

if [[ "${RUN_PROBES}" == "1" ]]; then
  for seed in ${SEEDS}; do
    signal_dir="${H7_RESULTS_DIR}/${RUN_TAG}_signals_seed${seed}"
    perturb_dir="${H7_RESULTS_DIR}/${RUN_TAG}_perturb_seed${seed}"

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
fi

if [[ "${RUN_PREDICTOR}" == "1" ]]; then
  python "${DATASET_BUILDER}" \
    --results-roots "${H6_RESULTS_DIR}" "${H7_RESULTS_DIR}" \
    --output "${H7_RESULTS_DIR}/precision_dataset_with_llama31_8b.csv"

  python "${PREDICTOR_TRAINER}" \
    --input "${H7_RESULTS_DIR}/precision_dataset_with_llama31_8b.csv" \
    --eval-mode both \
    --output "${H7_RESULTS_DIR}/predictor_metrics_with_llama31_8b.json" \
    --predictions-output "${H7_RESULTS_DIR}/predictions_with_llama31_8b.csv"

  python "${POLICY_SELECTOR}" \
    --predictions "${H7_RESULTS_DIR}/predictions_with_llama31_8b.csv" \
    --output "${H7_RESULTS_DIR}/selected_policy_llama31_8b_mlp_gate_up.json" \
    --modules-output "${H7_RESULTS_DIR}/selected_modules_llama31_8b_mlp_gate_up.txt" \
    --split cross_scale_0p5b_to_7b \
    --model-size-min 8 \
    --top-k 4 \
    --roles mlp_projection \
    --leaves gate_proj up_proj
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  if [[ ! -s "${H7_RESULTS_DIR}/selected_modules_llama31_8b_mlp_gate_up.txt" ]]; then
    echo "Missing selected module file. Run with RUN_PREDICTOR=1 first." >&2
    exit 1
  fi
  mapfile -t SELECTED_MODULES < "${H7_RESULTS_DIR}/selected_modules_llama31_8b_mlp_gate_up.txt"
  for seed in ${TRAIN_SEEDS}; do
    python "${TRAIN_RUNNER}" \
      --model-name "${MODEL_NAME}" \
      --precision-policy h6_custom_int8 \
      --fake-int8-modules "${SELECTED_MODULES[@]}" \
      --seed "${seed}" \
      --max-steps "${MAX_STEPS}" \
      --learning-rate "${LEARNING_RATE}" \
      --eval-every "${EVAL_EVERY}" \
      --seq-len "${SEQ_LEN}" \
      --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM}" \
      --train-size "${TRAIN_SIZE}" \
      --eval-size "${EVAL_SIZE}" \
      --eval-max-batches "${EVAL_MAX_BATCHES}" \
      --hardware-label "${HARDWARE_LABEL}" \
      --output-dir "${H7_RESULTS_DIR}/train_${RUN_TAG}_predictor_mlp_gate_up_seed${seed}_${MAX_STEPS}"
  done
fi

if [[ "${RUN_RESOURCE_BASELINES}" == "1" ]]; then
  for seed in ${TRAIN_SEEDS}; do
    python "${TRAIN_RUNNER}" \
      --model-name "${MODEL_NAME}" \
      --precision-policy bf16_baseline \
      --seed "${seed}" \
      --max-steps "${MAX_STEPS}" \
      --learning-rate "${LEARNING_RATE}" \
      --eval-every "${EVAL_EVERY}" \
      --seq-len "${SEQ_LEN}" \
      --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM}" \
      --train-size "${TRAIN_SIZE}" \
      --eval-size "${EVAL_SIZE}" \
      --eval-max-batches "${EVAL_MAX_BATCHES}" \
      --hardware-label "${HARDWARE_LABEL}" \
      --output-dir "${H7_RESULTS_DIR}/train_${RUN_TAG}_bf16_seed${seed}_${MAX_STEPS}"

    python "${TRAIN_RUNNER}" \
      --model-name "${MODEL_NAME}" \
      --precision-policy qlora_4bit_nf4 \
      --seed "${seed}" \
      --max-steps "${MAX_STEPS}" \
      --learning-rate "${LEARNING_RATE}" \
      --eval-every "${EVAL_EVERY}" \
      --seq-len "${SEQ_LEN}" \
      --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM}" \
      --train-size "${TRAIN_SIZE}" \
      --eval-size "${EVAL_SIZE}" \
      --eval-max-batches "${EVAL_MAX_BATCHES}" \
      --hardware-label "${HARDWARE_LABEL}" \
      --output-dir "${H7_RESULTS_DIR}/train_${RUN_TAG}_qlora_4bit_nf4_seed${seed}_${MAX_STEPS}"
  done
fi
