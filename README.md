# Mixed Precision LoRA Pilot

This project tests whether selective and eventually adaptive precision islands improve stability, quality, memory use, or throughput during small LLM LoRA fine-tuning compared with standard bf16 mixed precision. No paper is being written yet; the current goal is to make H1 executable and scientifically defensible before extending the work to adaptive precision assignment from stability signals.

## Install

```bash
pip install -r requirements.txt
```

## 1. Dtype Probe

Run this before the main experiment. H1 is only meaningful if the baseline autocast policy does not already run the target normalization modules in fp32.

```bash
python experiments/h1-selective-fp32-norms/code/probe_dtypes.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --seq-len 512 \
  --dtype bf16 \
  --output-json experiments/h1-selective-fp32-norms/results/dtype_probe.json
```

## 2. Smoke Baseline

```bash
python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --precision-policy bf16_baseline \
  --seed 42 \
  --max-steps 100 \
  --seq-len 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --eval-every 25 \
  --train-size 512 \
  --eval-size 64 \
  --eval-max-batches 8 \
  --output-dir experiments/h1-selective-fp32-norms/results/smoke_bf16
```

## 3. Smoke Treatment

```bash
python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --precision-policy fp32_norms \
  --seed 42 \
  --max-steps 100 \
  --seq-len 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --eval-every 25 \
  --train-size 512 \
  --eval-size 64 \
  --eval-max-batches 8 \
  --output-dir experiments/h1-selective-fp32-norms/results/smoke_fp32_norms
```

## 4. Main H1 Pair

These use the protocol defaults: 8,000 train examples, 1,000 validation examples, and full validation at each eval point.

```bash
python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --precision-policy bf16_baseline \
  --seed 42 \
  --max-steps 1000 \
  --seq-len 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 64 \
  --learning-rate 2e-4 \
  --eval-every 100 \
  --output-dir experiments/h1-selective-fp32-norms/results/h1_baseline_bf16_seed42
```

```bash
python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --precision-policy fp32_norms \
  --seed 42 \
  --max-steps 1000 \
  --seq-len 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 64 \
  --learning-rate 2e-4 \
  --eval-every 100 \
  --output-dir experiments/h1-selective-fp32-norms/results/h1_fp32_norms_seed42
```

## 5. Adaptive Precision Roadmap

H6 tracks the follow-on idea: derive per-operation precision from stability signals instead of hand-selecting one island at a time. The protocol is in `experiments/h6-adaptive-precision-assignment/protocol.md`.

H6 depends on H1-H5. Before testing it, the runner needs per-module instrumentation for activation outliers, quantization error, clipping or saturation rate, gradient spikes, and local loss deltas.

The first non-invasive H6 signal probe is now available:

```bash
python experiments/h6-adaptive-precision-assignment/code/probe_stability_signals.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --seed 42 \
  --seq-len 512 \
  --batch-size 1 \
  --calibration-batches 8 \
  --dataset-size 128 \
  --dtype bf16 \
  --output-dir experiments/h6-adaptive-precision-assignment/results/calibration_bf16_seed42
```

For the full H6 execution plan and the initial smoke result, see `experiments/h6-adaptive-precision-assignment/study_plan.md` and `experiments/h6-adaptive-precision-assignment/analysis.md`.

## Outputs

Each LoRA run writes:

- `metrics.jsonl`
- `summary.json`, including final eval loss, max gradient norm, peak memory, and train-only throughput

The dtype probe writes a JSON file and prints a readable table of observed module dtypes.
