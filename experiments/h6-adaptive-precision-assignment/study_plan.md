# H6 Study Plan: Adaptive Precision Assignment From Stability Signals

## Research Question

Can a short calibration pass derive a precision assignment from stability signals that preserves LoRA fine-tuning quality while improving memory or throughput compared with fixed bf16 and hand-selected fp32 precision islands?

## Study Design

The study has four phases.

1. Signal-only calibration: run `probe_stability_signals.py` on fixed batches with no optimizer step. Record activation outliers, finite checks, fake int8/int4 quantization error, saturation rate, calibration loss, peak memory, and a frozen policy trace.
2. Perturbation calibration: add one-island-at-a-time perturbations and record local loss deltas and gradient-norm deltas under matched seed, batch order, and sequence length.
3. Frozen policy comparison: compare `bf16_baseline`, `fp32_norms`, best static policy, and the derived H6 policy under fixed-step LoRA training.
4. Outer-loop synthesis: decide whether H6 should deepen, broaden, pivot, or conclude based on validation NLL, instability counts, throughput, memory, and whether the signal-to-decision story is coherent.

## Phase 0 Smoke Execution

Completed on 2026-05-13:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/h6-adaptive-precision-assignment/code/probe_stability_signals.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --seed 42 \
  --seq-len 64 \
  --batch-size 1 \
  --calibration-batches 1 \
  --dataset-size 2 \
  --dtype fp32 \
  --max-modules 8 \
  --output-dir experiments/h6-adaptive-precision-assignment/results/smoke_signals_seed42
```

Outputs:

- `experiments/h6-adaptive-precision-assignment/results/smoke_signals_seed42/stability_signals.json`
- `experiments/h6-adaptive-precision-assignment/results/smoke_signals_seed42/policy_trace.json`

The smoke should not be interpreted as a final policy. It verifies that the probe can load the model and dataset, observe candidate modules, compute stability signals, and write a policy trace.

## Next Executable Runs

Run a fuller calibration pass after H1 baseline/treatment runs or whenever GPU time is available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
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

Then repeat with seeds 43 and 44 if the policy trace is stable enough to justify a comparison run.

## Locked Decision Rule

A module is considered sensitive if it has non-finite activations, activation outlier score at least 12, or fake-int8 relative MSE at least `1e-3`. Norm and logits modules are promoted to fp32 when sensitive. Projection modules are considered int8 candidates only if outlier score is below 12 and int8 relative MSE is below `1e-3`. MLP projection modules are only marked as int4 candidates when output int4 relative MSE is below `5e-3`.

These thresholds are intentionally conservative for the first pass. They may be revised only after logging an outer-loop reflection and rerunning calibration, not after looking at final validation results.

## Success Criteria

H6 is supported if a frozen derived policy matches the best static policy within 1% validation NLL, does not increase NaN/Inf or loss-spike counts, and improves peak memory or train-only tokens/sec.

H6 is inconclusive if the derived policy preserves quality and stability but has no resource benefit.

H6 is refuted in this pilot regime if the derived policy worsens validation NLL or instability without a compensating resource benefit.
