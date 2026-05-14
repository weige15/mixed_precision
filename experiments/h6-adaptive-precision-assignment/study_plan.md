# H6 Study Plan: Quick Precision Check Before LoRA Fine-Tuning

## Research Question

Can a short pre-training precision check identify which model modules are precision-sensitive, so a frozen mixed-precision policy can match bf16 validation quality while saving resources or expanding the stable fine-tuning envelope?

Plain version: can we quickly test the model, find the fragile parts, and spend high precision only where it matters?

## Study Design

The study has four phases.

1. Signal-only calibration: run `probe_stability_signals.py` on fixed batches before training. Record activation outliers, finite checks, fake int8/int4 quantization error, saturation rate, calibration loss, peak memory, and a provisional policy trace.
2. Perturbation calibration: add one-island-at-a-time perturbations and record local loss deltas under matched seed, batch order, and sequence length. This tests whether the cheap signals actually predict precision sensitivity.
3. Frozen policy comparison: freeze one policy from calibration only, then compare `bf16_baseline`, `fp32_norms`, and the derived H6 policy under fixed-step LoRA training.
4. Outer-loop synthesis: decide whether H6 should deepen, broaden, pivot, or conclude based on validation NLL, instability counts, throughput, memory, stress tolerance, and whether the signal-to-decision story is coherent.

H1 and H5 are foundations for H6, but they now mainly explain why hand-written fp32 norms are not enough. H2, H3, and H4 are conditional ablations: run them only when H6 needs a stronger static anchor, a stressed regime, or seed-variance confirmation.

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

Run a fuller calibration pass after the H1/H5 synthesis or whenever GPU time is available:

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

After that, implement perturbation calibration:

```bash
# planned interface; not implemented yet
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/h6-adaptive-precision-assignment/code/probe_precision_perturbations.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --seed 42 \
  --seq-len 512 \
  --batch-size 1 \
  --calibration-batches 4 \
  --candidate-policy experiments/h6-adaptive-precision-assignment/results/calibration_bf16_seed42/policy_trace.json \
  --output-dir experiments/h6-adaptive-precision-assignment/results/perturbation_bf16_seed42
```

Recommended perturbation panel based on the three-seed Stage 1 calibration:

High-risk controls:

- `base_model.model.model.layers.2.mlp.down_proj`
- `base_model.model.model.layers.3.mlp.down_proj`
- `base_model.model.model.layers.21.mlp.down_proj`

Borderline/low-risk candidates:

- `base_model.model.model.layers.23.mlp.gate_proj`
- `base_model.model.model.layers.23.mlp.up_proj`
- `base_model.model.model.layers.22.mlp.gate_proj`
- `base_model.model.model.layers.22.mlp.up_proj`

Reduction/output controls:

- `base_model.model.model.layers.4.input_layernorm`
- `base_model.model.model.layers.4.post_attention_layernorm`
- `base_model.model.lm_head`

Do not freeze a training policy from Stage 1 alone. The current signal-only policy is stable but too conservative: it leaves all attention projections at bf16, promotes all norms/logits to fp32, and finds only one seed-specific int8 candidate.

## Locked Decision Rule

A module is considered sensitive if it has non-finite activations, activation outlier score at least 12, or fake-int8 relative MSE at least `1e-3`. Norm and logits modules are promoted to fp32 when sensitive. Projection modules are considered int8 candidates only if outlier score is below 12 and int8 relative MSE is below `1e-3`. MLP projection modules are only marked as int4 candidates when output int4 relative MSE is below `5e-3`.

These thresholds are intentionally conservative for the first pass. They may be revised only after logging an outer-loop reflection and rerunning calibration, not after looking at final validation results.

## Success Criteria

H6 is supported if the short precision check predicts perturbation sensitivity well enough to choose a frozen derived policy that matches bf16 within 1% validation NLL, does not increase NaN/Inf or loss-spike counts, and improves peak memory, train-only tokens/sec, or stress tolerance.

H6 is inconclusive if the derived policy preserves quality and stability but has no resource benefit.

H6 is refuted in this pilot regime if the derived policy worsens validation NLL or instability without a compensating resource benefit.
