# H6 Protocol: Adaptive Precision Assignment From Stability Signals

## Hypothesis

A stability-signal-driven precision policy can assign higher precision to sensitive operations and lower precision to tolerant operations while preserving validation loss and reducing memory or improving throughput relative to hand-selected static precision islands.

## Relationship To Current Work

H6 builds on the existing H1-H5 sequence. H1 tests a single hand-selected island, fp32 normalization. H2 and H3 test whether other static islands or stressed regimes reveal larger precision effects. H5 checks whether dtype probes are measuring real internal compute rather than only module-boundary tensors.

H6 should not replace H1. It depends on H1 because an adaptive policy needs a calibrated baseline, verified dtype behavior, and stability metrics before it can make defensible precision decisions.

## Candidate Policy Space

Initial policy space:

- RMSNorm / LayerNorm: `fp32`, `bf16`
- Logits and loss: `fp32`, `bf16`
- Attention softmax: `fp32`, `bf16`
- Attention Q/K/V/O projections: `bf16`, `int8`
- MLP gate/up/down projections: `bf16`, `int8`, optionally `int4` after int8 is stable
- LoRA adapter weights: `bf16`

Conservative constraint: do not start with `int4` attention softmax during training. Softmax is a high-sensitivity operation involving masking, exponentials, and normalization. If lower precision is explored around attention, begin with projection paths or calibrated score quantization, not the softmax reduction itself.

## Signals To Instrument

Run-level signals already present in the H1 runner:

- training loss
- validation loss
- gradient norm and max gradient norm
- loss-spike count
- NaN/Inf count
- peak CUDA memory
- tokens/sec

Per-module signals to add before H6 can be tested:

- activation absolute max and percentile outlier score
- activation mean, standard deviation, and RMS
- finite-value check on inputs and outputs
- approximate quantization error for candidate lower precision formats
- clipping or saturation rate under the candidate quantizer
- per-module gradient norm for trainable LoRA paths
- local loss delta from short precision perturbation trials

## Calibration Procedure

1. Run the baseline policy for a short calibration window.
2. For each candidate island, temporarily apply one precision perturbation while holding seed, batch, sequence length, optimizer, and data order fixed.
3. Record loss delta, grad-norm delta, activation outlier score, quantization error, clipping/saturation rate, NaN/Inf incidence, throughput, and peak memory.
4. Assign a sensitivity score:

```text
sensitivity =
  loss_delta_score
  + grad_spike_score
  + activation_outlier_score
  + quantization_error_score
  + clipping_saturation_score
  + nan_inf_penalty
```

5. Promote high-sensitivity islands to `fp32` or `bf16`.
6. Demote low-sensitivity islands only when memory or throughput improves enough to justify the added complexity.
7. Freeze the derived policy before the main comparison run.

## Decision Rule

H6 is supported if the derived policy matches the best static policy within 1% validation NLL, does not increase instability events, and improves either peak memory or train-only tokens/sec.

H6 is inconclusive if the derived policy matches quality and stability but has no measurable resource benefit.

H6 is refuted for this pilot regime if the derived policy worsens validation NLL or instability relative to the best static policy without a compensating memory or throughput benefit.

## Implementation Milestones

1. Finish or summarize H1 baseline/treatment results.
2. Add per-module activation and quantization-error probes without changing training behavior. Completed for the first non-invasive H6 probe in `code/probe_stability_signals.py`.
3. Add perturbation mode for one candidate island at a time.
4. Write a policy trace file that records each island, signal values, precision decision, and reason. Completed for the signal-only probe; perturbation reasons still need loss-delta evidence.
5. Compare `bf16_baseline`, `fp32_norms`, best static policy, and derived adaptive policy under matched training conditions.

## Phase 0 Smoke Result

On 2026-05-13, a one-batch smoke calibration ran on Qwen/Qwen2.5-0.5B with sequence length 64 and the first eight candidate modules. It produced `stability_signals.json` and `policy_trace.json` under `results/smoke_signals_seed42/`.

The probe observed no NaN/Inf events, mean calibration loss `1.7697`, and peak CUDA memory `2.1825 GiB`. The conservative decision rule assigned layer-0 input RMSNorm to `fp32` and kept the seven observed projection modules at `bf16` because activation outlier and fake-quantization error signals were high. This is a smoke verification of the instrumentation, not evidence for the final adaptive policy.

## Failure Criteria

Mark H6 as failed, not negative, if:

- candidate policies change trainable parameter count, data order, optimizer, learning rate, or effective batch size without being recorded;
- lower-precision simulation does not correspond to actual compute or a clearly labeled fake-quant proxy;
- per-module instrumentation changes training loss beyond measurement noise;
- precision decisions are changed after seeing final validation results;
- the adaptive policy is compared against an unmatched baseline.
