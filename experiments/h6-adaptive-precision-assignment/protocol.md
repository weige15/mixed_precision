# H6 Protocol: Calibration-Guided Precision Assignment

## Hypothesis

A short pre-training precision check can identify which model modules are fragile under reduced precision. A frozen policy derived from that check can match bf16 LoRA validation quality while reducing resource cost or expanding the stable fine-tuning envelope.

Plain-language version: quickly test the model before fine-tuning, find the fragile parts, and spend high precision only where it matters.

## Relationship To Current Work

H6 is the trunk research hypothesis. It builds on selected parts of the existing H1-H5 sequence rather than requiring every hypothesis to finish first. H1 tests a single hand-selected island, fp32 normalization, and gives H6 its first matched static anchor. H5 checks whether dtype probes are measuring real internal compute rather than only module-boundary tensors.

The H1/H5 results now simplify H6's motivation. H1 found that fp32 norms are only a weak static anchor in the default regime. H5 explains why: Qwen2RMSNorm already casts its reduction path to fp32 internally. Therefore the main research opportunity is not "which obvious module should be fp32?" but "can a short measured check discover the modules that actually matter?"

H2, H3, and H4 are conditional ablations. H2 is useful if logits/loss become a stronger static anchor than fp32 norms. H3 is useful if the default bf16 LoRA regime is too stable to reveal policy differences. H4 is useful after a candidate policy exists and seed variance needs to be measured.

## Simple Experimental Story

1. Run a standard bf16 LoRA baseline.
2. Before training, run a short calibration pass on a few fixed batches.
3. Measure which modules look numerically fragile.
4. Freeze a precision policy from those measurements.
5. Fine-tune with that policy.
6. Compare against bf16 on validation loss, stability, memory, speed, and stress tolerance.

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

1. Run the baseline policy for a short calibration window before training.
2. Record signal-only statistics: activation outliers, fake-quantization error, clipping/saturation rate, finite checks, and calibration loss.
3. For each candidate island, temporarily apply one precision perturbation while holding seed, batch, sequence length, optimizer, and data order fixed.
4. Record local loss delta, grad-norm delta when available, activation outlier score, quantization error, clipping/saturation rate, NaN/Inf incidence, throughput, and peak memory.
5. Test whether cheap signal-only scores predict perturbation loss deltas. This predictive-validity check is the core science question.
6. Assign a sensitivity score:

```text
sensitivity =
  loss_delta_score
  + grad_spike_score
  + activation_outlier_score
  + quantization_error_score
  + clipping_saturation_score
  + nan_inf_penalty
```

7. Keep high-sensitivity islands at `fp32` or `bf16`.
8. Demote low-sensitivity islands only when memory or throughput improves enough to justify the added complexity.
9. Freeze the derived policy before the main comparison run.

## Decision Rule

H6 is supported if the calibration signals predict perturbation sensitivity well enough to choose a frozen policy that matches bf16 validation NLL within 1%, does not increase instability events, and improves either peak memory, train-only tokens/sec, or the maximum stress level completed without collapse.

H6 is inconclusive if the precision check produces a stable ranking but the derived policy matches quality and stability with no measurable resource or stress-envelope benefit.

H6 is refuted for this pilot regime if the derived policy worsens validation NLL or instability relative to the best static policy without a compensating memory or throughput benefit.

## Implementation Milestones

1. Finish or summarize H1 baseline/treatment results as the first static anchor.
2. Add per-module activation and quantization-error probes without changing training behavior. Completed for the first non-invasive H6 probe in `code/probe_stability_signals.py`.
3. Add perturbation mode for one candidate island at a time.
4. Write a policy trace file that records each island, signal values, precision decision, and reason. Completed for the signal-only probe; perturbation reasons still need loss-delta evidence.
5. Compare `bf16_baseline`, `fp32_norms`, and the derived adaptive policy under matched training conditions. Add H2/H3/H4 only if the H6 evidence needs a stronger static anchor, stressed regime, or variance confirmation.

## H6.1 Protocol: SNIP-Style Budgeted Policy Expansion

### Hypothesis

A SNIP-style multi-signal risk score can expand the current four-module H6 late-MLP fake-int8 policy to a wider budgeted set of low-risk MLP projection outputs while preserving validation loss within 1% of matched bf16 and without increasing instability events.

Plain-language version: after the quick precision check finds which modules look fragile, rank the remaining projection modules by measured risk and demote only the safest `k` modules.

### Motivation

The current H6 policy is intentionally narrow: it demotes only `layers.22/23.mlp.gate_proj` and `layers.22/23.mlp.up_proj`. This preserves quality across seeds and stress settings, but it is too small to make a credible resource-saving claim. The next experiment should test whether the calibration evidence can safely support a larger demotion budget.

This is SNIP-inspired, not a full SNIP reproduction. SNIP is used here as a design pattern: compute a per-module saliency or risk score before training, sort candidates under a fixed budget, freeze the policy, and then validate the selected policy under matched training.

### Candidate Set

Use only MLP `gate_proj` and `up_proj` modules for the first expansion. Exclude:

- `down_proj`, because Stage 2 found several high-risk down projections with large perturbation loss deltas.
- attention projections, because the literature and current signals flag attention as a high-risk low-bit region.
- norms and logits, because output fake quantization is not equivalent to reducing their internal arithmetic precision.

### Scoring Rule

For each candidate module, aggregate calibration signals across seeds 42, 43, and 44:

```text
risk =
  normalized(max activation outlier score)
  + normalized(max int8 relative MSE)
  + normalized(max int4 relative MSE)
  + normalized(max int8 saturation rate)
  + finite-value penalty
  + perturbation-loss penalty when available
```

The score is conservative: maxima across seeds are preferred over means when deciding safety. Available perturbation loss deltas dominate the score, but modules without perturbation measurements are still ranked by calibration signals so the policy can propose wider budgets.

### Frozen Policy Budgets

Generate candidate policies for `k = 4, 8, 16, 24` demoted MLP gate/up modules:

- `k=4` should reproduce the current late-layer H6 policy or a close equivalent.
- `k=8` is the first meaningful expansion.
- `k=16` and `k=24` test how quickly quality degrades as the policy becomes more aggressive.

### Decision Rule

H6.1 is supported if a wider policy, preferably `k >= 8`, preserves final validation loss within 1% of matched bf16 across the seed-42 500-step screen, then across seeds 43 and 44 for the best candidate, with zero added NaN/Inf or loss-spike events.

H6.1 is inconclusive if only `k=4` remains safe or wider policies are quality-preserving but still provide no real throughput or memory path.

H6.1 is refuted for this pilot setting if `k=8` or larger consistently violates the 1% validation-loss gate or introduces instability without evidence of resource benefit.

### Execution Plan

1. Build the SNIP-style score table from existing Stage 1 calibration and Stage 2 perturbation artifacts.
2. Write frozen module lists for `k = 4, 8, 16, 24`.
3. Run a seed-42 500-step screen comparing bf16 against each policy.
4. Replicate only the best safe wider policy on seeds 43 and 44.
5. If no wider policy is safe, pivot toward hardware-realistic kernels for the already-supported narrow policy or conclude the quality-preserving sensitivity-ranking claim.

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
