# Research Findings

## Research Question

During LoRA fine-tuning of small open LLMs, can selective or adaptively assigned precision islands improve stability, validation quality, memory use, or throughput over standard bf16 mixed precision?

## Current Understanding

The initial working assumption is that blanket bf16 mixed precision may already be strong for small LoRA fine-tuning, but some operations may be disproportionately sensitive to reduced precision. Normalization layers are the first candidate because they compute activation statistics, have low computational cost compared with matrix multiplications, and can be kept in fp32 without changing the model architecture or dataset.

The broader research direction is adaptive precision assignment from stability signals. In that framing, H1 is the first calibration experiment rather than the final goal: it tests whether one manually chosen sensitive island matters. A later adaptive policy should use measured activation outliers, gradient spikes, loss deltas, quantization error, clipping or saturation rate, NaN/Inf incidence, memory, and throughput to decide which operations should be promoted to higher precision or demoted to lower precision.

The refreshed literature suggests three especially useful signal families for H6. First, LLM.int8() and SmoothQuant indicate that activation outliers are a strong predictor of INT8 sensitivity. Second, FP4 training and Attn-QAT suggest low-bit training is limited by quantization noise, rounding, heavy-tailed attention activations, and hidden backward-pass precision assumptions. Third, HAWQ, HAWQ-V3, convergence-aware mixed precision, and SNIP frame precision assignment as constrained optimization over predicted quality loss and hardware cost.

## Key Results

No main H1 experiment result exists yet. H1 is protocol-locked as the first experiment, and H6 has been added as a follow-on adaptive precision assignment hypothesis.

The first H6 signal-only smoke probe completed on 2026-05-13. It ran Qwen/Qwen2.5-0.5B on one Alpaca calibration batch with sequence length 64, fp32 dtype, and the first eight candidate modules. The probe wrote `stability_signals.json` and `policy_trace.json` under `experiments/h6-adaptive-precision-assignment/results/smoke_signals_seed42/`. It observed mean calibration loss `1.7697`, zero NaN/Inf events, and peak CUDA memory `2.1825 GiB`.

Under the conservative H6 decision rule, the smoke policy promoted layer-0 input RMSNorm to `fp32` and kept the seven observed projection modules at `bf16`. The largest early signals were `mlp.down_proj` input outlier score `72.95`, `self_attn.o_proj` output outlier score `35.92`, and `input_layernorm` output int8 relative MSE `0.00318`. This is instrumentation evidence, not final adaptive-policy evidence.

## Patterns and Insights

- Static precision islands are a prerequisite for adaptive assignment: without baseline and treatment traces, the adaptive policy has no calibrated notion of sensitivity.
- The current runner already logs useful run-level stability signals: loss spikes, NaN/Inf count, max gradient norm, peak memory, and train-only throughput.
- H6 now has initial per-module signal instrumentation for activation outliers, fake-quantization error, saturation, finite checks, and policy trace generation.
- INT8/INT4 inference papers are useful for signal design but should not be treated as direct evidence for training-time LoRA stability.
- Attention and logits/loss should be handled conservatively under subbyte precision because the literature flags heavy-tailed activations and backward precision assumptions as instability sources.
- The first H6 smoke suggests early-layer activation outlier scores can be large enough that naive int8/int4 demotion would be unsafe without perturbation loss-delta checks.

## Lessons and Constraints

- The pilot must avoid full pretraining and large benchmark suites.
- Experiments should use fixed-step or fixed-wall-clock budgets so precision policies are comparable on limited GPU resources.
- A useful negative result is possible: standard bf16 may be Pareto-optimal for this LoRA regime, which would redirect the project toward stressed settings or other precision targets.
- Aggressive low-precision candidates should be staged conservatively. Projection paths are better first targets for int8/int4 exploration than normalization reductions or attention softmax.
- Adaptive decisions must be frozen before final validation comparison. Otherwise the policy can overfit to the observed result.
- Signal-only calibration should be treated as a ranking/prior. It needs perturbation loss deltas before it can justify an adaptive training policy.
- Low-bit perturbation probes can identify sensitivity, but real throughput or memory claims require hardware-supported kernels on the target machine.

## Open Questions

- Does fp32 normalization measurably reduce loss spikes or validation loss compared with standard bf16 autocast?
- Is any quality or stability gain large enough to justify throughput overhead?
- If H1 is inconclusive, should the next precision island be logits/loss computation, attention softmax, optimizer state precision, or a stressed fp16 setting?
- Which per-module signals best predict precision sensitivity during LoRA fine-tuning?
- Does the H6 signal ranking remain stable across bf16 autocast, longer sequence length, more batches, and seeds 42-44?
- Do one-island perturbation loss deltas agree with the activation outlier and fake-quantization-error ranking?
- Can a short calibration pass derive a precision assignment that matches the best static policy while improving memory or throughput?
- Are int8 or int4 candidates real compute improvements on the available RTX 4050, or only fake-quant research probes?

## Optimization Trajectory

No training trajectory yet. The first H6 calibration artifact is a signal-only smoke run, not an optimizer-step result. The first training trajectory point will be the bf16 baseline from H1.
