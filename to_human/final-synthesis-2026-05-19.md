# Final Synthesis: Calibration-Guided Precision Assignment for LoRA Fine-Tuning

Date: 2026-05-19

## Bottom Line

This project produced a coherent result: a short calibration pass can identify precision-sensitive and precision-tolerant modules before LoRA fine-tuning, and a frozen policy derived from those measurements can preserve bf16 validation quality during actual updates.

There are two supported claims.

First, at Qwen2.5-0.5B, calibration signals and one-module perturbation tests identify safe fake-int8 projection candidates. The selected modules preserve validation quality across three 500-step LoRA seeds.

Second, at Qwen2.5-7B, the same rank-and-perturbation workflow transfers when thresholds are recalibrated by ranking rather than copied directly from 0.5B. A conservative selected fake-int8 policy preserves quality across three 500-step LoRA seeds. Separately, generic QLoRA 4-bit NF4 gives a robust 7B memory-capacity trade-off: lower peak memory with acceptable quality degradation, but slower throughput.

The result should not be framed as a universal resource-saving method yet. The project supports sensitivity prediction and quality-preserving policy selection. Hardware savings are demonstrated only by generic QLoRA at 7B, not by the custom selective fake-int8 policy.

## Research Question

Before LoRA fine-tuning, can a short precision check identify which model modules are precision-sensitive, so a frozen mixed-precision policy can match bf16 validation quality while saving resources or expanding the stable fine-tuning envelope?

The answer is partially yes.

A short precision check can identify module sensitivity well enough to choose harmless low-precision candidates. It does not automatically produce resource savings unless the selected policy is backed by real low-precision kernels. Existing bitsandbytes QLoRA becomes useful at 7B for memory capacity, but with a consistent throughput penalty.

## Main Claims

### Claim 1: Hand-picked fp32 norms are not the right core intervention

H1 tested fp32 normalization islands against a bf16 LoRA baseline. Seed 42 showed only a `0.20%` validation-loss improvement, below the locked 1% support threshold, with no instability events in either policy.

H5 explains why. The installed Qwen2RMSNorm implementation casts hidden states to fp32 internally for `pow`, `mean`, and `rsqrt`, then casts back to the input dtype. So bf16 boundary tensors do not imply bf16 reduction arithmetic. This weakens the original rationale for wrapping norms in fp32.

Inference: static norm promotion is not a compelling contribution here. Dtype-validity checks are necessary before claiming a precision island matters.

### Claim 2: Calibration signals predict perturbation sensitivity for projection modules

At 0.5B, H6 Stage 1 collected activation outliers and fake-quantization errors across candidate modules. H6 Stage 2 then measured one-module fake-int8 perturbation loss deltas.

The ranking was meaningful:

- High-risk MLP down projections produced large positive loss deltas across seeds.
- Late-layer gate/up candidates stayed near zero.
- Across MLP projection perturbations, max outlier score correlated with absolute loss delta at about `0.91`.

The norm/logit controls did not show the same behavior under output fake quantization, which is expected: output perturbation is not equivalent to changing internal reduction precision or loss precision.

Inference: activation outlier statistics are a useful ranking signal for projection-output precision sensitivity. Fake-quantization error alone is less reliable.

### Claim 3: Calibration-selected fake-int8 policies preserve quality during training

At 0.5B, the first narrow H6 policy fake-int8 quantized four late-layer MLP gate/up modules selected from low perturbation deltas. Across seeds 42, 43, and 44, the paired eval degradations were:

| Seed | BF16 eval | H6 eval | Relative delta |
|---:|---:|---:|---:|
| 42 | `1.62949` | `1.63112` | `+0.100%` |
| 43 | `1.63444` | `1.63621` | `+0.108%` |
| 44 | `1.63247` | `1.63493` | `+0.151%` |

All stayed far inside the 1% gate with zero loss spikes and zero NaN/Inf events.

H6.1 then expanded the policy with a SNIP-style ranker. The best tested budget, `k=24`, also replicated across seeds:

| Seed | BF16 eval | k=24 eval | Relative delta |
|---:|---:|---:|---:|
| 42 | `1.62978` | `1.63177` | `+0.122%` |
| 43 | `1.61701` | `1.61975` | `+0.169%` |
| 44 | `1.62089` | `1.62248` | `+0.098%` |

Inference: a calibration-derived ranking can safely widen a low-precision candidate set while preserving bf16-quality LoRA updates, at least under fake-int8 output quantization.

### Claim 4: Generic low-bit backends are scale-dependent

H6.2 tested hardware-backed low-bit baselines on RTX 3090 at 0.5B. The result was negative:

| Policy | Eval delta vs bf16 | Memory delta | Throughput delta |
|---|---:|---:|---:|
| fake-int8 k=24 | `+0.268%` | `+0.20%` | `-2.16%` |
| bitsandbytes 8-bit LoRA | `+0.102%` | `+16.80%` | `-40.16%` |
| QLoRA 4-bit NF4 | `+2.832%` | `+15.79%` | `-37.58%` |

At 1.5B, the resource picture was still not useful: low-bit paths did not provide a good memory/quality/speed trade-off.

At 7B, QLoRA becomes useful for memory:

| Seed | BF16 eval | QLoRA eval | Eval delta | Memory delta | Throughput delta |
|---:|---:|---:|---:|---:|---:|
| 42 | `1.37747` | `1.38524` | `+0.564%` | `-23.32%` | `-19.95%` |
| 43 | `1.36653` | `1.37427` | `+0.566%` | `-23.32%` | `-20.18%` |
| 44 | `1.34233` | `1.35478` | `+0.927%` | `-23.32%` | `-19.98%` |

Mean eval degradation is `+0.686%`, worst seed is `+0.927%`, and all runs have zero loss spikes and zero NaN/Inf events.

Inference: generic QLoRA is a robust 7B memory-capacity trade-off on RTX 3090, not a throughput improvement. It should be reported separately from the custom calibration-guided fake-int8 policy.

### Claim 5: The calibration-to-training story transfers to 7B if thresholds are rank-based

H6.4 tested a targeted 7B transfer panel across seeds 42, 43, and 44. The 0.5B fixed thresholds did not transfer directly: Stage 1 assigned all projections to bf16 because 7B activation outliers and fake-quantization errors were larger.

But the ranking still transferred. Across seeds:

- `layers.4.post_attention_layernorm` was consistently very sensitive, mean abs loss delta `0.2416`.
- `layers.3.mlp.down_proj` was consistently sensitive, mean abs loss delta `0.0464`.
- Four modules were consistently low-delta, with max abs delta below `0.005`:
  - `layers.26.mlp.gate_proj`
  - `layers.26.mlp.up_proj`
  - `layers.27.mlp.gate_proj`
  - `layers.26.self_attn.o_proj`
- Projection-only outlier score correlated with absolute perturbation delta at about `0.78`.
- Int8 relative MSE was not predictive in this 7B panel.

The conservative rank-selected 7B fake-int8 policy used those four low-delta modules. During actual 500-step LoRA updates:

| Seed | BF16 eval | H6.4 eval | Eval delta | Instability |
|---:|---:|---:|---:|---|
| 42 | `1.37747` | `1.37969` | `+0.161%` | none |
| 43 | `1.36653` | `1.36958` | `+0.223%` | none |
| 44 | `1.34233` | `1.34370` | `+0.102%` | none |

Mean eval degradation is `+0.162%`, with zero loss spikes and zero NaN/Inf events.

Inference: the calibration-to-training pipeline works at 7B for a small conservative module set. The rank-based version is the right formulation; fixed thresholds from 0.5B should not be reused at 7B.

## What the Paper Story Should Be

A good paper framing is:

> Low-precision LoRA recipes are usually chosen by broad dtype rules or backend defaults. We show that a short pre-training calibration pass can identify which modules are tolerant or sensitive to reduced precision. On Qwen2.5 LoRA, activation-outlier rankings and perturbation deltas predict which projection modules can be fake-int8 quantized without harming training. The selected policies preserve bf16 validation quality across seeds at both 0.5B and 7B. Hardware-backed QLoRA at 7B separately demonstrates that low-bit training can provide memory-capacity gains, but selective fake-int8 remains a sensitivity method until implemented with real kernels.

The contribution should be presented as empirical and methodological:

- A calibration-and-perturbation workflow for module precision sensitivity.
- Evidence that projection outlier ranking predicts perturbation sensitivity.
- Evidence that selected modules remain harmless during actual LoRA updates.
- A scale-dependent hardware-backend study showing QLoRA memory savings only become useful at 7B in this setup.

## What Not To Claim

Do not claim:

- That fp32 norms improve LoRA fine-tuning in this setup.
- That fake-int8 provides memory savings.
- That the H6/H6.4 selective policies are hardware-efficient.
- That generic bitsandbytes is Pareto-better than bf16.
- That fixed thresholds transfer across model scales.

The correct resource claim is narrower:

> QLoRA 4-bit NF4 at Qwen2.5-7B reduces peak memory by 23.32% while keeping eval degradation below 1%, but it is about 20% slower than bf16.

The correct calibration claim is:

> Rank/perturbation-selected fake-int8 module policies preserve quality and stability during LoRA updates at 0.5B and 7B.

## Limitations

- Fake-int8 output quantization is a sensitivity proxy, not a deployable memory-saving kernel.
- Only Qwen2.5 models and Alpaca-style LoRA fine-tuning were tested.
- The 7B selective policy is small and conservative; wider selective policies remain untested.
- The 7B calibration panel is targeted, not all-module.
- No full downstream benchmark suite was run; validation NLL is the primary metric.
- Throughput measurements under Python hooks should not be overinterpreted.
- QLoRA and selective fake-int8 are separate interventions; they have not yet been combined.

## Remaining Open Questions

The most useful open questions are:

1. How wide can the 7B rank-selected fake-int8 policy become before eval loss crosses the 1% gate?
2. Can the selected fake-int8 policy be implemented with real low-precision kernels?
3. Does calibration-guided selection improve or complement QLoRA, rather than merely coexist with it?
4. Do the sensitivity rankings transfer to other models, datasets, sequence lengths, or higher learning-rate stress settings?
5. Which signal should replace fixed thresholds across scale? Current evidence favors rank and perturbation deltas over absolute int8 relative MSE.

## Recommended Next Step

For a research artifact, stop here and write the paper-style report. The story is already coherent.

For one more experiment, run a cautious 7B width expansion:

- Start from the four validated H6.4 low-delta modules.
- Add exactly one or two borderline modules, not a large sweep.
- Best next candidates:
  - `layers.27.mlp.up_proj`
  - `layers.26.self_attn.q_proj`
- Avoid:
  - `layers.4.post_attention_layernorm`
  - `layers.3.mlp.down_proj`
  - `layers.24.mlp.down_proj`
  - `lm_head`

However, the highest-value next action is paper/report writing, not another run. The core empirical claims are supported across seeds.

## Final Status

Conclusion: H6 is supported as a calibration-guided sensitivity-ranking method for LoRA precision assignment. H6.3 is supported as a 7B QLoRA memory-capacity result. H6.4 is supported as calibration-to-training transfer at 7B for a conservative selected fake-int8 module set.

The project is ready for final write-up.
