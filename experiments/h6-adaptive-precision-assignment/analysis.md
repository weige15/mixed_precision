# H6 Analysis

## 2026-05-14 Brainstorm Integration

The H6 research question has been simplified around a quick precision check before LoRA fine-tuning:

> Can we quickly test the model before fine-tuning, find the fragile modules, and spend high precision only where it matters?

This framing is stronger than a generic "adaptive precision assignment" claim because it makes the central test concrete. The project must show that a short calibration pass predicts module-level precision sensitivity well enough to freeze a useful policy before training.

The current H1/H5 results motivate this pivot. H1 found fp32 norms to be an inconclusive static island under the default bf16 LoRA regime, and H5 suggests Qwen2RMSNorm already performs its reduction arithmetic in fp32 internally. Therefore, the next useful research step is not another hand-picked fp32 island. It is a predictive-validity test:

1. Collect cheap per-module signals such as activation outliers and fake-quantization error.
2. Perturb one island at a time and measure local loss deltas.
3. Check whether the cheap signals predict the perturbation deltas.
4. Freeze a policy using only calibration evidence.
5. Compare the frozen policy against bf16.

This keeps the research accessible: the "precision check" is the method, the perturbation deltas are the validation of the method, and the frozen policy comparison is the final practical test.

## 2026-05-14 BF16 Calibration Across Seeds

Stage 1 calibration completed for seeds 42, 43, and 44 using bf16 autocast, sequence length 512, batch size 1, and 8 calibration batches per seed. Each run observed 218 candidate modules on CUDA with zero NaN/Inf events and peak CUDA memory `2.7303 GiB`.

Run-level summary:

| seed | mean calibration loss | NaN/Inf | elapsed sec | role-level policy counts |
|---:|---:|---:|---:|---|
| 42 | `2.0182` | `0` | `6.42` | attention projections: 96 bf16; MLP projections: 71 bf16 + 1 int8 candidate; norms: 49 fp32; logits: 1 fp32 |
| 43 | `2.1414` | `0` | `6.14` | attention projections: 96 bf16; MLP projections: 72 bf16; norms: 49 fp32; logits: 1 fp32 |
| 44 | `2.3657` | `0` | `6.39` | attention projections: 96 bf16; MLP projections: 72 bf16; norms: 49 fp32; logits: 1 fp32 |

The policy decisions are highly stable across seeds: 217 of 218 common modules received the same assignment in all three runs. The only unstable assignment was `layers.23.mlp.gate_proj`, which was an `int8_candidate` for seed 42 but remained `bf16` for seeds 43 and 44. It is therefore a borderline candidate, not a safe frozen-policy decision.

The strongest high-risk modules are also stable across seeds. The top activation-outlier paths include `layers.2.mlp.down_proj`, `layers.3.mlp.down_proj`, `layers.21.mlp.down_proj`, and the layer 4-7 norm paths. For example, `layers.2.mlp.down_proj` has a mean outlier score around `909.7` across seeds and mean int8 relative MSE around `0.016`, far above the current int8 candidate threshold.

Using a relaxed screening rule of mean outlier score below 20 and mean int8 relative MSE below `1e-3`, only four projection modules look plausibly tolerant enough to test next:

- `layers.23.mlp.gate_proj`
- `layers.23.mlp.up_proj`
- `layers.22.mlp.gate_proj`
- `layers.22.mlp.up_proj`

Interpretation: Stage 1 gives a strong negative message against naive low-precision demotion. Most projections look sensitive under the current signals, and the conservative policy does not yet produce a resource-saving policy. This does not refute H6; it says the next step must be perturbation validation, not training a policy directly. The perturbation experiment should test a small panel: the four tolerant candidates above, the three extreme high-risk modules, and representative norm/logits paths.

## 2026-05-15 Stage 2 Perturbation Probe

Stage 2 perturbation probes ran on seeds 42, 43, and 44 with bf16 autocast, sequence length 512, batch size 1, and 4 calibration batches. Each run used fake int8 output quantization one module at a time, with no weight updates.

The replicated result supports the core H6 predictive-validity claim for MLP projections. The three modules Stage 1 ranked as extreme high-risk had the largest positive loss deltas in every seed:

| module | mean int8 loss delta over seeds | max abs mean loss delta | sign pattern |
|---|---:|---:|---|
| `layers.2.mlp.down_proj` | `+0.3699` | `0.4630` | `+++` |
| `layers.21.mlp.down_proj` | `+0.2665` | `0.3271` | `+++` |
| `layers.3.mlp.down_proj` | `+0.2209` | `0.2972` | `+++` |

The relaxed low-risk MLP candidates stayed near zero across seeds:

| module | mean int8 loss delta over seeds | mean abs loss delta | max abs mean loss delta |
|---|---:|---:|---:|
| `layers.23.mlp.up_proj` | `+0.0031` | `0.0031` | `0.0070` |
| `layers.23.mlp.gate_proj` | `+0.0011` | `0.0035` | `0.0048` |
| `layers.22.mlp.up_proj` | `-0.0014` | `0.0024` | `0.0042` |
| `layers.22.mlp.gate_proj` | `+0.0009` | `0.0016` | `0.0034` |

Simple correlation checks are encouraging. Across all 30 perturbations, max outlier score correlates with absolute loss delta at about `0.72`, and output int8 relative MSE correlates at about `0.56`. Restricting to the 21 MLP projection perturbations, max outlier score is much stronger at about `0.91`; output int8 relative MSE is weaker at about `0.53` pooled across seeds, though it was strong in seed 42. This suggests activation outlier score is the more reliable current ranking signal.

The norm/logit controls are more nuanced. Stage 1 marked layer-4 norms and `lm_head` as fp32-sensitive, but output fake-int8 perturbation produced small local loss deltas across all three seeds. This does not prove norms/logits are safe to demote. It shows that the current perturbation target, output fake quantization, is not equivalent to reducing the internal reduction or loss-computation arithmetic. Norm/logit sensitivity needs a different perturbation design if it remains scientifically important.

Interpretation: H6 now has replicated positive evidence. Cheap Stage 1 signals, especially activation outlier score, predict which MLP projection outputs are harmed by int8 perturbation. The next step is to freeze a narrow candidate policy around the consistently low-delta late-layer MLP gate/up projections and run a short training comparison against bf16.

## 2026-05-15 Stage 3 Narrow Policy Training

A matched 500-step seed-42 training comparison tested bf16 baseline against the frozen H6 late-layer MLP int8 candidate policy. The H6 policy applied straight-through fake int8 output quantization only to `layers.22/23.mlp.gate_proj` and `layers.22/23.mlp.up_proj`.

| metric | bf16 baseline | H6 narrow policy | delta |
|---|---:|---:|---:|
| final eval loss | `1.62949` | `1.63112` | `+0.00163` (`+0.10%`) |
| final train loss | `1.40481` | `1.40745` | `+0.00264` |
| max grad norm | `4.09634` | `4.09257` | `-0.00377` |
| loss spikes | `0` | `0` | `0` |
| NaN/Inf events | `0` | `0` | `0` |
| peak CUDA memory GiB | `2.77945` | `2.77884` | `-0.00061` |
| train tokens/sec | `455.14` | `393.86` | `-13.47%` |

Interpretation: the first frozen H6 policy is quality-preserving under the locked 1% validation-loss gate and does not add instability. This is an important positive result because the perturbation-selected modules remained harmless during actual LoRA updates, not just in no-update forward probes. However, the implementation is a Python-level fake-quant hook, so the measured throughput regression is not evidence against hardware-realistic low precision; it only shows the emulation is slower. H6 currently supports the sensitivity-prediction claim, not yet a resource-saving claim.

The H6 narrow-policy treatment has now also completed for seeds 43 and 44, but the matched 500-step bf16 controls for those seeds are not present under the expected results directory. The three H6 treatment runs are stable:

| seed | final eval loss | final train loss | max grad norm | loss spikes | NaN/Inf | train tokens/sec |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | `1.63112` | `1.40745` | `4.0926` | `0` | `0` | `393.86` |
| 43 | `1.63621` | `1.75036` | `4.6385` | `0` | `0` | `433.89` |
| 44 | `1.63493` | `1.64997` | `6.1208` | `0` | `0` | `433.79` |

Across the three H6 treatment seeds, final eval loss has mean `1.63409` and standard deviation `0.00265`, with zero loss spikes and zero NaN/Inf events. This strengthens the stability side of the H6 result. It does not replace the missing paired controls: to claim multi-seed quality matching, run 500-step bf16 baselines for seeds 43 and 44 with the same train/eval split and settings.

## 2026-05-13 Smoke Calibration

The first H6 smoke probe ran on Qwen/Qwen2.5-0.5B with one Alpaca calibration batch, sequence length 64, fp32 dtype, and the first eight candidate modules. It completed on CUDA and wrote both `stability_signals.json` and `policy_trace.json`.

Observed run-level signals:

- Mean calibration loss: `1.7697092294692993`
- NaN/Inf count: `0`
- Peak CUDA memory: `2.182478427886963 GiB`
- Elapsed time: `10.444058656692505 sec`

The first eight modules covered layer-0 Q/K/V/O projections, MLP gate/up/down projections, and input RMSNorm. Under the conservative smoke thresholds, all seven projection modules stayed at `bf16` and the input RMSNorm was promoted to `fp32`.

The strongest observed signals were heavy activation outliers and fake-quant error in early layer paths:

- `layer.0.mlp.down_proj` input outlier score: `72.95`
- `layer.0.self_attn.o_proj` output outlier score: `35.92`
- `layer.0.input_layernorm` output outlier score: `24.83`
- `layer.0.input_layernorm` output int8 relative MSE: `0.00318`

Interpretation: the smoke result is consistent with the current conservative hypothesis that normalization outputs and early projection paths can show high sensitivity signals. It does not yet prove that fp32 norms improve training, nor does it justify demoting any path to int8/int4. The next step is a fuller bf16 calibration pass across all candidate modules and more batches, followed by perturbation-based loss-delta checks.

## Limitations

- One batch and one layer slice are insufficient for a stable policy.
- The run used fp32 rather than bf16 because it was a smoke check. The next calibration should use bf16 autocast.
- Fake quantization error is a sensitivity proxy, not a hardware speed or memory claim.
- The probe currently records activation sensitivity but does not yet record one-island perturbation loss deltas or update divergence.
