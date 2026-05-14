# H1 Analysis: fp32 Normalization During bf16 LoRA Fine-Tuning

## 2026-05-14 Main Seed-42 Result

The first matched 1,000-step H1 comparison has completed for seed 42.

| policy | final eval NLL | train tokens/sec | peak memory GiB | max grad norm | loss spikes | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|
| `bf16_baseline` | `2.0906627425` | `457.0637` | `4.5319` | `3.7313` | `0` | `0` |
| `fp32_norms` | `2.0864177777` | `453.2701` | `4.5469` | `3.7405` | `0` | `0` |

The fp32_norms treatment improves validation NLL by `0.0042449648` absolute, about `0.20%` relative to the bf16 baseline. Train-only throughput is about `0.83%` lower, and peak memory is about `0.33%` higher.

Under the locked decision rule in the protocol, this is **inconclusive** rather than supportive. The quality improvement is below the required 1% threshold, and neither policy shows instability events for fp32 norms to remove.

Additional bf16 baseline runs completed for seed 43 and seed 44:

| policy | seed | final eval NLL | train tokens/sec | loss spikes | NaN/Inf |
|---|---:|---:|---:|---:|---:|
| `bf16_baseline` | 43 | `2.0687621786` | `499.2433` | `0` | `0` |
| `bf16_baseline` | 44 | `2.0959309117` | `503.8679` | `0` | `0` |

This baseline seed spread is larger than the seed-42 fp32_norms improvement, so the current evidence should not be interpreted as a real precision-island gain without paired treatment runs or a stronger stressed-regime effect.

## Dtype Validity

H1 is scientifically meaningful only if all of the following hold:

- The dtype probe shows the baseline does not already run the target normalization operation in fp32.
- The `fp32_norms` policy changes the actual computation dtype for RMSNorm / LayerNorm-like modules or another explicitly targeted normalization operation.
- The `fp32_norms` run improves loss stability, reduces spike count, or preserves validation loss while keeping memory and throughput overhead reasonable.

The existing boundary dtype probe shows Qwen RMSNorm module inputs, outputs, and parameters are bf16 under the earlier CUDA baseline autocast probe, so the intervention was not an obvious boundary-level no-op.

The H5 internal probe changes the interpretation. The installed `Qwen2RMSNorm.forward` source explicitly casts `hidden_states` to `float32` before `pow`, `mean`, and `rsqrt`, then casts the normalized activations back to the input dtype before multiplying by the norm weight. The H5 execution itself ran on CPU because CUDA was unavailable at probe time, but the reference implementation matched actual module output exactly with max absolute difference `0.0`.

Best current interpretation: Qwen2RMSNorm likely already performs the sensitive reduction arithmetic in fp32 under the baseline, even when boundary tensors are bf16. That makes the H1 fp32_norms wrapper a weaker intervention than originally assumed and is a plausible explanation for the inconclusive seed-42 result.

## Implementation Readiness

The LoRA runner now defaults to the H1 protocol split: 8,000 shuffled training examples and 1,000 validation examples. Smoke runs can still use smaller explicit `--train-size`, `--eval-size`, and `--eval-max-batches` overrides.

`summary.json` now records final validation loss, max gradient norm, effective batch size, train/eval split sizes, train-only throughput, train-only throughput excluding the first step, peak CUDA memory, NaN/Inf count, and loss-spike count.

Two one-step development smoke runs completed on CUDA:

- `bf16_baseline`: final eval loss `2.3294625282287598`, max grad norm `5.4556355476379395`, no NaN/Inf.
- `fp32_norms`: final eval loss `2.3311660289764404`, max grad norm `5.404299736022949`, no NaN/Inf.

These development smoke runs are implementation checks only. They are not evidence for or against H1.
