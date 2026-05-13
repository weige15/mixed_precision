# H1 Analysis: fp32 Normalization During bf16 LoRA Fine-Tuning

No main experiment result exists yet.

The first required result is the dtype probe. Before comparing `bf16_baseline` against `fp32_norms`, we must verify whether the baseline PyTorch autocast policy already runs Qwen RMSNorm / LayerNorm or related normalization and loss operations in fp32.

H1 is scientifically meaningful only if all of the following hold:

- The dtype probe shows the baseline does not already run the target normalization operation in fp32.
- The `fp32_norms` policy changes the actual computation dtype for RMSNorm / LayerNorm-like modules or another explicitly targeted normalization operation.
- The `fp32_norms` run improves loss stability, reduces spike count, or preserves validation loss while keeping memory and throughput overhead reasonable.

Until those conditions are checked, H1 should be treated as an executable protocol rather than a supported result.

## Implementation Readiness

The LoRA runner now defaults to the H1 protocol split: 8,000 shuffled training examples and 1,000 validation examples. Smoke runs can still use smaller explicit `--train-size`, `--eval-size`, and `--eval-max-batches` overrides.

`summary.json` now records final validation loss, max gradient norm, effective batch size, train/eval split sizes, train-only throughput, train-only throughput excluding the first step, peak CUDA memory, NaN/Inf count, and loss-spike count.

Two one-step development smoke runs completed on CUDA:

- `bf16_baseline`: final eval loss `2.3294625282287598`, max grad norm `5.4556355476379395`, no NaN/Inf.
- `fp32_norms`: final eval loss `2.3311660289764404`, max grad norm `5.404299736022949`, no NaN/Inf.

These development smoke runs are implementation checks only. They are not evidence for or against H1.
