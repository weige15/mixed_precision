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
