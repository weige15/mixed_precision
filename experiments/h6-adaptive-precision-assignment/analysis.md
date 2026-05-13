# H6 Analysis

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
