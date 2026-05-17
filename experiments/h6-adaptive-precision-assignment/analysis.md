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

The paired 500-step comparison is now complete for seeds 42, 43, and 44:

| seed | bf16 eval loss | H6 eval loss | H6 delta | H6 rel delta | instability |
|---:|---:|---:|---:|---:|---|
| 42 | `1.62949` | `1.63112` | `+0.00163` | `+0.100%` | none in either run |
| 43 | `1.63444` | `1.63621` | `+0.00177` | `+0.108%` | none in either run |
| 44 | `1.63247` | `1.63493` | `+0.00246` | `+0.151%` | none in either run |

Across seeds, the mean paired eval-loss delta is `+0.00195`, or `+0.120%` relative, with every seed well inside the locked 1% quality gate. BF16 eval loss has mean `1.63213` and standard deviation `0.00249`; H6 eval loss has mean `1.63409` and standard deviation `0.00265`. Both policies have zero total loss spikes and zero total NaN/Inf events across all three seeds.

The cost story is still not positive under the current emulation. H6 changes peak memory by only `-0.00061 GiB` and has mean train-throughput delta `-5.80%`, with high seed-to-seed variation (`-13.47%`, `-4.87%`, `+0.93%`). Because this policy uses Python-level fake quantization hooks, these speed and memory numbers should not be treated as hardware-realistic low-precision performance. The supported claim is now: calibration-guided precision selection preserved bf16 validation quality and stability across three seeds for the selected late-layer MLP modules.

## 2026-05-16 H6.1 SNIP-Style Width Screen

H6.1 tests whether a SNIP-style score can safely expand the fake-int8 policy beyond the original four late-layer MLP gate/up modules. The policy builder aggregates calibration signals and perturbation deltas, anchors `k=4` to the validated H6 narrow policy, and adds ranked MLP `gate_proj` / `up_proj` modules for larger budgets.

The seed-42 500-step screen completed for matched bf16 and `k=4/8/16/24`.

| policy | fake-int8 modules | final eval loss | delta vs bf16 | rel delta | loss spikes | NaN/Inf | train tok/s ex-first | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 0 | `1.62978` | - | - | 0 | 0 | `448.35` | `2.77945` |
| k=4 | 4 | `1.63111` | `+0.00133` | `+0.082%` | 0 | 0 | `447.01` | `2.77884` |
| k=8 | 8 | `1.63173` | `+0.00195` | `+0.120%` | 0 | 0 | `462.16` | `2.77981` |
| k=16 | 16 | `1.63126` | `+0.00148` | `+0.091%` | 0 | 0 | `441.20` | `2.77664` |
| k=24 | 24 | `1.63177` | `+0.00199` | `+0.122%` | 0 | 0 | `437.81` | `2.78030` |

Interpretation: this is a positive width result. All SNIP-style budgets are far inside the locked 1% validation-loss gate, and none add instability. The `k=24` policy is the most informative next candidate because it demotes half of the eligible MLP gate/up module set and still preserves seed-42 quality and stability. The throughput and memory measurements should remain secondary because Python fake-quant hooks are not hardware-realistic; the supported claim is wider quality-preserving module selection, not resource savings.

Next step: replicate `k=24` on seeds 43 and 44 with matched bf16 controls. If it holds, H6.1 supports a stronger statement: calibration-guided scoring can widen the safe low-precision candidate set substantially, not just identify four handpicked late-layer modules.

The `k=24` replication is now complete for seeds 43 and 44. These runs used per-device batch size 2 and gradient accumulation 8, keeping the effective batch size at 16. Because each seed has a matched bf16 control with the same microbatching, the paired comparison remains valid.

| seed | microbatch x accum | bf16 eval | k=24 eval | delta | rel delta | instability | k=24 tok/s ex-first | bf16 tok/s ex-first |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 42 | 1 x 16 | `1.62978` | `1.63177` | `+0.00199` | `+0.122%` | none | `437.81` | `448.35` |
| 43 | 2 x 8 | `1.61701` | `1.61975` | `+0.00274` | `+0.169%` | none | `861.39` | `898.12` |
| 44 | 2 x 8 | `1.62089` | `1.62248` | `+0.00159` | `+0.098%` | none | `866.24` | `893.59` |

Across seeds, the mean paired eval-loss degradation is `+0.00211` absolute, or `+0.130%` relative. Both policies had zero loss spikes and zero NaN/Inf events in every run. This supports the H6.1 claim that the SNIP-style score can widen the safe fake-int8 MLP gate/up set to 24 modules while preserving bf16 validation quality and stability.

The resource story remains unresolved. Moving from batch size 1 to batch size 2 roughly doubled throughput for both policies, but this is a microbatching improvement rather than a low-precision policy improvement. Within the matched batch-size-2 runs, `k=24` was about `3-4%` slower than bf16 and used about `0.009 GiB` more peak memory due to the Python fake-quant hooks.

## 2026-05-17 H6.2 Hardware-Realistic Resource Screen

H6.2 tests whether existing hardware-backed low-precision paths produce a real resource benefit before investing in custom selective kernels. The completed screen ran on the lab RTX 3090 (`cuda_device_name=NVIDIA GeForce RTX 3090`, `CUDA_VISIBLE_DEVICES=3`) with per-device batch size 2, gradient accumulation 8, effective batch size 16, learning rate `2e-4`, and 100 optimizer steps.

| policy | eval loss | eval delta vs bf16 | peak GiB | memory delta | train tok/s ex-first | tok/s delta | instability |
|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | `1.64161` | - | `4.530` | - | `897.1` | - | none |
| fake-int8 k=24 | `1.64601` | `+0.268%` | `4.539` | `+0.20%` | `877.7` | `-2.16%` | none |
| bitsandbytes 8-bit LoRA | `1.64329` | `+0.102%` | `5.291` | `+16.80%` | `536.8` | `-40.16%` | none |
| QLoRA 4-bit NF4 | `1.68811` | `+2.832%` | `5.246` | `+15.79%` | `560.0` | `-37.58%` | none |

Interpretation: H6.2 is negative for resource savings on this RTX 3090/bitsandbytes setup. The 8-bit policy preserves quality but loses badly on memory and throughput. QLoRA fails the 1% quality gate and also loses on memory and throughput. Fake-int8 k=24 remains quality-preserving but is still a Python-hook sensitivity proxy, not a resource implementation. This strengthens the current framing: the supported contribution is calibration-guided precision sensitivity ranking and safe policy expansion, while hardware savings require a different implementation path.

## 2026-05-17 H6.3 7B Hardware Scale Screen

The 7B scale screen ran on the lab RTX 3090 (`cuda_device_name=NVIDIA GeForce RTX 3090`, `CUDA_VISIBLE_DEVICES=3`) with Qwen/Qwen2.5-7B, per-device batch size 1, gradient accumulation 16, effective batch size 16, sequence length 512, learning rate `2e-4`, and 100 optimizer steps.

| policy | eval loss | eval delta vs bf16 | peak GiB | memory delta | train tok/s ex-first | tok/s delta | instability |
|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | `1.39876` | - | `19.448` | - | `191.2` | - | none |
| bitsandbytes 8-bit LoRA | `1.40028` | `+0.108%` | `18.145` | `-6.70%` | `113.7` | `-40.53%` | none |
| QLoRA 4-bit NF4 | `1.40867` | `+0.709%` | `14.912` | `-23.32%` | `153.4` | `-19.78%` | none |

Interpretation: 7B is the first scale where existing bitsandbytes low-bit paths produce real peak-memory savings under this setup. Both 8-bit LoRA and QLoRA 4-bit NF4 stay inside the locked 1% validation-loss gate and have zero loss spikes or NaN/Inf events. The resource trade-off is still not Pareto-positive because both policies reduce throughput, with 8-bit LoRA especially slow. During 8-bit LoRA, bitsandbytes emitted `MatMul8bitLt` warnings that fp32 and bf16 inputs are cast to fp16 during quantization. This does not invalidate the run, but it means the 8-bit result should be described as a hardware-backed int8-weight path with fp16 activation matmul behavior, not as pure bf16 compute.

## 2026-05-18 H6.3 7B 500-Step QLoRA Replication

The 500-step 7B follow-up replicated the most promising 100-step policy, QLoRA 4-bit NF4, against a matched bf16 control on seed 42. The run used the same lab RTX 3090, Qwen/Qwen2.5-7B, sequence length 512, per-device batch size 1, gradient accumulation 16, effective batch size 16, and learning rate `2e-4`.

| policy | eval loss | eval delta vs bf16 | peak GiB | memory delta | train tok/s ex-first | tok/s delta | instability |
|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | `1.37747` | - | `19.448` | - | `191.6` | - | none |
| QLoRA 4-bit NF4 | `1.38524` | `+0.563%` | `14.912` | `-23.32%` | `153.4` | `-19.95%` | none |

Interpretation: the longer seed-42 run strengthens the 7B memory-capacity trade-off result. QLoRA remains inside the locked 1% validation-loss gate, preserves stability, and saves the same `23.32%` peak memory as the 100-step screen. The throughput penalty is also stable at about `20%`. This is now worth seed replication on seeds 43 and 44 before making a robust 7B claim.

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
