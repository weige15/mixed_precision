# Research Findings

## Research Question

Before LoRA fine-tuning, can a short precision check identify which model modules are precision-sensitive, so a frozen mixed-precision policy can match bf16 validation quality while saving resources or expanding the stable fine-tuning envelope?

## Current Understanding

The initial working assumption was that blanket bf16 mixed precision may already be strong for small LoRA fine-tuning, but some operations may be disproportionately sensitive to reduced precision. The first matched H1 run supports the "bf16 is already strong here" part more than the "fp32 norms are a useful static island" part: fp32 norms produce only a small seed-42 validation-loss improvement, with no stability events in either policy.

The broader research direction is now simplified as a pre-training precision check. In plain terms: quickly test the model before fine-tuning, find the fragile modules, spend high precision only where it matters, freeze that choice, and then fine-tune. H6 is the trunk hypothesis. H1 is now a weak static anchor and cautionary example rather than the central story. H5 is important because dtype claims need internal-compute validation. H2, H3, and H4 are useful conditional ablations, but they are not blockers for the first H6 policy experiment.

The next H6 policy should use measured activation outliers, local perturbation loss deltas, quantization error, clipping or saturation rate, NaN/Inf incidence, memory, and throughput to decide which operations should be kept safe and which are candidates for lower precision.

The refreshed literature suggests three especially useful signal families for H6. First, LLM.int8() and SmoothQuant indicate that activation outliers are a strong predictor of INT8 sensitivity. Second, FP4 training and Attn-QAT suggest low-bit training is limited by quantization noise, rounding, heavy-tailed attention activations, and hidden backward-pass precision assumptions. Third, HAWQ, HAWQ-V3, convergence-aware mixed precision, and SNIP frame precision assignment as constrained optimization over predicted quality loss and hardware cost.

## Key Results

H1 has a matched seed-42 1,000-step result. The bf16 baseline reaches final validation NLL `2.09066`, train-only throughput `457.06` tokens/sec, peak CUDA memory `4.5319 GiB`, zero loss spikes, and zero NaN/Inf events. The fp32_norms treatment reaches final validation NLL `2.08642`, train-only throughput `453.27` tokens/sec, peak CUDA memory `4.5469 GiB`, zero loss spikes, and zero NaN/Inf events.

The seed-42 fp32_norms delta is `-0.00425` validation NLL, about `0.20%` relative improvement, with about `0.83%` train-throughput overhead and about `0.33%` peak-memory overhead. Under the locked H1 decision rule, this is inconclusive rather than supportive because the quality delta is below 1% and there are no instability events to remove.

The extra bf16 baseline seeds show that baseline run-to-run variation is larger than the seed-42 fp32_norms gain: seed 43 reaches `2.06876`, and seed 44 reaches `2.09593`. This strengthens the caution behind H4 even before matched fp32_norms seeds 43 and 44 are available.

H5 now explains why H1 may have little effect. The targeted RMSNorm internal probe recorded the installed `Qwen2RMSNorm.forward` source, which casts `hidden_states` to `float32` before `pow`, `mean`, and `rsqrt`, then casts the normalized activations back to the input dtype before multiplying by the norm weight. The executed H5 run used CPU because CUDA was unavailable at probe time, but its reference implementation matched actual module output exactly (`max_abs_diff = 0.0`). Combined with the earlier CUDA boundary probe showing bf16 module inputs/outputs, the likely interpretation is that baseline Qwen RMSNorm already performs its reduction arithmetic in fp32 even when its boundary tensors are bf16.

The first H6 signal-only smoke probe completed on 2026-05-13. It ran Qwen/Qwen2.5-0.5B on one Alpaca calibration batch with sequence length 64, fp32 dtype, and the first eight candidate modules. The probe wrote `stability_signals.json` and `policy_trace.json` under `experiments/h6-adaptive-precision-assignment/results/smoke_signals_seed42/`. It observed mean calibration loss `1.7697`, zero NaN/Inf events, and peak CUDA memory `2.1825 GiB`.

Under the conservative H6 decision rule, the smoke policy promoted layer-0 input RMSNorm to `fp32` and kept the seven observed projection modules at `bf16`. The largest early signals were `mlp.down_proj` input outlier score `72.95`, `self_attn.o_proj` output outlier score `35.92`, and `input_layernorm` output int8 relative MSE `0.00318`. This is instrumentation evidence, not final adaptive-policy evidence.

The fuller H6 Stage 1 calibration now exists for bf16 seeds 42, 43, and 44. Each run used sequence length 512, 8 calibration batches, all 218 candidate modules, CUDA bf16 autocast, and zero NaN/Inf events. Policy decisions are highly stable: 217 of 218 common modules received the same assignment across all three seeds. The only unstable assignment was `layers.23.mlp.gate_proj`, which was an int8 candidate for seed 42 but bf16 for seeds 43 and 44.

The signal-only policy is very conservative. All 96 attention projections remain bf16 in all seeds; all 49 norm/logit paths are promoted to fp32; and almost every MLP projection remains bf16. The strongest high-risk modules are stable across seeds, especially `layers.2.mlp.down_proj`, `layers.3.mlp.down_proj`, and `layers.21.mlp.down_proj`, which show extreme activation outlier scores and int8 relative MSE well above the current candidate threshold. A relaxed screen identifies only four plausible low-risk projection modules for perturbation testing: `layers.23.mlp.gate_proj`, `layers.23.mlp.up_proj`, `layers.22.mlp.gate_proj`, and `layers.22.mlp.up_proj`.

The Stage 2 perturbation probe now replicates across seeds 42, 43, and 44. Fake int8 output perturbation of the three high-risk MLP down projections caused large positive mean loss increases in every seed: averaged across seeds, `layers.2.mlp.down_proj` changed by `+0.3699`, `layers.21.mlp.down_proj` by `+0.2665`, and `layers.3.mlp.down_proj` by `+0.2209`. The four relaxed low-risk late-layer MLP candidates stayed near zero, with mean absolute loss deltas between `0.0016` and `0.0035`, and worst-case absolute mean delta at most `0.0070`. Across all 21 MLP projection perturbations, max outlier score correlates with absolute loss delta at about `0.91`; output int8 relative MSE is weaker pooled across seeds at about `0.53`.

The norm/logit controls did not show large output-quantization loss deltas, despite Stage 1 marking them as sensitive. This should not be overread: output fake quantization is not the same as reducing internal RMSNorm reduction arithmetic or loss-computation precision. The norm/logit result mainly says the current Stage 2 perturbation design is best suited to projection-output sensitivity.

The first frozen H6 policy training test is positive on quality and stability but negative on emulated speed. A matched 500-step seed-42 comparison gives bf16 final eval loss `1.62949` and H6 late-layer MLP fake-int8 final eval loss `1.63112`, a `+0.00163` absolute degradation or about `+0.10%` relative. This is inside the locked 1% quality gate. Both runs had zero loss spikes and zero NaN/Inf events, and max grad norm was essentially unchanged. Peak memory was also essentially unchanged. Train throughput fell from `455.14` to `393.86` tokens/sec, a `13.47%` slowdown, because the current policy uses Python-level fake-quant hooks rather than hardware-supported low-precision kernels.

The H6 narrow-policy comparison is now paired across seeds 42, 43, and 44. BF16 eval losses were `1.62949`, `1.63444`, and `1.63247`; H6 eval losses were `1.63112`, `1.63621`, and `1.63493`. The paired H6 deltas were `+0.00163`, `+0.00177`, and `+0.00246`, corresponding to relative degradations of `+0.100%`, `+0.108%`, and `+0.151%`. The mean paired degradation is `+0.00195`, or `+0.120%`, with every seed inside the locked 1% quality gate. Both policies had zero total loss spikes and zero total NaN/Inf events across the three seeds. The current fake-quant implementation still does not provide a resource-saving result: peak memory is effectively unchanged, and mean train throughput is about `5.8%` lower with substantial per-seed variation.

H6.1 is now implemented as the next policy-expansion branch. The SNIP-style policy builder aggregates seed 42-44 calibration signals and perturbation deltas, ranks eligible MLP `gate_proj`/`up_proj` modules by conservative risk, anchors `k=4` to the already validated late-layer policy, and emits frozen `k=4/8/16/24` module lists for `h6_custom_int8` training. A one-step CUDA smoke confirmed the generated `k=4` list resolves to the validated four modules and trains without NaN/Inf. The next real experiment is a seed-42 500-step width sweep over `k=8`, `k=16`, and `k=24` against matched bf16, followed by seed 43/44 replication only for the best safe wider policy.

The H6.1 seed-42 width sweep is now complete and positive. The matched bf16 control reached final eval loss `1.62978`. SNIP-style `k=4`, `k=8`, `k=16`, and `k=24` reached `1.63111`, `1.63173`, `1.63126`, and `1.63177`, respectively. These are relative degradations of `+0.082%`, `+0.120%`, `+0.091%`, and `+0.122%`, all far inside the locked 1% quality gate. Every run had zero loss spikes and zero NaN/Inf events. This means the conservative score can safely widen from the original four late-layer MLP modules to at least 24 MLP gate/up modules on seed 42. The cost metrics are not yet a resource claim: peak memory changed by less than `0.003 GiB`, and throughput moved between about `-2.35%` and `+3.08%` under Python fake-quant hooks.

The H6.1 `k=24` result now replicates across seeds 42, 43, and 44. Seed 42 used per-device batch size 1 with gradient accumulation 16; seeds 43 and 44 used per-device batch size 2 with gradient accumulation 8, preserving effective batch size 16. Each seed has a matched bf16 control under the same microbatch setting. BF16 eval losses were `1.62978`, `1.61701`, and `1.62089`; `k=24` eval losses were `1.63177`, `1.61975`, and `1.62248`. The paired relative degradations were `+0.122%`, `+0.169%`, and `+0.098%`, with mean `+0.130%`, all far inside the locked 1% quality gate. All six runs had zero loss spikes and zero NaN/Inf events. The batch-size-2 runs roughly doubled throughput versus the earlier batch-size-1 run, but `k=24` remained about `3-4%` slower than matched bf16 and used about `0.009 GiB` more peak memory, so the result still supports policy selection quality, not real resource savings.

H6.2 is now the active resource-feasibility branch. The implementation adds a bitsandbytes 8-bit LoRA baseline and a hardware matrix that compares matched bf16, H6.1 fake-int8 `k=24`, QLoRA 4-bit NF4, and 8-bit LoRA under the same hardware label and microbatching. The purpose is not yet to build custom selective kernels; it is to test whether supported hardware-backed low precision gives any real memory or throughput win on the RTX 4050 while staying inside the 1% quality gate.

The H6.2 lab RTX 3090 feasibility screen is complete and negative for resource savings at Qwen2.5-0.5B. The run metadata confirms `cuda_device_name = NVIDIA GeForce RTX 3090` and `CUDA_VISIBLE_DEVICES = 3`, with `HARDWARE_LABEL=rtx3090-lab`, per-device batch size 2, gradient accumulation 8, effective batch size 16, and 100 optimizer steps. Matched bf16 reached eval loss `1.64161`, peak memory `4.530 GiB`, and `897.1` train tokens/sec excluding the first step. Fake-int8 `k=24` stayed within the quality gate at `1.64601` (`+0.268%`) but used slightly more memory (`+0.20%`) and was slower (`-2.16%`). bitsandbytes 8-bit LoRA reached `1.64329` (`+0.102%`) but used more memory (`+16.8%`) and was much slower (`-40.2%`). QLoRA 4-bit NF4 reached `1.68811` (`+2.83%`), failing the 1% quality gate, while also using more memory (`+15.8%`) and running slower (`-37.6%`). All four policies had zero loss spikes and zero NaN/Inf events.

The H6.3 7B scale screen changes the resource story from uniformly negative to a memory-throughput trade-off. On Qwen/Qwen2.5-7B, matched bf16 reached eval loss `1.39876`, peak memory `19.448 GiB`, and `191.2` train tokens/sec excluding the first step. bitsandbytes 8-bit LoRA reached `1.40028` (`+0.108%`), saved `6.70%` peak memory, but was `40.53%` slower. QLoRA 4-bit NF4 reached `1.40867` (`+0.709%`), saved `23.32%` peak memory, but was `19.78%` slower. Both low-bit policies stayed within the 1% quality gate with zero instability events. The 8-bit run emitted `MatMul8bitLt` warnings showing fp32/bf16 inputs are cast to fp16 during quantization, so this result is best described as int8-weight training with fp16 activation matmul behavior rather than pure bf16 compute.

The H6.3 7B 500-step seed-42 follow-up confirms QLoRA as the useful hardware-backed baseline. Matched bf16 reached eval loss `1.37747`, peak memory `19.448 GiB`, and `191.6` train tokens/sec excluding the first step. QLoRA 4-bit NF4 reached eval loss `1.38524`, a `+0.563%` relative degradation that remains inside the 1% quality gate, with the same `23.32%` peak-memory saving and a `19.95%` throughput penalty. Both runs had zero loss spikes and zero NaN/Inf events.

## Patterns and Insights

- The simplified H6 contribution is: replace hand-written dtype rules with a short measured precision check before training.
- Static precision islands are a prerequisite for adaptive assignment, but not every static-island hypothesis is mandatory. H6 needs at least one matched static anchor, currently H1, plus dtype-validity evidence from H5.
- H1 seed 42 is now best interpreted as a weak static anchor: fp32 norms are feasible and cheap, but they do not yet show a meaningful benefit over bf16 in the default stable regime.
- H5 weakens the original mechanistic rationale for H1. If Qwen2RMSNorm already upcasts internally, the fp32_norms wrapper changes boundary/call behavior less than expected and should not be treated as a strong precision intervention.
- The default bf16 LoRA recipe appears stable at the current scale. Absence of spikes/NaNs in both H1 policies means stability-sensitive precision effects may require stressed settings or perturbation-based H6 probes to become measurable.
- Baseline seed variation is already larger than the fp32_norms seed-42 delta, so future claims need paired seeds or a stronger effect size.
- The current runner already logs useful run-level stability signals: loss spikes, NaN/Inf count, max gradient norm, peak memory, and train-only throughput.
- H6 now has initial per-module signal instrumentation for activation outliers, fake-quantization error, saturation, finite checks, and policy trace generation.
- INT8/INT4 inference papers are useful for signal design but should not be treated as direct evidence for training-time LoRA stability.
- Attention and logits/loss should be handled conservatively under subbyte precision because the literature flags heavy-tailed activations and backward precision assumptions as instability sources.
- The first H6 smoke suggests early-layer activation outlier scores can be large enough that naive int8/int4 demotion would be unsafe without perturbation loss-delta checks.
- The central unresolved question is predictive validity: do the cheap calibration signals actually predict one-island loss deltas and later training outcomes?
- The three-seed bf16 calibration is stable enough to proceed to perturbation tests. It is not rich enough to freeze a resource-saving policy because almost no modules are robust low-precision candidates under the current thresholds.
- Stage 2 now replicates the predictive-validity story for MLP projections across seeds 42-44: high Stage 1 outlier signals map to large int8 perturbation loss deltas, while relaxed low-risk late-layer gate/up projections have near-zero deltas.
- Stage 3 shows the perturbation-selected low-risk modules preserve bf16 validation quality and stability during actual 500-step LoRA updates across seeds 42-44. The quality-preservation claim is now paired across three seeds.
- H6.1 shows that the safe fake-int8 MLP gate/up set is not limited to the original four late-layer modules. Across seeds 42-44, widening to 24 ranked modules preserved validation quality and stability, suggesting the calibration score is conservative enough for budgeted expansion.

## Lessons and Constraints

- The pilot must avoid full pretraining and large benchmark suites.
- Experiments should use fixed-step or fixed-wall-clock budgets so precision policies are comparable on limited GPU resources.
- A useful negative result is possible: standard bf16 may be Pareto-optimal for this LoRA regime, which would redirect the project toward stressed settings or other precision targets.
- Aggressive low-precision candidates should be staged conservatively. Projection paths are better first targets for int8/int4 exploration than normalization reductions or attention softmax.
- Adaptive decisions must be frozen before final validation comparison. Otherwise the policy can overfit to the observed result.
- Signal-only calibration should be treated as a ranking/prior. It needs perturbation loss deltas before it can justify an adaptive training policy.
- The policy must be frozen after calibration and before training. Otherwise, it becomes an exploratory tuning procedure rather than a test of whether the short precision check predicts sensitivity.
- The perturbation panel is now replicated enough to freeze the first narrow H6 policy. Keep high-risk down projections, attention projections, norms, and logits conservative; only test demoting the consistently low-delta `layers.22/23.mlp.gate_proj` and `layers.22/23.mlp.up_proj` paths.
- The 500-step LoRA paired comparison is complete across seeds 42-44. The next decision is whether to implement hardware-realistic low precision for resource claims or to stress the regime to test whether calibration-guided precision expands the stable fine-tuning envelope.
- The H6.1 SNIP-style expansion should be treated as a budgeted policy test, not a full SNIP reproduction. Keep the current four-module policy as the anchor and evaluate whether wider `k=8/16/24` MLP gate/up policies preserve the 1% validation-loss gate.
- H6.1 `k=24` is now the best quality-preserving policy result. It demotes half of the eligible MLP gate/up modules and holds across three seeds, but it should not be described as resource-saving until backed by hardware-supported lower-precision kernels.
- H6.2 should compare resource deltas only within matched hardware labels and microbatch settings. Do not mix old batch-size-1 results with batch-size-2 resource screens when making throughput claims.
- On the lab RTX 3090, existing bitsandbytes 4-bit/8-bit training paths do not provide the desired resource win for Qwen2.5-0.5B or 1.5B LoRA. At Qwen2.5-7B they do save memory, but both are slower than bf16, so the current evidence supports a memory-capacity trade-off rather than a Pareto improvement.
- The Qwen2.5-7B QLoRA memory-capacity trade-off holds for 500 steps on seed 42: quality stays inside the 1% gate, peak memory drops by 23.32%, and throughput remains about 20% slower.
- bitsandbytes 8-bit LoRA casts bf16/fp32 activations to fp16 inside `MatMul8bitLt`; reports should not describe the 8-bit path as pure bf16 compute.
- Low-bit perturbation probes can identify sensitivity, but real throughput or memory claims require hardware-supported kernels on the target machine.
- Boundary dtype probes are not enough for normalization layers. For Qwen2RMSNorm, source-level/internal-operation validation is required because bf16 boundaries can coexist with fp32 internal reductions.
- H2 should be run only if H6 needs a stronger logits/loss static anchor. H3 should be run only if the default bf16 regime is too stable to reveal meaningful policy differences. H4 should be run after a candidate H6 policy exists, not before.
- A single-line H3 stress artifact exists, but it is incomplete and should not be interpreted as evidence.

## Open Questions

- Does fp32 normalization produce a paired multi-seed effect, or is the seed-42 improvement entirely within ordinary seed noise?
- Does the lack of H1 instability mean the default regime is too stable, or that norms are simply not the sensitive operation?
- If H1 is inconclusive, should the next precision island be logits/loss computation, attention softmax, optimizer state precision, or a stressed fp16 setting?
- Which per-module signals best predict precision sensitivity during LoRA fine-tuning?
- Can a short pre-training precision check rank fragile and tolerant modules well enough to choose a frozen policy?
- Does the H6 signal ranking remain stable across bf16 autocast, longer sequence length, more batches, and seeds 42-44?
- Can the H6 policy produce real resource savings with hardware-supported low-precision kernels, or is the current result limited to sensitivity assignment under emulation?
- Can a short calibration pass derive a precision assignment that matches the best static policy while improving memory or throughput?
- Are int8 or int4 candidates real compute improvements on the available RTX 4050, or only fake-quant research probes?
- Are the four relaxed low-risk candidates actually harmless under one-island perturbation, or are the signal thresholds still missing important training-time sensitivity?
- Does the narrow candidate policy provide any real resource saving on available hardware, or only a sensitivity-ranking result under fake quantization?
- How wide can the SNIP-style H6.1 fake-int8 MLP gate/up policy become before validation loss or stability degrades?
- Can the H6.1 `k=24` policy be converted from Python fake-quant hooks into a hardware-realistic low-precision implementation, or is the current contribution best framed as a calibration and sensitivity-ranking method?
- Is there a hardware-backed implementation path beyond generic bitsandbytes k-bit loading that can exploit the H6.1 selected module set, or should the paper contribution stop at calibration-guided sensitivity ranking with resource savings left as future systems work?

## Optimization Trajectory

The first training trajectory points are now available:

- H1 bf16 baseline seed 42: final validation NLL `2.09066`
- H1 fp32_norms seed 42: final validation NLL `2.08642`
- H1 bf16 baseline seed 43: final validation NLL `2.06876`
- H1 bf16 baseline seed 44: final validation NLL `2.09593`
- H5 RMSNorm internal dtype probe: Qwen2RMSNorm source/reference path uses fp32 for reduction operations; reference and actual outputs matched with max absolute difference `0.0` on the CPU probe.

The H6 calibration artifact remains signal-only and should be plotted separately from optimizer-step training results.
