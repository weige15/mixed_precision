# Literature Survey: Adaptive Precision Assignment From Stability Signals

## Research Question

During LLM training, fine-tuning, and inference, can precision be assigned adaptively from stability signals such as activation outliers, gradient spikes, loss deltas, quantization error, clipping/saturation rate, NaN/Inf incidence, memory, and throughput, instead of relying on a single global dtype or hand-selected precision islands?

## Scope

This survey focuses on mixed precision and low-precision execution for LLMs and adjacent neural-network training systems: FP16, BF16, FP8, INT8, INT4/FP4, and adaptive precision assignment. Pure pruning and pure post-training quantization are not the main focus. Quantization papers are included only when they contribute useful evidence about mixed precision, sensitivity signals, outlier handling, or operation-wise precision choices.

## Key Papers

| ID | Paper | Precision Focus | Why It Matters |
|---|---|---|---|
| MP2017 | [Mixed Precision Training](mixed_precision_training_2017.md) | FP16 + FP32 | Establishes loss scaling and FP32 master weights as stability mechanisms for FP16 training. |
| BF162019 | [A Study of BFLOAT16 for Deep Learning Training](bfloat16_training_2019.md) | BF16 | Explains why BF16 is a strong baseline because it keeps FP32-like exponent range. |
| FP8FMT2022 | [FP8 Formats for Deep Learning](fp8_formats_2022.md) | FP8 | Defines E4M3/E5M2 tradeoffs and motivates role-specific format selection. |
| FP8LM2023 | [FP8-LM](fp8_lm_2023.md) | FP8 LLM training | Shows LLM-scale FP8 mixed precision across gradients, optimizer states, and distributed components. |
| FP8STAB2024 | [To FP8 and Back Again](to_fp8_and_back_2024.md) | FP8 stability | Frames reduced precision as a training-stability problem, not only final quality. |
| SR2025 | [Stochastic Rounding for LLM Training](stochastic_rounding_llm_training_2025.md) | Rounding + optimizer precision | Shows rounding behavior and optimizer interaction are first-class precision-design choices. |
| LLMINT8 | [LLM.int8()](llm_int8_2022.md) | INT8 + 16-bit outlier path | Uses activation outlier structure to split most matmul work into INT8 while preserving outliers in higher precision. |
| SMOOTHQ | [SmoothQuant](smoothquant_2022.md) | W8A8 INT8 | Uses activation statistics to migrate quantization difficulty from activations to weights. |
| ZEROQ | [ZeroQuant](zeroquant_2022.md) | INT8 and mixed INT4/INT8 | Demonstrates module-wise low-bit choices and backend-aware inference constraints. |
| QLORA2023 | [QLoRA](qlora_2023.md) | 4-bit storage + higher-precision training path | Shows low-resource LoRA-style fine-tuning with aggressive base-model quantization. |
| FP4TRAIN | [FP4 All the Way](fp4_all_the_way_2025.md) | FP4 LLM training | Connects FP4 training success to scaling, rounding, and gradient-norm-versus-quantization-noise thresholds. |
| ATTNQAT | [Attn-QAT](attn_qat_2026.md) | 4-bit attention | Identifies attention as a high-risk low-precision region because of heavy-tailed activations and backward precision assumptions. |
| HAWQ | [HAWQ](hawq_2019.md) | Mixed precision by Hessian sensitivity | Provides a principled sensitivity-based answer to which layers deserve more bits. |
| HAWQV3 | [HAWQ-V3](hawq_v3_2021.md) | Mixed INT4/INT8 + hardware constraints | Frames precision assignment as constrained optimization over perturbation, memory, and latency. |
| ADAPT | [AdaPT](adapt_adaptive_precision_training_2021.md) | Dynamic fixed-point training | Explicitly studies adaptive precision during training rather than only post-training inference. |
| CONVOP | [Convergence-Aware Operator-Wise Mixed-Precision Training](convergence_aware_operatorwise_mixed_precision_2025.md) | Operator-wise training precision | Emphasizes convergence-aware operator precision decisions under multiple hardware formats. |
| SNIP | [SNIP](snip_adaptive_subbyte_llm_training_2026.md) | Adaptive subbyte LLM training | Closest match to this project: collects activation/gradient/optimizer statistics and optimizes layer-wise subbyte precision. |
| BITNET | [BitNet b1.58](bitnet_b158_2024.md) | Native ternary LLMs | Boundary context: low-bit success may require architecture-native design rather than post-hoc dtype assignment. |
| LORA2021 | [LoRA](lora_2021.md) | Adapter fine-tuning | Defines the parameter-efficient fine-tuning setting for the local experiments. |
| RMSNORM2019 | [RMSNorm](rmsnorm_2019.md) | Normalization | Explains the operation targeted by fp32-norm precision islands. |

## Synthesis

The literature converges on a simple but important principle: low precision works when the policy preserves the numerically sensitive parts of the computation. FP16 mixed precision needed FP32 master weights and loss scaling. BF16 simplified training by preserving FP32-like exponent range. FP8 work introduces format selection and per-role policy choices. INT8 and INT4/FP4 work makes the same theme sharper: some tensors, channels, layers, or operations are tolerant, while outliers, attention, optimizer updates, or reductions may need special treatment.

For LLMs, the most useful evidence comes from papers that expose a measurable reason for precision sensitivity. LLM.int8() and SmoothQuant show activation outliers are central for INT8 inference. FP4 training work highlights quantization-noise-to-gradient-norm ratios and rounding mode. Attn-QAT shows attention can be unstable when forward and backward precision assumptions are inconsistent. SNIP gives the closest high-level template for this project: periodically collect activation, gradient, and optimizer-state statistics, estimate quality loss from lower precision, and optimize precision choices under efficiency constraints.

LoRA fine-tuning changes the transfer story. Most large-scale papers study pretraining or inference, while this project freezes the base model and updates small adapters. Precision sensitivity may therefore concentrate in adapter gradients, optimizer states, normalization, logits/loss, and activation paths through the frozen model rather than full base-weight updates. That makes the local pilot feasible, but also limits how directly H100-scale FP8/FP4 results transfer.

The current H1/H6 plan fits the literature. H1 tests a hand-selected precision island, fp32 normalization, as a cheap first calibration. H6 generalizes the goal: derive precision assignment from observed stability signals, then compare the derived policy against static baselines under a fixed training budget.

## Open Problems

- **Signal validity:** It is unclear which cheap signals best predict precision sensitivity during LoRA fine-tuning: activation outliers, gradient spikes, loss deltas, quantization error, saturation/clipping, update divergence, or combinations.
- **Training versus inference transfer:** INT8/INT4 inference evidence may not predict training stability because gradients, optimizer states, and backward recomputation create different failure modes.
- **Adapter-specific sensitivity:** LoRA freezes most weights, so sensitivity may differ from full pretraining. Existing FP8/FP4 work rarely isolates adapter gradients and adapter optimizer state precision.
- **Normalization and loss precision:** RMSNorm/LayerNorm, logits, and cross-entropy are plausible stability bottlenecks, but the literature does not isolate them in small BF16 LoRA fine-tuning.
- **Attention under subbyte precision:** Recent work suggests attention is a high-risk FP4/INT4 target, but there is no lightweight test for whether LoRA attention paths can be safely demoted.
- **Boundary versus internal dtype:** Module input/output dtype may not reveal internal accumulator dtype, especially for fused kernels, RMSNorm, attention, or loss functions.
- **Hardware realism:** Fake quantization or dtype emulation may detect sensitivity but fail to predict memory or throughput benefits on RTX-class hardware without matching kernels.
- **Policy freezing:** Adaptive policies can overfit calibration traces unless decisions are frozen before final validation comparison.

## Evidence Gaps

- No full 1,000-step H1 baseline-versus-fp32-norm run has been completed and analyzed.
- No multi-seed estimate exists for BF16 baseline, fp32 norms, or any adaptive policy.
- The dtype probe verifies module-boundary dtypes but not internal RMSNorm reduction accumulator dtype.
- No per-module activation outlier, gradient spike, quantization error, clipping/saturation, or loss-delta traces exist yet.
- No calibration procedure has ranked candidate precision islands by measured stability sensitivity.
- No adaptive precision policy has been compared against matched static policies under a fixed training budget.
- No stressed setting has tested whether precision effects only emerge under high learning rate, FP16, longer context, tiny batch, or low-bit perturbation.
- No local evidence shows that INT8/INT4/FP4 candidates produce real speed or memory improvements on the available hardware.

## Implications For The Next Research Step

The literature supports a staged path rather than jumping directly to aggressive INT4 or FP4 training. First, complete H1 and H5-style instrumentation so the project has trustworthy BF16/FP32 calibration data. Next, add per-module signals that are cheap enough to collect during short LoRA runs. Then derive a frozen adaptive policy and compare it against BF16 baseline plus hand-selected fp32 islands. INT8/INT4/FP4 candidates should initially be treated as perturbation probes unless matching kernels are available.
