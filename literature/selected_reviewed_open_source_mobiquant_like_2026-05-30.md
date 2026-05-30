# Selected Reviewed + Open-Source MoBiQuant-Like Papers

This file selects papers that are both close to the soft-pruning-style
mixed-precision topic and have stronger review signals, preferably with public
code. The ranking favors main-conference acceptance and reproducibility over
perfect topical match.

## Tier 1: Best Main Evidence

| Rank | Paper | Review signal | Code | Why selected |
|---|---|---|---|---|
| 1 | DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment | NeurIPS 2025 poster / OpenReview | https://github.com/SNU-ARC/DP-LLM | Closest reviewed and open-source match to runtime adaptive precision: input-conditioned layer-wise bitwidth selection. |
| 2 | Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs | ICML 2024 oral | https://github.com/SNU-ARC/any-precision-llm | Strong reviewed storage answer: multiple bitwidth LLMs overlaid into one memory-efficient representation with custom kernels. |
| 3 | NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs | NeurIPS 2025 / OpenReview | https://github.com/SNU-ARC/NestedFP | Strong reviewed systems answer for nested FP16/FP8 serving without duplicate model copies. |
| 4 | SliM-LLM: Salience-Driven Mixed-Precision Quantization for LLMs | ICML 2025 poster / OpenReview | https://github.com/Aaronhuang-778/SliM-LLM | Best reviewed LLM-specific salience-to-bitwidth paper; less dynamic than MoBiQuant but very aligned with importance-based mixed precision. |
| 5 | Bayesian Bits: Unifying Quantization and Pruning | NeurIPS 2020 with public reviews | https://github.com/Qualcomm-AI-research/BayesianBits | Best reviewed open-source conceptual bridge from soft pruning to bit allocation, including a 0-bit pruning option. |

## Tier 2: Good Supporting Evidence

| Paper | Review signal | Code | Why supporting rather than core |
|---|---|---|---|
| QuEPT: Quantized Elastic Precision Transformers | AAAI 2026 | https://github.com/xuke225/QuEPT | Highly relevant elastic multi-bit switching; useful, but less central than DP-LLM/Any-Precision for the prompt/storage framing. |
| HAQ: Hardware-Aware Automated Quantization with Mixed Precision | CVPR 2019 oral | https://github.com/mit-han-lab/haq | Excellent hardware-aware assignment baseline, but older and not LLM/token-adaptive. |
| Instance-Aware Dynamic Neural Network Quantization | CVPR 2022 | https://github.com/huawei-noah/Efficient-Computing | Strong input-conditioned precision precedent, but vision-focused rather than LLM-focused. |
| DiffQ | Published paper + official implementation | https://github.com/facebookresearch/diffq | Strong differentiable bit-allocation implementation, but not LLM-specific and weaker venue signal than Tier 1. |

## Tier 3: Relevant But Lower Confidence

| Paper | Status | Why not core |
|---|---|---|
| MoBiQuant | arXiv 2026, no code found | Most conceptually aligned with token-adaptive residual bit slices, but lower review/reproducibility confidence. |
| QAQ: Query-adaptive Mixed-precision Quantization | NeurIPS 2025 ML for Systems workshop, no code found | Very aligned with query-conditioned bit-plane activation and CPU/GPU loading, but workshop-level and not reproducible yet. |
| Prompt-Adaptive Quantization | AAAI 2026 AIR-FM workshop, no code found | Useful prompt-router baseline over 2/4/8/16-bit model variants, but stores/routes separate quantized models rather than solving compact nested storage. |
| MatGPTQ | arXiv 2026, code available | Strong storage/mixed-kernel idea, but currently preprint-level. |
| ARKV | arXiv 2026, code available | Excellent soft-pruning analogy for KV cache states, but preprint-level and cache-side rather than weight-side. |
| QAQ KV Cache | ICCV 2025 workshop, code available | Useful cache-side adaptive precision, but workshop-level. |
| Don't Waste Bits | arXiv 2026, no code found | Very relevant per-token KV bitwidth policy, but no code/review signal found. |

## Recommended Set For A Proposal

Use these as the main related-work spine:

1. **Bayesian Bits** for the soft-pruning equivalence:
   `0-bit = prune`, low-bit = coarse keep, high-bit = precise keep.
2. **SliM-LLM** for reviewed LLM salience-driven bit allocation.
3. **DP-LLM** for reviewed runtime/input-conditioned precision selection.
4. **Any-Precision LLM** for reviewed multi-bitwidth storage and serving.
5. **NestedFP** for reviewed nested precision systems support.

Then mention **MoBiQuant** as the closest emerging paper to the proposed idea,
but be explicit that it currently appears weaker on review/code availability.

## Objective-Alignment Audit

The user's objective is specifically soft-pruning-style mixed precision:

```text
importance signal -> number of retained bits -> storage/layout that supports
heterogeneous or elastic bitwidths
```

Under that stricter definition, the papers should not be treated as equally
central:

| Paper | Objective fit | Use in the project |
|---|---|---|
| Bayesian Bits | Very high | Core conceptual anchor. It explicitly unifies pruning and quantization with a 0-bit option and learned residual-bit gates. |
| SliM-LLM | Very high | Core LLM anchor. It assigns mixed precision from salience, which directly matches "important information gets more bits." |
| DP-LLM | High | Dynamic precision anchor. It chooses layer bitwidth at runtime from input-conditioned error, but it is less explicitly a soft-pruning method and less focused on storage disentanglement. |
| Any-Precision LLM | Medium-high | Storage/serving anchor. It solves the multi-bitwidth storage problem, but it does not decide bitwidth from prompt importance by itself. |
| NestedFP | Medium | Systems/storage anchor for nested FP16/FP8 serving. Useful for layout, but only dual precision and not importance-based soft pruning. |
| QuEPT | Medium-high | Elastic precision support. Useful if the proposal needs real-time multi-bit switching, but less directly tied to soft pruning or prompt importance. |
| HAQ | Medium | Hardware-aware bit allocation baseline. Important history, but not LLM/prompt-adaptive and not soft pruning. |
| Instance-Aware Dynamic Quantization | Medium | Input-conditioned bitwidth precedent. Strong review signal, but vision-focused and not LLM storage. |
| DiffQ | High conceptually, medium empirically | Differentiable bit allocation is highly aligned, but it is not LLM-specific and should support rather than lead the LLM argument. |
| MoBiQuant | Very high conceptually, lower confidence | Closest to token-adaptive residual-bit soft pruning, but currently lower review/code confidence. |

The safest core for the objective is therefore:

1. **Bayesian Bits**: soft pruning becomes bit allocation.
2. **SliM-LLM**: LLM salience determines mixed precision.
3. **DP-LLM**: runtime input-conditioned precision.
4. **Any-Precision LLM / NestedFP**: storage systems for elastic precision.
5. **MoBiQuant**: closest emerging target to beat or extend.
