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

